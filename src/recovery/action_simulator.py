"""
src/recovery/action_simulator.py
================================

Post-detection recovery protocol for MG-ECFM Stage 5, executed inside the
``W_rec = 24 h`` settlement window on top of the TD-TPF trace.

Protocol (all times in hours, ``tau_d`` = detection time)
--------------------------------------------------------
1. **Instrument block** at ``tau_b = tau_d + block_latency``: the victim's
   compromised instrument is blocked, so every *follow-on* fraudulent debit
   from ``u_0`` with ``tau >= tau_b`` (and inside the window) is prevented.
   Follow-on fraud settled before ``tau_b`` is added to the exposure.
2. **Account freeze** at ``tau_f = tau_d + freeze_latency``: the TD-TPF trace is
   evaluated at ``tau_f`` (funds that mules move during the reaction latency
   are followed, not assumed frozen) and each tainted mule wallet ``v`` is
   frozen for ``min(T(v), Balance(v)) * freeze_liquid_fraction``.
   ``freeze_policy="full_account"`` freezes the whole balance; the untainted
   part is collateral friction charged at ``collateral_rate``.
3. **Chargeback / recall** at ``tau_c = tau_d + chargeback_latency``: tainted
   value that reached merchants is disputed through the card network
   (``merchant_chargeback_rate``); external wires are recalled at
   ``wire_recall_rate``; cash-outs are irrecoverable (``cash_out_recovery_rate``).

An action is executed only if its time is ``<= tau_0 + W_rec``.

Metrics
-------
::

    Recovered          = frozen + charged_back/recalled + returned_to_victim
    Recovery Rate (%)  = 100 * Recovered / A_0
    Exposure           = A_0 + follow-on fraud attempted in (tau_0, tau_0 + W_rec]
    Prevented loss     = Recovered + follow-on fraud blocked
    Prevented-Loss Ratio (%) = 100 * Prevented loss / Exposure
    Net savings        = Prevented loss - operational cost - collateral penalty

``mode="expected"`` books expected values (deterministic); ``mode="stochastic"``
draws Bernoulli action successes from a seeded generator.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.recovery.tainted_flow import (
    EDGE_CASH_OUT,
    EDGE_EXTERNAL_WIRE,
    EDGE_MERCHANT,
    FlowEdge,
    TaintedFlowTracker,
    TaintTrace,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPERPARAMS_PATH = ROOT / "configs" / "hyperparams.json"
SEED = 42
ACTION_FREEZE = "account_freeze"
ACTION_BLOCK = "instrument_block"
ACTION_CHARGEBACK = "chargeback"
ACTION_RECALL = "wire_recall"
FREEZE_POLICIES = ("tainted_only", "full_account")
MODES = ("expected", "stochastic")


@dataclass(frozen=True)
class RecoveryPolicy:
    """Latencies, success rates and costs of the recovery protocol (monetary units of the dataset)."""

    window_hours: float = 24.0
    freeze_latency_hours: float = 0.5
    block_latency_hours: float = 0.25
    chargeback_latency_hours: float = 2.0
    freeze_liquid_fraction: float = 1.0
    freeze_success_rate: float = 1.0
    freeze_policy: str = "tainted_only"
    merchant_chargeback_rate: float = 0.70
    wire_recall_rate: float = 0.0
    cash_out_recovery_rate: float = 0.0
    freeze_cost: float = 10.0
    block_cost: float = 2.0
    chargeback_cost: float = 25.0
    collateral_rate: float = 0.02
    mode: str = "expected"
    seed: int = SEED

    def __post_init__(self) -> None:
        if not self.window_hours > 0.0:
            raise ValueError("window_hours must be > 0")
        for name in ("freeze_latency_hours", "block_latency_hours", "chargeback_latency_hours",
                     "freeze_cost", "block_cost", "chargeback_cost", "collateral_rate"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("freeze_liquid_fraction", "freeze_success_rate", "merchant_chargeback_rate",
                     "wire_recall_rate", "cash_out_recovery_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.freeze_policy not in FREEZE_POLICIES:
            raise ValueError(f"freeze_policy must be one of {FREEZE_POLICIES}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None,
                    project_config: Optional[Mapping] = None, **overrides) -> "RecoveryPolicy":
        if hyperparams is None:
            hyperparams = json.loads(DEFAULT_HYPERPARAMS_PATH.read_text(encoding="utf-8"))
        td = dict(hyperparams.get("tdtpf", {}))
        kwargs: Dict[str, object] = {}
        window = td.get("recovery_window_hours")
        if window is None and project_config is not None:
            window = project_config.get("recovery", {}).get("settlement_window_hours")
        if window is not None:
            kwargs["window_hours"] = float(window)
        if "freeze_liquid_fraction" in td:
            kwargs["freeze_liquid_fraction"] = float(td["freeze_liquid_fraction"])
        for key, val in dict(td.get("actions", {})).items():
            kwargs[key] = val
        if "seed" in hyperparams:
            kwargs["seed"] = int(hyperparams["seed"])
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass(frozen=True)
class RecoveryAction:
    """One executed (or attempted) recovery action."""

    kind: str
    target: str
    tau: float
    amount_targeted: float
    amount_recovered: float
    cost: float
    executed: bool
    succeeded: bool
    collateral_frozen: float = 0.0
    edge_id: Optional[int] = None


@dataclass
class RecoveryOutcome:
    """Result of the recovery protocol for one fraud ``e_0``."""

    fraud_edge: FlowEdge
    tau_detect: float
    deadline: float
    within_window: bool
    trace: Optional[TaintTrace]
    actions: List[RecoveryAction] = field(default_factory=list)
    frozen: float = 0.0
    charged_back: float = 0.0
    recalled: float = 0.0
    returned_to_victim: float = 0.0
    follow_on_exposure: float = 0.0
    follow_on_blocked: float = 0.0
    operational_cost: float = 0.0
    collateral_frozen: float = 0.0
    collateral_penalty: float = 0.0

    @property
    def fraud_amount(self) -> float:
        return float(self.fraud_edge.amount)

    @property
    def recovered(self) -> float:
        return self.frozen + self.charged_back + self.recalled + self.returned_to_victim

    @property
    def recovery_rate_pct(self) -> float:
        return 100.0 * self.recovered / self.fraud_amount

    @property
    def exposure(self) -> float:
        return self.fraud_amount + self.follow_on_exposure

    @property
    def prevented_loss(self) -> float:
        return self.recovered + self.follow_on_blocked

    @property
    def prevented_loss_ratio_pct(self) -> float:
        return 100.0 * self.prevented_loss / self.exposure

    @property
    def residual_loss(self) -> float:
        return self.exposure - self.prevented_loss

    @property
    def net_savings(self) -> float:
        return self.prevented_loss - self.operational_cost - self.collateral_penalty

    def metrics(self) -> Dict[str, float]:
        return {
            "fraud_edge_id": float(self.fraud_edge.edge_id),
            "fraud_amount": self.fraud_amount,
            "detection_delay_hours": self.tau_detect - self.fraud_edge.tau,
            "within_window": float(self.within_window),
            "frozen": self.frozen,
            "charged_back": self.charged_back,
            "recalled": self.recalled,
            "returned_to_victim": self.returned_to_victim,
            "recovered_amount": self.recovered,
            "recovery_rate_pct": self.recovery_rate_pct,
            "follow_on_exposure": self.follow_on_exposure,
            "follow_on_blocked": self.follow_on_blocked,
            "exposure": self.exposure,
            "prevented_loss": self.prevented_loss,
            "prevented_loss_ratio_pct": self.prevented_loss_ratio_pct,
            "residual_loss": self.residual_loss,
            "operational_cost": self.operational_cost,
            "collateral_frozen": self.collateral_frozen,
            "collateral_penalty": self.collateral_penalty,
            "net_savings": self.net_savings,
            "n_accounts_frozen": float(sum(1 for a in self.actions
                                           if a.kind == ACTION_FREEZE and a.succeeded)),
        }

    def actions_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(a) for a in self.actions])


class RecoverySimulator:
    """Simulated freeze / instrument-block / chargeback protocol over a :class:`TaintedFlowTracker`."""

    def __init__(self, tracker: TaintedFlowTracker, policy: Optional[RecoveryPolicy] = None) -> None:
        self.tracker = tracker
        self.policy = policy or RecoveryPolicy(window_hours=tracker.window_hours)
        if abs(self.policy.window_hours - tracker.window_hours) > 1e-9:
            raise ValueError("policy and tracker must share the same recovery window")
        self._rng = np.random.default_rng(self.policy.seed)

    @classmethod
    def from_config(cls, tracker: TaintedFlowTracker, hyperparams: Optional[Mapping] = None,
                    project_config: Optional[Mapping] = None, **overrides) -> "RecoverySimulator":
        overrides.setdefault("window_hours", tracker.window_hours)
        return cls(tracker, RecoveryPolicy.from_config(hyperparams, project_config, **overrides))

    def reset_rng(self) -> None:
        self._rng = np.random.default_rng(self.policy.seed)

    # ----------------------------------------------------------------- helpers
    def _success(self, p: float) -> bool:
        if self.policy.mode == "expected":
            return p > 0.0
        return bool(self._rng.random() < p)

    def _realised(self, amount: float, p: float) -> float:
        """Expected value ``p * amount`` or a Bernoulli all-or-nothing draw."""
        if self.policy.mode == "expected":
            return amount * p
        return amount if self._success(p) else 0.0

    def follow_on_fraud(self, fraud: FlowEdge) -> List[FlowEdge]:
        """Labelled fraudulent debits from the victim after ``e_0`` within the window."""
        deadline = fraud.tau + self.policy.window_hours
        return [e for e in self.tracker.window_edges(fraud.tau, deadline)
                if e.src == fraud.src and e.label == 1 and e.edge_id != fraud.edge_id
                and (e.tau, e.edge_id) > (fraud.tau, fraud.edge_id)]

    # --------------------------------------------------------------- protocol
    def simulate(self, fraud: Union[FlowEdge, int], tau_detect: float,
                 follow_on: Optional[Sequence[FlowEdge]] = None) -> RecoveryOutcome:
        """Run the protocol for fraud ``e_0`` detected at ``tau_detect``."""
        pol = self.policy
        e0 = self.tracker.edge(fraud) if isinstance(fraud, (int, np.integer)) else fraud
        tau_detect = float(tau_detect)
        if tau_detect < e0.tau:
            raise ValueError("tau_detect precedes the fraud transaction")
        deadline = e0.tau + pol.window_hours
        follow = list(self.follow_on_fraud(e0) if follow_on is None else follow_on)
        out = RecoveryOutcome(e0, tau_detect, deadline, tau_detect <= deadline, None)
        out.follow_on_exposure = float(sum(e.amount for e in follow))
        if not out.within_window:
            return out

        # 1. Instrument block --------------------------------------------- #
        tau_b = tau_detect + pol.block_latency_hours
        block_ok = tau_b <= deadline
        blocked = sum(e.amount for e in follow if e.tau >= tau_b) if block_ok else 0.0
        out.follow_on_blocked = float(blocked)
        out.actions.append(RecoveryAction(ACTION_BLOCK, e0.src, tau_b, out.follow_on_exposure,
                                          float(blocked), pol.block_cost if block_ok else 0.0,
                                          block_ok, block_ok))
        out.operational_cost += pol.block_cost if block_ok else 0.0

        # 2. Account freeze (trace evaluated at the freeze time) ------------ #
        tau_f = min(tau_detect + pol.freeze_latency_hours, deadline)
        freeze_ok = tau_detect + pol.freeze_latency_hours <= deadline
        trace = self.tracker.trace(e0, tau_detect=tau_f)
        out.trace = trace
        out.returned_to_victim = trace.returned_to_victim()
        taint = trace.node_taint()
        for node, liquid in trace.liquid_recoverable().items():
            targeted = liquid * pol.freeze_liquid_fraction
            balance = max(trace.balances.get(node, 0.0), 0.0)
            collateral = max(balance - taint[node], 0.0) if pol.freeze_policy == "full_account" else 0.0
            if not freeze_ok:
                out.actions.append(RecoveryAction(ACTION_FREEZE, node, tau_f, targeted, 0.0, 0.0,
                                                  False, False))
                continue
            got = self._realised(targeted, pol.freeze_success_rate)
            ok = got > 0.0
            col = collateral if ok else 0.0
            out.frozen += got
            out.collateral_frozen += col
            out.operational_cost += pol.freeze_cost
            out.actions.append(RecoveryAction(ACTION_FREEZE, node, tau_f, targeted, got,
                                              pol.freeze_cost, True, ok, col))
        out.collateral_penalty = pol.collateral_rate * out.collateral_frozen

        # 3. Chargeback / recall of value that already left the ledger ------ #
        tau_c = tau_detect + pol.chargeback_latency_hours
        cb_ok = tau_c <= deadline
        rates = {EDGE_MERCHANT: (ACTION_CHARGEBACK, pol.merchant_chargeback_rate),
                 EDGE_EXTERNAL_WIRE: (ACTION_RECALL, pol.wire_recall_rate),
                 EDGE_CASH_OUT: (ACTION_RECALL, pol.cash_out_recovery_rate)}
        for flow in trace.exit_flows:
            kind, rate = rates[flow.kind]
            if rate <= 0.0 or flow.amount <= 0.0:
                continue
            if not cb_ok:
                out.actions.append(RecoveryAction(kind, flow.edge.dst, tau_c, flow.amount, 0.0, 0.0,
                                                  False, False, edge_id=flow.edge.edge_id))
                continue
            got = self._realised(flow.amount, rate)
            if flow.kind == EDGE_MERCHANT:
                out.charged_back += got
            else:
                out.recalled += got
            out.operational_cost += pol.chargeback_cost
            out.actions.append(RecoveryAction(kind, flow.edge.dst, tau_c, flow.amount, got,
                                              pol.chargeback_cost, True, got > 0.0,
                                              edge_id=flow.edge.edge_id))
        return out

    # ------------------------------------------------------------- batches
    def simulate_many(self, frauds: Iterable[Union[FlowEdge, int]],
                      detection_delays_hours: Union[float, Sequence[float]]) -> pd.DataFrame:
        """One row of :meth:`RecoveryOutcome.metrics` per fraud (deterministic order)."""
        frauds = [self.tracker.edge(f) if isinstance(f, (int, np.integer)) else f for f in frauds]
        delays = ([float(detection_delays_hours)] * len(frauds) if np.isscalar(detection_delays_hours)
                  else [float(d) for d in detection_delays_hours])
        if len(delays) != len(frauds):
            raise ValueError("one detection delay per fraud is required")
        rows = [self.simulate(e, e.tau + d).metrics() for e, d in zip(frauds, delays)]
        return pd.DataFrame(rows)

    @staticmethod
    def aggregate(frame: pd.DataFrame) -> Dict[str, float]:
        """Portfolio-level Recovered ($), Recovery Rate (%), Prevented-Loss Ratio (%), net savings."""
        if frame.empty:
            return {"n_frauds": 0.0, "fraud_amount": 0.0, "recovered_amount": 0.0,
                    "recovery_rate_pct": 0.0, "prevented_loss_ratio_pct": 0.0, "net_savings": 0.0}
        a0 = float(frame["fraud_amount"].sum())
        exposure = float(frame["exposure"].sum())
        return {
            "n_frauds": float(len(frame)),
            "fraud_amount": a0,
            "recovered_amount": float(frame["recovered_amount"].sum()),
            "recovery_rate_pct": 100.0 * float(frame["recovered_amount"].sum()) / a0,
            "mean_recovery_rate_pct": float(frame["recovery_rate_pct"].mean()),
            "prevented_loss": float(frame["prevented_loss"].sum()),
            "prevented_loss_ratio_pct": 100.0 * float(frame["prevented_loss"].sum()) / exposure,
            "operational_cost": float(frame["operational_cost"].sum()),
            "net_savings": float(frame["net_savings"].sum()),
            "share_within_window": float(frame["within_window"].mean()),
        }

    def recovery_curve(self, fraud: Union[FlowEdge, int],
                       delays_hours: Sequence[float]) -> pd.DataFrame:
        """Recovery Rate / Prevented-Loss Ratio as a function of detection delay."""
        e0 = self.tracker.edge(fraud) if isinstance(fraud, (int, np.integer)) else fraud
        rows = []
        for d in delays_hours:
            m = self.simulate(e0, e0.tau + float(d)).metrics()
            rows.append({"delay_hours": float(d), "recovery_rate_pct": m["recovery_rate_pct"],
                         "prevented_loss_ratio_pct": m["prevented_loss_ratio_pct"],
                         "recovered_amount": m["recovered_amount"], "net_savings": m["net_savings"]})
        return pd.DataFrame(rows)


__all__ = [
    "ACTION_BLOCK", "ACTION_CHARGEBACK", "ACTION_FREEZE", "ACTION_RECALL", "RecoveryAction",
    "RecoveryOutcome", "RecoveryPolicy", "RecoverySimulator",
]
