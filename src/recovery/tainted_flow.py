"""
src/recovery/tainted_flow.py
============================

Time-Decaying Tainted Proportional Flow (TD-TPF) for post-detection fund tracing.

Setting
-------
A confirmed fraud ``e_0 = (u_0, v_0, A_0, tau_0)`` bypassed pre-authorisation and
is detected at ``tau_detect <= tau_0 + W_rec`` (``W_rec = 24 h``). The tracker
replays every directed edge ``e' = (v_a, v_b, A', tau')`` with
``tau_0 <= tau' <= tau_detect`` in chronological order and propagates taint
forward from the beneficiary ``v_0``.

Taint is a *mass* in monetary units, seeded as ``T(v_0, tau_0) = A_0``. When
``v_a`` (holding taint received at ``tau``) sends ``A'`` at ``tau'``, the taint
delivered to ``v_b`` is::

    T(v_b, tau') = T(v_a, tau) * [ A' / (Balance(v_a) + A') ] * exp(-kappa (tau' - tau))

where ``Balance(v_a)`` is the ledger balance of ``v_a`` *after* the debit, so
``Balance(v_a) + A'`` is its pre-transfer balance and the bracket is the
pro-rata ("haircut") share of ``v_a``'s holdings that left. The undecayed mass
``T(v_a) * share`` is debited from ``v_a``; the factor ``exp(-kappa dt)`` is the
attribution-confidence decay - taint that has rested and been commingled for
``dt`` hours is attributed less confidently, and the difference is booked as
``attribution_loss``. A node that receives taint at several times keeps one
*lot* per arrival, each decaying from its own arrival time (for a single chain
this reduces exactly to the formula above).

Termination
-----------
Tracking stops on an edge that moves funds out of the reachable ledger:

* PaySim: ``CASH_OUT`` (cash leaves the system); payments to merchants
  (``nameDest`` starting with ``M``) are recorded as *merchant sinks* - out of
  freeze reach but eligible for card-network chargeback.
* AMLSim: ``CASH_OUT`` and external wires (``WIRE`` / destinations in the
  external-account set).

Inflow types (``CASH_IN`` / ``DEPOSIT``) credit the *originator* (PaySim
semantics) and never carry taint.

Conservation
------------
For every trace (checked by :meth:`TaintTrace.conservation_residual`)::

    A_0 = held + exited + attribution_loss + pruned + beyond_horizon

Freezable value at ``v`` is the liquid balance attributable to taint,
``min(T(v), Balance(v))``.

Leakage note
------------
Ledger balances are used only in this *post-detection* forensic stage (the
issuer can read its own ledger once a fraud is confirmed); they are never fed
to the pre-authorisation detector, so Phase 1's causal-leakage guarantee is
unaffected.
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPERPARAMS_PATH = ROOT / "configs" / "hyperparams.json"
DEFAULT_PROJECT_CONFIG_PATH = ROOT / "configs" / "project_config.json"

EDGE_INTERNAL = "internal"
EDGE_CASH_OUT = "cash_out"
EDGE_EXTERNAL_WIRE = "external_wire"
EDGE_MERCHANT = "merchant"
EDGE_INFLOW = "inflow"
EXIT_KINDS: Tuple[str, ...] = (EDGE_CASH_OUT, EDGE_EXTERNAL_WIRE, EDGE_MERCHANT)
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Edges and dataset schemas                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FlowEdge:
    """Directed transaction edge ``(src -> dst, amount, tau)``; ``tau`` in hours."""

    edge_id: int
    src: str
    dst: str
    amount: float
    tau: float
    tx_type: str = "TRANSFER"
    label: int = -1

    def __post_init__(self) -> None:
        if not (self.amount >= 0.0 and math.isfinite(self.amount)):
            raise ValueError(f"edge {self.edge_id}: amount must be finite and >= 0")
        if not math.isfinite(self.tau):
            raise ValueError(f"edge {self.edge_id}: tau must be finite")


@dataclass(frozen=True)
class EdgeSchema:
    """How a dataset's transaction types map onto TD-TPF edge kinds."""

    name: str
    hours_per_step: float
    exit_types: FrozenSet[str]
    wire_types: FrozenSet[str] = frozenset()
    inflow_types: FrozenSet[str] = frozenset()
    merchant_prefixes: Tuple[str, ...] = ()
    wire_is_external: bool = True

    @classmethod
    def paysim(cls) -> "EdgeSchema":
        return cls("paysim", 1.0, frozenset({"CASH_OUT"}), frozenset(), frozenset({"CASH_IN"}), ("M",))

    @classmethod
    def amlsim(cls, wire_is_external: bool = True) -> "EdgeSchema":
        return cls("amlsim", 24.0, frozenset({"CASH_OUT"}), frozenset({"WIRE"}),
                   frozenset({"CASH_IN", "DEPOSIT"}), (), wire_is_external)

    @classmethod
    def generic(cls, hours_per_step: float = 1.0) -> "EdgeSchema":
        return cls("generic", float(hours_per_step), frozenset({"CASH_OUT"}), frozenset({"WIRE"}),
                   frozenset({"CASH_IN", "DEPOSIT"}), ())

    @classmethod
    def by_name(cls, name: str) -> "EdgeSchema":
        key = name.strip().lower()
        if key == "paysim":
            return cls.paysim()
        if key == "amlsim":
            return cls.amlsim()
        if key in ("generic", "canonical"):
            return cls.generic()
        raise ValueError(f"unknown edge schema '{name}'")

    def classify(self, edge: FlowEdge, external_accounts: FrozenSet[str] = frozenset()) -> str:
        """Edge kind: internal, inflow, cash_out, external_wire or merchant."""
        t = str(edge.tx_type).upper()
        if t in self.inflow_types:
            return EDGE_INFLOW
        if t in self.exit_types:
            return EDGE_CASH_OUT
        if edge.dst in external_accounts:
            return EDGE_EXTERNAL_WIRE
        if t in self.wire_types and self.wire_is_external:
            return EDGE_EXTERNAL_WIRE
        if self.merchant_prefixes and str(edge.dst).startswith(self.merchant_prefixes):
            return EDGE_MERCHANT
        return EDGE_INTERNAL


# --------------------------------------------------------------------------- #
# Trace containers                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class TaintLot:
    """Taint mass held by a node, received at ``tau_in`` after ``hop`` transfers."""

    amount: float
    tau_in: float
    hop: int
    path: Tuple[int, ...]


@dataclass(frozen=True)
class TaintedEdge:
    """One taint-carrying transfer and its TD-TPF factors."""

    edge: FlowEdge
    kind: str
    hop: int
    taint_before: float
    share: float
    decay: float
    moved: float
    delivered: float
    balance_after_debit: float
    propagated: bool = False


@dataclass(frozen=True)
class ExitFlow:
    """Taint that left the reachable ledger (cash-out, external wire or merchant)."""

    edge: FlowEdge
    kind: str
    amount: float
    hop: int


@dataclass
class TaintTrace:
    """Result of one TD-TPF traversal."""

    fraud_edge: FlowEdge
    tau_detect: float
    kappa: float
    window_hours: float
    lots: Dict[str, List[TaintLot]]
    balances: Dict[str, float]
    tainted_edges: List[TaintedEdge]
    exit_flows: List[ExitFlow]
    attribution_loss: float
    pruned: float
    beyond_horizon: float
    ledger_deficit: float
    n_edges_scanned: int

    # -- per-node views ------------------------------------------------------ #
    @property
    def victim(self) -> str:
        return self.fraud_edge.src

    @property
    def beneficiary(self) -> str:
        return self.fraud_edge.dst

    @property
    def initial_taint(self) -> float:
        return float(self.fraud_edge.amount)

    def node_taint(self) -> Dict[str, float]:
        """Attributed taint mass ``T(v, tau_detect)`` per node."""
        out = {v: sum(l.amount for l in lots) for v, lots in self.lots.items()}
        return {v: t for v, t in sorted(out.items()) if t > _EPS}

    def taint_concentration(self) -> Dict[str, float]:
        """``T(v) / Balance(v)`` (1.0 = wallet holds only tainted value)."""
        out: Dict[str, float] = {}
        for v, t in self.node_taint().items():
            b = self.balances.get(v, 0.0)
            out[v] = min(1.0, t / b) if b > _EPS else 0.0
        return out

    def min_hop(self) -> Dict[str, int]:
        hops: Dict[str, int] = {self.beneficiary: 0}
        for te in self.tainted_edges:
            if te.propagated:
                d = te.edge.dst
                hops[d] = min(hops.get(d, te.hop), te.hop)
        return dict(sorted(hops.items()))

    def liquid_recoverable(self) -> Dict[str, float]:
        """Freezable tainted value ``min(T(v), Balance(v))`` per mule wallet (victim excluded)."""
        out: Dict[str, float] = {}
        for v, t in self.node_taint().items():
            if v == self.victim:
                continue
            amt = min(t, max(self.balances.get(v, 0.0), 0.0))
            if amt > _EPS:
                out[v] = amt
        return out

    def returned_to_victim(self) -> float:
        """Taint that flowed back to the victim's own account (no action needed)."""
        t = sum(l.amount for l in self.lots.get(self.victim, []))
        return min(t, max(self.balances.get(self.victim, 0.0), 0.0))

    # -- aggregates ---------------------------------------------------------- #
    @property
    def held(self) -> float:
        return float(sum(l.amount for lots in self.lots.values() for l in lots))

    def exited(self) -> Dict[str, float]:
        out = {k: 0.0 for k in EXIT_KINDS}
        for f in self.exit_flows:
            out[f.kind] += f.amount
        return out

    @property
    def exited_total(self) -> float:
        return float(sum(f.amount for f in self.exit_flows))

    def conservation_residual(self) -> float:
        """``A_0 - (held + exited + attribution_loss + pruned + beyond_horizon)`` (should be ~0)."""
        return self.initial_taint - (self.held + self.exited_total + self.attribution_loss
                                     + self.pruned + self.beyond_horizon)

    def metrics(self) -> Dict[str, float]:
        """Recovered Amount ($) and Recovery Rate (%) under freeze-all-liquid at ``tau_detect``."""
        a0 = self.initial_taint
        recovered = sum(self.liquid_recoverable().values()) + self.returned_to_victim()
        exited = self.exited()
        return {
            "fraud_amount": a0,
            "recovered_amount": recovered,
            "recovery_rate_pct": 100.0 * recovered / a0 if a0 > 0 else 0.0,
            "held_taint": self.held,
            "exited_cash_out": exited[EDGE_CASH_OUT],
            "exited_external_wire": exited[EDGE_EXTERNAL_WIRE],
            "exited_merchant": exited[EDGE_MERCHANT],
            "exit_rate_pct": 100.0 * self.exited_total / a0 if a0 > 0 else 0.0,
            "attribution_loss": self.attribution_loss,
            "pruned": self.pruned,
            "beyond_horizon": self.beyond_horizon,
            "n_mule_wallets": float(len(self.liquid_recoverable())),
            "max_hop": float(max(self.min_hop().values())),
            "detection_delay_hours": self.tau_detect - self.fraud_edge.tau,
            "conservation_residual": self.conservation_residual(),
        }

    def edges_frame(self) -> pd.DataFrame:
        rows = [{"edge_id": te.edge.edge_id, "src": te.edge.src, "dst": te.edge.dst,
                 "amount": te.edge.amount, "tau": te.edge.tau, "tx_type": te.edge.tx_type,
                 "kind": te.kind, "hop": te.hop, "taint_before": te.taint_before,
                 "share": te.share, "decay": te.decay, "moved": te.moved,
                 "delivered": te.delivered, "balance_after_debit": te.balance_after_debit,
                 "propagated": te.propagated}
                for te in self.tainted_edges]
        return pd.DataFrame(rows)

    def nodes_frame(self) -> pd.DataFrame:
        taint, conc, liquid, hops = (self.node_taint(), self.taint_concentration(),
                                     self.liquid_recoverable(), self.min_hop())
        rows = [{"node": v, "taint": t, "balance": self.balances.get(v, 0.0),
                 "concentration": conc.get(v, 0.0), "liquid_recoverable": liquid.get(v, 0.0),
                 "min_hop": hops.get(v, -1), "is_victim": v == self.victim}
                for v, t in taint.items()]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tracker                                                                     #
# --------------------------------------------------------------------------- #
class TaintedFlowTracker:
    """TD-TPF traversal over a chronologically indexed directed multigraph.

    Parameters
    ----------
    edges : transaction edges (any order; sorted internally by ``(tau, edge_id)``).
    schema : :class:`EdgeSchema` or its name (``paysim``, ``amlsim``, ``generic``).
    kappa : decay rate per hour in ``exp(-kappa (tau' - tau))``.
    window_hours : recovery window ``W_rec``.
    min_taint : lots below ``min_taint * A_0`` are pruned (booked in ``pruned``).
    max_hops : taint is not propagated beyond this many transfers from ``v_0``.
    opening_balances : ledger balances just before ``tau_0`` (default 0).
    external_accounts : destinations treated as external wires.
    """

    def __init__(self, edges: Iterable[FlowEdge], schema: Union[EdgeSchema, str] = "generic",
                 kappa: float = 0.10, window_hours: float = 24.0, min_taint: float = 0.01,
                 max_hops: int = 6, opening_balances: Optional[Mapping[str, float]] = None,
                 external_accounts: Iterable[str] = ()) -> None:
        if kappa < 0.0 or not math.isfinite(kappa):
            raise ValueError("kappa must be finite and >= 0")
        if not window_hours > 0.0:
            raise ValueError("window_hours must be > 0")
        if not 0.0 <= min_taint < 1.0:
            raise ValueError("min_taint must lie in [0, 1)")
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        self.schema = EdgeSchema.by_name(schema) if isinstance(schema, str) else schema
        self.kappa = float(kappa)
        self.window_hours = float(window_hours)
        self.min_taint = float(min_taint)
        self.max_hops = int(max_hops)
        self.opening_balances: Dict[str, float] = {str(k): float(v) for k, v in
                                                   (opening_balances or {}).items()}
        self.external_accounts: FrozenSet[str] = frozenset(str(a) for a in external_accounts)
        ordered = sorted(edges, key=lambda e: (e.tau, e.edge_id))
        ids = [e.edge_id for e in ordered]
        if len(set(ids)) != len(ids):
            raise ValueError("edge_id values must be unique")
        self._edges: List[FlowEdge] = ordered
        self._taus: List[float] = [e.tau for e in ordered]
        self._pos: Dict[int, int] = {e.edge_id: i for i, e in enumerate(ordered)}

    # -- construction ------------------------------------------------------- #
    @staticmethod
    def config_kwargs(hyperparams: Optional[Mapping] = None,
                      project_config: Optional[Mapping] = None) -> Dict[str, float]:
        """TD-TPF keyword arguments from ``hyperparams.json['tdtpf']``."""
        if hyperparams is None:
            hyperparams = json.loads(DEFAULT_HYPERPARAMS_PATH.read_text(encoding="utf-8"))
        td = dict(hyperparams.get("tdtpf", {}))
        window = td.get("recovery_window_hours")
        if window is None and project_config is not None:
            window = project_config.get("recovery", {}).get("settlement_window_hours")
        return {"kappa": float(td.get("kappa_decay_per_hour", 0.10)),
                "window_hours": float(window if window is not None else 24.0),
                "min_taint": float(td.get("min_taint", 0.01)),
                "max_hops": int(td.get("max_hops", 6))}

    @classmethod
    def from_config(cls, edges: Iterable[FlowEdge], schema: Union[EdgeSchema, str] = "generic",
                    hyperparams: Optional[Mapping] = None, project_config: Optional[Mapping] = None,
                    **overrides) -> "TaintedFlowTracker":
        kwargs: Dict[str, object] = dict(cls.config_kwargs(hyperparams, project_config))
        kwargs.update(overrides)
        return cls(edges, schema, **kwargs)

    @staticmethod
    def edges_from_paysim(frame: pd.DataFrame) -> List[FlowEdge]:
        """Raw PaySim log (``step, type, amount, nameOrig, nameDest[, isFraud]``); 1 step = 1 h."""
        label = frame["isFraud"].to_numpy() if "isFraud" in frame.columns else np.full(len(frame), -1)
        return [FlowEdge(int(i), str(s), str(d), float(a), float(t), str(ty), int(y))
                for i, s, d, a, t, ty, y in zip(range(len(frame)), frame["nameOrig"], frame["nameDest"],
                                                frame["amount"], frame["step"], frame["type"], label)]

    @staticmethod
    def edges_from_amlsim(frame: pd.DataFrame) -> List[FlowEdge]:
        """Raw AMLSim log (v1 ``tx.csv`` or v2 ``transactions.csv``; aliases as in Phase 1).

        Integer timestamps are simulation days (x 24 h); datetime strings are
        converted to hours since the earliest timestamp.
        """
        from src.data.dataset_loaders import AMLSimLoader
        s = AMLSimLoader._resolve_schema(list(frame.columns))
        raw_t = frame[s["time"]]
        numeric = pd.to_numeric(raw_t, errors="coerce")
        if numeric.isna().any():
            ts = pd.to_datetime(raw_t, utc=True)
            tau = ((ts - ts.min()).dt.total_seconds() / 3600.0).to_numpy(dtype=np.float64)
        else:
            tau = numeric.to_numpy(dtype=np.float64) * 24.0
        ids = frame[s["id"]].to_numpy() if "id" in s else np.arange(len(frame))
        types = frame[s["type"]].astype(str).to_numpy() if "type" in s else np.full(len(frame), "TRANSFER")
        labels = (AMLSimLoader._parse_label(frame[s["label"]]) if "label" in s
                  else np.full(len(frame), -1))
        return [FlowEdge(int(i), str(a), str(b), float(m), float(t), str(ty), int(y))
                for i, a, b, m, t, ty, y in zip(ids, frame[s["src"]], frame[s["dst"]],
                                                frame[s["amount"]], tau, types, labels)]

    @staticmethod
    def edges_from_canonical(frame: pd.DataFrame) -> List[FlowEdge]:
        """Phase 1 canonical stream (``tx_id, event_time [s], src, dst, amount, tx_type, label``)."""
        ids = frame["tx_id"].to_numpy() if "tx_id" in frame.columns else np.arange(len(frame))
        types = frame["tx_type"] if "tx_type" in frame.columns else ["TRANSFER"] * len(frame)
        labels = frame["label"] if "label" in frame.columns else [-1] * len(frame)
        return [FlowEdge(int(i), str(s), str(d), float(a), float(t) / 3600.0, str(ty), int(y))
                for i, s, d, a, t, ty, y in zip(ids, frame["src"], frame["dst"], frame["amount"],
                                                frame["event_time"], types, labels)]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, dataset: str, **kwargs) -> "TaintedFlowTracker":
        """Build from a raw ``paysim`` / ``amlsim`` log or a Phase 1 ``canonical`` frame."""
        key = dataset.strip().lower()
        if key == "paysim":
            return cls(cls.edges_from_paysim(frame), EdgeSchema.paysim(), **kwargs)
        if key == "amlsim":
            return cls(cls.edges_from_amlsim(frame), kwargs.pop("schema", EdgeSchema.amlsim()), **kwargs)
        if key == "canonical":
            return cls(cls.edges_from_canonical(frame), kwargs.pop("schema", "generic"), **kwargs)
        raise ValueError("dataset must be 'paysim', 'amlsim' or 'canonical'")

    @staticmethod
    def paysim_opening_balances(frame: pd.DataFrame, before_step: float) -> Dict[str, float]:
        """Issuer-ledger balances just before ``before_step`` from the raw PaySim balance columns.

        Post-detection forensic use only (see module docstring). For each account
        the latest post-transaction balance strictly before ``before_step`` is used.
        """
        cols = {"step", "nameOrig", "nameDest", "newbalanceOrig", "newbalanceDest"}
        missing = cols - set(frame.columns)
        if missing:
            raise KeyError(f"PaySim balance columns missing: {sorted(missing)}")
        prior = frame[frame["step"] < before_step]
        out: Dict[str, float] = {}
        for o, d, bo, bd in zip(prior["nameOrig"], prior["nameDest"], prior["newbalanceOrig"],
                                prior["newbalanceDest"]):
            out[str(o)] = float(bo)
            if not str(d).startswith("M"):
                out[str(d)] = float(bd)
        return out

    # -- queries ------------------------------------------------------------ #
    @property
    def edges(self) -> List[FlowEdge]:
        return list(self._edges)

    def edge(self, edge_id: int) -> FlowEdge:
        return self._edges[self._pos[int(edge_id)]]

    def window_edges(self, tau_start: float, tau_end: float) -> List[FlowEdge]:
        lo = bisect.bisect_left(self._taus, tau_start)
        hi = bisect.bisect_right(self._taus, tau_end)
        return self._edges[lo:hi]

    def decay(self, dt_hours: float) -> float:
        """``exp(-kappa dt)``."""
        if dt_hours < 0.0:
            raise ValueError("decay is defined for non-negative elapsed time")
        return math.exp(-self.kappa * dt_hours)

    # -- traversal ---------------------------------------------------------- #
    def trace(self, fraud: Union[FlowEdge, int], tau_detect: Optional[float] = None) -> TaintTrace:
        """Propagate taint from ``fraud`` over edges with ``tau_0 <= tau' <= tau_detect``."""
        e0 = self.edge(fraud) if isinstance(fraud, (int, np.integer)) else fraud
        if e0.edge_id not in self._pos:
            raise KeyError(f"fraud edge {e0.edge_id} is not part of the indexed graph")
        tau0 = e0.tau
        deadline = tau0 + self.window_hours
        tau_detect = deadline if tau_detect is None else float(tau_detect)
        if tau_detect < tau0 - _EPS:
            raise ValueError("tau_detect precedes the fraud transaction")
        if tau_detect > deadline + _EPS:
            raise ValueError(f"tau_detect exceeds the recovery window tau_0 + W_rec = {deadline}")
        if e0.amount <= 0.0:
            raise ValueError("fraud amount must be > 0")
        if self.schema.classify(e0, self.external_accounts) != EDGE_INTERNAL:
            raise ValueError("the fraud edge must credit an in-ledger beneficiary")

        min_abs = self.min_taint * e0.amount
        ledger: Dict[str, float] = {}
        lots: Dict[str, List[TaintLot]] = {}
        tainted: List[TaintedEdge] = []
        exits: List[ExitFlow] = []
        attribution_loss = pruned = beyond = deficit = 0.0
        start = self._pos[e0.edge_id]
        lo = bisect.bisect_left(self._taus, tau0)
        hi = bisect.bisect_right(self._taus, tau_detect + _EPS)
        n_scanned = 0

        def bal(v: str) -> float:
            if v not in ledger:
                ledger[v] = self.opening_balances.get(v, 0.0)
            return ledger[v]

        for i in range(lo, hi):
            e = self._edges[i]
            n_scanned += 1
            kind = self.schema.classify(e, self.external_accounts)
            if kind == EDGE_INFLOW:
                ledger[e.src] = bal(e.src) + e.amount
                continue
            if i < start:  # same-timestamp edges preceding e_0: ledger only
                post = bal(e.src) - e.amount
                if post < 0.0:
                    deficit += -post
                ledger[e.src] = max(post, 0.0)
                if kind == EDGE_INTERNAL:
                    ledger[e.dst] = bal(e.dst) + e.amount
                continue
            if i == start:
                post = bal(e.src) - e.amount
                if post < 0.0:
                    deficit += -post
                ledger[e.src] = max(post, 0.0)
                ledger[e.dst] = bal(e.dst) + e.amount
                lots.setdefault(e.dst, []).append(TaintLot(e.amount, e.tau, 0, (e.edge_id,)))
                continue

            pre = bal(e.src)
            post = pre - e.amount
            if post < 0.0:  # opening balance unknown / insufficient: treat as fully funded
                deficit += -post
                post = 0.0
            ledger[e.src] = post
            if kind == EDGE_INTERNAL:
                ledger[e.dst] = bal(e.dst) + e.amount

            src_lots = lots.get(e.src)
            if not src_lots or e.src == e.dst or e.amount <= 0.0:
                continue
            share = e.amount / (post + e.amount)
            keep: List[TaintLot] = []
            for lot in src_lots:
                taint_before = lot.amount
                moved = taint_before * share
                d = self.decay(e.tau - lot.tau_in)
                delivered = moved * d
                lot.amount = taint_before - moved
                attribution_loss += moved - delivered
                hop = lot.hop + 1
                propagated = False
                if kind in EXIT_KINDS:
                    exits.append(ExitFlow(e, kind, delivered, hop))
                elif hop > self.max_hops:
                    beyond += delivered
                elif delivered < min_abs:
                    pruned += delivered
                else:
                    lots.setdefault(e.dst, []).append(TaintLot(delivered, e.tau, hop,
                                                               lot.path + (e.edge_id,)))
                    propagated = True
                tainted.append(TaintedEdge(e, kind, hop, taint_before, share, d, moved, delivered,
                                           post, propagated))
                if lot.amount >= min_abs:
                    keep.append(lot)
                else:
                    pruned += lot.amount
            lots[e.src] = keep

        for v in list(lots):
            if not lots[v]:
                del lots[v]
        balances = {v: ledger.get(v, self.opening_balances.get(v, 0.0)) for v in lots}
        balances.update({v: b for v, b in ledger.items()})
        return TaintTrace(e0, tau_detect, self.kappa, self.window_hours, lots, balances, tainted,
                          exits, attribution_loss, pruned, beyond, deficit, n_scanned)


__all__ = [
    "EDGE_CASH_OUT", "EDGE_EXTERNAL_WIRE", "EDGE_INFLOW", "EDGE_INTERNAL", "EDGE_MERCHANT",
    "EXIT_KINDS", "EdgeSchema", "ExitFlow", "FlowEdge", "TaintLot", "TaintTrace",
    "TaintedEdge", "TaintedFlowTracker",
]
