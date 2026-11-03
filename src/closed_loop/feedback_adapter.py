"""
src/closed_loop/feedback_adapter.py
===================================

Closed-loop adapter: verified outcomes released by the
:class:`~src.closed_loop.verification_queue.VerificationQueue` drive

1. **CS-ACRC recalibration** (Phase 3, :class:`OnlineAdaptiveCRC`): every released
   label ``(a_t, y_t, A_t)`` is applied in release order, giving
   ``lambda_{t+1} = Pi[lambda_t + gamma (alpha - ell_t)]`` (threshold coordinates;
   ``theta_{t+1} = theta_t - gamma (alpha - ell_t)`` in protection coordinates
   ``theta = 1 - lambda``). The ACI telescoping guarantee does not depend on
   when an outcome is observed, so delayed feedback keeps the long-run risk bound.
2. **Knowledge-evolution gate** (Phase 4, :class:`MartingaleEProcessGate`): the
   outcome of every rule that fired on a released transaction is buffered and
   fed to ``run_update_cycle`` once per gate interval, updating the admission
   wealth ``M^admit_t`` of quarantined rules and the rollback wealth
   ``M^roll_t`` of active rules (Ville threshold ``1/alpha``); rules whose
   precision degrades are rolled back atomically by the gate.
3. **Incremental retraining schedule**: a :class:`RetrainTrigger` is emitted when
   enough new verified evidence has accumulated, when the rolling CS-ACRC loss
   drifts above ``alpha + drift_tolerance``, or when the gate rolled a rule back
   (the detector's ``phi_KB`` inputs changed). Triggers respect a cool-down.

Immediate telemetry (step-up pass/fail, blocks) is logged separately and is
never used as a label.
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from src.closed_loop.verification_queue import ImmediateTelemetry, VerificationQueue, VerifiedLabel
from src.conformal.adaptive_crc import OnlineAdaptiveCRC
from src.conformal.loss_functions import Action

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPERPARAMS_PATH = ROOT / "configs" / "hyperparams.json"

REASON_VOLUME = "label_volume"
REASON_DRIFT = "risk_drift"
REASON_ROLLBACK = "rule_rollback"


@dataclass(frozen=True)
class AdapterConfig:
    """Scheduling parameters of the closed loop (hours)."""

    gate_cycle_interval_hours: float = 6.0
    retrain_min_labels: int = 500
    retrain_min_frauds: int = 10
    retrain_cooldown_hours: float = 24.0
    drift_tolerance: float = 0.01
    drift_min_samples: int = 200
    retrain_on_rollback: bool = True

    def __post_init__(self) -> None:
        if self.gate_cycle_interval_hours < 0.0 or self.retrain_cooldown_hours < 0.0:
            raise ValueError("intervals must be >= 0")
        if self.retrain_min_labels < 1 or self.retrain_min_frauds < 0 or self.drift_min_samples < 1:
            raise ValueError("invalid retraining thresholds")
        if self.drift_tolerance < 0.0:
            raise ValueError("drift_tolerance must be >= 0")

    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None, **overrides) -> "AdapterConfig":
        if hyperparams is None:
            hyperparams = json.loads(DEFAULT_HYPERPARAMS_PATH.read_text(encoding="utf-8"))
        cl = dict(hyperparams.get("closed_loop", {}))
        kwargs: Dict[str, object] = {}
        if "rollback_check_interval_steps" in cl:
            kwargs["gate_cycle_interval_hours"] = float(cl["rollback_check_interval_steps"])
        if "recalibration_interval_steps" in cl:
            kwargs["retrain_cooldown_hours"] = float(cl["recalibration_interval_steps"])
        for key in ("retrain_min_labels", "retrain_min_frauds", "drift_min_samples"):
            if key in cl:
                kwargs[key] = int(cl[key])
        for key in ("drift_tolerance",):
            if key in cl:
                kwargs[key] = float(cl[key])
        if "retrain_on_rollback" in cl:
            kwargs["retrain_on_rollback"] = bool(cl["retrain_on_rollback"])
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass(frozen=True)
class RetrainTrigger:
    """Scheduled incremental-retraining request."""

    tau: float
    reasons: Tuple[str, ...]
    n_labels: int
    n_frauds: int
    rolling_loss: float
    rolled_back_rules: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ThresholdUpdate:
    """Audit record of one CS-ACRC recalibration from a released label."""

    tx_id: str
    tau_release: float
    action: int
    y: int
    amount: float
    loss_normalized: float
    lambda_before: Tuple[float, float, float]
    lambda_after: Tuple[float, float, float]


@dataclass
class FeedbackStep:
    """Everything that happened in one :meth:`ClosedLoopAdapter.advance` call."""

    tau_now: float
    released: List[VerifiedLabel]
    lambda_before: Tuple[float, float, float]
    lambda_after: Tuple[float, float, float]
    gate_report: Optional[object] = None
    wealth: Dict[str, float] = field(default_factory=dict)
    retrain: Optional[RetrainTrigger] = None

    @property
    def n_released(self) -> int:
        return len(self.released)


class ClosedLoopAdapter:
    """Routes delayed verified outcomes to CS-ACRC and the martingale gate."""

    def __init__(self, crc: OnlineAdaptiveCRC, gate=None, queue: Optional[VerificationQueue] = None,
                 config: Optional[AdapterConfig] = None,
                 on_retrain: Optional[Callable[[RetrainTrigger], None]] = None,
                 on_fraud_confirmed: Optional[Callable[[VerifiedLabel], None]] = None) -> None:
        self.crc = crc
        self.gate = gate
        self.queue = queue or VerificationQueue()
        self.config = config or AdapterConfig()
        self.on_retrain = on_retrain
        self.on_fraud_confirmed = on_fraud_confirmed
        self._pending_rule_outcomes: "OrderedDict[str, List[int]]" = OrderedDict()
        self._last_gate_cycle = -math.inf
        self._last_retrain = -math.inf
        self._labels_since = 0
        self._frauds_since = 0
        self._rollbacks_since: List[str] = []
        self.threshold_log: List[ThresholdUpdate] = []
        self.gate_reports: List[object] = []
        self.retrain_schedule: List[RetrainTrigger] = []
        self.n_labels_applied = 0
        self.n_rule_outcomes_applied = 0

    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None,
                    project_config: Optional[Mapping] = None, gate=None,
                    crc: Optional[OnlineAdaptiveCRC] = None, **adapter_overrides) -> "ClosedLoopAdapter":
        if hyperparams is None:
            hyperparams = json.loads(DEFAULT_HYPERPARAMS_PATH.read_text(encoding="utf-8"))
        crc = crc or OnlineAdaptiveCRC.from_config(hyperparams, project_config)
        queue = VerificationQueue.from_config(hyperparams, project_config)
        return cls(crc, gate, queue, AdapterConfig.from_config(hyperparams, **adapter_overrides))

    # ------------------------------------------------------------- decisions
    def record_decision(self, tx_id: Union[str, int], tau: float, action: Union[Action, int, str],
                        amount: float, y_true: int, rule_ids: Sequence[str] = (),
                        score: Optional[float] = None, delay_hours: Optional[float] = None
                        ) -> ImmediateTelemetry:
        """Log a routed transaction; its label is withheld until the verification delay elapses."""
        return self.queue.submit(tx_id, tau, action, y_true, amount, rule_ids, score, delay_hours)

    def route_and_record(self, tx_id: Union[str, int], tau: float, score: float, amount: float,
                         y_true: int, rule_ids: Sequence[str] = (),
                         delay_hours: Optional[float] = None) -> Tuple[Action, ImmediateTelemetry]:
        """Route ``score`` with the *current* ``lambda_t`` and record the decision."""
        action = self.crc.action_for_score(float(score))
        tel = self.record_decision(tx_id, tau, action, amount, y_true, rule_ids, score, delay_hours)
        return action, tel

    # --------------------------------------------------------------- feedback
    def _apply_label(self, lab: VerifiedLabel) -> None:
        before = self.crc.thresholds
        step = self.crc.update(lab.action, lab.y, lab.amount)
        self.threshold_log.append(ThresholdUpdate(lab.tx_id, lab.tau_release, lab.action, lab.y,
                                                  lab.amount, step.loss_normalized, before,
                                                  step.thresholds_after))
        for rid in lab.rule_ids:
            self._pending_rule_outcomes.setdefault(rid, []).append(lab.y)
        self.n_labels_applied += 1
        self._labels_since += 1
        self._frauds_since += lab.y
        if lab.y == 1 and self.on_fraud_confirmed is not None:
            self.on_fraud_confirmed(lab)

    def flush_gate(self, tau_now: float):
        """Run one gate update cycle on all buffered rule outcomes (no-op without a gate)."""
        if self.gate is None or not self._pending_rule_outcomes:
            self._pending_rule_outcomes.clear()
            return None
        outcomes = {rid: list(ys) for rid, ys in self._pending_rule_outcomes.items()}
        self._pending_rule_outcomes.clear()
        report = self.gate.run_update_cycle(outcomes)
        self.n_rule_outcomes_applied += report.n_observations
        self.gate_reports.append(report)
        self._last_gate_cycle = float(tau_now)
        self._rollbacks_since.extend(report.rolled_back)
        return report

    def advance(self, tau_now: float, force_gate_cycle: bool = False) -> FeedbackStep:
        """Move the clock to ``tau_now``: release due labels, recalibrate, update the gate."""
        tau_now = float(tau_now)
        before = self.crc.thresholds
        released = self.queue.release_until(tau_now)
        for lab in released:
            self._apply_label(lab)
        report = None
        due = tau_now - self._last_gate_cycle >= self.config.gate_cycle_interval_hours
        if self._pending_rule_outcomes and (force_gate_cycle or due):
            report = self.flush_gate(tau_now)
        trigger = self._check_retrain(tau_now)
        return FeedbackStep(tau_now, released, before, self.crc.thresholds, report,
                            self.wealth_snapshot(), trigger)

    def drain(self) -> FeedbackStep:
        """Release every pending label and flush the gate (end of stream)."""
        nxt = [self.queue.clock]
        if self.queue.n_pending:
            nxt.append(max(t for t, _, _ in self.queue._heap))
        return self.advance(max(nxt), force_gate_cycle=True)

    # -------------------------------------------------------------- retraining
    def _check_retrain(self, tau_now: float) -> Optional[RetrainTrigger]:
        cfg = self.config
        reasons: List[str] = []
        if self._labels_since >= cfg.retrain_min_labels and self._frauds_since >= cfg.retrain_min_frauds:
            reasons.append(REASON_VOLUME)
        if (self.crc.t >= cfg.drift_min_samples
                and self.crc.rolling_coverage_gap > cfg.drift_tolerance):
            reasons.append(REASON_DRIFT)
        if cfg.retrain_on_rollback and self._rollbacks_since:
            reasons.append(REASON_ROLLBACK)
        if not reasons:
            return None
        cooling = tau_now - self._last_retrain < cfg.retrain_cooldown_hours
        if cooling and REASON_ROLLBACK not in reasons:
            return None
        trig = RetrainTrigger(tau_now, tuple(reasons), self._labels_since, self._frauds_since,
                              self.crc.rolling_loss, tuple(self._rollbacks_since))
        self.retrain_schedule.append(trig)
        self._last_retrain = tau_now
        self._labels_since = self._frauds_since = 0
        self._rollbacks_since = []
        if self.on_retrain is not None:
            self.on_retrain(trig)
        return trig

    # ----------------------------------------------------------------- views
    def wealth(self, rule_id: str) -> Optional[float]:
        """Current e-process wealth ``M_t`` of ``rule_id`` (admission if quarantined, rollback if active)."""
        if self.gate is None or str(rule_id) not in self.gate.records:
            return None
        rec = self.gate.records[str(rule_id)]
        proc = rec.admission if rec.status == "quarantined" else rec.rollback
        return None if proc is None else proc.wealth

    def wealth_snapshot(self) -> Dict[str, float]:
        if self.gate is None:
            return {}
        out: Dict[str, float] = {}
        for rid in sorted(self.gate.records):
            w = self.wealth(rid)
            if w is not None:
                out[rid] = w
        return out

    def threshold_frame(self) -> pd.DataFrame:
        rows = []
        for u in self.threshold_log:
            row = asdict(u)
            row.update({f"lambda_{n}_before": v for n, v in zip(("low", "mid", "high"), u.lambda_before)})
            row.update({f"lambda_{n}_after": v for n, v in zip(("low", "mid", "high"), u.lambda_after)})
            del row["lambda_before"], row["lambda_after"]
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, object]:
        return {
            "clock": self.queue.clock,
            "n_pending": self.queue.n_pending,
            "n_labels_applied": self.n_labels_applied,
            "n_rule_outcomes_applied": self.n_rule_outcomes_applied,
            "n_gate_cycles": len(self.gate_reports),
            "n_rollbacks": sum(len(r.rolled_back) for r in self.gate_reports),
            "n_promotions": sum(len(r.promoted) for r in self.gate_reports),
            "n_retrain_triggers": len(self.retrain_schedule),
            "thresholds": self.crc.threshold_dict(),
            "rolling_loss": self.crc.rolling_loss,
            "telemetry": self.queue.telemetry_stats(),
        }


__all__ = ["AdapterConfig", "ClosedLoopAdapter", "FeedbackStep", "REASON_DRIFT", "REASON_ROLLBACK",
           "REASON_VOLUME", "RetrainTrigger", "ThresholdUpdate"]
