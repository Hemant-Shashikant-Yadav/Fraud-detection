"""
src/closed_loop/verification_queue.py
=====================================

Chronological delayed ground-truth arrival for the MG-ECFM closed loop.

Two information channels are separated explicitly:

* **Immediate telemetry** - available at the decision time ``tau_t``: step-up
  challenge pass / fail, Hold-and-Verify escrow, velocity / hard blocks. It is
  *not* a label (a fraudster can pass a step-up; a genuine customer can fail
  one) and is never fed to the risk controller or the e-process gate.
* **Delayed labels** - confirmed ground truth released at
  ``tau_t + Delta_tau`` with ``Delta_tau in [1, 14]`` days:

  ======================  =========================================================
  source                  latency model (days, truncated to [min_days, max_days])
  ======================  =========================================================
  ``customer_dispute``    log-normal, median ``dispute_median_days``
  ``bank_chargeback``     log-normal, median ``chargeback_median_days``
  ``maturity``            genuine transactions are confirmed when the dispute
                          window closes without a claim (``max_days``), or drawn
                          from the dispute model when ``legit_policy="sampled"``
  ======================  =========================================================

A fraud is reported by the customer with probability ``dispute_share``,
otherwise through the issuer's chargeback pipeline. Time is in hours
(1 PaySim step = 1 h).

Causality: a label is never visible before its release time; the queue clock
is monotone, and decisions cannot be submitted before the current clock.
Releases are ordered by ``(tau_release, submission sequence)`` so equal-time
releases preserve submission order deterministically.
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.conformal.loss_functions import Action

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPERPARAMS_PATH = ROOT / "configs" / "hyperparams.json"
SEED = 42

SOURCE_DISPUTE = "customer_dispute"
SOURCE_CHARGEBACK = "bank_chargeback"
SOURCE_MATURITY = "maturity"
SOURCE_MANUAL = "manual"

TELEMETRY_APPROVED = "approved"
TELEMETRY_STEP_UP_PASS = "step_up_pass"
TELEMETRY_STEP_UP_FAIL = "step_up_fail"
TELEMETRY_HOLD = "hold_escrow"
TELEMETRY_BLOCK = "velocity_block"
LEGIT_POLICIES = ("maturity", "sampled")


@dataclass(frozen=True)
class LatencyModel:
    """Verification-latency distributions (days) and immediate-telemetry rates."""

    min_days: float = 1.0
    max_days: float = 14.0
    hours_per_day: float = 24.0
    dispute_share: float = 0.6
    dispute_median_days: float = 2.0
    dispute_sigma: float = 0.6
    chargeback_median_days: float = 7.0
    chargeback_sigma: float = 0.35
    legit_policy: str = "maturity"
    beta_bypass: float = 0.10
    legit_step_up_fail_rate: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.min_days <= self.max_days:
            raise ValueError("require 0 < min_days <= max_days")
        if not 0.0 <= self.dispute_share <= 1.0:
            raise ValueError("dispute_share must lie in [0, 1]")
        for name in ("dispute_median_days", "chargeback_median_days", "dispute_sigma",
                     "chargeback_sigma", "hours_per_day"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.legit_policy not in LEGIT_POLICIES:
            raise ValueError(f"legit_policy must be one of {LEGIT_POLICIES}")
        if not (0.0 <= self.beta_bypass <= 1.0 and 0.0 <= self.legit_step_up_fail_rate <= 1.0):
            raise ValueError("telemetry rates must lie in [0, 1]")

    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None,
                    project_config: Optional[Mapping] = None, **overrides) -> "LatencyModel":
        if hyperparams is None:
            hyperparams = json.loads(DEFAULT_HYPERPARAMS_PATH.read_text(encoding="utf-8"))
        kwargs: Dict[str, object] = {}
        days = dict(hyperparams.get("closed_loop", {})).get("label_delay_days")
        if days is None and project_config is not None:
            ver = project_config.get("verification", {})
            if "label_delay_days_min" in ver:
                days = (ver["label_delay_days_min"], ver["label_delay_days_max"])
        if days is not None:
            kwargs["min_days"], kwargs["max_days"] = float(days[0]), float(days[1])
        if project_config is not None and "beta_bypass" in project_config.get("costs", {}):
            kwargs["beta_bypass"] = float(project_config["costs"]["beta_bypass"])
        kwargs.update(dict(hyperparams.get("closed_loop", {}).get("latency", {})))
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def min_hours(self) -> float:
        return self.min_days * self.hours_per_day

    @property
    def max_hours(self) -> float:
        return self.max_days * self.hours_per_day

    def _truncated_lognormal_days(self, median: float, sigma: float,
                                  rng: np.random.Generator) -> float:
        """Inverse-CDF draw from ``LogNormal(log median, sigma)`` truncated to ``[min, max]`` days."""
        from statistics import NormalDist
        nd = NormalDist()
        mu = math.log(median)
        lo = nd.cdf((math.log(self.min_days) - mu) / sigma)
        hi = nd.cdf((math.log(self.max_days) - mu) / sigma)
        if hi - lo < 1e-12:
            return min(max(median, self.min_days), self.max_days)
        u = lo + (hi - lo) * float(rng.random())
        u = min(max(u, 1e-15), 1.0 - 1e-15)
        return min(max(math.exp(mu + sigma * nd.inv_cdf(u)), self.min_days), self.max_days)

    def sample(self, y: int, rng: np.random.Generator) -> Tuple[float, str]:
        """``(delay_hours, source)`` for a transaction with ground truth ``y``."""
        if int(y) == 1:
            if rng.random() < self.dispute_share:
                d = self._truncated_lognormal_days(self.dispute_median_days, self.dispute_sigma, rng)
                return d * self.hours_per_day, SOURCE_DISPUTE
            d = self._truncated_lognormal_days(self.chargeback_median_days, self.chargeback_sigma, rng)
            return d * self.hours_per_day, SOURCE_CHARGEBACK
        if self.legit_policy == "maturity":
            return self.max_hours, SOURCE_MATURITY
        d = self._truncated_lognormal_days(self.dispute_median_days, self.dispute_sigma, rng)
        return d * self.hours_per_day, SOURCE_MATURITY


@dataclass(frozen=True)
class ImmediateTelemetry:
    """Signal observable at decision time ``tau`` (not ground truth)."""

    tx_id: str
    tau: float
    action: int
    kind: str
    amount: float
    challenge_passed: Optional[bool] = None


@dataclass(frozen=True)
class VerifiedLabel:
    """Confirmed ground truth released from the queue."""

    tx_id: str
    tau_decision: float
    tau_release: float
    y: int
    amount: float
    action: int
    source: str
    rule_ids: Tuple[str, ...] = ()
    score: Optional[float] = None

    @property
    def latency_hours(self) -> float:
        return self.tau_release - self.tau_decision

    @property
    def latency_days(self) -> float:
        return self.latency_hours / 24.0


class VerificationQueue:
    """Priority queue of pending labels released in chronological order."""

    def __init__(self, latency: Optional[LatencyModel] = None, seed: int = SEED,
                 strict_bounds: bool = True) -> None:
        self.latency = latency or LatencyModel()
        self.seed = int(seed)
        self.strict_bounds = bool(strict_bounds)
        self._rng = np.random.default_rng(self.seed)
        self._telemetry_rng = np.random.default_rng(self.seed + 1)
        self._heap: List[Tuple[float, int, VerifiedLabel]] = []
        self._seq = 0
        self._clock = -math.inf
        self._known: Dict[str, float] = {}
        self._released: List[VerifiedLabel] = []
        self.telemetry_log: List[ImmediateTelemetry] = []

    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None,
                    project_config: Optional[Mapping] = None, **overrides) -> "VerificationQueue":
        seed = int((hyperparams or {}).get("seed", SEED))
        return cls(LatencyModel.from_config(hyperparams, project_config, **overrides), seed=seed)

    # ---------------------------------------------------------------- state
    @property
    def clock(self) -> float:
        """Latest time passed to :meth:`release_until` (``-inf`` before the first release)."""
        return self._clock

    @property
    def n_pending(self) -> int:
        return len(self._heap)

    @property
    def n_released(self) -> int:
        return len(self._released)

    @property
    def released(self) -> List[VerifiedLabel]:
        return list(self._released)

    def next_release_time(self) -> Optional[float]:
        return self._heap[0][0] if self._heap else None

    def label_available_at(self, tx_id: str) -> float:
        return self._known[str(tx_id)]

    def is_label_available(self, tx_id: str, tau: float) -> bool:
        """``True`` iff the label of ``tx_id`` has been released by time ``tau``."""
        tx_id = str(tx_id)
        return tx_id in self._known and self._known[tx_id] <= tau and self._known[tx_id] <= self._clock

    # ------------------------------------------------------------ telemetry
    def _telemetry(self, tx_id: str, tau: float, action: Action, y: int,
                   amount: float) -> ImmediateTelemetry:
        lat = self.latency
        if action == Action.STEP_UP:
            u = float(self._telemetry_rng.random())
            passed = u < lat.beta_bypass if y == 1 else u >= lat.legit_step_up_fail_rate
            kind = TELEMETRY_STEP_UP_PASS if passed else TELEMETRY_STEP_UP_FAIL
            return ImmediateTelemetry(tx_id, tau, int(action), kind, amount, passed)
        kind = {Action.APPROVE: TELEMETRY_APPROVED, Action.HOLD: TELEMETRY_HOLD,
                Action.BLOCK: TELEMETRY_BLOCK}[action]
        return ImmediateTelemetry(tx_id, tau, int(action), kind, amount, None)

    # ---------------------------------------------------------------- intake
    def submit(self, tx_id: Union[str, int], tau: float, action: Union[Action, int, str], y_true: int,
               amount: float, rule_ids: Sequence[str] = (), score: Optional[float] = None,
               delay_hours: Optional[float] = None, source: Optional[str] = None) -> ImmediateTelemetry:
        """Register a routed decision; returns its immediate telemetry and schedules its label."""
        tx_id = str(tx_id)
        tau = float(tau)
        y = int(y_true)
        if y not in (0, 1):
            raise ValueError("y_true must be 0 or 1")
        if tx_id in self._known:
            raise ValueError(f"transaction {tx_id} already submitted")
        if tau < self._clock:
            raise ValueError(f"decision at tau={tau} precedes the queue clock {self._clock}")
        act = Action.parse(action)
        if delay_hours is None:
            delay, src = self.latency.sample(y, self._rng)
        else:
            delay = float(delay_hours)
            src = source or SOURCE_MANUAL
            if self.strict_bounds and not (self.latency.min_hours - 1e-9 <= delay
                                           <= self.latency.max_hours + 1e-9):
                raise ValueError(f"delay {delay} h outside [{self.latency.min_hours}, "
                                 f"{self.latency.max_hours}] h")
        label = VerifiedLabel(tx_id, tau, tau + delay, y, float(amount), int(act), src,
                              tuple(str(r) for r in rule_ids), None if score is None else float(score))
        heapq.heappush(self._heap, (label.tau_release, self._seq, label))
        self._seq += 1
        self._known[tx_id] = label.tau_release
        tel = self._telemetry(tx_id, tau, act, y, float(amount))
        self.telemetry_log.append(tel)
        return tel

    # --------------------------------------------------------------- release
    def release_until(self, tau_now: float) -> List[VerifiedLabel]:
        """Pop every label with ``tau_release <= tau_now`` in chronological order."""
        tau_now = float(tau_now)
        if tau_now < self._clock:
            raise ValueError(f"time cannot move backwards ({tau_now} < {self._clock})")
        self._clock = tau_now
        out: List[VerifiedLabel] = []
        while self._heap and self._heap[0][0] <= tau_now:
            out.append(heapq.heappop(self._heap)[2])
        self._released.extend(out)
        return out

    def drain(self) -> List[VerifiedLabel]:
        """Release everything still pending (end of stream)."""
        if not self._heap:
            return []
        return self.release_until(max(self._clock, max(t for t, _, _ in self._heap)))

    # ----------------------------------------------------------------- stats
    def latency_stats(self) -> Dict[str, float]:
        if not self._released:
            return {"n": 0.0}
        lat = np.asarray([l.latency_days for l in self._released])
        fraud = np.asarray([l.latency_days for l in self._released if l.y == 1])
        return {"n": float(lat.size), "min_days": float(lat.min()), "max_days": float(lat.max()),
                "mean_days": float(lat.mean()),
                "fraud_median_days": float(np.median(fraud)) if fraud.size else float("nan"),
                "share_dispute": float(np.mean([l.source == SOURCE_DISPUTE for l in self._released
                                                if l.y == 1])) if fraud.size else float("nan")}

    def telemetry_stats(self) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        for t in self.telemetry_log:
            counts[t.kind] = counts.get(t.kind, 0.0) + 1.0
        n_su = counts.get(TELEMETRY_STEP_UP_PASS, 0.0) + counts.get(TELEMETRY_STEP_UP_FAIL, 0.0)
        counts["step_up_fail_rate"] = counts.get(TELEMETRY_STEP_UP_FAIL, 0.0) / n_su if n_su else 0.0
        return counts

    def released_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(l) for l in self._released])


__all__ = [
    "ImmediateTelemetry", "LatencyModel", "SOURCE_CHARGEBACK", "SOURCE_DISPUTE", "SOURCE_MANUAL",
    "SOURCE_MATURITY", "TELEMETRY_APPROVED", "TELEMETRY_BLOCK", "TELEMETRY_HOLD",
    "TELEMETRY_STEP_UP_FAIL", "TELEMETRY_STEP_UP_PASS", "VerificationQueue", "VerifiedLabel",
]
