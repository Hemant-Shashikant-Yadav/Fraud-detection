"""
src/epistemic_gate/martingale_eprocess.py
=========================================

Anytime-valid (SAVI) betting e-processes for the MG-ECFM knowledge-evolution
gate (Ramdas et al. 2023; Shafer & Vovk 2019).

Admission gate (per quarantined candidate ``c_k``; ``y_i = 1`` confirmed fraud)::

    H0^admit : Precision(c_k) <= p0     vs.   H1 : Precision(c_k) = p1 > p0
    M_t^admit = prod_i [ y_i p1/p0 + (1 - y_i) (1 - p1)/(1 - p0) ]

Rollback gate (per active rule; ``1 - y_i = 1`` is a false alarm)::

    H0^good : FPR(c_k) <= q0        vs.   H1^bad : FPR(c_k) = q1 > q0
    M_t^roll  = prod_i [ (1 - y_i) q1/q0 + y_i (1 - q1)/(1 - q0) ]

Both are instances of one Bernoulli likelihood-ratio betting process on an
indicator ``x_i`` (``x = y`` for admission, ``x = 1 - y`` for rollback) with
null rate ``theta0`` and alternative ``theta1 > theta0``. For every null rate
``theta <= theta0`` the one-step factor has conditional mean
``1 + (theta - theta0)(theta1 - theta0)/(theta0 (1 - theta0)) <= 1``, so
``M_t`` is a non-negative test supermartingale for the *composite* null and
Ville's maximal inequality gives

    P_H0( exists t >= 1 : M_t >= 1/alpha ) <= alpha.

Betting strategies
------------------
``likelihood_ratio``  fixed alternative ``theta1`` (synopsis equations).
``plugin``            predictable plug-in alternative
                      ``theta1_t = clip((k_{t-1} + s m)/(n_{t-1} + s), theta0 + delta, 1 - delta)``
                      using only past outcomes and an F_0-measurable prior mean
                      ``m`` (e.g. the Subjective-Logic expected probability).
                      Any predictable ``theta1_t > theta0`` keeps validity.

Monitors
--------
``martingale``  the product above; time-uniform false-alarm bound ``alpha``.
``e_detector``  CUSUM e-detector ``log D_t = max(0, log D_{t-1}) + log f_t``
                (Shin, Ramdas & Rinaldo 2023). It restarts after good history,
                so detection delay after a change does not grow with the rule's
                age; its guarantee is on the average run length,
                ``E_H0[T] >= 1/alpha``, instead of the infinite-horizon bound.

Analytical helpers give the minimum number of events needed to cross the
threshold and the *exact* probability of crossing within ``n`` observations
(dynamic programme over the event count), which certifies the ``<= 3`` update
cycle rollback requirement for a given firing volume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BETTING_STRATEGIES: Tuple[str, ...] = ("likelihood_ratio", "plugin")
MONITORS: Tuple[str, ...] = ("martingale", "e_detector")
ROLLBACK_POLICIES: Tuple[str, ...] = ("targeted", "snapshot")
SCREENING_MODES: Tuple[str, ...] = ("strict", "lenient", "off")


# --------------------------------------------------------------------------- #
# Closed-form helpers                                                         #
# --------------------------------------------------------------------------- #
def _validate_rates(theta0: float, theta1: float) -> Tuple[float, float]:
    theta0, theta1 = float(theta0), float(theta1)
    if not 0.0 < theta0 < theta1 < 1.0:
        raise ValueError(f"require 0 < theta0 < theta1 < 1 (got {theta0}, {theta1})")
    return theta0, theta1


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return alpha


def betting_factor(x: int, theta0: float, theta1: float) -> float:
    """One-step wealth multiplier ``x theta1/theta0 + (1 - x)(1 - theta1)/(1 - theta0)``."""
    return theta1 / theta0 if x else (1.0 - theta1) / (1.0 - theta0)


def log_factors(theta0: float, theta1: float) -> Tuple[float, float]:
    """``(log f(x=1), log f(x=0))``."""
    theta0, theta1 = _validate_rates(theta0, theta1)
    return math.log(theta1 / theta0), math.log((1.0 - theta1) / (1.0 - theta0))


def expected_factor(theta: float, theta0: float, theta1: float) -> float:
    """``E_theta[f]``; ``<= 1`` for all ``theta <= theta0`` (supermartingale)."""
    return theta * theta1 / theta0 + (1.0 - theta) * (1.0 - theta1) / (1.0 - theta0)


def expected_log_growth(theta: float, theta0: float, theta1: float) -> float:
    """Drift ``E_theta[log f]`` (positive => the test eventually rejects a.s.)."""
    l1, l0 = log_factors(theta0, theta1)
    return theta * l1 + (1.0 - theta) * l0


def min_events_to_cross(theta0: float, theta1: float, alpha: float) -> int:
    """Fewest observations (all events) after which ``M_t >= 1/alpha``."""
    l1, _ = log_factors(theta0, theta1)
    return int(math.ceil(math.log(1.0 / _validate_alpha(alpha)) / l1 - 1e-12))


def crossing_probability(n_steps: int, theta: float, theta0: float, theta1: float,
                         alpha: float) -> float:
    """Exact ``P_theta(exists t <= n : M_t >= 1/alpha)`` for the fixed-bet martingale.

    ``log M_t`` depends only on the event count ``k_t``, so the not-yet-crossed
    probability mass is propagated over ``k`` with absorption at the boundary.
    """
    n_steps = int(n_steps)
    if n_steps <= 0:
        return 0.0
    theta = float(theta)
    l1, l0 = log_factors(theta0, theta1)
    thresh = math.log(1.0 / _validate_alpha(alpha)) - 1e-12
    alive = np.zeros(n_steps + 2, dtype=np.float64)
    alive[0] = 1.0
    absorbed = 0.0
    ks = np.arange(n_steps + 2, dtype=np.float64)
    for n in range(1, n_steps + 1):
        nxt = np.empty_like(alive)
        nxt[0] = alive[0] * (1.0 - theta)
        nxt[1:] = alive[1:] * (1.0 - theta) + alive[:-1] * theta
        logw = ks * l1 + (n - ks) * l0
        hit = (logw >= thresh) & (ks <= n)
        absorbed += float(nxt[hit].sum())
        nxt[hit] = 0.0
        alive = nxt
    return min(1.0, absorbed)


def steps_for_power(theta: float, theta0: float, theta1: float, alpha: float,
                    power: float = 0.95, max_steps: int = 10_000) -> Optional[int]:
    """Smallest ``n`` with ``P_theta(cross within n) >= power`` (``None`` if > ``max_steps``)."""
    if expected_log_growth(theta, theta0, theta1) <= 0.0:
        return None
    lo, hi = 1, 1
    while crossing_probability(hi, theta, theta0, theta1, alpha) < power:
        lo, hi = hi + 1, hi * 2
        if hi > max_steps:
            return None
    while lo < hi:
        mid = (lo + hi) // 2
        if crossing_probability(mid, theta, theta0, theta1, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --------------------------------------------------------------------------- #
# Betting e-process                                                           #
# --------------------------------------------------------------------------- #
class BettingEProcess:
    """Sequential Bernoulli betting test with Ville stopping threshold ``1/alpha``."""

    def __init__(self, theta0: float, theta1: float, alpha: float,
                 betting: str = "likelihood_ratio", monitor: str = "martingale",
                 prior_mean: Optional[float] = None, prior_strength: float = 2.0,
                 delta: float = 1e-3, name: str = "") -> None:
        self.theta0, self.theta1 = _validate_rates(theta0, theta1)
        self.alpha = _validate_alpha(alpha)
        if betting not in BETTING_STRATEGIES:
            raise ValueError(f"betting must be one of {BETTING_STRATEGIES}")
        if monitor not in MONITORS:
            raise ValueError(f"monitor must be one of {MONITORS}")
        if monitor == "e_detector" and betting != "likelihood_ratio":
            raise ValueError("the CUSUM e-detector is defined for the fixed likelihood-ratio bet")
        if not 0.0 < delta < 0.5 * (1.0 - self.theta0):
            raise ValueError("delta must be a small positive clip margin")
        self.betting, self.monitor, self.name = betting, monitor, name
        self.delta = float(delta)
        self.prior_mean = self.theta1 if prior_mean is None else float(prior_mean)
        self.prior_strength = float(prior_strength)
        if self.prior_strength < 0.0:
            raise ValueError("prior_strength must be non-negative")
        self.log_threshold = math.log(1.0 / self.alpha)
        self._l1, self._l0 = log_factors(self.theta0, self.theta1)
        self.reset()

    # --------------------------------------------------------------- state
    def reset(self) -> None:
        self.n = 0
        self.n_events = 0
        self.log_wealth = 0.0
        self.max_log_wealth = 0.0
        self.crossed = False
        self.crossed_at: Optional[int] = None

    @property
    def wealth(self) -> float:
        return math.exp(min(self.log_wealth, 700.0))

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha

    @property
    def anytime_p_value(self) -> float:
        """``min(1, 1 / sup_s M_s)`` - valid at any data-dependent stopping time."""
        return min(1.0, math.exp(-self.max_log_wealth))

    def current_alternative(self) -> float:
        """Predictable alternative ``theta1_t`` used for the next bet."""
        if self.betting == "likelihood_ratio":
            return self.theta1
        s = self.prior_strength
        est = (self.n_events + s * self.prior_mean) / (self.n + s) if (self.n + s) > 0 else self.theta1
        return min(1.0 - self.delta, max(self.theta0 + self.delta, est))

    # -------------------------------------------------------------- update
    def _log_factor(self, x: int) -> float:
        if self.betting == "likelihood_ratio":
            return self._l1 if x else self._l0
        t1 = self.current_alternative()
        return math.log(t1 / self.theta0) if x else math.log((1.0 - t1) / (1.0 - self.theta0))

    def update(self, x: int) -> bool:
        """Observe indicator ``x``; returns ``True`` exactly once, at the first crossing."""
        if self.crossed:
            raise RuntimeError(f"e-process '{self.name}' already stopped at n={self.crossed_at}")
        x = int(x)
        if x not in (0, 1):
            raise ValueError("observations must be binary")
        lf = self._log_factor(x)
        if self.monitor == "e_detector":
            self.log_wealth = max(0.0, self.log_wealth) + lf
        else:
            self.log_wealth += lf
        self.n += 1
        self.n_events += x
        self.max_log_wealth = max(self.max_log_wealth, self.log_wealth)
        if self.log_wealth >= self.log_threshold - 1e-12:
            self.crossed, self.crossed_at = True, self.n
            return True
        return False

    def update_many(self, xs: Iterable[int]) -> Optional[int]:
        """Feed a batch; returns the 0-based index of the crossing observation, if any."""
        for i, x in enumerate(xs):
            if self.update(int(x)):
                return i
        return None

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "theta0": self.theta0, "theta1": self.theta1,
                "alpha": self.alpha, "betting": self.betting, "monitor": self.monitor,
                "n": self.n, "n_events": self.n_events, "log_wealth": self.log_wealth,
                "max_log_wealth": self.max_log_wealth, "log_threshold": self.log_threshold,
                "crossed": self.crossed, "crossed_at": self.crossed_at,
                "anytime_p_value": self.anytime_p_value,
                "current_alternative": self.current_alternative()}


def admission_process(p0: float, p1: float, alpha: float, **kwargs) -> BettingEProcess:
    """``M^admit``: indicator ``x_i = y_i`` (confirmed fraud)."""
    return BettingEProcess(p0, p1, alpha, name=kwargs.pop("name", "admission"), **kwargs)


def rollback_process(q0: float, q1: float, alpha: float, **kwargs) -> BettingEProcess:
    """``M^roll``: indicator ``x_i = 1 - y_i`` (false alarm)."""
    return BettingEProcess(q0, q1, alpha, name=kwargs.pop("name", "rollback"), **kwargs)


def simulate_crossing_rate(theta: float, theta0: float, theta1: float, alpha: float,
                           n_steps: int, n_reps: int, seed: int = 42,
                           monitor: str = "martingale") -> Dict[str, float]:
    """Vectorised Monte-Carlo crossing frequency and first-crossing times."""
    rng = np.random.default_rng(seed)
    l1, l0 = log_factors(theta0, theta1)
    thresh = math.log(1.0 / _validate_alpha(alpha)) - 1e-12
    x = rng.random((n_reps, n_steps)) < float(theta)
    inc = np.where(x, l1, l0)
    if monitor == "martingale":
        path = np.cumsum(inc, axis=1)
    elif monitor == "e_detector":
        path = np.empty_like(inc)
        state = np.zeros(n_reps)
        for t in range(n_steps):
            state = np.maximum(state, 0.0) + inc[:, t]
            path[:, t] = state
    else:
        raise ValueError(f"monitor must be one of {MONITORS}")
    hit = path >= thresh
    crossed = hit.any(axis=1)
    first = np.where(crossed, hit.argmax(axis=1) + 1, n_steps + 1)
    return {"rate": float(crossed.mean()), "n_reps": int(n_reps), "n_steps": int(n_steps),
            "mean_first_crossing": float(first[crossed].mean()) if crossed.any() else float("nan"),
            "censored_mean_run_length": float(first.mean())}


# --------------------------------------------------------------------------- #
# Knowledge-evolution gate                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateConfig:
    p0: float = 0.5
    p1: float = 0.8
    alpha_admit: float = 0.05
    q0: float = 0.20
    q1: float = 0.50
    alpha_roll: float = 0.05
    max_rollback_cycles: int = 3
    max_firings_per_candidate: int = 5000
    betting: str = "likelihood_ratio"
    rollback_monitor: str = "martingale"
    rollback_policy: str = "targeted"
    screening: str = "lenient"
    max_admissions_per_rule: int = 2
    plugin_prior_strength: float = 2.0

    def __post_init__(self) -> None:
        _validate_rates(self.p0, self.p1)
        _validate_rates(self.q0, self.q1)
        _validate_alpha(self.alpha_admit)
        _validate_alpha(self.alpha_roll)
        if self.betting not in BETTING_STRATEGIES:
            raise ValueError(f"betting must be one of {BETTING_STRATEGIES}")
        if self.rollback_monitor not in MONITORS:
            raise ValueError(f"rollback_monitor must be one of {MONITORS}")
        if self.rollback_policy not in ROLLBACK_POLICIES:
            raise ValueError(f"rollback_policy must be one of {ROLLBACK_POLICIES}")
        if self.screening not in SCREENING_MODES:
            raise ValueError(f"screening must be one of {SCREENING_MODES}")
        if self.max_rollback_cycles < 1 or self.max_firings_per_candidate < 1:
            raise ValueError("cycle / firing budgets must be positive")
        if self.max_admissions_per_rule < 1:
            raise ValueError("max_admissions_per_rule must be >= 1")

    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None, **overrides) -> "GateConfig":
        mp = dict((hyperparams or {}).get("martingale_eprocess", {}))
        adm, rb = dict(mp.get("admission", {})), dict(mp.get("rollback", {}))
        kwargs: Dict[str, object] = {}
        for key, src in (("p0", adm), ("p1", adm), ("alpha_admit", adm),
                         ("q0", rb), ("q1", rb), ("alpha_roll", rb)):
            if key in src:
                kwargs[key] = float(src[key])
        if "max_rollback_cycles" in rb:
            kwargs["max_rollback_cycles"] = int(rb["max_rollback_cycles"])
        if "monitor" in rb:
            kwargs["rollback_monitor"] = str(rb["monitor"])
        if "policy" in rb:
            kwargs["rollback_policy"] = str(rb["policy"])
        for key in ("max_firings_per_candidate", "max_admissions_per_rule"):
            if key in mp:
                kwargs[key] = int(mp[key])
        for key in ("betting", "screening"):
            if key in mp:
                kwargs[key] = str(mp[key])
        kwargs.update(overrides)
        return cls(**kwargs)

    def admission_min_hits(self) -> int:
        return min_events_to_cross(self.p0, self.p1, self.alpha_admit)

    def rollback_min_false_alarms(self) -> int:
        return min_events_to_cross(self.q0, self.q1, self.alpha_roll)

    def rollback_crossing_probability(self, n_firings: int, false_alarm_rate: float) -> float:
        """Exact P(rollback within ``n_firings``) for a rule with the given FPR."""
        return crossing_probability(n_firings, false_alarm_rate, self.q0, self.q1, self.alpha_roll)

    def firings_per_cycle_for_rollback(self, false_alarm_rate: float, power: float = 0.99,
                                       cycles: Optional[int] = None) -> Optional[int]:
        """Minimum firings per update cycle so a rule with this FPR is rolled back within
        ``max_rollback_cycles`` cycles with probability ``>= power`` (fresh monitor)."""
        cycles = int(cycles or self.max_rollback_cycles)
        n = steps_for_power(false_alarm_rate, self.q0, self.q1, self.alpha_roll, power)
        return None if n is None else int(math.ceil(n / cycles))


@dataclass
class RuleRecord:
    rule_id: str
    status: str
    rule: object
    prior_opinion: object
    opinion: object
    admission: Optional[BettingEProcess] = None
    rollback: Optional[BettingEProcess] = None
    n_admissions: int = 0
    n_rollbacks: int = 0
    admitted_cycle: Optional[int] = None
    rolled_back_cycle: Optional[int] = None
    pre_admission_snapshot: Optional[int] = None
    events: List[Tuple[int, str]] = field(default_factory=list)


@dataclass
class CycleReport:
    cycle: int
    n_observations: int
    promoted: List[str]
    rolled_back: List[str]
    rejected: List[str]
    head_snapshot_id: int

    def to_dict(self) -> Dict[str, object]:
        return {"cycle": self.cycle, "n_observations": self.n_observations,
                "promoted": list(self.promoted), "rolled_back": list(self.rolled_back),
                "rejected": list(self.rejected), "head_snapshot_id": self.head_snapshot_id}


class MartingaleEProcessGate:
    """Quarantine buffer + dual (admission / rollback) e-processes + versioned KB.

    Lifecycle of candidate ``c_k``::

        submit -> [Subjective-Logic screen] -> QUARANTINED --(M^admit >= 1/alpha_admit)--> ACTIVE
        ACTIVE --(M^roll >= 1/alpha_roll)--> atomic rollback to V_j, c_k -> QUARANTINED
        QUARANTINED --(budget exhausted / admissions exhausted)--> REJECTED

    Every promotion and rollback is committed to the :class:`KnowledgeStore` as a
    new immutable snapshot together with the e-value certificate that justified
    it. Outcomes arrive per *update cycle* (a batch of verified firings).
    """

    def __init__(self, store=None, config: Optional[GateConfig] = None, fusion_engine=None) -> None:
        from src.epistemic_gate.knowledge_store import KnowledgeStore
        from src.epistemic_gate.subjective_logic import EpistemicFusionEngine
        self.config = config or GateConfig()
        self.store = store if store is not None else KnowledgeStore()
        self.fusion = fusion_engine if fusion_engine is not None else EpistemicFusionEngine()
        self.records: Dict[str, RuleRecord] = {}
        self.cycle = 0

    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None, store=None,
                    fusion_overrides: Optional[Mapping] = None, **overrides) -> "MartingaleEProcessGate":
        from src.epistemic_gate.subjective_logic import EpistemicFusionEngine
        fusion = EpistemicFusionEngine.from_config(hyperparams, **dict(fusion_overrides or {}))
        return cls(store, GateConfig.from_config(hyperparams, **overrides), fusion)

    # ---------------------------------------------------------- processes
    def _new_admission(self, rec: RuleRecord) -> BettingEProcess:
        cfg = self.config
        prior = getattr(rec.prior_opinion, "expected_probability", None)
        return admission_process(cfg.p0, cfg.p1, cfg.alpha_admit, betting=cfg.betting,
                                 prior_mean=prior, prior_strength=cfg.plugin_prior_strength,
                                 name=f"admission:{rec.rule_id}")

    def _new_rollback(self, rec: RuleRecord) -> BettingEProcess:
        cfg = self.config
        betting = "likelihood_ratio" if cfg.rollback_monitor == "e_detector" else cfg.betting
        return rollback_process(cfg.q0, cfg.q1, cfg.alpha_roll, betting=betting,
                                monitor=cfg.rollback_monitor, prior_mean=cfg.q1,
                                prior_strength=cfg.plugin_prior_strength,
                                name=f"rollback:{rec.rule_id}")

    # --------------------------------------------------------------- intake
    def submit_candidate(self, rule, now: int):
        """Fuse provenance evidence, screen, and (if admissible) quarantine ``rule``."""
        from src.epistemic_gate.knowledge_store import rule_to_logic
        rid = str(rule.rule_id)
        if rid in self.records:
            raise ValueError(f"candidate {rid} already submitted")
        result = self.fusion.fuse_rule(rule, now)
        op = result.opinion
        mode = self.config.screening
        if mode == "strict":
            admit_to_buffer = result.decision == "quarantine"
        elif mode == "lenient":
            # Reject only *confident* disbelief; lack of evidence is resolved by the e-process.
            admit_to_buffer = not (op.uncertainty <= self.fusion.max_uncertainty
                                   and op.expected_probability <= self.config.p0)
        else:
            admit_to_buffer = True
        rec = RuleRecord(rid, "quarantined" if admit_to_buffer else "rejected", rule, op, op)
        self.records[rid] = rec
        logic = rule_to_logic(rule)
        logic["fusion"] = {"n_sources": len(result.sources), "n_groups": result.n_groups,
                           "screen": result.decision, "screening_mode": mode}
        if admit_to_buffer:
            rec.admission = self._new_admission(rec)
            self.store.quarantine_candidate(rid, logic, op, reason="submitted", step=self.cycle)
            rec.events.append((self.cycle, "quarantined"))
        else:
            self.store.reject_candidate(rid, logic, op, reason="epistemic screen", step=self.cycle)
            rec.events.append((self.cycle, "rejected:screen"))
        return result

    def inject_active(self, rule, opinion=None, reason: str = "injected") -> int:
        """Place ``rule`` directly in the active KB (legacy import / adversarial injection
        experiments); it is monitored by a fresh rollback e-process."""
        from src.epistemic_gate.knowledge_store import rule_to_logic
        from src.epistemic_gate.subjective_logic import SubjectiveOpinion
        rid = str(rule.rule_id)
        op = opinion if opinion is not None else SubjectiveOpinion.vacuous()
        rec = self.records.get(rid) or RuleRecord(rid, "quarantined", rule, op, op)
        self.records[rid] = rec
        self.store.quarantine_candidate(rid, rule_to_logic(rule), op, reason=reason, step=self.cycle)
        rec.pre_admission_snapshot = self.store.head_snapshot_id
        sid = self.store.admit_candidate(rid, op, reason=reason, step=self.cycle)
        rec.status, rec.admission = "active", None
        rec.rollback = self._new_rollback(rec)
        rec.admitted_cycle = self.cycle
        rec.n_admissions += 1
        rec.events.append((self.cycle, "active:injected"))
        return sid

    # ---------------------------------------------------------- transitions
    def _promote(self, rec: RuleRecord, step: int) -> None:
        proc = rec.admission
        rec.pre_admission_snapshot = self.store.head_snapshot_id
        sid = self.store.admit_candidate(rec.rule_id, rec.opinion, reason="M_admit >= 1/alpha_admit",
                                         step=step)
        self.store.record_certificate(rec.rule_id, sid, "admission", proc.to_dict(), step=step)
        rec.status, rec.admission = "active", None
        rec.rollback = self._new_rollback(rec)
        rec.admitted_cycle = step
        rec.n_admissions += 1
        rec.events.append((step, "promoted"))

    def _rollback(self, rec: RuleRecord, step: int) -> List[str]:
        proc = rec.rollback
        if self.config.rollback_policy == "snapshot" and rec.pre_admission_snapshot is not None:
            before = {r.rule_id for r in self.store.get_active_rules()}
            target = {r.rule_id for r in self.store.get_active_rules(rec.pre_admission_snapshot)}
            collateral = sorted(before - target - {rec.rule_id})
            sid = self.store.rollback_to_snapshot(rec.pre_admission_snapshot,
                                                  quarantine_rule_ids=[rec.rule_id] + collateral,
                                                  reason=f"M_roll >= 1/alpha_roll ({rec.rule_id})",
                                                  step=step, preserve_non_active=True)
            after = {r.rule_id for r in self.store.get_active_rules()}
            demoted = sorted(before - after)
        else:
            sid = self.store.revert_rule(rec.rule_id, reason="M_roll >= 1/alpha_roll", step=step,
                                         restored_from=rec.pre_admission_snapshot)
            demoted = [rec.rule_id]
        self.store.record_certificate(rec.rule_id, sid, "rollback", proc.to_dict(), step=step)
        for rid in demoted:
            other = self.records.get(rid)
            if other is None:
                continue
            other.n_rollbacks += 1
            other.rolled_back_cycle = step
            other.rollback = None
            other.events.append((step, "rolled_back" if rid == rec.rule_id else "demoted:snapshot"))
            if other.n_admissions >= self.config.max_admissions_per_rule:
                other.status = "rejected"
                self.store.reject_candidate(rid, reason="admission budget exhausted", step=step)
            else:
                other.status = "quarantined"
                other.admission = self._new_admission(other)
        # Rules the snapshot restore re-activated are not expected under either policy, but
        # keep the in-memory view consistent with the store regardless.
        self._resync_from_store()
        return demoted

    def _resync_from_store(self) -> None:
        active = {r.rule_id for r in self.store.get_active_rules()}
        for rid, rec in self.records.items():
            if rid in active and rec.status != "active":
                rec.status, rec.admission = "active", None
                rec.rollback = self._new_rollback(rec)

    # ------------------------------------------------------------- updates
    def observe(self, rule_id: str, y: int, step: Optional[int] = None) -> Optional[str]:
        """Feed one verified firing outcome ``y`` (1 = fraud, 0 = false alarm)."""
        rec = self.records[str(rule_id)]
        step = self.cycle if step is None else step
        y = int(y)
        if rec.status == "quarantined":
            if rec.admission.update(y):
                self._promote(rec, step)
                return "promoted"
            if rec.admission.n >= self.config.max_firings_per_candidate:
                rec.status, rec.admission = "rejected", None
                self.store.reject_candidate(rec.rule_id, reason="firing budget exhausted", step=step)
                rec.events.append((step, "rejected:budget"))
                return "rejected"
        elif rec.status == "active":
            if rec.rollback.update(1 - y):
                self._rollback(rec, step)
                return "rolled_back"
        return None

    def run_update_cycle(self, outcomes: Mapping[str, Sequence[int]]) -> CycleReport:
        """Process one validation cycle of verified outcomes per rule (sorted, deterministic)."""
        self.cycle += 1
        step = self.cycle
        promoted: List[str] = []
        rolled: List[str] = []
        rejected: List[str] = []
        n_obs = 0
        touched: Dict[str, Tuple[int, int]] = {}
        for rid in sorted(outcomes):
            rec = self.records.get(str(rid))
            if rec is None or rec.status == "rejected":
                continue
            ys = [int(v) for v in outcomes[rid]]
            hits = fa = 0
            for y in ys:
                if rec.status == "rejected":
                    break
                status_before = rec.status
                event = self.observe(rec.rule_id, y, step)
                n_obs += 1
                hits += y
                fa += 1 - y
                if event == "promoted":
                    promoted.append(rec.rule_id)
                elif event == "rolled_back":
                    rolled.append(rec.rule_id)
                    break  # remaining firings of this cycle predate the quarantine decision
                elif event == "rejected":
                    rejected.append(rec.rule_id)
                if status_before == "active" and rec.status != "active":
                    break
            if hits or fa:
                rec.opinion = rec.opinion.observe(hits, fa)
                touched[rec.rule_id] = (hits, fa)
        if touched:
            self.store.record_opinions({rid: self.records[rid].opinion for rid in touched}, step=step)
        return CycleReport(step, n_obs, promoted, rolled, rejected, self.store.head_snapshot_id)

    # -------------------------------------------------------------- queries
    @property
    def quarantine_buffer(self) -> List[str]:
        return sorted(r for r, rec in self.records.items() if rec.status == "quarantined")

    @property
    def active_rule_ids(self) -> List[str]:
        return sorted(r for r, rec in self.records.items() if rec.status == "active")

    def status(self, rule_id: str) -> str:
        return self.records[str(rule_id)].status

    def ville_report(self, rule_id: str) -> Dict[str, object]:
        """Current wealth, threshold ``1/alpha`` and anytime-valid p-value of the live test."""
        rec = self.records[str(rule_id)]
        proc = rec.admission if rec.status == "quarantined" else rec.rollback
        out: Dict[str, object] = {"rule_id": rec.rule_id, "status": rec.status}
        if proc is not None:
            out.update(proc.to_dict())
            out["ville_bound"] = proc.alpha
        return out

    def summary(self) -> Dict[str, object]:
        counts: Dict[str, int] = {}
        for rec in self.records.values():
            counts[rec.status] = counts.get(rec.status, 0) + 1
        return {"cycle": self.cycle, "counts": counts,
                "n_promotions": sum(rec.n_admissions for rec in self.records.values()),
                "n_rollbacks": sum(rec.n_rollbacks for rec in self.records.values()),
                "head_snapshot_id": self.store.head_snapshot_id}
