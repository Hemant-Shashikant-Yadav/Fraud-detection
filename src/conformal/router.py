"""
src/conformal/router.py
=======================

Pre-transaction prevention router (MG-ECFM Stage 1B, component C4).

Input: the calibrated risk tuple ``(p_hat_t, Delta_p_hat_t, A_t, active_rules)``

* ``p_hat_t``        calibrated fraud probability from the Stage 1A fusion scorer;
* ``Delta_p_hat_t``  epistemic width of the calibration (e.g. Venn-Abers
                     ``p1 - p0``), ``>= 0``;
* ``A_t``            transaction amount (carried to the loss on feedback);
* ``active_rules``   admitted knowledge-base rules firing on ``e_t``
                     (ids, ``{id: belief}`` mapping, or a count).

Decision rule (in order):

1. Conservative score ``s_t = clip(p_hat_t + kappa * Delta_p_hat_t, 0, 1)``.
2. Threshold rule on the online CS-ACRC thresholds ``lambda_t``
   (see :mod:`src.conformal.adaptive_crc`).
3. Optional amount-aware escalation: Approve -> Step-Up when the plug-in
   conditional risk of approving exceeds that of a step-up,
   ``s_t * A_t * (m - beta_bypass) > c_step``.
4. Knowledge-base floor: if any admitted rule fires, the action is at least
   ``rule_floor_action`` (default Step-Up).
5. Epistemic guard: a Block whose calibration width exceeds
   ``block_max_uncertainty`` is downgraded to Hold-and-Verify, whose loss is
   ``c_hold`` whatever the outcome (an uncertain Block risks ``c_block_churn``).

The hot path is pure-Python scalar arithmetic (no NumPy / allocation beyond
the decision record), well inside the 1 ms routing budget.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.conformal.adaptive_crc import DEFAULT_HYPERPARAMS_PATH, CRCStep, OnlineAdaptiveCRC
from src.conformal.loss_functions import Action

ActiveRules = Union[None, int, Sequence, Mapping]

REASON_THRESHOLD = "threshold"
REASON_AMOUNT = "amount_escalation"
REASON_RULE_FLOOR = "kb_rule_floor"
REASON_UNCERTAIN_BLOCK = "uncertain_block_to_hold"


class RoutingDecision:
    """Outcome of :meth:`PreTransactionRouter.route` (slot-based, cheap to build)."""

    __slots__ = ("action", "score", "p_hat", "delta_p_hat", "amount", "thresholds",
                 "reason", "n_active_rules", "latency_ms")

    def __init__(self, action: Action, score: float, p_hat: float, delta_p_hat: float,
                 amount: float, thresholds: Tuple[float, float, float], reason: str,
                 n_active_rules: int, latency_ms: float) -> None:
        self.action = action
        self.score = score
        self.p_hat = p_hat
        self.delta_p_hat = delta_p_hat
        self.amount = amount
        self.thresholds = thresholds
        self.reason = reason
        self.n_active_rules = n_active_rules
        self.latency_ms = latency_ms

    @property
    def label(self) -> str:
        return self.action.label

    def to_dict(self) -> Dict[str, object]:
        d = {s: getattr(self, s) for s in self.__slots__}
        d["action"] = int(self.action)
        d["action_label"] = self.label
        return d

    def __repr__(self) -> str:
        return (f"RoutingDecision(action={self.action.name}, score={self.score:.4f}, "
                f"reason={self.reason}, latency_ms={self.latency_ms:.4f})")


def _count_rules(active_rules: ActiveRules) -> int:
    if active_rules is None:
        return 0
    if isinstance(active_rules, bool):
        return int(active_rules)
    if isinstance(active_rules, (int, np.integer)):
        if active_rules < 0:
            raise ValueError("active rule count must be >= 0")
        return int(active_rules)
    return len(active_rules)


class PreTransactionRouter:
    """Route calibrated risk tuples to {Approve, Step-Up, Hold, Block} under CS-ACRC.

    Parameters
    ----------
    controller : the online threshold controller (shared state with feedback).
    kappa_uncertainty : weight ``kappa`` of ``Delta_p_hat`` in the conservative score.
    block_max_uncertainty : maximum calibration width for which Block is allowed.
    rule_floor_action : minimum action when an admitted KB rule fires (``None`` disables).
    amount_aware_escalation : enable step 3 of the decision rule.
    latency_budget_ms : routing budget checked by :meth:`latency_profile`.
    """

    def __init__(self, controller: Optional[OnlineAdaptiveCRC] = None,
                 kappa_uncertainty: float = 0.5, block_max_uncertainty: float = 0.25,
                 rule_floor_action: Optional[Union[Action, int, str]] = Action.STEP_UP,
                 amount_aware_escalation: bool = False,
                 latency_budget_ms: float = 1.0) -> None:
        if not (kappa_uncertainty >= 0.0 and math.isfinite(kappa_uncertainty)):
            raise ValueError("kappa_uncertainty must be finite and >= 0")
        if not (block_max_uncertainty >= 0.0):
            raise ValueError("block_max_uncertainty must be >= 0")
        self.controller = controller or OnlineAdaptiveCRC()
        self.kappa = float(kappa_uncertainty)
        self.block_max_uncertainty = float(block_max_uncertainty)
        self.rule_floor: Optional[Action] = (None if rule_floor_action is None
                                             else Action.parse(rule_floor_action))
        self.amount_aware = bool(amount_aware_escalation)
        self.latency_budget_ms = float(latency_budget_ms)
        costs = self.controller.loss_fn.costs
        self._c_step = costs.c_step
        self._esc_coef = costs.undetected_fraud_multiplier - costs.beta_bypass
        self.action_counts = [0, 0, 0, 0]
        self.n_routed = 0

    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None,
                    project_config: Optional[Mapping] = None,
                    controller: Optional[OnlineAdaptiveCRC] = None, **overrides) -> "PreTransactionRouter":
        """Build from ``hyperparams.json['conformal_crc']['router']``."""
        if hyperparams is None:
            with open(DEFAULT_HYPERPARAMS_PATH, "r", encoding="utf-8") as fh:
                hyperparams = json.load(fh)
        controller = controller or OnlineAdaptiveCRC.from_config(hyperparams, project_config)
        cfg = dict(hyperparams.get("conformal_crc", {}).get("router", {}))
        cfg.update(overrides)
        return cls(controller=controller, **cfg)

    # -- hot path ----------------------------------------------------------- #
    def route(self, p_hat: float, delta_p_hat: float = 0.0, amount: float = 0.0,
              active_rules: ActiveRules = None) -> RoutingDecision:
        """Select the pre-settlement action for one transaction."""
        t0 = time.perf_counter()
        p = float(p_hat)
        d = float(delta_p_hat)
        amt = float(amount)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p_hat must lie in [0, 1], got {p_hat!r}")
        if not (d >= 0.0 and d <= 1.0):
            raise ValueError(f"delta_p_hat must lie in [0, 1], got {delta_p_hat!r}")
        if not (amt >= 0.0 and amt < math.inf):
            raise ValueError(f"amount must be finite and >= 0, got {amount!r}")
        n_rules = _count_rules(active_rules)

        s = p + self.kappa * d
        if s > 1.0:
            s = 1.0
        lam = self.controller._lam  # read-only view of the live thresholds
        lo, mid, hi = lam[0], lam[1], lam[2]
        if s < lo:
            a = 0
        elif s < mid:
            a = 1
        elif s < hi:
            a = 2
        else:
            a = 3
        reason = REASON_THRESHOLD
        if a == 0 and self.amount_aware and s * amt * self._esc_coef > self._c_step:
            a, reason = 1, REASON_AMOUNT
        if n_rules and self.rule_floor is not None and a < self.rule_floor:
            a, reason = int(self.rule_floor), REASON_RULE_FLOOR
        if a == 3 and d > self.block_max_uncertainty:
            a, reason = 2, REASON_UNCERTAIN_BLOCK

        self.action_counts[a] += 1
        self.n_routed += 1
        action = Action(a)
        latency = (time.perf_counter() - t0) * 1000.0
        return RoutingDecision(action, s, p, d, amt, (lo, mid, hi), reason, n_rules, latency)

    def route_tuple(self, risk_tuple: Sequence) -> RoutingDecision:
        """Route ``(p_hat, Delta_p_hat, A_t, active_rules)``."""
        p, d, amt, rules = risk_tuple
        return self.route(p, d, amt, rules)

    def route_scored_event(self, score: Mapping[str, object], amount: float,
                           active_rules: ActiveRules = None) -> RoutingDecision:
        """Route the dict returned by ``StackedFusionClassifier.score_event``.

        ``Delta_p_hat`` is taken from ``score['delta_p_hat']`` or from a
        Venn-Abers interval ``score['p_interval'] = (p0, p1)`` when present.
        """
        d = score.get("delta_p_hat")
        if d is None and score.get("p_interval") is not None:
            p0, p1 = score["p_interval"]
            d = float(p1) - float(p0)
        return self.route(float(score["p_hat"]), float(d or 0.0), amount, active_rules)

    def route_batch(self, p_hat: Sequence[float], delta_p_hat: Optional[Sequence[float]] = None,
                    amounts: Optional[Sequence[float]] = None,
                    active_rules: Optional[Sequence[ActiveRules]] = None) -> np.ndarray:
        """Route a sequence against the *current* thresholds (no feedback in between)."""
        n = len(p_hat)
        d = delta_p_hat if delta_p_hat is not None else [0.0] * n
        amt = amounts if amounts is not None else [0.0] * n
        rules = active_rules if active_rules is not None else [None] * n
        return np.fromiter((int(self.route(p_hat[i], d[i], amt[i], rules[i]).action)
                            for i in range(n)), dtype=np.int64, count=n)

    # -- feedback ----------------------------------------------------------- #
    def feedback(self, decision: RoutingDecision, y: int) -> CRCStep:
        """Report the verified label of a routed transaction to the controller."""
        return self.controller.update(decision.action, y, decision.amount)

    # -- monitoring --------------------------------------------------------- #
    def action_distribution(self) -> Dict[str, float]:
        n = max(self.n_routed, 1)
        return {a.name.lower(): self.action_counts[int(a)] / n for a in Action}

    def latency_profile(self, tuples: Iterable[Sequence], warmup: int = 200) -> Dict[str, float]:
        """Per-call routing latency percentiles (ms) over ``tuples``."""
        tuples = list(tuples)
        for tup in tuples[:warmup]:
            self.route_tuple(tup)
        lat = np.empty(len(tuples), dtype=np.float64)
        for i, tup in enumerate(tuples):
            t0 = time.perf_counter()
            self.route_tuple(tup)
            lat[i] = (time.perf_counter() - t0) * 1000.0
        return {"n": float(lat.size), "mean_ms": float(lat.mean()),
                "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                "p99_ms": float(np.percentile(lat, 99)), "max_ms": float(lat.max()),
                "within_budget": bool(np.percentile(lat, 99) < self.latency_budget_ms)}
