"""
src/conformal/budgeted_crc.py
=============================

Budgeted cost-sensitive adaptive conformal risk control (CS-ACRC, Bayes rule).

Motivation
----------
The threshold controller of :mod:`src.conformal.adaptive_crc` routes on the
score alone, with fixed bands ``lambda``. On PaySim it has three defects: the
risk target ``alpha`` lies above the fraud rate (so "approve everything" is
inside the budget), the loss cap ``min(L / Z, 1)`` removes the transaction
amount, and the narrow bands leave Step-Up and Hold almost unused. This module
replaces the bands with the cost matrix itself and adds two online
controllers with anytime (ACI-type) guarantees.

Decision rule
-------------
For a calibrated probability ``p_t`` and amount ``A_t``::

    q_t   = sigmoid(logit(p_t) + theta_t)                 (drift-corrected risk)
    E_0   = m q_t A_t                                     Auto-Approve
    E_1   = c_step + beta q_t A_t            + nu_t       Step-Up
    E_2   = c_hold                           + nu_t       Hold-and-Verify
    E_3   = (1 - q_t) c_block                + nu_t       Immediate Block
    a_t   = argmin_a E_a

``nu_t = exp(eta_t) - 1 >= 0`` is the Lagrange price of one intervention.
With ``nu = 0`` and ``theta = 0`` the rule is the Bayes-optimal action under
the asymmetric loss. ``E_a`` has slopes ``(m A, beta A, 0, -c_block)`` in
``q``, so the chosen action is non-decreasing in ``q`` (monotone routing) and
non-increasing in ``nu``.

Online updates (from released feedback only)
--------------------------------------------
* Intervention budget ``rho`` (dual ascent on the constraint E[I] <= rho)::

      eta_{t+1} = clip(eta_t + gamma_b (I_t - rho), 0, eta_max),  I_t = 1[a_t >= Step-Up]

  Telescoping gives ``gamma_b sum_t (I_t - rho) = eta_{T+1} - eta_1 - C_T``,
  hence ``(1/T) sum_t I_t <= rho + (eta_max + C^+_T) / (gamma_b T)``, where
  ``C^+_T`` is the displacement of the upper clip. The lower clip at 0 only
  fires when the plain Bayes rule already uses less than the budget.
* Calibration-in-the-large under drift::

      theta_{t+1} = clip(theta_t + gamma_c (y_t - q_t), -theta_max, theta_max)

  which gives ``|(1/T) sum_t (y_t - q_t)| <= (2 theta_max + |C_T|) / (gamma_c T)``
  for every label sequence (no exchangeability assumption).

Both guarantees hold under bounded label delay because the identities do not
depend on when each outcome is observed.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.conformal.loss_functions import (
    DEFAULT_PROJECT_CONFIG_PATH,
    Action,
    AsymmetricLoss,
    CostParameters,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPERPARAMS_PATH = ROOT / "configs" / "hyperparams.json"
_P_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _P_EPS), 1.0 - _P_EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class BudgetStep:
    """Record of one feedback update."""

    t: int
    action: int
    y: int
    amount: float
    loss_raw: float
    loss_normalized: float
    intervened: bool
    nu_before: float
    nu_after: float
    theta_before: float
    theta_after: float


class BudgetedBayesCRC:
    """Cost-sensitive Bayes routing with an online intervention budget and drift shift.

    Parameters
    ----------
    intervention_budget : ``rho``, target long-run share of transactions that
        receive Step-Up, Hold or Block.
    gamma_budget : step size ``gamma_b`` of the budget price (log scale).
    gamma_calibration : step size ``gamma_c`` of the intercept shift ``theta``.
    eta_max : upper bound of ``eta`` (``nu <= exp(eta_max) - 1``).
    theta_max : bound of ``|theta|``.
    calibration_shift : enable the ``theta`` controller.
    mode : ``"adaptive"`` (both controllers on) or ``"static"`` (frozen state).
    reference_amount : amount used to report probability cut points
        (``thresholds``) for logging and figures.
    loss : asymmetric loss (cost matrix) of the routing problem.
    """

    def __init__(self, intervention_budget: float = 0.05, gamma_budget: float = 0.05,
                 gamma_calibration: float = 0.05, eta_max: float = 20.0, theta_max: float = 4.0,
                 calibration_shift: bool = True, mode: str = "adaptive",
                 reference_amount: float = 100_000.0, eta_init: float = 0.0,
                 loss: Optional[AsymmetricLoss] = None, rolling_window: int = 1000,
                 record_history: bool = False) -> None:
        if not (0.0 < intervention_budget <= 1.0):
            raise ValueError("intervention_budget must lie in (0, 1]")
        if gamma_budget <= 0.0 or gamma_calibration <= 0.0:
            raise ValueError("step sizes must be > 0")
        if eta_max <= 0.0 or theta_max <= 0.0:
            raise ValueError("eta_max and theta_max must be > 0")
        if mode not in ("adaptive", "static"):
            raise ValueError("mode must be 'adaptive' or 'static'")
        if reference_amount <= 0.0:
            raise ValueError("reference_amount must be > 0")
        self.rho = float(intervention_budget)
        self.gamma_b = float(gamma_budget)
        self.gamma_c = float(gamma_calibration)
        self.eta_max = float(eta_max)
        self.theta_max = float(theta_max)
        self.calibration_shift = bool(calibration_shift)
        self.mode = mode
        self.reference_amount = float(reference_amount)
        self.loss_fn = loss or AsymmetricLoss(CostParameters.from_config())
        self.rolling_window = int(rolling_window)
        self.record_history = bool(record_history)
        self._eta_init = min(max(float(eta_init), 0.0), self.eta_max)
        c = self.loss_fn.costs
        self._m, self._cs, self._ch = c.undetected_fraud_multiplier, c.c_step, c.c_hold
        self._cb, self._beta = c.c_block_churn, c.beta_bypass
        self.reset()

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None,
                    project_config: Optional[Mapping] = None, **overrides) -> "BudgetedBayesCRC":
        """Build from ``hyperparams.json['conformal_crc']['budgeted_bayes']``."""
        if hyperparams is None:
            with open(DEFAULT_HYPERPARAMS_PATH, "r", encoding="utf-8") as fh:
                hyperparams = json.load(fh)
        if project_config is None:
            with open(DEFAULT_PROJECT_CONFIG_PATH, "r", encoding="utf-8") as fh:
                project_config = json.load(fh)
        crc = hyperparams.get("conformal_crc", hyperparams)
        cfg = dict(crc.get("budgeted_bayes", {}))
        kwargs = dict(
            intervention_budget=float(cfg.get("intervention_budget", 0.05)),
            gamma_budget=float(cfg.get("gamma_budget", 0.05)),
            gamma_calibration=float(cfg.get("gamma_calibration", 0.05)),
            eta_max=float(cfg.get("eta_max", 20.0)),
            theta_max=float(cfg.get("theta_max", 4.0)),
            calibration_shift=bool(cfg.get("calibration_shift", True)),
            mode=str(crc.get("mode", "adaptive")),
            reference_amount=float(cfg.get("reference_amount", 100_000.0)),
            loss=AsymmetricLoss(CostParameters.from_config(project_config)),
            rolling_window=int(crc.get("rolling_window", 1000)),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def reset(self) -> None:
        self._eta = self._eta_init
        self._theta = 0.0
        self._eta_start, self._theta_start = self._eta, self._theta
        self._clip_eta_up = 0.0
        self._clip_eta = 0.0
        self._clip_theta = 0.0
        self.t = 0
        self.n_adaptive = 0
        self._sum_loss = 0.0
        self._sum_ell = 0.0
        self._sum_int = 0.0
        self._sum_int_ad = 0.0
        self._sum_resid = 0.0
        self._n_resid = 0
        self._n_fraud = 0
        self._window: Deque[Tuple[float, int, int]] = deque()
        self._win_sum = 0.0
        self._win_int = 0
        self._win_fraud = 0
        self._hist: list = []

    # -- state -------------------------------------------------------------- #
    @property
    def nu(self) -> float:
        """Current price of one intervention (monetary units)."""
        return math.expm1(self._eta)

    @property
    def theta(self) -> float:
        return self._theta

    def adjusted_probability(self, p: float) -> float:
        return _sigmoid(_logit(p) + self._theta) if self.calibration_shift else min(max(p, 0.0), 1.0)

    def expected_losses(self, p: float, amount: float) -> Tuple[float, float, float, float]:
        q = self.adjusted_probability(p)
        nu = self.nu
        return (self._m * q * amount,
                self._cs + self._beta * q * amount + nu,
                self._ch + nu,
                (1.0 - q) * self._cb + nu)

    def decide(self, p: float, amount: float) -> int:
        """Bayes action under the current price ``nu`` and shift ``theta``."""
        e = self.expected_losses(p, amount)
        best = 0
        for k in (1, 2, 3):
            if e[k] < e[best]:
                best = k
        return best

    def decide_batch(self, p: Sequence[float], amounts: Sequence[float]) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), _P_EPS, 1.0 - _P_EPS)
        a = np.asarray(amounts, dtype=np.float64)
        if self.calibration_shift:
            q = 1.0 / (1.0 + np.exp(-(np.log(p / (1.0 - p)) + self._theta)))
        else:
            q = p
        nu = self.nu
        e = np.stack([self._m * q * a, self._cs + self._beta * q * a + nu,
                      np.full_like(q, self._ch + nu), (1.0 - q) * self._cb + nu])
        return e.argmin(axis=0).astype(np.int64)

    def action_for_score(self, score: float) -> Action:
        """Action at the reference amount (compatibility with the threshold controller)."""
        return Action(self.decide(float(score), self.reference_amount))

    @property
    def thresholds(self) -> Tuple[float, float, float]:
        """Probability cut points (Step-Up, Hold, Block) at ``reference_amount``.

        Reported for logging and figures only; the decision uses each
        transaction's own amount. Unused actions get the next float above the
        previous cut point so that the triple is strictly increasing.
        """
        key = (self._eta, self._theta)
        if getattr(self, "_thr_key", None) == key:
            return self._thr_val
        grid = np.linspace(0.0, 1.0, 4001)
        acts = self.decide_batch(grid, np.full_like(grid, self.reference_amount))
        cuts = []
        prev = 0.0
        for k in (1, 2, 3):
            idx = np.flatnonzero(acts >= k)
            c = float(grid[idx[0]]) if idx.size else 1.0
            c = max(c, math.nextafter(prev, 2.0)) if cuts else c
            cuts.append(c)
            prev = c
        self._thr_key, self._thr_val = key, (cuts[0], cuts[1], cuts[2])
        return self._thr_val

    # -- warm start --------------------------------------------------------- #
    def warm_start(self, p: Sequence[float], amounts: Sequence[float]) -> float:
        """Set ``eta`` so that the Bayes rule meets the budget on past scores.

        Bisection on ``eta`` over historical (label-free) scores, e.g. the
        calibration split. Returns the resulting intervention rate.
        """
        p = np.asarray(p, dtype=np.float64)
        a = np.asarray(amounts, dtype=np.float64)
        if p.size == 0:
            return float("nan")
        lo, hi = 0.0, self.eta_max
        self._eta = lo
        if (self.decide_batch(p, a) >= 1).mean() <= self.rho:
            self._eta_start = self._eta
            return float((self.decide_batch(p, a) >= 1).mean())
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            self._eta = mid
            if (self.decide_batch(p, a) >= 1).mean() > self.rho:
                lo = mid
            else:
                hi = mid
        self._eta = hi
        self._eta_start = self._eta
        return float((self.decide_batch(p, a) >= 1).mean())

    # -- online update ------------------------------------------------------ #
    def update(self, action: Union[Action, int, str], y: int, amount: float,
               p: Optional[float] = None) -> BudgetStep:
        """Observe one released outcome and adapt ``nu`` and ``theta``."""
        a = int(Action.parse(action))
        y = int(y)
        loss = self.loss_fn.loss(a, y, amount)
        ell = loss / (float(amount) + self._cb)
        intervened = a >= int(Action.STEP_UP)
        nu_b, th_b = self.nu, self._theta
        self.t += 1
        self._sum_loss += loss
        self._sum_ell += ell
        self._sum_int += intervened
        self._n_fraud += y
        self._window.append((ell, int(intervened), y))
        self._win_sum += ell
        self._win_int += int(intervened)
        self._win_fraud += y
        if len(self._window) > self.rolling_window:
            o_ell, o_int, o_y = self._window.popleft()
            self._win_sum -= o_ell
            self._win_int -= o_int
            self._win_fraud -= o_y
        if self.mode == "adaptive":
            self.n_adaptive += 1
            self._sum_int_ad += intervened
            raw = self._eta + self.gamma_b * (float(intervened) - self.rho)
            new = min(max(raw, 0.0), self.eta_max)
            d = new - raw
            self._clip_eta += d
            if d < 0.0:
                self._clip_eta_up += -d
            self._eta = new
            if self.calibration_shift and p is not None:
                q = self.adjusted_probability(float(p))
                self._sum_resid += y - q
                self._n_resid += 1
                raw = self._theta + self.gamma_c * (y - q)
                new = min(max(raw, -self.theta_max), self.theta_max)
                self._clip_theta += new - raw
                self._theta = new
        if self.record_history:
            self._hist.append((self.nu, self._theta, ell, int(intervened)))
        return BudgetStep(self.t, a, y, float(amount), loss, ell, intervened, nu_b, self.nu,
                          th_b, self._theta)

    def update_batch(self, actions: Sequence, y: Sequence[int], amounts: Sequence[float],
                     p: Optional[Sequence[float]] = None) -> np.ndarray:
        out = np.empty(len(y), dtype=np.float64)
        for i in range(len(y)):
            pi = None if p is None else float(p[i])
            out[i] = self.update(actions[i], int(y[i]), float(amounts[i]), pi).loss_normalized
        return out

    # -- monitoring --------------------------------------------------------- #
    @property
    def rolling_loss(self) -> float:
        return self._win_sum / len(self._window) if self._window else 0.0

    @property
    def rolling_intervention_rate(self) -> float:
        return self._win_int / len(self._window) if self._window else 0.0

    @property
    def rolling_fraud_rate(self) -> float:
        return self._win_fraud / len(self._window) if self._window else 0.0

    @property
    def cumulative_raw_loss(self) -> float:
        return self._sum_loss

    def guarantee(self) -> Dict[str, float]:
        """Anytime certificates of the two online controllers."""
        n = self.n_adaptive
        rate = self._sum_int_ad / n if n else float("nan")
        budget_bound = (self.rho + (self.eta_max + self._clip_eta_up) / (self.gamma_b * n)
                        if n else math.inf)
        nr = self._n_resid
        resid_mean = self._sum_resid / nr if nr else float("nan")
        calib_bound = ((2.0 * self.theta_max + abs(self._clip_theta)) / (self.gamma_c * nr)
                       if nr else math.inf)
        identity = abs(self.gamma_b * (self._sum_int_ad - n * self.rho)
                       - (self._eta - self._eta_start - self._clip_eta)) if n else 0.0
        return {
            "T_adaptive": float(n), "intervention_budget": self.rho,
            "intervention_rate": rate, "intervention_upper_bound": budget_bound,
            "within_budget_bound": bool(n and rate <= budget_bound + 1e-12),
            "calibration_residual_mean": resid_mean, "calibration_bound": calib_bound,
            "within_calibration_bound": bool(nr and abs(resid_mean) <= calib_bound + 1e-12)
            if self.calibration_shift else True,
            "budget_identity_residual": identity,
            "nu": self.nu, "theta": self._theta,
        }

    def summary(self) -> Dict[str, object]:
        return {"t": self.t, "mode": self.mode, "nu": self.nu, "theta": self._theta,
                "thresholds": dict(zip(("low", "mid", "high"), self.thresholds)),
                "rolling_loss": self.rolling_loss,
                "rolling_intervention_rate": self.rolling_intervention_rate,
                "rolling_fraud_rate": self.rolling_fraud_rate,
                "cumulative_raw_loss": self._sum_loss,
                **{f"guarantee_{k}": v for k, v in self.guarantee().items()}}

    def history(self) -> Dict[str, np.ndarray]:
        h = np.asarray(self._hist, dtype=np.float64).reshape(-1, 4)
        return {"nu": h[:, 0], "theta": h[:, 1], "loss": h[:, 2], "intervened": h[:, 3]}

    def state_dict(self) -> Dict[str, object]:
        return {"kind": "budgeted_bayes", "eta": self._eta, "theta": self._theta, "t": self.t,
                "n_adaptive": self.n_adaptive, "mode": self.mode, "eta_start": self._eta_start,
                "sum_int": self._sum_int, "sum_int_ad": self._sum_int_ad,
                "sum_resid": self._sum_resid, "n_resid": self._n_resid,
                "clip_eta": self._clip_eta, "clip_eta_up": self._clip_eta_up,
                "clip_theta": self._clip_theta}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("kind") != "budgeted_bayes":
            raise ValueError("state does not belong to a BudgetedBayesCRC")
        self._eta = min(max(float(state["eta"]), 0.0), self.eta_max)
        self._theta = float(state["theta"])
        self.t = int(state.get("t", 0))
        self.n_adaptive = int(state.get("n_adaptive", 0))
        self.mode = str(state.get("mode", self.mode))
        self._eta_start = float(state.get("eta_start", self._eta))
        for key in ("sum_int", "sum_int_ad", "sum_resid", "clip_eta", "clip_eta_up", "clip_theta"):
            setattr(self, f"_{key}", float(state.get(key, 0.0)))
        self._n_resid = int(state.get("n_resid", 0))


def build_controller(hyperparams: Optional[Mapping] = None, project_config: Optional[Mapping] = None,
                     **overrides):
    """Factory: ``conformal_crc.decision_rule`` selects the controller.

    ``"threshold"`` (legacy) -> :class:`OnlineAdaptiveCRC`; ``"budgeted_bayes"``
    (default when configured) -> :class:`BudgetedBayesCRC`.
    """
    from src.conformal.adaptive_crc import OnlineAdaptiveCRC
    if hyperparams is None:
        with open(DEFAULT_HYPERPARAMS_PATH, "r", encoding="utf-8") as fh:
            hyperparams = json.load(fh)
    rule = str(hyperparams.get("conformal_crc", {}).get("decision_rule", "threshold"))
    if rule == "budgeted_bayes":
        return BudgetedBayesCRC.from_config(hyperparams, project_config, **overrides)
    if rule == "threshold":
        return OnlineAdaptiveCRC.from_config(hyperparams, project_config, **overrides)
    raise ValueError("conformal_crc.decision_rule must be 'threshold' or 'budgeted_bayes'")
