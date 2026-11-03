"""
src/conformal/adaptive_crc.py
=============================

Cost-Sensitive Online Adaptive Conformal Risk Control (CS-ACRC).

The router maps a calibrated risk score ``s_t`` to one of four actions with an
ordered threshold vector ``lambda_t = (lambda^low, lambda^mid, lambda^high)``::

    s_t <  lambda^low                 -> Auto-Approve
    lambda^low  <= s_t < lambda^mid   -> Step-Up Auth
    lambda^mid  <= s_t < lambda^high  -> Hold-and-Verify Escrow
    s_t >= lambda^high                -> Immediate Block

Online update (Gibbs & Candes, 2021, Adaptive Conformal Inference, applied to
the bounded asymmetric loss ``ell_t = min(L(a_t, y_t, A_t) / Z, B)``)
-------------------------------------------------------------------------
Let ``theta_t = 1 - lambda_t`` be the *protection level* (larger = more
interventions; the analogue of ``1 - alpha_t`` in ACI). The synopsis update is
applied to it verbatim::

    theta_{t+1} = theta_t - gamma * (alpha - ell_t)

which in threshold coordinates is ``lambda_{t+1} = lambda_t + gamma (alpha - ell_t)``:
a loss above target lowers every threshold (more step-ups / holds / blocks),
a loss below target raises them (less friction). The vector is then projected
(Euclidean) onto the feasible set

    lb <= lambda^low,  lambda^low + g <= lambda^mid,  lambda^mid + g <= lambda^high <= ub

with ``g > 0`` (``min_threshold_gap``), which enforces *strict* monotonicity.

Finite-sample guarantee (no exchangeability assumption)
-------------------------------------------------------
Telescoping the update for any component ``k`` gives the exact identity

    gamma_k * sum_{t<=T} (alpha - ell_t) = lambda^k_{T+1} - lambda^k_1 - C^k_T

where ``C^k_T`` is the cumulative projection displacement. Hence, for every
data sequence (adversarial, drifting, delayed feedback),

    | (1/T) sum ell_t - alpha | <= (range_k + |C^k_T|) / (gamma_k * T) =: eps_T.

Splitting ``C^k_T = C^{k,+}_T + C^{k,-}_T`` into upward (protective-extreme)
and downward (permissive-extreme) projection displacement, and noting that a
permissive clip only ever *lowers* the loss, gives the one-sided risk bound

    (1/T) sum ell_t <= alpha + (range_k + C^{k,+}_T) / (gamma_k * T).

When the protective extreme has loss below ``alpha`` (see :meth:`feasibility`)
``C^{k,+}_T`` stays O(gamma) and the excess risk is O(1/T): this is the
long-run expected-loss control of CS-ACRC. Thresholds resting at the
permissive bound simply mean the stream is cheaper than the risk budget.
:meth:`OnlineAdaptiveCRC.guarantee` reports both bounds and the identity residual.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.conformal.loss_functions import (
    DEFAULT_PROJECT_CONFIG_PATH,
    Action,
    AsymmetricLoss,
    CostParameters,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPERPARAMS_PATH = ROOT / "configs" / "hyperparams.json"
THRESHOLD_NAMES: Tuple[str, str, str] = ("low", "mid", "high")


# --------------------------------------------------------------------------- #
# Projection                                                                  #
# --------------------------------------------------------------------------- #
def _pava_nondecreasing(x: Sequence[float]) -> List[float]:
    """Pool-adjacent-violators: L2 projection onto ``{z_1 <= ... <= z_K}``."""
    blocks: List[List[float]] = []  # [sum, count]
    for v in x:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out: List[float] = []
    for s, c in blocks:
        out.extend([s / c] * int(c))
    return out


def project_ordered_thresholds(lam: Sequence[float], lower: float, upper: float,
                               gap: float) -> List[float]:
    """Euclidean projection onto ``{lower <= l_1, l_{k+1} - l_k >= gap, l_K <= upper}``.

    Substituting ``mu_k = l_k - (k-1) gap`` turns the gap constraints into
    ``mu_1 <= ... <= mu_K`` and the bounds into ``mu_k in [lower, upper-(K-1)gap]``;
    the projection onto (isotonic cone) n (uniform box) is the box-clip of the
    isotonic (PAVA) fit, and the substitution is an isometric translation.
    """
    k = len(lam)
    hi = upper - (k - 1) * gap
    if gap <= 0.0:
        raise ValueError("gap must be > 0 for strict monotonicity")
    if hi < lower:
        raise ValueError("infeasible: lower + (K-1)*gap > upper")
    mu = [float(v) - i * gap for i, v in enumerate(lam)]
    fit = _pava_nondecreasing(mu)
    return [min(max(m, lower), hi) + i * gap for i, m in enumerate(fit)]


# --------------------------------------------------------------------------- #
# Controller                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CRCStep:
    """Result of one online update."""

    t: int
    action: int
    y: int
    amount: float
    loss: float
    loss_normalized: float
    thresholds_before: Tuple[float, float, float]
    thresholds_after: Tuple[float, float, float]
    projected: bool
    adapted: bool


class OnlineAdaptiveCRC:
    """Online adaptive conformal risk controller over ``(lambda^low, lambda^mid, lambda^high)``.

    Parameters
    ----------
    alpha : target long-run normalised loss ``E[ell_t]``.
    gamma : ACI step size, scalar or one value per threshold.
    lambda_init : initial thresholds (projected onto the feasible set).
    lambda_bounds : ``(lb, ub)`` box for the thresholds.
    min_gap : strict-monotonicity margin ``g > 0``.
    warmup : number of initial feedback events during which thresholds are frozen
        (losses are still tracked; the guarantee covers the adaptive segment).
    mode : ``"adaptive"`` (CS-ACRC) or ``"static"`` (frozen thresholds, ablation).
    loss : :class:`AsymmetricLoss` or :class:`CostParameters`.
    rolling_window : length of the rolling empirical-loss window.
    record_history : keep the threshold / loss trajectory for analysis.
    """

    MODES = ("adaptive", "static")

    def __init__(self, alpha: float = 0.02, gamma: Union[float, Sequence[float]] = 0.005,
                 lambda_init: Sequence[float] = (0.20, 0.50, 0.85),
                 lambda_bounds: Sequence[float] = (0.0, 1.0), min_gap: float = 0.01,
                 warmup: int = 0, mode: str = "adaptive",
                 loss: Optional[Union[AsymmetricLoss, CostParameters]] = None,
                 rolling_window: int = 1000, record_history: bool = True) -> None:
        if not (0.0 < alpha):
            raise ValueError("alpha must be > 0")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        gam = [float(gamma)] * 3 if np.isscalar(gamma) else [float(g) for g in gamma]
        if len(gam) != 3 or any(not (g > 0.0 and math.isfinite(g)) for g in gam):
            raise ValueError("gamma must be a positive scalar or 3 positive values")
        lb, ub = float(lambda_bounds[0]), float(lambda_bounds[1])
        if not (lb < ub):
            raise ValueError("lambda_bounds must satisfy lb < ub")
        if not (min_gap > 0.0 and lb + 2.0 * min_gap <= ub):
            raise ValueError("min_gap must be > 0 with lb + 2*min_gap <= ub")
        if len(lambda_init) != 3:
            raise ValueError("lambda_init must have 3 entries (low, mid, high)")
        if rolling_window < 1 or warmup < 0:
            raise ValueError("rolling_window >= 1 and warmup >= 0 required")

        self.alpha = float(alpha)
        self.gamma: Tuple[float, float, float] = (gam[0], gam[1], gam[2])
        self.lower, self.upper, self.min_gap = lb, ub, float(min_gap)
        self.warmup = int(warmup)
        self.mode = mode
        if isinstance(loss, CostParameters) or loss is None:
            loss = AsymmetricLoss(loss)
        self.loss_fn: AsymmetricLoss = loss
        self.rolling_window = int(rolling_window)
        self.record_history = bool(record_history)
        self._lambda_init = tuple(float(v) for v in lambda_init)
        self.reset()

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None,
                    project_config: Optional[Mapping] = None, **overrides) -> "OnlineAdaptiveCRC":
        """Build from ``hyperparams.json['conformal_crc']`` and ``project_config.json['costs']``."""
        if hyperparams is None:
            with open(DEFAULT_HYPERPARAMS_PATH, "r", encoding="utf-8") as fh:
                hyperparams = json.load(fh)
        if project_config is None:
            with open(DEFAULT_PROJECT_CONFIG_PATH, "r", encoding="utf-8") as fh:
                project_config = json.load(fh)
        crc = hyperparams.get("conformal_crc", hyperparams)
        init = crc.get("lambda_init", {"low": 0.2, "mid": 0.5, "high": 0.85})
        if isinstance(init, Mapping):
            init = [init[n] for n in THRESHOLD_NAMES]
        cost_overrides = {}
        if "loss_cap" in crc:
            cost_overrides["loss_cap"] = float(crc["loss_cap"])
        kwargs = dict(
            alpha=float(crc.get("target_risk_alpha", 0.02)),
            gamma=crc.get("gamma_step_size", 0.005),
            lambda_init=init,
            lambda_bounds=tuple(crc.get("lambda_bounds", (0.0, 1.0))),
            min_gap=float(crc.get("min_threshold_gap", 0.01)),
            warmup=int(crc.get("warmup_transactions", 0)),
            mode=str(crc.get("mode", "adaptive")),
            loss=AsymmetricLoss(CostParameters.from_config(project_config, **cost_overrides)),
            rolling_window=int(crc.get("rolling_window", 1000)),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def reset(self) -> None:
        """Restore the initial thresholds and clear all statistics."""
        lam = project_ordered_thresholds(self._lambda_init, self.lower, self.upper, self.min_gap)
        self._lam: List[float] = lam
        self._lam_adapt_start: List[float] = list(lam)
        self._clip: List[float] = [0.0, 0.0, 0.0]
        self._clip_up: List[float] = [0.0, 0.0, 0.0]
        self.t = 0
        self.n_adaptive = 0
        self.n_projections = 0
        self._sum_loss = 0.0
        self._sum_ell = 0.0
        self._sum_ell_adaptive = 0.0
        self._max_ell = 0.0
        self._n_fraud = 0
        self._window: Deque[Tuple[float, int]] = deque()
        self._win_sum = 0.0
        self._win_fraud = 0
        self._hist_lam: List[Tuple[float, float, float]] = [tuple(lam)] if self.record_history else []
        self._hist_ell: List[float] = []
        self._hist_rolling: List[float] = []

    # -- thresholds --------------------------------------------------------- #
    @property
    def thresholds(self) -> Tuple[float, float, float]:
        return (self._lam[0], self._lam[1], self._lam[2])

    @property
    def lambda_low(self) -> float:
        return self._lam[0]

    @property
    def lambda_mid(self) -> float:
        return self._lam[1]

    @property
    def lambda_high(self) -> float:
        return self._lam[2]

    @property
    def protection_levels(self) -> Tuple[float, float, float]:
        """``theta_t = 1 - lambda_t`` (the ACI state of the synopsis update)."""
        return (1.0 - self._lam[0], 1.0 - self._lam[1], 1.0 - self._lam[2])

    def threshold_dict(self) -> Dict[str, float]:
        return dict(zip(THRESHOLD_NAMES, self.thresholds))

    def is_strictly_monotone(self, tol: float = 1e-12) -> bool:
        lo, mid, hi = self._lam
        return (self.lower - tol <= lo and lo + self.min_gap - tol <= mid
                and mid + self.min_gap - tol <= hi and hi <= self.upper + tol and lo < mid < hi)

    def action_for_score(self, score: float) -> Action:
        """Threshold rule ``s -> action`` under the current ``lambda_t``."""
        lo, mid, hi = self._lam
        if score < lo:
            return Action.APPROVE
        if score < mid:
            return Action.STEP_UP
        if score < hi:
            return Action.HOLD
        return Action.BLOCK

    # -- online update ------------------------------------------------------ #
    def update(self, action: Union[Action, int, str], y: int, amount: float) -> CRCStep:
        """Observe the realised outcome of a routed transaction and adapt ``lambda``.

        Valid for delayed feedback: the telescoping guarantee does not depend
        on when, relative to routing, each outcome is observed.
        """
        a = int(Action.parse(action))
        y = int(y)
        loss = self.loss_fn.loss(a, y, amount)
        ell = loss / self.loss_fn.costs.loss_normalizer
        cap = self.loss_fn.costs.loss_cap
        if ell > cap:
            ell = cap
        before = (self._lam[0], self._lam[1], self._lam[2])

        self.t += 1
        self._sum_loss += loss
        self._sum_ell += ell
        if ell > self._max_ell:
            self._max_ell = ell
        self._n_fraud += y
        self._window.append((ell, y))
        self._win_sum += ell
        self._win_fraud += y
        if len(self._window) > self.rolling_window:
            old_ell, old_y = self._window.popleft()
            self._win_sum -= old_ell
            self._win_fraud -= old_y

        adapted = projected = False
        if self.mode == "adaptive" and self.t > self.warmup:
            if self.n_adaptive == 0:
                self._lam_adapt_start = list(self._lam)
            self.n_adaptive += 1
            self._sum_ell_adaptive += ell
            g = self.alpha - ell
            raw = [self._lam[k] + self.gamma[k] * g for k in range(3)]
            proj = project_ordered_thresholds(raw, self.lower, self.upper, self.min_gap)
            for k in range(3):
                d = proj[k] - raw[k]
                if d != 0.0:
                    projected = True
                    if d > 0.0:
                        self._clip_up[k] += d
                self._clip[k] += d
            if projected:
                self.n_projections += 1
            self._lam = proj
            adapted = True

        if self.record_history:
            self._hist_lam.append((self._lam[0], self._lam[1], self._lam[2]))
            self._hist_ell.append(ell)
            self._hist_rolling.append(self._win_sum / len(self._window))
        return CRCStep(self.t, a, y, float(amount), loss, ell, before,
                       (self._lam[0], self._lam[1], self._lam[2]), projected, adapted)

    def update_batch(self, actions: Sequence, y: Sequence[int], amounts: Sequence[float]) -> np.ndarray:
        """Sequential updates over aligned arrays; returns the normalised losses."""
        out = np.empty(len(y), dtype=np.float64)
        for i, (a, yy, amt) in enumerate(zip(actions, y, amounts)):
            out[i] = self.update(a, int(yy), float(amt)).loss_normalized
        return out

    # -- monitoring --------------------------------------------------------- #
    @property
    def rolling_loss(self) -> float:
        """Mean normalised loss over the last ``rolling_window`` outcomes."""
        return self._win_sum / len(self._window) if self._window else 0.0

    @property
    def rolling_coverage_gap(self) -> float:
        """``rolling_loss - alpha`` (> 0: currently over the risk budget)."""
        return self.rolling_loss - self.alpha

    @property
    def rolling_fraud_rate(self) -> float:
        return self._win_fraud / len(self._window) if self._window else 0.0

    @property
    def cumulative_loss(self) -> float:
        """Mean normalised loss over all observed outcomes."""
        return self._sum_ell / self.t if self.t else 0.0

    @property
    def cumulative_raw_loss(self) -> float:
        """Total raw monetary + friction loss."""
        return self._sum_loss

    def threshold_ranges(self) -> Tuple[float, float, float]:
        """Admissible range of each threshold under bounds and gaps."""
        span = self.upper - self.lower - 2.0 * self.min_gap
        return (span, span, span)

    def epsilon_bound(self) -> float:
        """Deterministic ``eps_T`` with ``|mean_adaptive ell - alpha| <= eps_T``."""
        if self.n_adaptive == 0:
            return math.inf
        rng = self.threshold_ranges()
        return min((rng[k] + abs(self._clip[k])) / (self.gamma[k] * self.n_adaptive)
                   for k in range(3))

    def risk_upper_bound(self) -> float:
        """One-sided certificate: ``mean_adaptive ell <= alpha + min_k (range_k + C^{k,+})/(gamma_k T)``."""
        if self.n_adaptive == 0:
            return math.inf
        rng = self.threshold_ranges()
        return self.alpha + min((rng[k] + self._clip_up[k]) / (self.gamma[k] * self.n_adaptive)
                                for k in range(3))

    def window_bound(self, window: int) -> float:
        """Projection-free bound on the mean loss gap over any ``window`` adaptive steps."""
        return min(r / (g * window) for r, g in zip(self.threshold_ranges(), self.gamma))

    def guarantee(self) -> Dict[str, float]:
        """ACI certificate for the adaptive segment of the stream."""
        n = self.n_adaptive
        mean = self._sum_ell_adaptive / n if n else float("nan")
        resid = 0.0
        if n:
            s = n * self.alpha - self._sum_ell_adaptive
            for k in range(3):
                lhs = self.gamma[k] * s
                rhs = self._lam[k] - self._lam_adapt_start[k] - self._clip[k]
                resid = max(resid, abs(lhs - rhs))
        return {
            "T_adaptive": float(n),
            "alpha": self.alpha,
            "mean_loss": mean,
            "coverage_gap": mean - self.alpha if n else float("nan"),
            "epsilon_T": self.epsilon_bound(),
            "within_bound": bool(n and abs(mean - self.alpha) <= self.epsilon_bound() + 1e-12),
            "risk_upper_bound": self.risk_upper_bound(),
            "within_risk_bound": bool(n and mean <= self.risk_upper_bound() + 1e-12),
            "projection_displacement": float(max(abs(c) for c in self._clip)),
            "protective_displacement": float(max(self._clip_up)),
            "n_projections": float(self.n_projections),
            "identity_residual": resid,
        }

    def feasibility(self) -> Dict[str, object]:
        """Check the self-correction preconditions of the ACI guarantee.

        At the protective extreme every genuine customer is blocked or held, so
        the per-transaction loss there is at most ``max(c_block, c_hold)/Z``.
        If that is ``>= alpha`` the target is not attainable and thresholds will
        pin to the lower bound (``epsilon_T`` then grows with ``|C_T|``).
        """
        c = self.loss_fn.costs
        z = c.loss_normalizer
        worst_legit = min(max(c.c_block_churn, c.c_hold) / z, c.loss_cap)
        return {
            "protective_extreme_legit_loss": worst_legit,
            "alpha": self.alpha,
            "feasible": worst_legit < self.alpha,
            "note": ("ok" if worst_legit < self.alpha else
                     "alpha below the friction cost of maximal protection; raise alpha or Z"),
        }

    def history(self) -> Dict[str, np.ndarray]:
        lam = np.asarray(self._hist_lam, dtype=np.float64).reshape(-1, 3)
        return {"lambda": lam, "loss": np.asarray(self._hist_ell, dtype=np.float64),
                "rolling_loss": np.asarray(self._hist_rolling, dtype=np.float64)}

    def summary(self) -> Dict[str, object]:
        return {
            "t": self.t, "mode": self.mode, "thresholds": self.threshold_dict(),
            "rolling_loss": self.rolling_loss, "rolling_coverage_gap": self.rolling_coverage_gap,
            "rolling_fraud_rate": self.rolling_fraud_rate,
            "cumulative_loss": self.cumulative_loss, "cumulative_raw_loss": self._sum_loss,
            "max_normalized_loss": self._max_ell, **{f"guarantee_{k}": v for k, v in
                                                    self.guarantee().items()},
        }

    # -- persistence (snapshot / rollback for the closed loop) --------------- #
    def state_dict(self) -> Dict[str, object]:
        return {
            "lambda": list(self._lam), "lambda_adapt_start": list(self._lam_adapt_start),
            "clip": list(self._clip), "clip_up": list(self._clip_up), "t": self.t, "n_adaptive": self.n_adaptive,
            "n_projections": self.n_projections, "sum_loss": self._sum_loss,
            "sum_ell": self._sum_ell, "sum_ell_adaptive": self._sum_ell_adaptive,
            "max_ell": self._max_ell, "n_fraud": self._n_fraud,
            "window": [list(w) for w in self._window],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        lam = [float(v) for v in state["lambda"]]
        proj = project_ordered_thresholds(lam, self.lower, self.upper, self.min_gap)
        if max(abs(p - v) for p, v in zip(proj, lam)) > 1e-12:
            raise ValueError("state thresholds violate the monotonicity constraints")
        self._lam = lam
        self._lam_adapt_start = [float(v) for v in state["lambda_adapt_start"]]
        self._clip = [float(v) for v in state["clip"]]
        self._clip_up = [float(v) for v in state.get("clip_up", [0.0, 0.0, 0.0])]
        self.t, self.n_adaptive = int(state["t"]), int(state["n_adaptive"])
        self.n_projections = int(state["n_projections"])
        self._sum_loss, self._sum_ell = float(state["sum_loss"]), float(state["sum_ell"])
        self._sum_ell_adaptive = float(state["sum_ell_adaptive"])
        self._max_ell, self._n_fraud = float(state["max_ell"]), int(state["n_fraud"])
        self._window = deque((float(e), int(y)) for e, y in state["window"])
        self._win_sum = sum(e for e, _ in self._window)
        self._win_fraud = sum(y for _, y in self._window)
