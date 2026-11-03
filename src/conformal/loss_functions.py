"""
src/conformal/loss_functions.py
===============================

Asymmetric monetary and customer-friction loss ``L(a_t, y_t, A_t)`` for the
MG-ECFM Stage 1 pre-transaction router (CS-ACRC).

Actions (pre-settlement friction tiers, ordered by severity)::

    0  APPROVE   Auto-Approve
    1  STEP_UP   Step-Up Authentication (biometric / OTP challenge)
    2  HOLD      Hold-and-Verify Escrow (funds held until analyst review)
    3  BLOCK     Immediate Block

Loss matrix (``A_t`` = transaction amount, ``y_t = 1`` fraud)::

                     y = 0 (legitimate)      y = 1 (fraud)
    APPROVE          0                       m * A_t
    STEP_UP          c_step                  c_step + beta_bypass * A_t
    HOLD             c_hold                  c_hold      (fraud intercepted
                                                          before settlement)
    BLOCK            c_block_churn           0

``m = undetected_fraud_multiplier`` (default 1.0, i.e. exactly ``A_t``).

Hold-and-Verify is the only action whose loss does not depend on ``y_t``:
it caps the realised loss at ``c_hold`` irrespective of the amount at risk,
saving ``m * A_t - c_hold`` relative to approving a fraudulent transaction.

The online controller (``adaptive_crc.py``) consumes the *normalised, capped*
loss ``ell_t = min(L_t / Z, B)``, where ``Z = loss_normalizer`` puts monetary
units on the scale of the target risk ``alpha`` and ``B = loss_cap`` makes the
loss bounded, which the adaptive-conformal guarantee requires.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_CONFIG_PATH = ROOT / "configs" / "project_config.json"

ArrayLike = Union[Sequence[float], np.ndarray]


class Action(IntEnum):
    """Pre-settlement routing actions, ordered by increasing severity."""

    APPROVE = 0
    STEP_UP = 1
    HOLD = 2
    BLOCK = 3

    @property
    def label(self) -> str:
        return _ACTION_LABELS[self]

    @classmethod
    def parse(cls, value: Union["Action", int, str]) -> "Action":
        """Accept an ``Action``, its integer code or a name/alias string."""
        if isinstance(value, Action):
            return value
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return cls(int(value))
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "_").replace(" ", "_")
            if key in _ACTION_ALIASES:
                return _ACTION_ALIASES[key]
        raise ValueError(f"unknown action {value!r}")


_ACTION_LABELS: Dict[Action, str] = {
    Action.APPROVE: "Auto-Approve",
    Action.STEP_UP: "Step-Up Auth",
    Action.HOLD: "Hold-and-Verify Escrow",
    Action.BLOCK: "Immediate Block",
}
_ACTION_ALIASES: Dict[str, Action] = {
    "approve": Action.APPROVE, "auto_approve": Action.APPROVE, "allow": Action.APPROVE,
    "step_up": Action.STEP_UP, "stepup": Action.STEP_UP, "step_up_auth": Action.STEP_UP,
    "challenge": Action.STEP_UP,
    "hold": Action.HOLD, "hold_and_verify": Action.HOLD, "escrow": Action.HOLD,
    "hold_and_verify_escrow": Action.HOLD,
    "block": Action.BLOCK, "immediate_block": Action.BLOCK, "decline": Action.BLOCK,
}
N_ACTIONS = len(Action)


@dataclass(frozen=True)
class CostParameters:
    """Cost constants of the asymmetric loss (monetary units of the dataset).

    Attributes
    ----------
    c_step : friction penalty of a step-up challenge.
    c_hold : operational analyst-review cost of a Hold-and-Verify escrow.
    c_block_churn : false-positive churn penalty of blocking a genuine customer.
    beta_bypass : fraction of fraudulent value that defeats the step-up
        challenge, ``0 <= beta_bypass <= 1``.
    undetected_fraud_multiplier : ``m`` in ``L(APPROVE, 1) = m * A_t``.
    loss_normalizer : ``Z`` in ``ell_t = min(L_t / Z, B)``.
    loss_cap : ``B`` in ``ell_t = min(L_t / Z, B)`` (bounded-loss requirement).
    """

    c_step: float = 5.0
    c_hold: float = 25.0
    c_block_churn: float = 150.0
    beta_bypass: float = 0.10
    undetected_fraud_multiplier: float = 1.0
    loss_normalizer: float = 10000.0
    loss_cap: float = 1.0

    def __post_init__(self) -> None:
        for name in ("c_step", "c_hold", "c_block_churn", "undetected_fraud_multiplier"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {v!r}")
        if not (0.0 <= self.beta_bypass <= 1.0):
            raise ValueError(f"beta_bypass must lie in [0, 1], got {self.beta_bypass!r}")
        if not (math.isfinite(self.loss_normalizer) and self.loss_normalizer > 0.0):
            raise ValueError("loss_normalizer must be finite and > 0")
        if not (self.loss_cap > 0.0):
            raise ValueError("loss_cap must be > 0 (use math.inf for no cap)")

    @classmethod
    def from_config(cls, config: Optional[Mapping] = None,
                    path: Optional[Path] = None, **overrides: float) -> "CostParameters":
        """Build from ``project_config.json['costs']`` (or a ``costs`` mapping)."""
        if config is None:
            with open(path or DEFAULT_PROJECT_CONFIG_PATH, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        costs = config.get("costs", config)
        kwargs = {f: float(costs[f]) for f in cls.__dataclass_fields__ if f in costs}
        kwargs.update({k: float(v) for k, v in overrides.items()})
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Scalar (single-transaction) evaluation                                      #
# --------------------------------------------------------------------------- #
def _check_scalar(y: int, amount: float) -> None:
    if y not in (0, 1):
        raise ValueError(f"y must be 0 or 1, got {y!r}")
    if not (math.isfinite(amount) and amount >= 0.0):
        raise ValueError(f"amount A_t must be finite and >= 0, got {amount!r}")


def asymmetric_loss(action: Union[Action, int, str], y: int, amount: float,
                    c_step: float, c_hold: float, c_block_churn: float,
                    beta_bypass: float, undetected_fraud_multiplier: float = 1.0) -> float:
    """``L(action, y, A_t; c_step, c_hold, c_block_churn, beta_bypass)``.

    Exact implementation of the eight-cell CS-ACRC loss matrix (module doc).
    """
    a = Action.parse(action)
    y = int(y)
    amount = float(amount)
    _check_scalar(y, amount)
    if a is Action.APPROVE:
        return undetected_fraud_multiplier * amount if y == 1 else 0.0
    if a is Action.STEP_UP:
        return c_step + beta_bypass * amount if y == 1 else c_step
    if a is Action.HOLD:
        return c_hold  # y = 1: fraud intercepted before settlement, only review cost
    return 0.0 if y == 1 else c_block_churn  # BLOCK


class AsymmetricLoss:
    """Callable loss object bound to a :class:`CostParameters` instance.

    ``__call__`` / :meth:`loss` evaluate a single transaction in pure Python
    (no NumPy dispatch, ~1 us); :meth:`batch` evaluates whole arrays.
    """

    __slots__ = ("costs", "_c_step", "_c_hold", "_c_block", "_beta", "_m", "_z", "_cap")

    def __init__(self, costs: Optional[CostParameters] = None) -> None:
        self.costs = costs or CostParameters()
        c = self.costs
        self._c_step, self._c_hold, self._c_block = c.c_step, c.c_hold, c.c_block_churn
        self._beta, self._m = c.beta_bypass, c.undetected_fraud_multiplier
        self._z, self._cap = c.loss_normalizer, c.loss_cap

    # -- single transaction ------------------------------------------------- #
    def loss(self, action: Union[Action, int, str], y: int, amount: float) -> float:
        """Raw monetary/friction loss ``L(a, y, A)``."""
        a = int(Action.parse(action))
        y = int(y)
        amount = float(amount)
        _check_scalar(y, amount)
        if a == 0:
            return self._m * amount if y else 0.0
        if a == 1:
            return self._c_step + self._beta * amount if y else self._c_step
        if a == 2:
            return self._c_hold
        return 0.0 if y else self._c_block

    __call__ = loss

    def normalized(self, action: Union[Action, int, str], y: int, amount: float) -> float:
        """Bounded controller loss ``ell = min(L / Z, B)``."""
        v = self.loss(action, y, amount) / self._z
        return v if v < self._cap else self._cap

    def matrix(self, amount: float) -> np.ndarray:
        """``(4, 2)`` loss matrix ``M[a, y]`` for a given amount."""
        return np.array([[self.loss(a, y, amount) for y in (0, 1)] for a in Action],
                        dtype=np.float64)

    def hold_savings(self, amount: float) -> float:
        """Loss avoided by Hold-and-Verify vs Auto-Approve on a fraud: ``m A - c_hold``."""
        return self.loss(Action.APPROVE, 1, amount) - self.loss(Action.HOLD, 1, amount)

    # -- vectorised --------------------------------------------------------- #
    def batch(self, actions: ArrayLike, y: ArrayLike, amounts: ArrayLike) -> np.ndarray:
        """Vectorised ``L`` over aligned arrays (actions as int codes or names)."""
        a = _as_action_array(actions)
        yy = np.asarray(y)
        amt = np.asarray(amounts, dtype=np.float64)
        a, yy, amt = np.broadcast_arrays(a, yy, amt)
        if yy.size and not np.isin(yy, (0, 1)).all():
            raise ValueError("y must contain only 0/1")
        if amt.size and (not np.isfinite(amt).all() or (amt < 0).any()):
            raise ValueError("amounts must be finite and >= 0")
        yy = yy.astype(bool)
        fixed = np.array([[0.0, 0.0],
                          [self._c_step, self._c_step],
                          [self._c_hold, self._c_hold],
                          [self._c_block, 0.0]], dtype=np.float64)
        amount_coef = np.array([[0.0, self._m],
                                [0.0, self._beta],
                                [0.0, 0.0],
                                [0.0, 0.0]], dtype=np.float64)
        yi = yy.astype(np.intp)
        return fixed[a, yi] + amount_coef[a, yi] * amt

    def batch_normalized(self, actions: ArrayLike, y: ArrayLike, amounts: ArrayLike) -> np.ndarray:
        return np.minimum(self.batch(actions, y, amounts) / self._z, self._cap)

    # -- decision-theoretic helpers ----------------------------------------- #
    def expected_losses(self, p: float, amount: float) -> np.ndarray:
        """Conditional risk ``E[L(a, Y, A) | p]`` for every action, ``Y ~ Bern(p)``."""
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must lie in [0, 1]")
        m = self.matrix(amount)
        return (1.0 - p) * m[:, 0] + p * m[:, 1]

    def bayes_action(self, p: float, amount: float) -> Action:
        """Plug-in Bayes action ``argmin_a E[L | p]`` (ties -> less severe action)."""
        return Action(int(np.argmin(self.expected_losses(p, amount))))


def _as_action_array(actions: ArrayLike) -> np.ndarray:
    arr = np.asarray(actions)
    if arr.dtype.kind in ("U", "S", "O"):
        arr = np.vectorize(lambda v: int(Action.parse(v)), otypes=[np.intp])(arr)
    arr = arr.astype(np.intp, copy=False)
    if arr.size and ((arr < 0).any() or (arr >= N_ACTIONS).any()):
        raise ValueError("action codes must lie in {0, 1, 2, 3}")
    return arr


def batch_asymmetric_loss(actions: ArrayLike, y: ArrayLike, amounts: ArrayLike,
                          costs: Optional[CostParameters] = None) -> np.ndarray:
    """Functional form of :meth:`AsymmetricLoss.batch`."""
    return AsymmetricLoss(costs).batch(actions, y, amounts)
