"""
src/graph/temporal_memory.py
============================

Persistent per-entity temporal memory for the MG-ECFM relational fast path.

Each entity node ``v`` keeps a compact state ``s_v(t^-) in R^{d_s}`` and the
timestamp ``tau_v^last`` of its last update. On a new event
``e_t = (u_t, v_t, x_t, A_t, tau_t)`` touching ``v`` the state evolves as

    s_v(t) = s_v(t^-) * exp(-delta * (tau_t - tau_v^last)) + W_mem [x_t || A_t]

Reads and writes are O(1) per node: entity keys are resolved through a hash
map to a row of a pre-allocated tensor whose capacity grows geometrically
(amortised O(1) registration).

Batch semantics (consistent with the Phase-1 strict step causality)
-------------------------------------------------------------------
* :meth:`TemporalNodeMemory.event_states` returns, for every event in a batch,
  the entity's *pre-batch* state decayed to the event time plus the event's
  own message. Events of the same batch do not see each other, so the output
  is invariant to within-batch permutation. Gradients flow into ``W_mem``.
* :meth:`TemporalNodeMemory.commit` writes the exact closed form of applying
  every event of the batch sequentially in time order:

      s_v(T_v) = s_v(t^-) e^{-delta (T_v - tau_v^last)}
                 + sum_i W_mem m_i e^{-delta (T_v - tau_i)},  T_v = max_i tau_i

  which equals event-by-event updates bit-for-bit up to float rounding.

The stored memory is detached from the autograd graph (TGN-style), so no
back-propagation through time is required.

The module also provides :class:`SignedLogStandardizer`, the shared,
leakage-safe feature scaler (``sign(x) log1p|x|`` followed by z-scoring with
statistics fitted on the training split only) used by the graph and
autoencoder engines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

SEED = 42
ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


# --------------------------------------------------------------------------- #
# Shared feature scaling                                                      #
# --------------------------------------------------------------------------- #
class SignedLogStandardizer:
    """``z = (sign(x) log1p|x| - mu) / sigma`` with train-only statistics.

    Heavy-tailed monetary and ledger features (PaySim amounts span seven
    orders of magnitude) are compressed by the signed log before z-scoring.
    Columns with zero variance get ``sigma = 1``. Non-finite inputs map to 0.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = float(eps)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    @staticmethod
    def signed_log(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.where(np.isfinite(x), x, 0.0)
        return np.sign(x) * np.log1p(np.abs(x))

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    @property
    def n_features(self) -> int:
        if self.mean_ is None:
            raise RuntimeError("SignedLogStandardizer is not fitted")
        return int(self.mean_.shape[0])

    def fit(self, x: np.ndarray) -> "SignedLogStandardizer":
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] == 0:
            raise ValueError("Cannot fit a scaler on an empty matrix")
        z = self.signed_log(arr)
        self.mean_ = z.mean(axis=0)
        std = z.std(axis=0)
        self.scale_ = np.where(std > self.eps, std, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("SignedLogStandardizer is not fitted")
        arr = np.asarray(x, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None, :] if arr.shape[0] == self.mean_.shape[0] else arr[:, None]
        if arr.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"Expected {self.mean_.shape[0]} features, received {arr.shape[1]}"
            )
        out = ((self.signed_log(arr) - self.mean_) / self.scale_).astype(np.float32)
        return out[0] if squeeze and out.shape[0] == 1 else out

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> Dict[str, object]:
        return {
            "eps": self.eps,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> "SignedLogStandardizer":
        self.eps = float(state.get("eps", 1e-6))
        mean, scale = state.get("mean"), state.get("scale")
        self.mean_ = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.scale_ = None if scale is None else np.asarray(scale, dtype=np.float64)
        return self


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TemporalMemoryConfig:
    """Hyper-parameters of :class:`TemporalNodeMemory`."""

    feature_dim: int
    memory_dim: int = 64
    decay_delta: float = 0.05
    message_includes_amount: bool = True
    max_staleness_steps: Optional[float] = 720.0
    initial_capacity: int = 1024
    init: str = "xavier_uniform"

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.memory_dim <= 0:
            raise ValueError("memory_dim must be positive")
        if self.decay_delta < 0:
            raise ValueError("decay_delta must be non-negative")
        if self.initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        if self.max_staleness_steps is not None and self.max_staleness_steps <= 0:
            raise ValueError("max_staleness_steps must be positive or None")
        if self.init not in ("xavier_uniform", "orthogonal", "normal"):
            raise ValueError(f"Unsupported init '{self.init}'")

    @classmethod
    def from_hyperparams(cls, hyperparams: Mapping, feature_dim: int) -> "TemporalMemoryConfig":
        tm = hyperparams.get("temporal_memory", {}) if hyperparams else {}
        staleness = tm.get("max_staleness_steps", 720)
        return cls(
            feature_dim=int(feature_dim),
            memory_dim=int(tm.get("memory_dim", 64)),
            decay_delta=float(tm.get("decay_delta", 0.05)),
            message_includes_amount=bool(tm.get("message_includes_amount", True)),
            max_staleness_steps=None if staleness is None else float(staleness),
            init=str(tm.get("init", "xavier_uniform")),
        )


# --------------------------------------------------------------------------- #
# Temporal node memory                                                        #
# --------------------------------------------------------------------------- #
class TemporalNodeMemory(nn.Module):
    """In-memory entity state store with exponential temporal decay.

    Args:
        config: :class:`TemporalMemoryConfig`.
        seed: Seed for the deterministic initialisation of ``W_mem``.
    """

    def __init__(self, config: TemporalMemoryConfig, seed: int = SEED) -> None:
        super().__init__()
        self.config = config
        self.delta = float(config.decay_delta)
        msg_dim = config.feature_dim + (1 if config.message_includes_amount else 0)
        self.message_dim = msg_dim
        self.W_mem = nn.Linear(msg_dim, config.memory_dim, bias=False)
        gen = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            if config.init == "xavier_uniform":
                bound = math.sqrt(6.0 / (msg_dim + config.memory_dim))
                self.W_mem.weight.copy_(
                    torch.rand(self.W_mem.weight.shape, generator=gen) * 2 * bound - bound
                )
            elif config.init == "orthogonal":
                a = torch.randn(max(msg_dim, config.memory_dim), min(msg_dim, config.memory_dim),
                                generator=gen)
                q, _ = torch.linalg.qr(a)
                w = q if config.memory_dim >= msg_dim else q.T
                self.W_mem.weight.copy_(w[: config.memory_dim, :msg_dim])
            else:
                self.W_mem.weight.copy_(
                    torch.randn(self.W_mem.weight.shape, generator=gen) / math.sqrt(msg_dim)
                )
        cap = int(config.initial_capacity)
        self.register_buffer("memory", torch.zeros(cap, config.memory_dim), persistent=True)
        self.register_buffer("last_update", torch.zeros(cap, dtype=torch.float64), persistent=True)
        self.register_buffer("n_updates", torch.zeros(cap, dtype=torch.long), persistent=True)
        self._key_to_index: Dict[Hashable, int] = {}
        self._index_to_key: List[Hashable] = []
        self.feature_scaler = SignedLogStandardizer()
        self.amount_mean = 0.0
        self.amount_scale = 1.0

    # --------------------------------------------------------------- registry
    @property
    def n_nodes(self) -> int:
        return len(self._index_to_key)

    @property
    def capacity(self) -> int:
        return int(self.memory.shape[0])

    @property
    def memory_dim(self) -> int:
        return int(self.config.memory_dim)

    def _ensure_capacity(self, n: int) -> None:
        cap = self.capacity
        if n <= cap:
            return
        new_cap = cap
        while new_cap < n:
            new_cap *= 2
        mem = torch.zeros(new_cap, self.memory_dim, dtype=self.memory.dtype)
        mem[:cap] = self.memory
        last = torch.zeros(new_cap, dtype=torch.float64)
        last[:cap] = self.last_update
        cnt = torch.zeros(new_cap, dtype=torch.long)
        cnt[:cap] = self.n_updates
        self.memory, self.last_update, self.n_updates = mem, last, cnt

    def register(self, keys: Iterable[Hashable]) -> torch.Tensor:
        """Map entity keys to row indices, allocating rows for unseen keys."""
        table = self._key_to_index
        out: List[int] = []
        for key in keys:
            idx = table.get(key)
            if idx is None:
                idx = len(self._index_to_key)
                table[key] = idx
                self._index_to_key.append(key)
            out.append(idx)
        self._ensure_capacity(len(self._index_to_key))
        return torch.as_tensor(out, dtype=torch.long)

    def register_one(self, key: Hashable) -> int:
        """O(1) single-key registration (fast path)."""
        idx = self._key_to_index.get(key)
        if idx is None:
            idx = len(self._index_to_key)
            self._key_to_index[key] = idx
            self._index_to_key.append(key)
            self._ensure_capacity(idx + 1)
        return idx

    def lookup(self, key: Hashable) -> Optional[int]:
        return self._key_to_index.get(key)

    def key_of(self, index: int) -> Hashable:
        return self._index_to_key[int(index)]

    def reset_state(self) -> None:
        """Forget all node states (weights and scaler are kept)."""
        self.memory.zero_()
        self.last_update.zero_()
        self.n_updates.zero_()
        self._key_to_index = {}
        self._index_to_key = []

    # -------------------------------------------------------------- messages
    def fit_normalizer(self, features: np.ndarray, amounts: Optional[np.ndarray] = None) -> None:
        """Fit the message scaler on training-split data only."""
        self.feature_scaler.fit(features)
        if self.config.message_includes_amount:
            if amounts is None:
                raise ValueError("amounts are required when message_includes_amount=True")
            la = np.log1p(np.maximum(np.asarray(amounts, dtype=np.float64), 0.0))
            self.amount_mean = float(la.mean())
            sd = float(la.std())
            self.amount_scale = sd if sd > 1e-6 else 1.0

    def build_message_input(self, features: np.ndarray, amounts: Optional[np.ndarray]) -> torch.Tensor:
        """Return the scaled ``[x_t || A_t]`` tensor of shape ``(n, message_dim)``."""
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim == 1:
            feats = feats[None, :]
        if feats.shape[1] != self.config.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.config.feature_dim}, received {feats.shape[1]}"
            )
        if self.feature_scaler.is_fitted:
            x = self.feature_scaler.transform(feats)
        else:
            x = SignedLogStandardizer.signed_log(feats).astype(np.float32)
        if self.config.message_includes_amount:
            if amounts is None:
                raise ValueError("amounts are required when message_includes_amount=True")
            a = np.log1p(np.maximum(np.asarray(amounts, dtype=np.float64).reshape(-1), 0.0))
            a = ((a - self.amount_mean) / self.amount_scale).astype(np.float32)[:, None]
            x = np.concatenate([x, a], axis=1)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    def message(self, message_input: torch.Tensor) -> torch.Tensor:
        """``W_mem [x_t || A_t]``."""
        return self.W_mem(message_input)

    # ------------------------------------------------------------------ decay
    def _decay_factor(self, idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``exp(-delta * (t - tau_last))`` (0 for never-seen or stale nodes)."""
        dt = (t.to(torch.float64) - self.last_update[idx]).clamp_min(0.0)
        factor = torch.exp(-self.delta * dt)
        seen = self.n_updates[idx] > 0
        if self.config.max_staleness_steps is not None:
            seen = seen & (dt <= self.config.max_staleness_steps)
        return torch.where(seen, factor, torch.zeros_like(factor)).to(self.memory.dtype)

    @staticmethod
    def _as_time(t: Union[float, ArrayLike], n: int) -> torch.Tensor:
        tt = torch.as_tensor(t, dtype=torch.float64)
        if tt.dim() == 0:
            tt = tt.expand(n)
        if tt.shape[0] != n:
            raise ValueError("timestamps must be scalar or match the number of nodes")
        return tt

    def read(self, idx: Union[torch.Tensor, Sequence[int]], t: Union[float, ArrayLike]) -> torch.Tensor:
        """Decayed states ``s_v(t^-) * exp(-delta * (t - tau_v^last))`` (detached, O(1)/node)."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        if idx.numel() == 0:
            return torch.zeros(0, self.memory_dim)
        tt = self._as_time(t, idx.shape[0])
        return self.memory[idx] * self._decay_factor(idx, tt).unsqueeze(1)

    def read_one(self, index: int, t: float) -> torch.Tensor:
        """Scalar fast path of :meth:`read` returning a ``(d_s,)`` tensor."""
        if int(self.n_updates[index]) == 0:
            return torch.zeros(self.memory_dim)
        dt = max(0.0, float(t) - float(self.last_update[index]))
        if self.config.max_staleness_steps is not None and dt > self.config.max_staleness_steps:
            return torch.zeros(self.memory_dim)
        return self.memory[index] * math.exp(-self.delta * dt)

    def event_states(
        self,
        idx: torch.Tensor,
        t: Union[float, ArrayLike],
        message_input: torch.Tensor,
    ) -> torch.Tensor:
        """Per-event post-event states with gradient through ``W_mem``.

        ``s_v(t) = read(v, t) + W_mem m_t`` where ``read`` uses the pre-batch
        state, i.e. events in one batch do not observe each other.
        """
        idx = torch.as_tensor(idx, dtype=torch.long)
        return self.read(idx, t) + self.message(message_input)

    @torch.no_grad()
    def commit(
        self,
        idx: torch.Tensor,
        t: Union[float, ArrayLike],
        message_input: torch.Tensor,
    ) -> None:
        """Write the exact sequential update of all batch events to memory."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        n = idx.shape[0]
        if n == 0:
            return
        tt = self._as_time(t, n)
        msgs = self.message(message_input).detach().to(self.memory.dtype)
        uniq, inverse = torch.unique(idx, return_inverse=True)
        t_final = torch.full((uniq.shape[0],), -math.inf, dtype=torch.float64)
        t_final = t_final.scatter_reduce(0, inverse, tt, reduce="amax", include_self=True)
        base = self.memory[uniq] * self._decay_factor(uniq, t_final).unsqueeze(1)
        weights = torch.exp(-self.delta * (t_final[inverse] - tt)).to(self.memory.dtype)
        base.index_add_(0, inverse, msgs * weights.unsqueeze(1))
        counts = torch.bincount(inverse, minlength=uniq.shape[0])
        self.memory[uniq] = base
        self.last_update[uniq] = t_final
        self.n_updates[uniq] += counts

    def update(
        self,
        idx: torch.Tensor,
        t: Union[float, ArrayLike],
        message_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-event states, then commit the batch. Returns the states."""
        states = self.event_states(idx, t, message_input)
        self.commit(idx, t, message_input)
        return states

    @torch.no_grad()
    def update_one(self, index: int, t: float, message_input: torch.Tensor) -> torch.Tensor:
        """O(1) single-event update; returns the new state ``s_v(t)``."""
        msg = self.W_mem(message_input.reshape(1, -1))[0]
        new_state = self.read_one(index, t) + msg
        self.memory[index] = new_state
        self.last_update[index] = float(t)
        self.n_updates[index] += 1
        return new_state

    # ------------------------------------------------------------ persistence
    def export_state(self) -> Dict[str, object]:
        n = self.n_nodes
        return {
            "keys": list(self._index_to_key),
            "memory": self.memory[:n].clone(),
            "last_update": self.last_update[:n].clone(),
            "n_updates": self.n_updates[:n].clone(),
            "scaler": self.feature_scaler.state_dict(),
            "amount_mean": self.amount_mean,
            "amount_scale": self.amount_scale,
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        keys = list(state["keys"])
        self.reset_state()
        self.register(keys)
        n = len(keys)
        self.memory[:n] = torch.as_tensor(state["memory"])
        self.last_update[:n] = torch.as_tensor(state["last_update"], dtype=torch.float64)
        self.n_updates[:n] = torch.as_tensor(state["n_updates"], dtype=torch.long)
        self.feature_scaler.load_state_dict(state["scaler"])
        self.amount_mean = float(state["amount_mean"])
        self.amount_scale = float(state["amount_scale"])
