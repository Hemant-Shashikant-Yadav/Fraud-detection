"""
src/data/causal_preprocessor.py
===============================

Causal, leakage-free feature engineering for MG-ECFM transaction streams.

Guarantees
----------
1. **No post-decision leakage.** PaySim balance columns (``oldbalanceOrg``,
   ``newbalanceOrig``, ``oldbalanceDest``, ``newbalanceDest``) and the
   rule-artefact column ``isFlaggedFraud`` are forbidden. Any frame containing
   them is rejected with :class:`LeakageError`.
2. **Strict step causality.** Features of a transaction at step ``tau_t`` are
   computed exclusively from transactions with ``step < tau_t``. Transactions
   sharing the same step never see each other (PaySim steps are one-hour
   buckets with no reliable intra-step ordering), so features are invariant
   to within-step row permutation.
3. **Label blindness.** The label column is carried through untouched and is
   never read by any feature computation.
4. **Streaming exactness.** Chunked processing yields bit-identical features
   to whole-frame processing: trailing rows of a step that may continue into
   the next chunk are buffered until that step is complete.

Feature set (per transaction ``e_t = (u_t, v_t, x_t, A_t, tau_t)``)
--------------------------------------------------------------------
* ``log1p_amount``, ``hour_of_day``, one-hot ``type_<T>``
* Source (outgoing) and destination (incoming) role features, prefix ``src_`` /
  ``dst_``: rolling velocity ``vel_w`` over steps ``[tau_t - w, tau_t - 1]``,
  historical count / mean / std of amounts, amount-to-mean ratio, amount
  z-score, steps since last activity, new-entity flag.
* Pre-decision observed-ledger balance features: the entity's observed net
  position accumulated strictly before ``tau_t`` (``*_ledger_pre``), the
  projected causal balance delta after this transaction
  (``*_balance_delta``), and amount-to-ledger ratio.

Usage
-----
    python -m src.data.causal_preprocessor --dataset paysim
    python -m src.data.causal_preprocessor --dataset paysim --max-rows 200000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("mg_ecfm.causal_preprocessor")

SEED = 42
PAYSIM_LEAKAGE_COLUMNS: Tuple[str, ...] = (
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
)
PAYSIM_RULE_ARTIFACT_COLUMNS: Tuple[str, ...] = ("isFlaggedFraud",)
FORBIDDEN_FEATURE_COLUMNS: Tuple[str, ...] = (
    PAYSIM_LEAKAGE_COLUMNS + PAYSIM_RULE_ARTIFACT_COLUMNS
)
CANONICAL_COLUMNS: Tuple[str, ...] = (
    "tx_id", "source_tx_id", "step", "event_time", "src", "dst",
    "amount", "tx_type", "label",
)
REQUIRED_ENGINE_COLUMNS: Tuple[str, ...] = ("step", "src", "amount", "tx_type")
DEFAULT_VELOCITY_WINDOWS: Tuple[int, ...] = (1, 5, 20)
DEFAULT_TX_TYPES: Tuple[str, ...] = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
DEFAULT_INFLOW_TYPES: Tuple[str, ...] = ("CASH_IN",)
MISSING_ENTITY = ""
PASSTHROUGH_PREFIX = "raw_"


class LeakageError(RuntimeError):
    """Raised when post-decision / leakage-prone columns reach the feature layer."""


class TemporalOrderError(RuntimeError):
    """Raised when a transaction stream violates chronological ordering."""


# --------------------------------------------------------------------------- #
# Reproducibility and leakage utilities                                       #
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = SEED) -> None:
    """Fix Python, NumPy and (if installed) PyTorch RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: WPS433 (optional dependency)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def strip_leakage_columns(
    frame: pd.DataFrame,
    forbidden: Sequence[str] = FORBIDDEN_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return ``frame`` without any forbidden (post-decision) columns."""
    present = [c for c in forbidden if c in frame.columns]
    if not present:
        return frame
    return frame.drop(columns=present)


def assert_no_leakage(
    frame: pd.DataFrame,
    forbidden: Sequence[str] = FORBIDDEN_FEATURE_COLUMNS,
) -> None:
    """Raise :class:`LeakageError` if any forbidden column is present."""
    present = [c for c in frame.columns if c in set(forbidden)]
    present += [
        c for c in frame.columns
        if c.startswith(PASSTHROUGH_PREFIX) and c[len(PASSTHROUGH_PREFIX):] in set(forbidden)
    ]
    if present:
        raise LeakageError(
            "Post-decision leakage columns detected and refused: "
            + ", ".join(sorted(set(present)))
        )


def hash_entity_keys(keys: np.ndarray) -> np.ndarray:
    """Deterministically hash entity identifiers to uint64.

    Uses pandas' fixed-key SipHash, which is stable across processes and
    platforms. With 64-bit hashes the collision probability for 10^7 entities
    is ~2.7e-6, negligible for feature statistics.
    """
    arr = np.asarray(keys, dtype=object)
    return pd.util.hash_array(arr, categorize=True)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CausalFeatureConfig:
    """Configuration of the causal feature engineer."""

    velocity_windows: Tuple[int, ...] = DEFAULT_VELOCITY_WINDOWS
    tx_types: Tuple[str, ...] = DEFAULT_TX_TYPES
    inflow_types: Tuple[str, ...] = DEFAULT_INFLOW_TYPES
    include_destination: bool = True
    ratio_smoothing: float = 1.0
    hours_per_day: int = 24

    def __post_init__(self) -> None:
        windows = tuple(int(w) for w in self.velocity_windows)
        if not windows or any(w <= 0 for w in windows):
            raise ValueError("velocity_windows must be non-empty positive integers")
        if len(set(windows)) != len(windows):
            raise ValueError("velocity_windows must be unique")
        object.__setattr__(self, "velocity_windows", tuple(sorted(windows)))
        object.__setattr__(self, "tx_types", tuple(str(t) for t in self.tx_types))
        object.__setattr__(self, "inflow_types", tuple(str(t) for t in self.inflow_types))
        if self.ratio_smoothing <= 0:
            raise ValueError("ratio_smoothing must be > 0")
        if self.hours_per_day <= 0:
            raise ValueError("hours_per_day must be > 0")

    @classmethod
    def from_project_config(cls, project_config: Mapping, dataset: str) -> "CausalFeatureConfig":
        """Build from ``configs/project_config.json`` for ``dataset``."""
        feats = project_config.get("causal_features", {})
        ds = project_config.get("datasets", {}).get(dataset, {})
        return cls(
            velocity_windows=tuple(feats.get("velocity_windows", DEFAULT_VELOCITY_WINDOWS)),
            tx_types=tuple(ds.get("tx_types", DEFAULT_TX_TYPES)),
            inflow_types=tuple(ds.get("inflow_types", DEFAULT_INFLOW_TYPES)),
            include_destination=bool(ds.get("include_destination", True)),
            ratio_smoothing=float(feats.get("ratio_smoothing", 1.0)),
            hours_per_day=int(feats.get("hours_per_day", 24)),
        )


# --------------------------------------------------------------------------- #
# Internal state containers                                                   #
# --------------------------------------------------------------------------- #
def _grow(array: np.ndarray, capacity: int, fill) -> np.ndarray:
    out = np.full(capacity, fill, dtype=array.dtype)
    out[: array.shape[0]] = array
    return out


def _scatter_add(target: np.ndarray, ids: np.ndarray, values: np.ndarray) -> None:
    """``target[ids] += values`` with correct accumulation of duplicate ids."""
    if ids.size == 0:
        return
    ids, values = _canonical_order(ids, values)
    uniq, inverse = np.unique(ids, return_inverse=True)
    sums = np.bincount(inverse.ravel(), weights=values, minlength=uniq.size)
    target[uniq] += sums


def _canonical_order(ids: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by (id, value) so floating-point accumulation is row-order invariant.

    Guarantees bit-identical features under any permutation of rows inside
    a step, not merely equality up to rounding.
    """
    order = np.lexsort((values, ids))
    return ids[order], values[order]


class _EntityRegistry:
    """Maps hashed entity identifiers to dense integer ids (append-only)."""

    def __init__(self) -> None:
        self._codes: Dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._codes)

    def encode(self, keys: np.ndarray) -> np.ndarray:
        if len(keys) == 0:
            return np.empty(0, dtype=np.int64)
        hashed = hash_entity_keys(keys)
        uniq, inverse = np.unique(hashed, return_inverse=True)
        codes = self._codes
        mapped = np.fromiter(
            (codes.setdefault(h, len(codes)) for h in uniq.tolist()),
            dtype=np.int64,
            count=uniq.size,
        )
        return mapped[inverse.ravel()]


class _RoleState:
    """Per-entity sufficient statistics for one transaction role (src / dst).

    Historical amount moments use Chan et al.'s parallel Welford merge, which
    is numerically stable for amounts up to 1e8. Rolling-window counters hold
    the number of events in steps ``[s - w, s - 1]`` after :meth:`evict`.
    """

    def __init__(self, windows: Tuple[int, ...]) -> None:
        self.windows = windows
        self.capacity = 0
        self.count = np.zeros(0, dtype=np.int64)
        self.mean = np.zeros(0, dtype=np.float64)
        self.m2 = np.zeros(0, dtype=np.float64)
        self.last_step = np.zeros(0, dtype=np.int64)
        self.window_counts: Dict[int, np.ndarray] = {
            w: np.zeros(0, dtype=np.int64) for w in windows
        }
        self.window_blocks: Dict[int, Deque[Tuple[int, np.ndarray, np.ndarray]]] = {
            w: deque() for w in windows
        }

    def ensure_capacity(self, n_entities: int) -> None:
        if n_entities <= self.capacity:
            return
        new_cap = max(n_entities, int(self.capacity * 1.5) + 1024)
        self.count = _grow(self.count, new_cap, 0)
        self.mean = _grow(self.mean, new_cap, 0.0)
        self.m2 = _grow(self.m2, new_cap, 0.0)
        self.last_step = _grow(self.last_step, new_cap, -1)
        for w in self.windows:
            self.window_counts[w] = _grow(self.window_counts[w], new_cap, 0)
        self.capacity = new_cap

    def evict(self, current_step: int) -> None:
        """Drop events older than each window: keep steps ``>= current - w``."""
        for w in self.windows:
            blocks = self.window_blocks[w]
            counts = self.window_counts[w]
            threshold = current_step - w
            while blocks and blocks[0][0] < threshold:
                _, ids, n = blocks.popleft()
                counts[ids] -= n

    def snapshot(self, ids: np.ndarray, current_step: int) -> Dict[str, np.ndarray]:
        count = self.count[ids]
        has_hist = count > 0
        mean = np.where(has_hist, self.mean[ids], 0.0)
        var = np.zeros(ids.shape[0], dtype=np.float64)
        np.divide(self.m2[ids], count, out=var, where=has_hist)
        std = np.sqrt(np.maximum(var, 0.0))
        last = self.last_step[ids]
        since = np.where(last >= 0, current_step - last, -1)
        out = {f"vel_{w}": self.window_counts[w][ids].copy() for w in self.windows}
        out.update({"hist_count": count, "hist_mean": mean, "hist_std": std,
                    "steps_since_last": since.astype(np.int64)})
        return out

    def update(self, ids: np.ndarray, amounts: np.ndarray, step: int) -> None:
        if ids.size == 0:
            return
        ids, amounts = _canonical_order(ids, amounts)
        uniq, inverse = np.unique(ids, return_inverse=True)
        inverse = inverse.ravel()
        n_b = np.bincount(inverse, minlength=uniq.size).astype(np.int64)
        sum_b = np.bincount(inverse, weights=amounts, minlength=uniq.size)
        mean_b = sum_b / n_b
        dev = amounts - mean_b[inverse]
        m2_b = np.bincount(inverse, weights=dev * dev, minlength=uniq.size)

        n_a = self.count[uniq]
        mean_a = self.mean[uniq]
        m2_a = self.m2[uniq]
        n_tot = n_a + n_b
        delta = mean_b - mean_a
        self.mean[uniq] = mean_a + delta * (n_b / n_tot)
        self.m2[uniq] = m2_a + m2_b + delta * delta * (n_a * n_b / n_tot)
        self.count[uniq] = n_tot
        self.last_step[uniq] = step
        for w in self.windows:
            self.window_counts[w][uniq] += n_b
            self.window_blocks[w].append((step, uniq, n_b))


# --------------------------------------------------------------------------- #
# Causal feature engineer                                                     #
# --------------------------------------------------------------------------- #
class CausalFeatureEngineer:
    """Stateful, streaming, strictly causal feature engineer.

    The engineer consumes canonical transaction frames (see
    :data:`CANONICAL_COLUMNS`) sorted by ``step`` and emits one feature row per
    input row. State persists across calls, so a stream can be processed in
    arbitrary chunk sizes.
    """

    def __init__(self, config: Optional[CausalFeatureConfig] = None) -> None:
        self.config = config or CausalFeatureConfig()
        self.feature_names: List[str] = self._build_feature_names()
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        """Forget all accumulated history."""
        cfg = self.config
        self._registry = _EntityRegistry()
        self._src = _RoleState(cfg.velocity_windows)
        self._dst = _RoleState(cfg.velocity_windows)
        self._ledger = np.zeros(0, dtype=np.float64)
        self._pending: Optional[pd.DataFrame] = None
        self._last_finalized_step: Optional[int] = None
        self.rows_emitted = 0

    @property
    def n_entities(self) -> int:
        return len(self._registry)

    @property
    def last_finalized_step(self) -> Optional[int]:
        return self._last_finalized_step

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk; return features for every *completed* step.

        Rows belonging to the chunk's final step are buffered because that step
        may continue in the next chunk. Call :meth:`flush` at end of stream.
        """
        self._validate_input(chunk)
        frame = chunk if self._pending is None else pd.concat(
            [self._pending, chunk], ignore_index=True
        )
        self._pending = None
        if frame.empty:
            return self._empty_output(frame)
        steps = frame["step"].to_numpy(dtype=np.int64)
        self._check_order(steps)
        last_step = steps[-1]
        cut = int(np.searchsorted(steps, last_step, side="left"))
        self._pending = frame.iloc[cut:].reset_index(drop=True)
        complete = frame.iloc[:cut].reset_index(drop=True)
        return self._process_frame(complete)

    def flush(self) -> pd.DataFrame:
        """Finalise and return features for any buffered trailing step."""
        if self._pending is None or self._pending.empty:
            self._pending = None
            return self._empty_output(pd.DataFrame(columns=list(CANONICAL_COLUMNS)))
        frame = self._pending
        self._pending = None
        return self._process_frame(frame)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Process a full, step-sorted frame (continues existing state)."""
        head = self.transform_chunk(frame)
        tail = self.flush()
        parts = [p for p in (head, tail) if not p.empty]
        if not parts:
            return head
        return pd.concat(parts, ignore_index=True)

    def transform_stream(self, chunks: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Lazily transform an iterable of step-sorted chunks."""
        for chunk in chunks:
            out = self.transform_chunk(chunk)
            if not out.empty:
                yield out
        tail = self.flush()
        if not tail.empty:
            yield tail

    # ------------------------------------------------------------- internals
    def _build_feature_names(self) -> List[str]:
        cfg = self.config
        names = ["log1p_amount", "hour_of_day"] + [f"type_{t}" for t in cfg.tx_types]
        roles = ["src", "dst"] if cfg.include_destination else ["src"]
        for role in roles:
            names += [f"{role}_vel_{w}" for w in cfg.velocity_windows]
            names += [
                f"{role}_hist_count", f"{role}_hist_mean", f"{role}_hist_std",
                f"{role}_amount_to_mean", f"{role}_amount_zscore",
                f"{role}_steps_since_last", f"{role}_is_new",
            ]
        names += ["src_ledger_pre", "src_balance_delta", "src_amount_to_ledger"]
        if cfg.include_destination:
            names += ["dst_ledger_pre", "dst_balance_delta"]
        return names

    def _validate_input(self, chunk: pd.DataFrame) -> None:
        assert_no_leakage(chunk)
        required = list(REQUIRED_ENGINE_COLUMNS)
        if self.config.include_destination:
            required.append("dst")
        missing = [c for c in required if c not in chunk.columns]
        if missing:
            raise KeyError(f"Input frame missing required columns: {missing}")
        if chunk["amount"].isna().any():
            raise ValueError("Transaction amounts must not contain NaN")
        if (chunk["amount"].to_numpy(dtype=np.float64) < 0).any():
            raise ValueError("Transaction amounts must be non-negative")

    def _check_order(self, steps: np.ndarray) -> None:
        if steps.size > 1 and np.any(np.diff(steps) < 0):
            bad = int(np.flatnonzero(np.diff(steps) < 0)[0])
            raise TemporalOrderError(
                f"Steps not chronologically sorted at row {bad + 1}: "
                f"{steps[bad]} -> {steps[bad + 1]}"
            )
        if self._last_finalized_step is not None and steps[0] <= self._last_finalized_step:
            raise TemporalOrderError(
                f"Received step {steps[0]} after step {self._last_finalized_step} "
                "was already finalised (late or out-of-order arrival)."
            )

    def _carry_columns(self, frame: pd.DataFrame) -> List[str]:
        carry = [c for c in CANONICAL_COLUMNS if c in frame.columns]
        carry += [c for c in frame.columns if c.startswith(PASSTHROUGH_PREFIX)]
        return carry

    def _empty_output(self, frame: pd.DataFrame) -> pd.DataFrame:
        carry = self._carry_columns(frame)
        out = pd.DataFrame({c: pd.Series(dtype=frame[c].dtype) for c in carry})
        for name in self.feature_names:
            out[name] = pd.Series(dtype=np.float64)
        return out

    def _process_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return self._empty_output(frame)
        steps = frame["step"].to_numpy(dtype=np.int64)
        boundaries = np.flatnonzero(np.diff(steps)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [steps.size]))

        collected: Dict[str, List[np.ndarray]] = {n: [] for n in self.feature_names}
        for start, end in zip(starts, ends):
            block_feats = self._process_block(frame.iloc[start:end], int(steps[start]))
            for name in self.feature_names:
                collected[name].append(block_feats[name])

        features = pd.DataFrame(
            {name: np.concatenate(collected[name]) for name in self.feature_names}
        )
        carry = frame[self._carry_columns(frame)].reset_index(drop=True)
        out = pd.concat([carry, features], axis=1)
        assert_no_leakage(out)
        self.rows_emitted += len(out)
        return out

    def _process_block(self, block: pd.DataFrame, step: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        smooth = cfg.ratio_smoothing
        n_rows = len(block)
        amounts = block["amount"].to_numpy(dtype=np.float64)
        tx_types = block["tx_type"].astype(str).to_numpy()

        # 1. Age out rolling windows so they cover exactly [step - w, step - 1].
        self._src.evict(step)
        self._dst.evict(step)

        # 2. Register entities (new entities start with empty history).
        src_ids = self._registry.encode(block["src"].astype(str).to_numpy())
        if cfg.include_destination:
            dst_raw = block["dst"]
            dst_keys = dst_raw.fillna(MISSING_ENTITY).astype(str).to_numpy()
            dst_valid = dst_keys != MISSING_ENTITY
            dst_ids = np.full(n_rows, -1, dtype=np.int64)
            dst_ids[dst_valid] = self._registry.encode(dst_keys[dst_valid])
        else:
            dst_valid = np.zeros(n_rows, dtype=bool)
            dst_ids = np.full(n_rows, -1, dtype=np.int64)

        n_ent = len(self._registry)
        self._src.ensure_capacity(n_ent)
        self._dst.ensure_capacity(n_ent)
        if self._ledger.shape[0] < n_ent:
            self._ledger = _grow(self._ledger, max(n_ent, int(self._ledger.shape[0] * 1.5) + 1024), 0.0)

        # 3. Read state (strictly prior steps only) BEFORE any update.
        feats: Dict[str, np.ndarray] = {
            "log1p_amount": np.log1p(amounts),
            "hour_of_day": np.full(n_rows, step % cfg.hours_per_day, dtype=np.int64),
        }
        for t in cfg.tx_types:
            feats[f"type_{t}"] = (tx_types == t).astype(np.int64)

        src_snap = self._src.snapshot(src_ids, step)
        self._emit_role_features(feats, "src", src_snap, amounts, np.ones(n_rows, bool), smooth)

        if cfg.include_destination:
            safe_dst = np.where(dst_valid, dst_ids, 0)
            dst_snap = self._dst.snapshot(safe_dst, step)
            self._emit_role_features(feats, "dst", dst_snap, amounts, dst_valid, smooth)

        inflow = np.isin(tx_types, np.asarray(cfg.inflow_types, dtype=object)) \
            if cfg.inflow_types else np.zeros(n_rows, dtype=bool)
        src_flow = np.where(inflow, amounts, -amounts)
        dst_flow = -src_flow
        src_ledger_pre = self._ledger[src_ids].copy()
        feats["src_ledger_pre"] = src_ledger_pre
        feats["src_balance_delta"] = src_ledger_pre + src_flow
        feats["src_amount_to_ledger"] = amounts / (np.abs(src_ledger_pre) + smooth)
        if cfg.include_destination:
            dst_ledger_pre = np.where(dst_valid, self._ledger[np.where(dst_valid, dst_ids, 0)], 0.0)
            feats["dst_ledger_pre"] = dst_ledger_pre
            feats["dst_balance_delta"] = np.where(dst_valid, dst_ledger_pre + dst_flow, 0.0)

        # 4. Commit this step's events to state (visible from step + 1 onward).
        self._src.update(src_ids, amounts, step)
        if cfg.include_destination and dst_valid.any():
            self._dst.update(dst_ids[dst_valid], amounts[dst_valid], step)
            _scatter_add(self._ledger, dst_ids[dst_valid], dst_flow[dst_valid])
        _scatter_add(self._ledger, src_ids, src_flow)
        self._last_finalized_step = step
        return feats

    @staticmethod
    def _emit_role_features(
        feats: Dict[str, np.ndarray],
        role: str,
        snap: Dict[str, np.ndarray],
        amounts: np.ndarray,
        valid: np.ndarray,
        smooth: float,
    ) -> None:
        count = np.where(valid, snap["hist_count"], 0)
        has_hist = count > 0
        mean = np.where(valid, snap["hist_mean"], 0.0)
        std = np.where(valid, snap["hist_std"], 0.0)
        for key, values in snap.items():
            if key.startswith("vel_"):
                feats[f"{role}_{key}"] = np.where(valid, values, 0).astype(np.int64)
        feats[f"{role}_hist_count"] = count.astype(np.int64)
        feats[f"{role}_hist_mean"] = mean
        feats[f"{role}_hist_std"] = std
        feats[f"{role}_amount_to_mean"] = np.where(has_hist, amounts / (mean + smooth), 1.0)
        feats[f"{role}_amount_zscore"] = np.where(has_hist, (amounts - mean) / (std + smooth), 0.0)
        feats[f"{role}_steps_since_last"] = np.where(valid, snap["steps_since_last"], -1).astype(np.int64)
        feats[f"{role}_is_new"] = (~has_hist & valid).astype(np.int64)


# --------------------------------------------------------------------------- #
# Temporal splits                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TemporalSplit:
    """Inclusive step interval ``[start_step, end_step]`` with a name."""

    name: str
    start_step: int
    end_step: int

    def contains(self, steps: np.ndarray) -> np.ndarray:
        return (steps >= self.start_step) & (steps <= self.end_step)


def load_temporal_splits(spec: Mapping[str, Mapping[str, int]]) -> List[TemporalSplit]:
    """Parse and validate chronologically ordered, non-overlapping splits."""
    splits = [
        TemporalSplit(name, int(v["start_step"]), int(v["end_step"]))
        for name, v in spec.items()
    ]
    splits.sort(key=lambda s: s.start_step)
    for s in splits:
        if s.end_step < s.start_step:
            raise ValueError(f"Split {s.name} has end_step < start_step")
    for prev, nxt in zip(splits, splits[1:]):
        if nxt.start_step <= prev.end_step:
            raise ValueError(f"Splits {prev.name} and {nxt.name} overlap")
    return splits


def assign_temporal_split(
    steps: np.ndarray, splits: Sequence[TemporalSplit], default: str = "out_of_range"
) -> np.ndarray:
    """Vectorised split assignment by step."""
    steps = np.asarray(steps, dtype=np.int64)
    labels = np.full(steps.shape[0], default, dtype=object)
    for split in splits:
        labels[split.contains(steps)] = split.name
    return labels


# --------------------------------------------------------------------------- #
# Causality audit                                                             #
# --------------------------------------------------------------------------- #
def _perturb_future(frame: pd.DataFrame, cutoff_step: int, rng: np.random.Generator) -> pd.DataFrame:
    """Adversarially rewrite every row with ``step > cutoff_step``.

    Future amounts are rescaled, labels flipped, types shuffled, and extra
    future transactions are injected for entities already active in the past.
    """
    past = frame[frame["step"] <= cutoff_step]
    future = frame[frame["step"] > cutoff_step].copy()
    if future.empty:
        return frame.copy()
    future["amount"] = future["amount"].to_numpy(dtype=np.float64) * rng.uniform(0.1, 10.0, len(future))
    if "label" in future.columns:
        future["label"] = 1 - future["label"].to_numpy()
    future["tx_type"] = rng.permutation(future["tx_type"].to_numpy())

    n_inject = max(1, len(future) // 4)
    injected = future.sample(n=n_inject, replace=True, random_state=int(rng.integers(0, 2**31 - 1))).copy()
    past_entities = past["src"].astype(str).unique() if not past.empty else frame["src"].astype(str).unique()
    injected["src"] = rng.choice(past_entities, size=n_inject)
    if "tx_id" in injected.columns:
        injected["tx_id"] = np.arange(n_inject, dtype=np.int64) + int(frame["tx_id"].max()) + 1
    out = pd.concat([past, future, injected], ignore_index=True)
    return out.sort_values("step", kind="mergesort").reset_index(drop=True)


def audit_temporal_causality(
    frame: pd.DataFrame,
    cutoff_step: int,
    config: Optional[CausalFeatureConfig] = None,
    seed: int = SEED,
    atol: float = 0.0,
) -> Dict[str, object]:
    """Verify that features up to ``cutoff_step`` are independent of the future.

    Features are computed on the original frame and on a copy whose future is
    adversarially rewritten. Rows with ``step <= cutoff_step`` must match
    exactly (``atol=0``). Requires a unique ``tx_id`` column.
    """
    if "tx_id" not in frame.columns:
        raise KeyError("audit_temporal_causality requires a 'tx_id' column")
    rng = np.random.default_rng(seed)
    base = CausalFeatureEngineer(config).transform(frame)
    alt = CausalFeatureEngineer(config).transform(_perturb_future(frame, cutoff_step, rng))
    names = CausalFeatureEngineer(config).feature_names

    base_past = base[base["step"] <= cutoff_step].sort_values("tx_id").reset_index(drop=True)
    alt_past = alt[alt["step"] <= cutoff_step].sort_values("tx_id").reset_index(drop=True)
    if len(base_past) != len(alt_past) or not np.array_equal(
        base_past["tx_id"].to_numpy(), alt_past["tx_id"].to_numpy()
    ):
        return {"passed": False, "reason": "row sets differ", "max_abs_diff": float("inf")}
    a = base_past[names].to_numpy(dtype=np.float64)
    b = alt_past[names].to_numpy(dtype=np.float64)
    diff = np.abs(a - b)
    max_diff = float(diff.max()) if diff.size else 0.0
    worst = names[int(diff.max(axis=0).argmax())] if diff.size else None
    return {
        "passed": bool(max_diff <= atol),
        "cutoff_step": int(cutoff_step),
        "rows_checked": int(len(base_past)),
        "max_abs_diff": max_diff,
        "worst_feature": worst,
    }


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #
class SplitPartitionedWriter:
    """Append feature chunks to one file per split (Parquet if available, else CSV)."""

    def __init__(self, out_dir: Path, dataset: str, fmt: str = "parquet") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.fmt = fmt
        if fmt == "parquet":
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                LOGGER.warning("pyarrow not installed; falling back to CSV output")
                self.fmt = "csv"
        if self.fmt not in {"parquet", "csv"}:
            raise ValueError("fmt must be 'parquet' or 'csv'")
        self._writers: Dict[str, object] = {}
        self._schemas: Dict[str, object] = {}
        self.rows_written: Dict[str, int] = {}
        self.frauds_written: Dict[str, int] = {}
        self.paths: Dict[str, str] = {}

    def _path(self, split: str) -> Path:
        ext = "parquet" if self.fmt == "parquet" else "csv"
        return self.out_dir / f"{self.dataset}_{split}_causal.{ext}"

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        for split, part in frame.groupby("split", sort=False):
            self._write_part(str(split), part)

    def _write_part(self, split: str, part: pd.DataFrame) -> None:
        path = self._path(split)
        if self.fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            if split not in self._writers:
                table = pa.Table.from_pandas(part, preserve_index=False)
                self._schemas[split] = table.schema
                self._writers[split] = pq.ParquetWriter(str(path), table.schema)
            else:
                table = pa.Table.from_pandas(part, schema=self._schemas[split], preserve_index=False)
            self._writers[split].write_table(table)
        else:
            header = split not in self._writers
            part.to_csv(path, mode="w" if header else "a", header=header, index=False)
            self._writers[split] = True
        self.rows_written[split] = self.rows_written.get(split, 0) + len(part)
        if "label" in part.columns:
            self.frauds_written[split] = self.frauds_written.get(split, 0) + int(
                (part["label"].to_numpy() == 1).sum()
            )
        self.paths[split] = str(path)

    def close(self) -> None:
        if self.fmt == "parquet":
            for writer in self._writers.values():
                writer.close()
        self._writers = {}


def _config_fingerprint(*objs: Mapping) -> str:
    payload = json.dumps(objs, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def preprocess_dataset(
    dataset: str,
    project_config: Mapping,
    repo_root: Path,
    chunksize: Optional[int] = None,
    max_rows: Optional[int] = None,
    fmt: str = "parquet",
) -> Dict[str, object]:
    """Stream a raw dataset through the causal engineer to ``processed_dir``.

    Returns the run manifest (also written as JSON next to the outputs).
    """
    from src.data.dataset_loaders import build_loader  # local import avoids cycle

    seed = int(project_config.get("project", {}).get("seed", SEED))
    set_global_seed(seed)
    loader = build_loader(dataset, project_config, repo_root=repo_root,
                          chunksize=chunksize, max_rows=max_rows)
    feat_cfg = CausalFeatureConfig.from_project_config(project_config, dataset)
    engineer = CausalFeatureEngineer(feat_cfg)
    split_spec = project_config.get("temporal_splits", {}).get(dataset)
    splits = load_temporal_splits(split_spec) if split_spec else None

    out_dir = repo_root / project_config.get("paths", {}).get("processed_dir", "data/processed")
    writer = SplitPartitionedWriter(out_dir, dataset, fmt=fmt)
    t0 = time.perf_counter()
    try:
        for out in engineer.transform_stream(loader.iter_chunks()):
            out["split"] = (assign_temporal_split(out["step"].to_numpy(), splits)
                            if splits else "stream")
            writer.write(out)
            LOGGER.info("processed %d rows (last step %s)", engineer.rows_emitted,
                        engineer.last_finalized_step)
    finally:
        writer.close()

    manifest = {
        "dataset": dataset,
        "seed": seed,
        "rows_total": engineer.rows_emitted,
        "rows_per_split": writer.rows_written,
        "frauds_per_split": writer.frauds_written,
        "outputs": writer.paths,
        "format": writer.fmt,
        "feature_names": engineer.feature_names,
        "stripped_leakage_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        "feature_config": asdict(feat_cfg),
        "temporal_splits": [asdict(s) for s in splits] if splits else None,
        "n_entities": engineer.n_entities,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "config_fingerprint": _config_fingerprint(project_config),
    }
    manifest_path = out_dir / f"{dataset}_causal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MG-ECFM causal preprocessing")
    parser.add_argument("--dataset", required=True, choices=["paysim", "ieee_cis", "amlsim"])
    parser.add_argument("--config", type=Path, default=Path("configs/project_config.json"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    cfg_path = args.config if args.config.is_absolute() else args.root / args.config
    project_config = json.loads(cfg_path.read_text(encoding="utf-8"))
    manifest = preprocess_dataset(args.dataset, project_config, args.root.resolve(),
                                  chunksize=args.chunksize, max_rows=args.max_rows,
                                  fmt=args.format)
    print(json.dumps({k: manifest[k] for k in
                      ("dataset", "rows_total", "rows_per_split", "frauds_per_split",
                       "elapsed_seconds", "manifest_path")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
