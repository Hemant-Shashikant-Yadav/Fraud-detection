"""
src/data/perturbation_generator.py
==================================

Synthetic candidate-rule feeds with controllable adversarial contamination,
used to stress-test the Subjective-Logic / martingale e-process knowledge gate.

Each candidate rule ``c_k`` carries

* an executable predicate over the causal feature space produced by
  :mod:`src.data.causal_preprocessor` (``rule.evaluate(frame) -> bool mask``);
* per-source evidence ``(r_plus, r_minus)`` with source reliability ``pi_m``,
  observation step (for freshness decay ``exp(-eta * delta_tau)``), source
  age and optional Sybil-cluster membership - exactly the inputs of the
  Dirichlet opinion ``omega_k^(m) = (b, d, u, a)``;
* latent ground truth: the rule's *current* precision and an ``is_harmful``
  flag (``true_precision <= p0``) against which gate decisions are scored;
* a Bernoulli outcome simulator for firing-level martingale updates.

Categories
----------
genuine      honest sources, current precision well above ``p0``.
stale        evidence was honest but observed beyond the staleness horizon;
             the pattern has since decayed (historical precision high, current
             precision low).
conflicting  honest sources disagree on the *same* predicate (often a copy of
             a genuine rule's predicate); true precision straddles ``p0``.
poisoned     one low-reliability source fabricates strongly positive evidence
             for a useless predicate.
sybil        a poisoned rule backed by a cluster of young, low-reliability,
             near-duplicate sources attempting to dominate consensus fusion.

Poison rate = (poisoned + sybil) / n_candidates, controllable in [5%, 25%]
(``0.0`` is accepted as the clean control).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

SEED = 42
RULE_CATEGORIES: Tuple[str, ...] = ("genuine", "stale", "conflicting", "poisoned", "sybil")
ADVERSARIAL_CATEGORIES: Tuple[str, ...] = ("poisoned", "sybil")
HONEST_SOURCE_TYPES: Tuple[str, ...] = ("tree_path", "graph_motif", "analyst_alert", "consortium_feed")
_OPS = {
    ">": np.greater, ">=": np.greater_equal, "<": np.less,
    "<=": np.less_equal, "==": np.equal, "!=": np.not_equal,
}


# --------------------------------------------------------------------------- #
# Rule primitives                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    """Atomic predicate ``feature <op> value``."""

    feature: str
    op: str
    value: Union[float, str]

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"Unsupported operator '{self.op}'")

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        if self.feature not in frame.columns:
            raise KeyError(f"Rule feature '{self.feature}' not present in frame")
        column = frame[self.feature]
        if isinstance(self.value, str):
            values = column.astype(str).to_numpy()
            return np.asarray(_OPS[self.op](values, self.value), dtype=bool)
        return np.asarray(_OPS[self.op](column.to_numpy(dtype=np.float64), float(self.value)), dtype=bool)

    def describe(self) -> str:
        val = f"'{self.value}'" if isinstance(self.value, str) else f"{self.value:g}"
        return f"{self.feature} {self.op} {val}"


@dataclass(frozen=True)
class SourceEvidence:
    """Evidence reported by one source ``m`` for candidate ``c_k``."""

    source_id: str
    source_type: str
    reliability: float
    r_plus: int
    r_minus: int
    observed_at_step: int
    source_age_steps: int
    sybil_cluster_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must lie in [0, 1]")
        if self.r_plus < 0 or self.r_minus < 0:
            raise ValueError("evidence counts must be non-negative")

    @property
    def n_observations(self) -> int:
        return self.r_plus + self.r_minus

    @property
    def claimed_precision(self) -> float:
        """Laplace-smoothed precision implied by the reported counts."""
        return (self.r_plus + 1.0) / (self.r_plus + self.r_minus + 2.0)


@dataclass
class CandidateRule:
    """Candidate fraud pattern ``c_k`` with evidence and latent ground truth."""

    rule_id: str
    category: str
    conditions: Tuple[Condition, ...]
    evidence: Tuple[SourceEvidence, ...]
    emitted_at_step: int
    true_precision: float
    historical_precision: float
    is_harmful: bool
    base_rate: float = 0.5
    conflict_target: Optional[str] = None

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        """Boolean mask of rows on which the rule fires (conjunction)."""
        mask = np.ones(len(frame), dtype=bool)
        for cond in self.conditions:
            mask &= cond.evaluate(frame)
        return mask

    def simulate_outcomes(self, n_firings: int, rng: np.random.Generator) -> np.ndarray:
        """Labels ``y_i ~ Bernoulli(true_precision)`` for ``n_firings`` firings."""
        return (rng.random(n_firings) < self.true_precision).astype(np.int8)

    @property
    def is_adversarial(self) -> bool:
        return self.category in ADVERSARIAL_CATEGORIES

    def describe(self) -> str:
        return " AND ".join(c.describe() for c in self.conditions)

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["predicate"] = self.describe()
        return out


@dataclass(frozen=True)
class FeatureSpec:
    """Range of a numeric feature from which rule thresholds are sampled."""

    name: str
    low: float
    high: float
    integer: bool
    ops: Tuple[str, ...]


DEFAULT_FEATURE_SPACE: Tuple[FeatureSpec, ...] = (
    FeatureSpec("log1p_amount", 6.0, 14.0, False, (">", ">=")),
    FeatureSpec("src_vel_1", 1, 4, True, (">=",)),
    FeatureSpec("src_vel_5", 1, 10, True, (">=",)),
    FeatureSpec("src_vel_20", 2, 30, True, (">=",)),
    FeatureSpec("dst_vel_1", 1, 6, True, (">=",)),
    FeatureSpec("dst_vel_5", 2, 15, True, (">=",)),
    FeatureSpec("dst_vel_20", 3, 40, True, (">=",)),
    FeatureSpec("src_amount_to_mean", 1.5, 15.0, False, (">",)),
    FeatureSpec("dst_amount_zscore", 1.0, 8.0, False, (">",)),
    FeatureSpec("src_steps_since_last", 0, 24, True, ("<=",)),
    FeatureSpec("dst_hist_count", 1, 50, True, (">=",)),
    FeatureSpec("src_amount_to_ledger", 1.0, 50.0, False, (">",)),
    FeatureSpec("hour_of_day", 0, 6, True, ("<=",)),
)
DEFAULT_CATEGORICAL_SPACE: Mapping[str, Tuple[str, ...]] = {
    "tx_type": ("TRANSFER", "CASH_OUT"),
}


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PerturbationConfig:
    n_candidates: int = 200
    stale_rate: float = 0.10
    conflict_rate: float = 0.10
    sybil_share_of_poison: float = 0.5
    sybil_cluster_size: Tuple[int, int] = (3, 8)
    sybil_rules_per_cluster: int = 3
    min_poison_rate: float = 0.05
    max_poison_rate: float = 0.25
    staleness_horizon_steps: int = 168
    current_step: int = 632
    p0: float = 0.5
    p1: float = 0.8
    honest_reliability: Tuple[float, float] = (0.60, 0.95)
    adversarial_reliability: Tuple[float, float] = (0.10, 0.35)
    honest_observation_range: Tuple[int, int] = (20, 200)
    fabricated_observation_range: Tuple[int, int] = (100, 400)
    max_conditions: int = 3
    base_rate: float = 0.5
    poison_rates: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)

    def __post_init__(self) -> None:
        if self.n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        if not 0.0 < self.p0 < self.p1 < 1.0:
            raise ValueError("require 0 < p0 < p1 < 1")
        if not 0.0 <= self.sybil_share_of_poison <= 1.0:
            raise ValueError("sybil_share_of_poison must lie in [0, 1]")
        if self.sybil_cluster_size[0] < 2 or self.sybil_cluster_size[1] < self.sybil_cluster_size[0]:
            raise ValueError("sybil_cluster_size must be (k_min >= 2, k_max >= k_min)")
        if self.staleness_horizon_steps <= 0:
            raise ValueError("staleness_horizon_steps must be positive")
        if self.max_conditions < 1:
            raise ValueError("max_conditions must be >= 1")

    @classmethod
    def from_configs(cls, project_config: Mapping, hyperparams: Optional[Mapping] = None) -> "PerturbationConfig":
        p = dict(project_config.get("perturbation", {}))
        adm = (hyperparams or {}).get("martingale_eprocess", {}).get("admission", {})
        sl = (hyperparams or {}).get("subjective_logic", {})
        tuples = {"sybil_cluster_size", "honest_reliability", "adversarial_reliability",
                  "honest_observation_range", "fabricated_observation_range", "poison_rates"}
        kwargs = {k: (tuple(v) if k in tuples else v) for k, v in p.items()
                  if k in cls.__dataclass_fields__}
        if "p0" in adm:
            kwargs["p0"] = float(adm["p0"])
        if "p1" in adm:
            kwargs["p1"] = float(adm["p1"])
        if "base_rate_a" in sl:
            kwargs["base_rate"] = float(sl["base_rate_a"])
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Feed container                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class CandidateFeed:
    rules: List[CandidateRule]
    poison_rate: float
    stale_rate: float
    conflict_rate: float
    current_step: int
    seed: int
    p0: float
    metadata: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rules)

    def by_category(self, category: str) -> List[CandidateRule]:
        return [r for r in self.rules if r.category == category]

    def summary(self) -> Dict[str, object]:
        n = len(self.rules)
        counts = {c: sum(r.category == c for r in self.rules) for c in RULE_CATEGORIES}
        adversarial = counts["poisoned"] + counts["sybil"]
        return {
            "n_candidates": n,
            "counts": counts,
            "target_poison_rate": self.poison_rate,
            "realized_poison_rate": adversarial / n if n else 0.0,
            "realized_stale_rate": counts["stale"] / n if n else 0.0,
            "realized_conflict_rate": counts["conflicting"] / n if n else 0.0,
            "n_harmful": sum(r.is_harmful for r in self.rules),
            "n_sybil_sources": len({e.source_id for r in self.rules for e in r.evidence
                                    if e.sybil_cluster_id is not None}),
            "p0": self.p0,
            "current_step": self.current_step,
            "seed": self.seed,
        }

    def to_frame(self) -> pd.DataFrame:
        """Evidence-level long table (one row per rule x source)."""
        rows = []
        for r in self.rules:
            for e in r.evidence:
                rows.append({
                    "rule_id": r.rule_id, "category": r.category, "predicate": r.describe(),
                    "true_precision": r.true_precision, "is_harmful": r.is_harmful,
                    "emitted_at_step": r.emitted_at_step, **asdict(e),
                    "claimed_precision": e.claimed_precision,
                })
        return pd.DataFrame(rows)

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "metadata": self.metadata,
                   "rules": [r.to_dict() for r in self.rules]}
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


# --------------------------------------------------------------------------- #
# Generator                                                                   #
# --------------------------------------------------------------------------- #
class PerturbationGenerator:
    """Deterministic generator of contaminated candidate-rule feeds."""

    def __init__(
        self,
        config: Optional[PerturbationConfig] = None,
        feature_space: Sequence[FeatureSpec] = DEFAULT_FEATURE_SPACE,
        categorical_space: Mapping[str, Sequence[str]] = DEFAULT_CATEGORICAL_SPACE,
        seed: int = SEED,
    ) -> None:
        self.config = config or PerturbationConfig()
        self.feature_space = tuple(feature_space)
        self.categorical_space = {k: tuple(v) for k, v in categorical_space.items()}
        if not self.feature_space:
            raise ValueError("feature_space must not be empty")
        self.seed = int(seed)

    # -------------------------------------------------------------- public
    def validate_poison_rate(self, rate: float) -> float:
        cfg = self.config
        rate = float(rate)
        if rate == 0.0:
            return rate
        eps = 1e-12
        if not (cfg.min_poison_rate - eps <= rate <= cfg.max_poison_rate + eps):
            raise ValueError(
                f"poison_rate {rate} outside [{cfg.min_poison_rate}, {cfg.max_poison_rate}] "
                "(use 0.0 for a clean control feed)"
            )
        return rate

    def category_counts(self, n: int, poison_rate: float, stale_rate: float,
                        conflict_rate: float) -> Dict[str, int]:
        n_poison_total = int(round(poison_rate * n))
        n_sybil = int(round(self.config.sybil_share_of_poison * n_poison_total))
        counts = {
            "poisoned": n_poison_total - n_sybil,
            "sybil": n_sybil,
            "stale": int(round(stale_rate * n)),
            "conflicting": int(round(conflict_rate * n)),
        }
        counts["genuine"] = n - sum(counts.values())
        if counts["genuine"] < 0:
            raise ValueError("poison + stale + conflict rates exceed 1.0")
        return counts

    def generate_feed(
        self,
        poison_rate: float,
        n_candidates: Optional[int] = None,
        stale_rate: Optional[float] = None,
        conflict_rate: Optional[float] = None,
        current_step: Optional[int] = None,
    ) -> CandidateFeed:
        """Generate one feed. Same arguments + seed => identical feed."""
        cfg = self.config
        poison_rate = self.validate_poison_rate(poison_rate)
        n = int(n_candidates or cfg.n_candidates)
        stale_rate = cfg.stale_rate if stale_rate is None else float(stale_rate)
        conflict_rate = cfg.conflict_rate if conflict_rate is None else float(conflict_rate)
        for name, r in (("stale_rate", stale_rate), ("conflict_rate", conflict_rate)):
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        now = int(cfg.current_step if current_step is None else current_step)

        rng = np.random.default_rng(
            [self.seed, int(round(poison_rate * 10_000)), n,
             int(round(stale_rate * 10_000)), int(round(conflict_rate * 10_000)), now]
        )
        counts = self.category_counts(n, poison_rate, stale_rate, conflict_rate)
        labels = np.array([c for c in RULE_CATEGORIES for _ in range(counts[c])], dtype=object)
        labels = labels[rng.permutation(n)]
        ids = [f"R{idx:05d}" for idx in range(n)]

        sybil_clusters = self._make_sybil_clusters(counts["sybil"], now, rng)
        rules: Dict[int, CandidateRule] = {}
        genuine_positions = [i for i in range(n) if labels[i] == "genuine"]
        for i in genuine_positions:
            rules[i] = self._make_genuine(ids[i], now, rng)
        sybil_cursor = 0
        for i in range(n):
            cat = labels[i]
            if cat == "genuine":
                continue
            if cat == "stale":
                rules[i] = self._make_stale(ids[i], now, rng)
            elif cat == "conflicting":
                target = rules[int(rng.choice(genuine_positions))] if genuine_positions else None
                rules[i] = self._make_conflicting(ids[i], now, rng, target)
            elif cat == "poisoned":
                rules[i] = self._make_poisoned(ids[i], now, rng)
            elif cat == "sybil":
                cluster = sybil_clusters[sybil_cursor % len(sybil_clusters)]
                rules[i] = self._make_sybil(ids[i], now, rng, cluster, sybil_cursor)
                sybil_cursor += 1

        feed = CandidateFeed(
            rules=[rules[i] for i in range(n)],
            poison_rate=poison_rate, stale_rate=stale_rate, conflict_rate=conflict_rate,
            current_step=now, seed=self.seed, p0=cfg.p0,
            metadata={"category_counts": counts, "n_sybil_clusters": len(sybil_clusters),
                      "staleness_horizon_steps": cfg.staleness_horizon_steps},
        )
        return feed

    def poison_rate_sweep(self, rates: Optional[Sequence[float]] = None, **kwargs) -> Dict[float, CandidateFeed]:
        """Feeds for each poisoning rate (default 5%, 10%, ..., 25%)."""
        rates = tuple(rates) if rates is not None else self.config.poison_rates
        return {float(r): self.generate_feed(float(r), **kwargs) for r in rates}

    # ------------------------------------------------------------ builders
    def _predicate(self, rng: np.random.Generator) -> Tuple[Condition, ...]:
        k = int(rng.integers(1, self.config.max_conditions + 1))
        conds: List[Condition] = []
        if self.categorical_space and rng.random() < 0.6:
            feat = sorted(self.categorical_space)[int(rng.integers(0, len(self.categorical_space)))]
            choices = self.categorical_space[feat]
            conds.append(Condition(feat, "==", str(choices[int(rng.integers(0, len(choices)))])))
        n_numeric = max(1, k - len(conds))
        n_numeric = min(n_numeric, len(self.feature_space))
        picked = rng.choice(len(self.feature_space), size=n_numeric, replace=False)
        for j in sorted(int(p) for p in picked):
            spec = self.feature_space[j]
            raw = rng.uniform(spec.low, spec.high)
            value = float(int(round(raw))) if spec.integer else float(np.round(raw, 3))
            op = spec.ops[int(rng.integers(0, len(spec.ops)))]
            conds.append(Condition(spec.name, op, value))
        return tuple(conds)

    def _honest_source(self, rng: np.random.Generator, precision: float, observed_at: int,
                       idx: int) -> SourceEvidence:
        cfg = self.config
        n_obs = int(rng.integers(cfg.honest_observation_range[0], cfg.honest_observation_range[1] + 1))
        r_plus = int(rng.binomial(n_obs, precision))
        src_type = HONEST_SOURCE_TYPES[int(rng.integers(0, len(HONEST_SOURCE_TYPES)))]
        return SourceEvidence(
            source_id=f"{src_type}_{int(rng.integers(0, 50)):03d}",
            source_type=src_type,
            reliability=float(np.round(rng.uniform(*cfg.honest_reliability), 4)),
            r_plus=r_plus, r_minus=n_obs - r_plus,
            observed_at_step=int(observed_at),
            source_age_steps=int(rng.integers(cfg.staleness_horizon_steps, 10 * cfg.staleness_horizon_steps)),
        )

    def _fresh_step(self, now: int, rng: np.random.Generator) -> int:
        return now - int(rng.integers(0, max(1, self.config.staleness_horizon_steps // 4)))

    def _make_genuine(self, rid: str, now: int, rng: np.random.Generator) -> CandidateRule:
        cfg = self.config
        lo = min(0.97, max(cfg.p0 + 0.10, cfg.p1 - 0.10))
        precision = float(rng.uniform(lo, 0.98))
        n_src = int(rng.integers(1, 4))
        evidence = tuple(self._honest_source(rng, precision, self._fresh_step(now, rng), j)
                         for j in range(n_src))
        return CandidateRule(rid, "genuine", self._predicate(rng), evidence,
                             emitted_at_step=max(e.observed_at_step for e in evidence),
                             true_precision=precision, historical_precision=precision,
                             is_harmful=precision <= cfg.p0, base_rate=cfg.base_rate)

    def _make_stale(self, rid: str, now: int, rng: np.random.Generator) -> CandidateRule:
        cfg = self.config
        hist = float(rng.uniform(max(cfg.p1 - 0.1, cfg.p0 + 0.05), 0.97))
        current = float(rng.uniform(0.02, cfg.p0 * 0.6))
        h = cfg.staleness_horizon_steps
        n_src = int(rng.integers(1, 3))
        evidence = tuple(
            self._honest_source(rng, hist, now - int(rng.integers(h, 3 * h + 1)), j)
            for j in range(n_src)
        )
        return CandidateRule(rid, "stale", self._predicate(rng), evidence,
                             emitted_at_step=max(e.observed_at_step for e in evidence),
                             true_precision=current, historical_precision=hist,
                             is_harmful=True, base_rate=cfg.base_rate)

    def _make_conflicting(self, rid: str, now: int, rng: np.random.Generator,
                          target: Optional[CandidateRule]) -> CandidateRule:
        cfg = self.config
        precision = float(np.clip(rng.uniform(cfg.p0 - 0.15, cfg.p0 + 0.15), 0.01, 0.99))
        predicate = target.conditions if target is not None else self._predicate(rng)
        supportive = self._honest_source(rng, float(rng.uniform(0.80, 0.95)), self._fresh_step(now, rng), 0)
        opposing = self._honest_source(rng, float(rng.uniform(0.05, 0.20)), self._fresh_step(now, rng), 1)
        evidence: List[SourceEvidence] = [supportive, opposing]
        if rng.random() < 0.5:
            evidence.append(self._honest_source(rng, precision, self._fresh_step(now, rng), 2))
        order = rng.permutation(len(evidence))
        evidence_t = tuple(evidence[int(i)] for i in order)
        return CandidateRule(rid, "conflicting", predicate, evidence_t,
                             emitted_at_step=max(e.observed_at_step for e in evidence_t),
                             true_precision=precision, historical_precision=precision,
                             is_harmful=precision <= cfg.p0, base_rate=cfg.base_rate,
                             conflict_target=target.rule_id if target is not None else None)

    def _fabricated_counts(self, rng: np.random.Generator) -> Tuple[int, int]:
        cfg = self.config
        n_obs = int(rng.integers(cfg.fabricated_observation_range[0], cfg.fabricated_observation_range[1] + 1))
        claimed = float(rng.uniform(0.90, 0.99))
        r_plus = int(round(n_obs * claimed))
        return r_plus, n_obs - r_plus

    def _make_poisoned(self, rid: str, now: int, rng: np.random.Generator) -> CandidateRule:
        cfg = self.config
        precision = float(rng.uniform(0.02, cfg.p0 * 0.5))
        r_plus, r_minus = self._fabricated_counts(rng)
        evidence = (SourceEvidence(
            source_id=f"adversary_{int(rng.integers(0, 10_000)):05d}",
            source_type="external_feed",
            reliability=float(np.round(rng.uniform(*cfg.adversarial_reliability), 4)),
            r_plus=r_plus, r_minus=r_minus,
            observed_at_step=now - int(rng.integers(0, 6)),
            source_age_steps=int(rng.integers(1, 48)),
        ),)
        return CandidateRule(rid, "poisoned", self._predicate(rng), evidence,
                             emitted_at_step=evidence[0].observed_at_step,
                             true_precision=precision, historical_precision=precision,
                             is_harmful=True, base_rate=cfg.base_rate)

    def _make_sybil_clusters(self, n_sybil_rules: int, now: int,
                             rng: np.random.Generator) -> List[List[Dict[str, object]]]:
        if n_sybil_rules == 0:
            return []
        cfg = self.config
        n_clusters = max(1, math.ceil(n_sybil_rules / max(1, cfg.sybil_rules_per_cluster)))
        clusters = []
        for c in range(n_clusters):
            k = int(rng.integers(cfg.sybil_cluster_size[0], cfg.sybil_cluster_size[1] + 1))
            created = now - int(rng.integers(1, 24))
            members = [{
                "source_id": f"sybil_c{c:02d}_{j:02d}",
                "cluster_id": c,
                "reliability": float(np.round(rng.uniform(*cfg.adversarial_reliability), 4)),
                "age": int(now - created + rng.integers(0, 3)),
            } for j in range(k)]
            clusters.append(members)
        return clusters

    def _make_sybil(self, rid: str, now: int, rng: np.random.Generator,
                    cluster: List[Dict[str, object]], cursor: int) -> CandidateRule:
        cfg = self.config
        precision = float(rng.uniform(0.02, cfg.p0 * 0.5))
        base_plus, base_minus = self._fabricated_counts(rng)
        observed = now - int(rng.integers(0, 3))
        evidence = []
        for member in cluster:
            jitter = int(rng.integers(-2, 3))
            evidence.append(SourceEvidence(
                source_id=str(member["source_id"]),
                source_type="sybil_feed",
                reliability=float(member["reliability"]),
                r_plus=max(0, base_plus + jitter),
                r_minus=max(0, base_minus + int(rng.integers(0, 2))),
                observed_at_step=observed,
                source_age_steps=int(member["age"]),
                sybil_cluster_id=int(member["cluster_id"]),
            ))
        return CandidateRule(rid, "sybil", self._predicate(rng), tuple(evidence),
                             emitted_at_step=observed, true_precision=precision,
                             historical_precision=precision, is_harmful=True,
                             base_rate=cfg.base_rate)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate adversarial candidate-rule feeds")
    root_default = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--config", type=Path, default=Path("configs/project_config.json"))
    parser.add_argument("--hyperparams", type=Path, default=Path("configs/hyperparams.json"))
    parser.add_argument("--rates", type=float, nargs="*", default=None)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    project = json.loads((root / args.config).read_text(encoding="utf-8"))
    hyper = json.loads((root / args.hyperparams).read_text(encoding="utf-8"))
    cfg = PerturbationConfig.from_configs(project, hyper)
    gen = PerturbationGenerator(cfg, seed=int(project.get("project", {}).get("seed", SEED)))
    out_dir = root / project.get("paths", {}).get("rule_feed_dir", "data/synthetic/rule_feeds")
    for rate, feed in gen.poison_rate_sweep(args.rates).items():
        path = feed.to_json(out_dir / f"rule_feed_poison_{int(round(rate * 100)):02d}.json")
        s = feed.summary()
        print(f"poison={rate:.2f} -> {path.name}: counts={s['counts']} "
              f"harmful={s['n_harmful']} sybil_sources={s['n_sybil_sources']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
