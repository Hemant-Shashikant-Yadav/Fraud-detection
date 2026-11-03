"""
src/epistemic_gate/subjective_logic.py
======================================

Subjective-Logic epistemic calculus for candidate fraud rules (Josang 2016).

A binomial opinion about candidate rule ``c_k`` is the tuple

    omega_k = (b_k, d_k, u_k, a_k),      b + d + u = 1,  a in [0, 1]

with the Beta/Dirichlet evidence mapping (non-informative prior weight ``W``)

    b = r+ / (r+ + r- + W),   d = r- / (r+ + r- + W),   u = W / (r+ + r- + W)

where ``r+`` counts confirmed-fraud firings (hits) and ``r-`` false alarms.
For ``W = 2`` this is exactly the synopsis mapping. The projected probability
(posterior mean of the Beta) is ``E = b + a * u``.

Operators
---------
* Trust (provenance) discounting by source reliability ``pi in [0, 1]``::

      pi (x) omega = (pi * b, pi * d, 1 - pi * (b + d), a)

* Temporal freshness decay ``psi(delta_tau) = exp(-eta * delta_tau)``, applied
  with the same discount operator, as in the synopsis aggregation
  ``omega~_k = (+)_m (pi_m psi_{k,m} (x) omega_k^(m))``. Discounts compose
  multiplicatively: ``pi (x) (psi (x) omega) = (pi psi) (x) omega``.

* Josang's cumulative consensus (``(+)``, aleatory cumulative belief fusion)
  for independent sources::

      kappa = u_A + u_B - u_A u_B
      b     = (b_A u_B + b_B u_A) / kappa
      u     = u_A u_B / kappa
      a     = (a_A u_B + a_B u_A - (a_A + a_B) u_A u_B) / (u_A + u_B - 2 u_A u_B)

  with the dogmatic limits (``u_A = u_B = 0``: equal relative dogmatism) and
  the both-vacuous base-rate limit handled explicitly. For non-dogmatic
  opinions with a common base rate, cumulative fusion equals *evidence
  addition* (``r = sum r_m``), which gives an exact, associative n-ary form.

* Averaging fusion (for *dependent* sources, e.g. a correlated Sybil cluster)
  equals *evidence averaging*; it is idempotent, so ``k`` copies of the same
  report carry the evidence of one report.

:class:`EpistemicFusionEngine` assembles the per-source opinions of a
:class:`src.data.perturbation_generator.CandidateRule` into the fused opinion
``omega~_k`` and screens it before the candidate enters the martingale
quarantine buffer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ATOL = 1e-12
DEFAULT_PRIOR_WEIGHT = 2.0


def _check_unit(name: str, value: float) -> float:
    value = float(value)
    if not (math.isfinite(value) and -ATOL <= value <= 1.0 + ATOL):
        raise ValueError(f"{name}={value} must lie in [0, 1]")
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class SubjectiveOpinion:
    """Binomial subjective opinion ``omega = (b, d, u, a)``.

    Instances are immutable; every operator returns a new opinion whose
    additivity ``b + d + u = 1`` is enforced to machine precision.
    """

    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float = 0.5
    prior_weight: float = DEFAULT_PRIOR_WEIGHT

    def __post_init__(self) -> None:
        b = _check_unit("belief", self.belief)
        d = _check_unit("disbelief", self.disbelief)
        u = _check_unit("uncertainty", self.uncertainty)
        a = _check_unit("base_rate", self.base_rate)
        if not (math.isfinite(self.prior_weight) and self.prior_weight > 0.0):
            raise ValueError("prior_weight W must be a positive finite number")
        total = b + d + u
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"b + d + u = {total!r} != 1")
        # Renormalise away floating-point drift so additivity holds to machine precision;
        # a residual uncertainty at rounding level is snapped to an exact dogmatic opinion.
        u = 1.0 - b - d
        if u <= 4.0 * 2.220446049250313e-16:
            u, d = 0.0, max(0.0, 1.0 - b)
        object.__setattr__(self, "belief", b)
        object.__setattr__(self, "disbelief", d)
        object.__setattr__(self, "uncertainty", u)
        object.__setattr__(self, "base_rate", a)
        object.__setattr__(self, "prior_weight", float(self.prior_weight))

    # ------------------------------------------------------------ builders
    @classmethod
    def from_evidence(cls, r_plus: float, r_minus: float, base_rate: float = 0.5,
                      prior_weight: float = DEFAULT_PRIOR_WEIGHT) -> "SubjectiveOpinion":
        """Beta-Dirichlet mapping ``(r+, r-) -> (b, d, u, a)``."""
        r_plus, r_minus, w = float(r_plus), float(r_minus), float(prior_weight)
        if r_plus < 0.0 or r_minus < 0.0 or not (math.isfinite(r_plus) and math.isfinite(r_minus)):
            raise ValueError("evidence counts must be finite and non-negative")
        if not w > 0.0:
            raise ValueError("prior_weight W must be positive")
        denom = r_plus + r_minus + w
        b, d = r_plus / denom, r_minus / denom
        return cls(b, d, 1.0 - b - d, base_rate, w)

    @classmethod
    def vacuous(cls, base_rate: float = 0.5,
                prior_weight: float = DEFAULT_PRIOR_WEIGHT) -> "SubjectiveOpinion":
        """Total ignorance ``(0, 0, 1, a)`` - neutral element of consensus."""
        return cls(0.0, 0.0, 1.0, base_rate, prior_weight)

    @classmethod
    def dogmatic(cls, probability: float, base_rate: float = 0.5,
                 prior_weight: float = DEFAULT_PRIOR_WEIGHT) -> "SubjectiveOpinion":
        """Zero-uncertainty opinion ``(p, 1 - p, 0, a)``."""
        p = _check_unit("probability", probability)
        return cls(p, 1.0 - p, 0.0, base_rate, prior_weight)

    @classmethod
    def from_tuple(cls, values: Sequence[float],
                   prior_weight: float = DEFAULT_PRIOR_WEIGHT) -> "SubjectiveOpinion":
        b, d, u, a = (float(v) for v in values)
        return cls(b, d, u, a, prior_weight)

    @classmethod
    def from_dict(cls, payload: Mapping[str, float]) -> "SubjectiveOpinion":
        return cls(float(payload["b"]), float(payload["d"]), float(payload["u"]),
                   float(payload.get("a", 0.5)), float(payload.get("W", DEFAULT_PRIOR_WEIGHT)))

    # ---------------------------------------------------------- extractors
    @property
    def expected_probability(self) -> float:
        """Projected probability ``E[omega] = b + a u`` (Beta posterior mean)."""
        return self.belief + self.base_rate * self.uncertainty

    @property
    def epistemic_uncertainty(self) -> float:
        """Uncertainty mass ``u`` (lack of evidence, *not* aleatoric variance)."""
        return self.uncertainty

    @property
    def is_dogmatic(self) -> bool:
        return self.uncertainty <= ATOL

    @property
    def is_vacuous(self) -> bool:
        return self.uncertainty >= 1.0 - ATOL

    def evidence(self) -> Tuple[float, float]:
        """Inverse mapping ``(b, d, u) -> (r+, r-)`` (requires ``u > 0``)."""
        if self.is_dogmatic:
            raise ValueError("dogmatic opinion corresponds to infinite evidence")
        w = self.prior_weight
        return w * self.belief / self.uncertainty, w * self.disbelief / self.uncertainty

    def beta_parameters(self) -> Tuple[float, float]:
        """Beta(alpha, beta) with ``alpha = r+ + a W`` and ``beta = r- + (1 - a) W``."""
        r_plus, r_minus = self.evidence()
        w = self.prior_weight
        return r_plus + self.base_rate * w, r_minus + (1.0 - self.base_rate) * w

    def variance(self) -> float:
        """Variance of the equivalent Beta pdf (``0`` for dogmatic opinions)."""
        if self.is_dogmatic:
            return 0.0
        alpha, beta = self.beta_parameters()
        s = alpha + beta
        return alpha * beta / (s * s * (s + 1.0))

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.belief, self.disbelief, self.uncertainty, self.base_rate

    def to_dict(self) -> Dict[str, float]:
        return {"b": self.belief, "d": self.disbelief, "u": self.uncertainty,
                "a": self.base_rate, "W": self.prior_weight,
                "E": self.expected_probability}

    # ----------------------------------------------------------- operators
    def trust_discount(self, pi: float) -> "SubjectiveOpinion":
        """Provenance discounting ``pi (x) omega = (pi b, pi d, 1 - pi (b + d), a)``."""
        pi = _check_unit("pi", pi)
        b, d = pi * self.belief, pi * self.disbelief
        return SubjectiveOpinion(b, d, 1.0 - b - d, self.base_rate, self.prior_weight)

    def time_decay(self, delta_tau: float, eta: float) -> "SubjectiveOpinion":
        """Freshness decay ``psi(delta_tau) (x) omega`` with ``psi = exp(-eta delta_tau)``."""
        return self.trust_discount(freshness(delta_tau, eta))

    def consensus_fuse(self, other: "SubjectiveOpinion") -> "SubjectiveOpinion":
        """Josang's cumulative consensus operator ``self (+) other``."""
        return cumulative_fuse_pair(self, other)

    def __add__(self, other: "SubjectiveOpinion") -> "SubjectiveOpinion":
        if not isinstance(other, SubjectiveOpinion):
            return NotImplemented
        return cumulative_fuse_pair(self, other)

    def average_fuse(self, other: "SubjectiveOpinion") -> "SubjectiveOpinion":
        """Averaging fusion for dependent sources."""
        return averaging_fuse_pair(self, other)

    def observe(self, hits: float, false_alarms: float) -> "SubjectiveOpinion":
        """Bayesian update with first-hand firing outcomes (local source, ``pi = 1``)."""
        local = SubjectiveOpinion.from_evidence(hits, false_alarms, self.base_rate, self.prior_weight)
        return cumulative_fuse_pair(self, local)

    def is_close(self, other: "SubjectiveOpinion", tol: float = 1e-9) -> bool:
        return all(abs(x - y) <= tol for x, y in zip(self.as_tuple(), other.as_tuple()))

    def __repr__(self) -> str:
        return (f"SubjectiveOpinion(b={self.belief:.4f}, d={self.disbelief:.4f}, "
                f"u={self.uncertainty:.4f}, a={self.base_rate:.3f}, E={self.expected_probability:.4f})")


# --------------------------------------------------------------------------- #
# Fusion operators                                                            #
# --------------------------------------------------------------------------- #
def freshness(delta_tau: float, eta: float) -> float:
    """``psi(delta_tau) = exp(-eta * delta_tau)`` (negative ages are clipped to 0)."""
    delta_tau, eta = float(delta_tau), float(eta)
    if eta < 0.0 or not math.isfinite(eta):
        raise ValueError("eta must be a non-negative finite decay rate")
    return math.exp(-eta * max(0.0, delta_tau))


def _check_weight(a: SubjectiveOpinion, b: SubjectiveOpinion) -> float:
    if abs(a.prior_weight - b.prior_weight) > 1e-12:
        raise ValueError("cannot fuse opinions with different prior weights W")
    return a.prior_weight


def cumulative_fuse_pair(op_a: SubjectiveOpinion, op_b: SubjectiveOpinion) -> SubjectiveOpinion:
    """Cumulative consensus ``omega_A (+) omega_B`` (Josang 2016, Def. 12.5)."""
    w = _check_weight(op_a, op_b)
    ua, ub = op_a.uncertainty, op_b.uncertainty
    if ua <= ATOL and ub <= ATOL:
        # Both dogmatic: limit with equal relative dogmatism gamma_A = gamma_B = 1/2.
        b = 0.5 * (op_a.belief + op_b.belief)
        d = 0.5 * (op_a.disbelief + op_b.disbelief)
        return SubjectiveOpinion(b, d, 0.0, 0.5 * (op_a.base_rate + op_b.base_rate), w)
    kappa = ua + ub - ua * ub
    b = (op_a.belief * ub + op_b.belief * ua) / kappa
    d = (op_a.disbelief * ub + op_b.disbelief * ua) / kappa
    u = (ua * ub) / kappa
    denom_a = ua + ub - 2.0 * ua * ub
    if denom_a <= ATOL:  # both vacuous (or both dogmatic, handled above)
        a = 0.5 * (op_a.base_rate + op_b.base_rate)
    else:
        a = (op_a.base_rate * ub + op_b.base_rate * ua
             - (op_a.base_rate + op_b.base_rate) * ua * ub) / denom_a
    total = b + d + u
    return SubjectiveOpinion(b / total, d / total, 1.0 - (b + d) / total, min(1.0, max(0.0, a)), w)


def averaging_fuse_pair(op_a: SubjectiveOpinion, op_b: SubjectiveOpinion) -> SubjectiveOpinion:
    """Averaging fusion ``omega_A (_) omega_B`` (Josang 2016, Def. 12.7)."""
    w = _check_weight(op_a, op_b)
    ua, ub = op_a.uncertainty, op_b.uncertainty
    a = 0.5 * (op_a.base_rate + op_b.base_rate)
    if ua <= ATOL and ub <= ATOL:
        return SubjectiveOpinion(0.5 * (op_a.belief + op_b.belief),
                                 0.5 * (op_a.disbelief + op_b.disbelief), 0.0, a, w)
    s = ua + ub
    b = (op_a.belief * ub + op_b.belief * ua) / s
    d = (op_a.disbelief * ub + op_b.disbelief * ua) / s
    return SubjectiveOpinion(b, d, max(0.0, 1.0 - b - d), a, w)


def _split_dogmatic(opinions: Sequence[SubjectiveOpinion]):
    dogmatic = [o for o in opinions if o.is_dogmatic]
    return dogmatic, [o for o in opinions if not o.is_dogmatic]


def cumulative_fuse(opinions: Iterable[SubjectiveOpinion]) -> SubjectiveOpinion:
    """n-ary cumulative consensus ``(+)_m omega_m``.

    Non-dogmatic opinions are fused exactly by evidence addition (associative
    and order-independent); base rates are combined with the pairwise formula.
    If any opinion is dogmatic, the dogmatic opinions dominate and are averaged.
    """
    ops = list(opinions)
    if not ops:
        raise ValueError("cumulative_fuse needs at least one opinion")
    w = ops[0].prior_weight
    for o in ops[1:]:
        _check_weight(ops[0], o)
    dogmatic, regular = _split_dogmatic(ops)
    if dogmatic:
        out = dogmatic[0]
        for o in dogmatic[1:]:
            out = cumulative_fuse_pair(out, o)
        return out
    r_plus = r_minus = 0.0
    for o in regular:
        rp, rm = o.evidence()
        r_plus += rp
        r_minus += rm
    base = regular[0]
    for o in regular[1:]:
        base = cumulative_fuse_pair(base, o)
    return SubjectiveOpinion.from_evidence(r_plus, r_minus, base.base_rate, w)


def averaging_fuse(opinions: Iterable[SubjectiveOpinion]) -> SubjectiveOpinion:
    """n-ary averaging fusion (evidence averaging; idempotent)."""
    ops = list(opinions)
    if not ops:
        raise ValueError("averaging_fuse needs at least one opinion")
    w = ops[0].prior_weight
    for o in ops[1:]:
        _check_weight(ops[0], o)
    a = sum(o.base_rate for o in ops) / len(ops)
    dogmatic, regular = _split_dogmatic(ops)
    if dogmatic:
        b = sum(o.belief for o in dogmatic) / len(dogmatic)
        return SubjectiveOpinion(b, 1.0 - b, 0.0, a, w)
    n = float(len(regular))
    r_plus = sum(o.evidence()[0] for o in regular) / n
    r_minus = sum(o.evidence()[1] for o in regular) / n
    return SubjectiveOpinion.from_evidence(r_plus, r_minus, a, w)


# --------------------------------------------------------------------------- #
# Candidate-level fusion engine                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceOpinion:
    """One source's discounted contribution to ``omega~_k`` (audit record)."""

    source_id: str
    raw: SubjectiveOpinion
    pi: float
    psi: float
    discounted: SubjectiveOpinion
    group: str


@dataclass(frozen=True)
class FusionResult:
    opinion: SubjectiveOpinion
    sources: Tuple[SourceOpinion, ...]
    n_groups: int
    decision: str  # "quarantine" | "reject"

    def to_dict(self) -> Dict[str, object]:
        return {"opinion": self.opinion.to_dict(), "n_sources": len(self.sources),
                "n_groups": self.n_groups, "decision": self.decision,
                "sources": [{"source_id": s.source_id, "pi": s.pi, "psi": s.psi,
                             "group": s.group, "discounted": s.discounted.to_dict()}
                            for s in self.sources]}


class EpistemicFusionEngine:
    """Builds ``omega~_k = (+)_g [ (_)_{m in g} (pi_m psi_{k,m} (x) omega_k^(m)) ]``.

    Sources are first grouped into *dependence groups*: sources flagged as one
    correlated cluster (``sybil_cluster_id`` set by the upstream source-
    correlation detector) form one group and are combined by averaging fusion,
    so ``k`` near-duplicate reports count as one. Independent groups are then
    combined by Josang's cumulative consensus. With ``cluster_aware=False``
    every source is its own group (plain synopsis aggregation).

    Screening: a candidate enters the martingale quarantine buffer iff
    ``b >= min_belief`` and ``u <= max_uncertainty``; otherwise it is rejected.
    """

    def __init__(self, prior_weight: float = DEFAULT_PRIOR_WEIGHT, eta: float = 0.01,
                 min_belief: float = 0.55, max_uncertainty: float = 0.35,
                 default_reliability: float = 0.5, cluster_aware: bool = True) -> None:
        if eta < 0.0:
            raise ValueError("eta must be non-negative")
        self.prior_weight = float(prior_weight)
        self.eta = float(eta)
        self.min_belief = _check_unit("min_belief", min_belief)
        self.max_uncertainty = _check_unit("max_uncertainty", max_uncertainty)
        self.default_reliability = _check_unit("default_reliability", default_reliability)
        self.cluster_aware = bool(cluster_aware)

    @classmethod
    def from_config(cls, hyperparams: Optional[Mapping] = None, **overrides) -> "EpistemicFusionEngine":
        sl = dict((hyperparams or {}).get("subjective_logic", {}))
        kwargs = {
            "prior_weight": float(sl.get("prior_weight_W", DEFAULT_PRIOR_WEIGHT)),
            "eta": float(sl.get("freshness_eta", 0.01)),
            "min_belief": float(sl.get("min_belief_for_quarantine", 0.55)),
            "max_uncertainty": float(sl.get("max_uncertainty_for_quarantine", 0.35)),
            "default_reliability": float(sl.get("default_source_reliability", 0.5)),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def source_opinion(self, evidence, now: int, base_rate: float = 0.5) -> SourceOpinion:
        """Discounted opinion ``pi_m psi_{k,m} (x) omega_k^(m)`` of one source."""
        raw = SubjectiveOpinion.from_evidence(evidence.r_plus, evidence.r_minus, base_rate,
                                              self.prior_weight)
        pi = getattr(evidence, "reliability", None)
        pi = self.default_reliability if pi is None else float(pi)
        psi = freshness(float(now) - float(evidence.observed_at_step), self.eta)
        cluster = getattr(evidence, "sybil_cluster_id", None)
        group = (f"cluster:{cluster}" if (self.cluster_aware and cluster is not None)
                 else f"source:{evidence.source_id}")
        return SourceOpinion(str(evidence.source_id), raw, pi, psi,
                             raw.trust_discount(pi * psi), group)

    def fuse_evidence(self, evidence: Sequence, now: int, base_rate: float = 0.5) -> FusionResult:
        if not evidence:
            op = SubjectiveOpinion.vacuous(base_rate, self.prior_weight)
            return FusionResult(op, (), 0, self.screen(op))
        per_source = tuple(self.source_opinion(e, now, base_rate) for e in evidence)
        groups: Dict[str, List[SubjectiveOpinion]] = {}
        for s in per_source:
            groups.setdefault(s.group, []).append(s.discounted)
        group_ops = [averaging_fuse(ops) if len(ops) > 1 else ops[0]
                     for _, ops in sorted(groups.items())]
        fused = cumulative_fuse(group_ops)
        return FusionResult(fused, per_source, len(group_ops), self.screen(fused))

    def fuse_rule(self, rule, now: int) -> FusionResult:
        """Fused opinion for a :class:`CandidateRule` at logical step ``now``."""
        return self.fuse_evidence(tuple(rule.evidence), now, float(getattr(rule, "base_rate", 0.5)))

    def screen(self, opinion: SubjectiveOpinion) -> str:
        ok = opinion.belief >= self.min_belief and opinion.uncertainty <= self.max_uncertainty
        return "quarantine" if ok else "reject"
