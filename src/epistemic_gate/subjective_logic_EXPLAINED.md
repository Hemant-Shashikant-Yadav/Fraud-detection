# Code Walkthrough: `src/epistemic_gate/subjective_logic.py`

**Purpose:** Subjective Logic (Jøsang 2016) for *candidate rules*. It answers: "how much trustworthy evidence do we have for this rule, and how much is simply unknown?" It also **screens** candidates before they enter the martingale quarantine buffer.

---

## 1. Concepts
A **binomial opinion** is `ω = (b, d, u, a)` with `b + d + u = 1`:
* `b` belief, `d` disbelief, `u` **uncertainty** (lack of evidence), `a` base rate.
* From evidence counts: `b = r⁺/(r⁺+r⁻+W)`, `d = r⁻/(…)`, `u = W/(…)`, with prior weight `W = 2`. `r⁺` = confirmed-fraud firings (hits), `r⁻` = false alarms.
* Expected probability (Beta posterior mean) `E = b + a·u`.

Operators used:
* **Trust discount** `π ⊗ ω = (πb, πd, 1 − π(b+d), a)` — unreliable sources contribute less belief and more uncertainty.
* **Freshness decay** `ψ(Δτ) = exp(−η·Δτ)`, applied with the same discount.
* **Cumulative fusion (⊕)** for *independent* sources — equals adding evidence.
* **Averaging fusion** for *dependent* sources (e.g. a Sybil cluster) — idempotent, so `k` copies of one report count once.

## 2. Section-by-section

### 2.1 Constants and `_check_unit` (lines 58–66)
`ATOL`, `DEFAULT_PRIOR_WEIGHT = 2.0`; helper that validates a number is in [0, 1] and clamps rounding errors.

### 2.2 `SubjectiveOpinion` (lines 69–224)
Frozen dataclass with `belief, disbelief, uncertainty, base_rate, prior_weight`.
* `__post_init__`: validates ranges, requires `b+d+u = 1` (tolerance 1e-9), re-normalises away floating-point drift, and snaps tiny residual uncertainty to an exactly dogmatic opinion.
* Builders: `from_evidence(r⁺, r⁻)` (Beta-Dirichlet mapping), `vacuous()` (total ignorance `(0,0,1,a)`, neutral element of fusion), `dogmatic(p)` (zero uncertainty), `from_tuple`, `from_dict`.
* Extractors: `expected_probability`, `epistemic_uncertainty`, `is_dogmatic`, `is_vacuous`, `evidence()` (inverse mapping), `beta_parameters()`, `variance()`, `as_tuple`, `to_dict`.
* Operators: `trust_discount(π)`, `time_decay(Δτ, η)`, `consensus_fuse` / `+` (cumulative), `average_fuse`, and `observe(hits, false_alarms)` — a Bayesian update from first-hand firings (a local source with π = 1).

### 2.3 Fusion operators (lines 230–332)
* `freshness(Δτ, η)`: `exp(−η·max(0, Δτ))`.
* `cumulative_fuse_pair`: Jøsang's formulas with `κ = uA + uB − uA·uB`, handling the **both-dogmatic** limit (equal relative dogmatism) and the **both-vacuous** base-rate limit explicitly; re-normalises at the end. Opinions must share the same prior weight `W`.
* `averaging_fuse_pair`: averaging operator with the corresponding dogmatic case.
* `cumulative_fuse(opinions)`: n-ary version; non-dogmatic opinions are fused **exactly by evidence addition** (associative, order-independent); if any is dogmatic, dogmatic opinions dominate.
* `averaging_fuse(opinions)`: n-ary evidence averaging.

### 2.4 `SourceOpinion` and `FusionResult` (lines 338–362)
Audit records: each source's raw and discounted opinion, its reliability `π`, freshness `ψ` and dependence group; the fused opinion, number of groups, and the screening decision (`quarantine` or `reject`).

### 2.5 `EpistemicFusionEngine` (lines 365–436)
Builds `ω̃_k = ⊕_g [ ⊙_{m∈g} (π_m ψ_{k,m} ⊗ ω_k^(m)) ]`:
1. For each evidence source: opinion from its `(r⁺, r⁻)`, discount by reliability `π` (default 0.5) and freshness `ψ` (η = 0.01 per step).
2. **Group** sources: those with the same `sybil_cluster_id` form one dependence group (if `cluster_aware`); each group is combined by *averaging* fusion.
3. Combine independent groups by *cumulative* fusion.
4. `screen`: a candidate is admitted to quarantine iff `b ≥ min_belief (0.55)` and `u ≤ max_uncertainty (0.35)`.
`from_config` reads `hyperparams.json['subjective_logic']`; `source_opinion`, `fuse_evidence`, `fuse_rule` expose the steps.

## 3. Why it matters for the paper
* Sybil-resistance: coordinated duplicate reports (poisoning) cannot add evidence beyond one report's worth.
* The expected probability can serve as the prior mean for the **plug-in** betting strategy in the e-process, linking Subjective Logic and the martingale test.
* Uncertainty `u` is *epistemic* (not enough evidence), which is distinct from the calibrated Venn-Abers width on transactions.
