"""
experiments/run_adversarial_stress_test.py
==========================================

Phase 7 - adversarial knowledge-poisoning stress test of the MG-ECFM gate.

For every poisoning rate ``rho in {5%, 10%, 15%, 20%, 25%}`` a contaminated
candidate-rule feed (genuine, stale, conflicting, poisoned and Sybil-backed
rules) is generated with :mod:`src.data.perturbation_generator` and processed
by three knowledge-admission policies:

``mg_ecfm_gate``     Subjective-Logic screening + martingale e-process admission
                     (``H0: Precision <= p0``) + rollback monitoring
                     (``H0^good: FPR <= q0``) - the proposed gate.
``sl_screen_only``   Subjective-Logic fused-opinion screening alone
                     (``b >= min_belief``, ``u <= max_uncertainty``); no
                     sequential testing, no rollback.
``auto_admit``       Gate OFF: every candidate enters the KB.

Two experiments are run per rate (firing outcomes are Bernoulli draws from
each rule's latent *current* precision, one batch of ``firings_per_cycle``
verified firings per rule per update cycle):

1. **Admission** - all candidates are submitted; after ``n_cycles`` cycles the
   empirical false-admission rate ``#{harmful ever admitted} / #{harmful}``
   (harmful = latent precision ``<= p0``), the adversarial-only rate (poisoned
   + Sybil), the genuine admission rate and the admission accuracy are recorded.
2. **Injection / rollback** - the harmful rules of the feed are *injected
   directly* into the active KB (an attacker or a stale legacy import that
   bypasses admission), together with the genuine rules. The number of update
   cycles until each harmful rule is rolled back (time-to-rollback) is
   recorded, and the false-rollback frequency of the null-good rules
   (``FPR <= q0``) is measured.

Ville's maximal inequality
--------------------------
``P_{H0}(exists t >= 1 : M_t^roll >= 1/alpha_roll) <= alpha_roll`` is verified
three ways for every rate: (i) the exact crossing probability at the least
favourable null ``FPR = q0`` (dynamic programme over the event count, horizon
``ville_horizon``); (ii) a Monte-Carlo estimate over ``n_ville_reps``
independent monitors, with a Clopper-Pearson 95% interval and the one-sided
binomial p-value of ``H: rate > alpha``; (iii) the empirical false-rollback
frequency of the null-good rules monitored inside the full gate. The bound is
declared satisfied when the exact probability is ``<= alpha`` and neither
empirical frequency is significantly above ``alpha`` (one-sided binomial test
at the 5% level).

Output: ``reports/adversarial_robustness_results.csv`` (one row per
rate x policy) and ``reports/adversarial_time_to_rollback.csv`` (one row per
injected harmful rule).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_drift_eval import load_configs  # noqa: E402
from src.data.perturbation_generator import (  # noqa: E402
    CandidateFeed,
    PerturbationConfig,
    PerturbationGenerator,
)
from src.epistemic_gate.martingale_eprocess import (  # noqa: E402
    GateConfig,
    MartingaleEProcessGate,
    crossing_probability,
    min_events_to_cross,
    simulate_crossing_rate,
)
from src.epistemic_gate.subjective_logic import EpistemicFusionEngine  # noqa: E402

SEED = 42
REPORTS_DIR = ROOT / "reports"
DEFAULT_RESULTS_PATH = REPORTS_DIR / "adversarial_robustness_results.csv"
DEFAULT_TTR_PATH = REPORTS_DIR / "adversarial_time_to_rollback.csv"
POISON_RATES: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)
POLICIES: Tuple[str, ...] = ("mg_ecfm_gate", "sl_screen_only", "auto_admit")


# --------------------------------------------------------------------------- #
# Exact binomial helpers (no SciPy dependency)                                #
# --------------------------------------------------------------------------- #
def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binomial_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Bin(n, p)``."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = [_log_binom_pmf(i, n, p) for i in range(k, n + 1)]
    m = max(logs)
    if m == -math.inf:
        return 0.0
    return float(min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs)))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)``."""
    return 1.0 - binomial_sf(k + 1, n, p) if k < n else 1.0


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval for a binomial proportion (bisection)."""
    if n <= 0:
        return 0.0, 1.0
    a = (1.0 - confidence) / 2.0

    def solve(f, target: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > target:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    lower = 0.0 if k == 0 else solve(lambda p: binomial_sf(k, n, p), a)
    upper = 1.0 if k == n else solve(lambda p: 1.0 - binomial_cdf(k, n, p), 1.0 - a)
    return float(lower), float(upper)


# --------------------------------------------------------------------------- #
# Simulation primitives                                                       #
# --------------------------------------------------------------------------- #
def simulate_cycles(gate: MartingaleEProcessGate, precision_of: Mapping[str, float], n_cycles: int,
                    firings_per_cycle: int, rng: np.random.Generator) -> List[Any]:
    """Run ``n_cycles`` update cycles with Bernoulli(precision) firing outcomes."""
    reports = []
    for _ in range(n_cycles):
        outcomes = {rid: (rng.random(firings_per_cycle) < precision_of[rid]).astype(int).tolist()
                    for rid, rec in sorted(gate.records.items())
                    if rec.status in ("quarantined", "active")}
        reports.append(gate.run_update_cycle(outcomes))
    return reports


def admission_experiment(feed: CandidateFeed, hyperparams: Mapping[str, Any], policy: str,
                         n_cycles: int, firings_per_cycle: int, seed: int) -> Dict[str, Any]:
    """False-admission statistics of one admission policy on one feed."""
    rules = feed.rules
    harmful = [r for r in rules if r.is_harmful]
    adversarial = [r for r in rules if r.is_adversarial]
    good = [r for r in rules if not r.is_harmful]
    out: Dict[str, Any] = {}
    t0 = time.perf_counter()
    if policy == "mg_ecfm_gate":
        gate = MartingaleEProcessGate.from_config(hyperparams)
        for r in rules:
            gate.submit_candidate(r, feed.current_step)
        truth = {r.rule_id: r.true_precision for r in rules}
        reports = simulate_cycles(gate, truth, n_cycles, firings_per_cycle,
                                  np.random.default_rng([seed, 1, int(round(feed.poison_rate * 1e4))]))
        ever = {rid for rep in reports for rid in rep.promoted}
        active = set(gate.active_rule_ids)
        first_admit = {}
        for rep in reports:
            for rid in rep.promoted:
                first_admit.setdefault(rid, rep.cycle)
        tta = [first_admit[r.rule_id] for r in good if r.rule_id in first_admit]
        out.update({"n_screen_rejected": sum(1 for rec in gate.records.values()
                                             if ("rejected:screen" in [e for _, e in rec.events])),
                    "mean_cycles_to_admission_good": float(np.mean(tta)) if tta else float("nan"),
                    "kb_snapshots": int(gate.store.head_snapshot_id),
                    "store_integrity_ok": bool(gate.store.verify_integrity()["ok"]),
                    "n_rollbacks_during_admission": int(sum(len(rep.rolled_back) for rep in reports))})
    elif policy == "sl_screen_only":
        engine = EpistemicFusionEngine.from_config(hyperparams)
        ever = {r.rule_id for r in rules if engine.fuse_rule(r, feed.current_step).decision == "quarantine"}
        active = set(ever)
        out.update({"n_screen_rejected": len(rules) - len(ever)})
    elif policy == "auto_admit":
        ever = {r.rule_id for r in rules}
        active = set(ever)
        out.update({"n_screen_rejected": 0})
    else:
        raise ValueError(f"unknown policy {policy}")
    far = sum(r.rule_id in ever for r in harmful) / len(harmful) if harmful else float("nan")
    adv_far = sum(r.rule_id in ever for r in adversarial) / len(adversarial) if adversarial else float("nan")
    accuracy = float(np.mean([(r.rule_id in active) != r.is_harmful for r in rules]))
    out.update({
        "n_candidates": len(rules), "n_harmful": len(harmful), "n_adversarial": len(adversarial),
        "n_ever_admitted": len(ever), "n_active_end": len(active),
        "false_admission_rate": float(far),
        "adversarial_false_admission_rate": float(adv_far),
        "harmful_active_end": int(sum(r.rule_id in active for r in harmful)),
        "genuine_admission_rate": float(sum(r.rule_id in ever for r in good) / len(good)) if good else float("nan"),
        "admission_accuracy": accuracy,
        "admission_seconds": time.perf_counter() - t0,
    })
    for cat in ("poisoned", "sybil", "stale", "conflicting"):
        group = [r for r in rules if r.category == cat and r.is_harmful]
        out[f"far_{cat}"] = float(sum(r.rule_id in ever for r in group) / len(group)) if group else float("nan")
    return out


def injection_experiment(feed: CandidateFeed, hyperparams: Mapping[str, Any], gate_cfg: GateConfig,
                         n_cycles: int, firings_per_cycle: int, seed: int
                         ) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Inject harmful + genuine rules as ACTIVE; measure time-to-rollback and false rollbacks."""
    gate = MartingaleEProcessGate.from_config(hyperparams)
    engine = gate.fusion
    harmful = [r for r in feed.rules if r.is_harmful]
    good = [r for r in feed.rules if not r.is_harmful]
    for r in harmful + good:
        gate.inject_active(r, engine.fuse_rule(r, feed.current_step).opinion, reason="adversarial injection")
    truth = {r.rule_id: r.true_precision for r in feed.rules}
    reports = simulate_cycles(gate, truth, n_cycles, firings_per_cycle,
                              np.random.default_rng([seed, 2, int(round(feed.poison_rate * 1e4))]))
    first_rb: Dict[str, int] = {}
    for rep in reports:
        for rid in rep.rolled_back:
            first_rb.setdefault(rid, rep.cycle)
    rows = []
    for r in harmful:
        cyc = first_rb.get(r.rule_id)
        rows.append({"poison_rate": feed.poison_rate, "rule_id": r.rule_id, "category": r.category,
                     "true_precision": r.true_precision, "false_alarm_rate": 1.0 - r.true_precision,
                     "rolled_back": cyc is not None,
                     "time_to_rollback_cycles": float(cyc) if cyc is not None else float("nan"),
                     "firings_to_rollback": float(cyc * firings_per_cycle) if cyc is not None else float("nan")})
    ttr = pd.DataFrame(rows)
    null_good = [r for r in good if 1.0 - r.true_precision <= gate_cfg.q0]
    n_false_rb = sum(r.rule_id in first_rb for r in null_good)
    adv_ttr = ttr[ttr["category"].isin(["poisoned", "sybil"])]["time_to_rollback_cycles"]
    times = ttr["time_to_rollback_cycles"]
    k = gate_cfg.max_rollback_cycles
    summary = {
        "n_injected_harmful": len(harmful), "n_injected_good": len(good),
        "rollback_rate_harmful": float(ttr["rolled_back"].mean()) if len(ttr) else float("nan"),
        "time_to_rollback_mean": float(times.mean()) if times.notna().any() else float("nan"),
        "time_to_rollback_median": float(times.median()) if times.notna().any() else float("nan"),
        "time_to_rollback_max": float(times.max()) if times.notna().any() else float("nan"),
        "adversarial_time_to_rollback_mean": float(adv_ttr.mean()) if adv_ttr.notna().any() else float("nan"),
        "adversarial_time_to_rollback_max": float(adv_ttr.max()) if adv_ttr.notna().any() else float("nan"),
        "share_rolled_back_within_max_cycles": float(((times <= k) & times.notna()).mean()) if len(ttr) else float("nan"),
        "adversarial_share_within_max_cycles": float(((adv_ttr <= k) & adv_ttr.notna()).mean())
        if len(adv_ttr) else float("nan"),
        "max_rollback_cycles": int(k),
        "n_null_good_rules": len(null_good), "n_null_good_false_rollbacks": int(n_false_rb),
        "harmful_active_after_injection_end": int(sum(r.rule_id in set(gate.active_rule_ids) for r in harmful)),
        "injection_store_integrity_ok": bool(gate.store.verify_integrity()["ok"]),
    }
    return summary, ttr


def ville_verification(gate_cfg: GateConfig, horizon: int, n_reps: int, seed: int,
                       n_null_rules: int, n_null_false_rollbacks: int) -> Dict[str, Any]:
    """Exact, Monte-Carlo and in-gate checks of ``P(sup_t M_t^roll >= 1/alpha) <= alpha``."""
    a = gate_cfg.alpha_roll
    exact = crossing_probability(horizon, gate_cfg.q0, gate_cfg.q0, gate_cfg.q1, a)
    sim = simulate_crossing_rate(gate_cfg.q0, gate_cfg.q0, gate_cfg.q1, a, horizon, n_reps, seed=seed)
    k_mc = int(round(sim["rate"] * n_reps))
    lo, hi = clopper_pearson(k_mc, n_reps)
    p_mc = binomial_sf(k_mc, n_reps, a)
    if n_null_rules > 0:
        ig_rate = n_null_false_rollbacks / n_null_rules
        ig_lo, ig_hi = clopper_pearson(n_null_false_rollbacks, n_null_rules)
        p_ig = binomial_sf(n_null_false_rollbacks, n_null_rules, a)
    else:
        ig_rate = ig_lo = ig_hi = float("nan")
        p_ig = 1.0
    satisfied = bool(exact <= a + 1e-12 and p_mc > 0.05 and p_ig > 0.05)
    return {
        "ville_alpha_roll": a, "ville_threshold": 1.0 / a, "ville_horizon": int(horizon),
        "ville_exact_crossing_prob": float(exact),
        "ville_mc_rate": float(sim["rate"]), "ville_mc_reps": int(n_reps),
        "ville_mc_ci_low": lo, "ville_mc_ci_high": hi, "ville_mc_pvalue_exceeds_alpha": float(p_mc),
        "ville_ingate_rate": float(ig_rate), "ville_ingate_ci_low": float(ig_lo),
        "ville_ingate_ci_high": float(ig_hi), "ville_ingate_pvalue_exceeds_alpha": float(p_ig),
        "ville_min_false_alarms_to_cross": int(min_events_to_cross(gate_cfg.q0, gate_cfg.q1, a)),
        "ville_bound_satisfied": satisfied,
    }


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #
def run_adversarial_stress_test(rates: Sequence[float] = POISON_RATES, n_candidates: Optional[int] = None,
                                n_cycles: int = 40, firings_per_cycle: int = 8,
                                ville_horizon: int = 2000, n_ville_reps: int = 10_000,
                                policies: Sequence[str] = POLICIES,
                                results_path: Optional[Path] = DEFAULT_RESULTS_PATH,
                                ttr_path: Optional[Path] = DEFAULT_TTR_PATH, seed: int = SEED,
                                verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Poisoning-rate sweep; returns ``(results, time_to_rollback)`` and writes the CSVs."""
    hyperparams, project_config = load_configs()
    gate_cfg = GateConfig.from_config(hyperparams)
    gen = PerturbationGenerator(PerturbationConfig.from_configs(project_config, hyperparams), seed=seed)
    unknown = [p for p in policies if p not in POLICIES]
    if unknown:
        raise ValueError(f"unknown policies {unknown}")
    rows: List[Dict[str, Any]] = []
    ttr_frames: List[pd.DataFrame] = []
    for i, rate in enumerate(rates):
        feed = gen.generate_feed(float(rate), n_candidates=n_candidates)
        summary = feed.summary()
        inj, ttr = injection_experiment(feed, hyperparams, gate_cfg, n_cycles, firings_per_cycle, seed)
        ttr_frames.append(ttr)
        ville = ville_verification(gate_cfg, ville_horizon, n_ville_reps, seed + 101 * (i + 1),
                                   inj["n_null_good_rules"], inj["n_null_good_false_rollbacks"])
        for policy in policies:
            adm = admission_experiment(feed, hyperparams, policy, n_cycles, firings_per_cycle, seed)
            row: Dict[str, Any] = {"poison_rate": float(rate), "policy": policy,
                                   "realized_poison_rate": summary["realized_poison_rate"],
                                   "n_cycles": n_cycles, "firings_per_cycle": firings_per_cycle,
                                   "p0": gate_cfg.p0, "alpha_admit": gate_cfg.alpha_admit,
                                   "q0": gate_cfg.q0, "q1": gate_cfg.q1}
            row.update(adm)
            if policy == "mg_ecfm_gate":
                row.update(inj)
                row.update(ville)
                row["false_admission_within_alpha"] = bool(row["false_admission_rate"] <= gate_cfg.alpha_admit)
            else:
                # Without a rollback monitor injected harmful rules are never removed.
                row.update({k: (float("nan") if isinstance(v, float) else v) for k, v in inj.items()})
                row.update({"rollback_rate_harmful": 0.0,
                            "share_rolled_back_within_max_cycles": 0.0,
                            "adversarial_share_within_max_cycles": 0.0,
                            "time_to_rollback_mean": float("inf"), "time_to_rollback_median": float("inf"),
                            "time_to_rollback_max": float("inf"),
                            "adversarial_time_to_rollback_mean": float("inf"),
                            "adversarial_time_to_rollback_max": float("inf"),
                            "harmful_active_after_injection_end": inj["n_injected_harmful"],
                            "n_null_good_false_rollbacks": 0})
                row.update({k: float("nan") for k in ville if k != "ville_bound_satisfied"})
                row["ville_bound_satisfied"] = float("nan")
                row["false_admission_within_alpha"] = bool(row["false_admission_rate"] <= gate_cfg.alpha_admit)
            rows.append(row)
        if verbose:
            g = rows[-len(policies)]
            print(f"[rho={rate:.2f}] gate FAR={g['false_admission_rate']:.3f} "
                  f"adv-FAR={g['adversarial_false_admission_rate']:.3f} "
                  f"TTR mean={g['time_to_rollback_mean']:.2f} max={g['time_to_rollback_max']:.0f} cycles  "
                  f"Ville exact={g['ville_exact_crossing_prob']:.4f} MC={g['ville_mc_rate']:.4f} "
                  f"in-gate={g['ville_ingate_rate']:.4f} ok={g['ville_bound_satisfied']}", flush=True)
    results = pd.DataFrame(rows)
    ttr_all = pd.concat(ttr_frames, ignore_index=True) if ttr_frames else pd.DataFrame()
    if results_path is not None:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_path, index=False)
    if ttr_path is not None:
        Path(ttr_path).parent.mkdir(parents=True, exist_ok=True)
        ttr_all.to_csv(ttr_path, index=False)
    if verbose and results_path is not None:
        print(f"[saved] {results_path}")
    return results, ttr_all


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MG-ECFM Phase-7 adversarial poisoning stress test")
    ap.add_argument("--rates", type=float, nargs="+", default=list(POISON_RATES))
    ap.add_argument("--n-candidates", type=int, default=None)
    ap.add_argument("--n-cycles", type=int, default=40)
    ap.add_argument("--firings-per-cycle", type=int, default=8)
    ap.add_argument("--ville-horizon", type=int, default=2000)
    ap.add_argument("--ville-reps", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = ap.parse_args(argv)
    run_adversarial_stress_test(args.rates, args.n_candidates, args.n_cycles, args.firings_per_cycle,
                                args.ville_horizon, args.ville_reps, results_path=args.output,
                                ttr_path=args.output.with_name("adversarial_time_to_rollback.csv"),
                                seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
