"""
tests/test_phase4_epistemic_martingale.py
=========================================

Phase 4 verification: Stage 3 of MG-ECFM - Subjective-Logic epistemic fusion
and the anytime-valid martingale e-process knowledge-evolution gate.

1. Subjective Logic: ``b + d + u = 1`` is preserved by every operator, trust
   discounting inflates ``u``, freshness decay composes with provenance, and
   cumulative consensus aggregates corroboration (= evidence addition);
   cluster-aware fusion neutralises Sybil amplification.
2. The admission e-process promotes high-precision rules once
   ``M_t >= 1/alpha_admit`` and never promotes useless ones.
3. The rollback e-process rolls back injected poisoned / degraded rules within
   ``<= 3`` update cycles (Monte-Carlo + exact dynamic-programme certificate).
4. Ville's inequality: empirical false-admission and false-rollback frequencies
   at the null boundary stay below ``alpha`` (and match the exact DP value).
5. The SQLite knowledge store creates snapshots, commits additions, performs
   atomic rollbacks, survives injected faults and concurrent writers, and feeds
   Phase 2's ``phi_KB`` encoder and Phase 3's router.

Run from the repository root:

    python tests/test_phase4_epistemic_martingale.py          # unittest runner
    python -m pytest tests/test_phase4_epistemic_martingale.py -q
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal.router import REASON_RULE_FLOOR, PreTransactionRouter  # noqa: E402
from src.data.perturbation_generator import (  # noqa: E402
    CandidateRule,
    Condition,
    PerturbationConfig,
    PerturbationGenerator,
    SourceEvidence,
)
from src.epistemic_gate.knowledge_store import (  # noqa: E402
    RULE_VERSION_COLUMNS,
    KnowledgeStore,
    KnowledgeStoreError,
    rule_to_logic,
)
from src.epistemic_gate.martingale_eprocess import (  # noqa: E402
    BettingEProcess,
    GateConfig,
    MartingaleEProcessGate,
    admission_process,
    betting_factor,
    crossing_probability,
    expected_factor,
    expected_log_growth,
    min_events_to_cross,
    rollback_process,
    simulate_crossing_rate,
)
from src.epistemic_gate.subjective_logic import (  # noqa: E402
    EpistemicFusionEngine,
    SubjectiveOpinion,
    averaging_fuse,
    cumulative_fuse,
    cumulative_fuse_pair,
    freshness,
)

SEED = 42
CONFIG_DIR = ROOT / "configs"
HYPERPARAMS = json.loads((CONFIG_DIR / "hyperparams.json").read_text(encoding="utf-8"))
PROJECT_CONFIG = json.loads((CONFIG_DIR / "project_config.json").read_text(encoding="utf-8"))
GATE_CFG = GateConfig.from_config(HYPERPARAMS)
FIRINGS_PER_CYCLE = 8


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def random_opinion(rng: np.random.Generator) -> SubjectiveOpinion:
    kind = rng.random()
    if kind < 0.05:
        return SubjectiveOpinion.vacuous(float(rng.random()))
    if kind < 0.10:
        return SubjectiveOpinion.dogmatic(float(rng.random()), float(rng.random()))
    return SubjectiveOpinion.from_evidence(float(rng.gamma(1.0, 50.0)), float(rng.gamma(1.0, 50.0)),
                                           float(rng.random()))


def make_rule(rule_id: str, precision: float, conditions=None, category: str = "genuine",
              evidence=None) -> CandidateRule:
    conditions = conditions or (Condition("log1p_amount", ">", 9.0), Condition("tx_type", "==", "TRANSFER"))
    evidence = evidence or (SourceEvidence("analyst_alert_001", "analyst_alert", 0.9, 90, 10, 630, 500),)
    return CandidateRule(rule_id, category, tuple(conditions), tuple(evidence), emitted_at_step=630,
                         true_precision=float(precision), historical_precision=float(precision),
                         is_harmful=precision <= GATE_CFG.p0)


def firing_event(rule) -> dict:
    """Event satisfying every condition of ``rule`` (plus neutral defaults)."""
    event = {"log1p_amount": 7.0, "tx_type": "PAYMENT", "src_vel_1": 0}
    for c in rule.conditions:
        if isinstance(c.value, str):
            event[c.feature] = c.value if c.op == "==" else c.value + "_other"
        elif c.op in (">", ">="):
            event[c.feature] = float(c.value) + 1.0
        elif c.op in ("<", "<="):
            event[c.feature] = float(c.value) - 1.0
        else:
            event[c.feature] = float(c.value)
    return event


def simulate_cycles(gate: MartingaleEProcessGate, precision_of, n_cycles: int,
                    rng: np.random.Generator, firings: int = FIRINGS_PER_CYCLE):
    reports = []
    for _ in range(n_cycles):
        outcomes = {rid: (rng.random(firings) < precision_of(rid)).astype(int).tolist()
                    for rid, rec in sorted(gate.records.items()) if rec.status in ("quarantined", "active")}
        reports.append(gate.run_update_cycle(outcomes))
    return reports


def build_feed(poison_rate: float = 0.20):
    gen = PerturbationGenerator(PerturbationConfig.from_configs(PROJECT_CONFIG, HYPERPARAMS), seed=SEED)
    return gen.generate_feed(poison_rate)


# --------------------------------------------------------------------------- #
# 1. Subjective Logic                                                         #
# --------------------------------------------------------------------------- #
class TestSubjectiveLogic(unittest.TestCase):

    def test_evidence_mapping_matches_synopsis(self) -> None:
        for r_plus, r_minus in ((0, 0), (3, 1), (40, 10), (250, 7)):
            op = SubjectiveOpinion.from_evidence(r_plus, r_minus, 0.3)
            n = r_plus + r_minus + 2.0
            self.assertAlmostEqual(op.belief, r_plus / n, places=15)
            self.assertAlmostEqual(op.disbelief, r_minus / n, places=15)
            self.assertAlmostEqual(op.uncertainty, 2.0 / n, places=15)
            self.assertEqual(op.belief + op.disbelief + op.uncertainty, 1.0)
            self.assertAlmostEqual(op.expected_probability, op.belief + 0.3 * op.uncertainty, places=15)
            self.assertEqual(op.epistemic_uncertainty, op.uncertainty)
            if r_plus + r_minus:
                rp, rm = op.evidence()
                self.assertAlmostEqual(rp, r_plus, places=9)
                self.assertAlmostEqual(rm, r_minus, places=9)
        # Beta posterior mean = (r+ + aW) / (r+ + r- + W)
        op = SubjectiveOpinion.from_evidence(7, 3, 0.25)
        self.assertAlmostEqual(op.expected_probability, (7 + 0.5) / 12.0, places=14)
        alpha, beta = op.beta_parameters()
        self.assertAlmostEqual(alpha / (alpha + beta), op.expected_probability, places=14)

    def test_additivity_strictly_preserved_under_all_operators(self) -> None:
        rng = np.random.default_rng(SEED)
        op = random_opinion(rng)
        for _ in range(3000):
            choice = int(rng.integers(0, 5))
            if choice == 0:
                op = op.trust_discount(float(rng.random()))
            elif choice == 1:
                op = op.time_decay(float(rng.uniform(0, 500)), float(rng.uniform(0, 0.05)))
            elif choice == 2:
                op = op.consensus_fuse(random_opinion(rng))
            elif choice == 3:
                op = op.average_fuse(random_opinion(rng))
            else:
                op = op.observe(int(rng.integers(0, 10)), int(rng.integers(0, 10)))
            self.assertAlmostEqual(op.belief + op.disbelief + op.uncertainty, 1.0, delta=4.5e-16)
            for v in op.as_tuple():
                self.assertTrue(0.0 <= v <= 1.0)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            SubjectiveOpinion(0.6, 0.6, 0.0)
        with self.assertRaises(ValueError):
            SubjectiveOpinion(-0.1, 0.6, 0.5)
        with self.assertRaises(ValueError):
            SubjectiveOpinion.from_evidence(-1, 2)
        with self.assertRaises(ValueError):
            SubjectiveOpinion.from_evidence(1, 2).trust_discount(1.5)
        with self.assertRaises(ValueError):
            freshness(10, -0.1)
        with self.assertRaises(ValueError):
            SubjectiveOpinion.dogmatic(1.0).evidence()
        with self.assertRaises(ValueError):
            cumulative_fuse_pair(SubjectiveOpinion.from_evidence(1, 1, prior_weight=2.0),
                                 SubjectiveOpinion.from_evidence(1, 1, prior_weight=10.0))

    def test_trust_discount_inflates_uncertainty(self) -> None:
        op = SubjectiveOpinion.from_evidence(180, 20, 0.5)
        d = op.trust_discount(0.6)
        self.assertAlmostEqual(d.belief, 0.6 * op.belief, places=15)
        self.assertAlmostEqual(d.disbelief, 0.6 * op.disbelief, places=15)
        self.assertAlmostEqual(d.uncertainty, 1.0 - 0.6 * (op.belief + op.disbelief), places=15)
        self.assertEqual(d.base_rate, op.base_rate)
        self.assertGreater(d.uncertainty, op.uncertainty)
        self.assertTrue(op.trust_discount(1.0).is_close(op, 1e-15))
        self.assertTrue(op.trust_discount(0.0).is_vacuous)
        us = [op.trust_discount(pi).uncertainty for pi in np.linspace(0.0, 1.0, 21)]
        self.assertTrue(all(a >= b for a, b in zip(us, us[1:])))  # monotone in pi
        # Belief/disbelief ratio (direction of evidence) is preserved by discounting.
        self.assertAlmostEqual(d.belief / d.disbelief, op.belief / op.disbelief, places=12)

    def test_time_decay_and_composition(self) -> None:
        op = SubjectiveOpinion.from_evidence(60, 5, 0.5)
        eta = HYPERPARAMS["subjective_logic"]["freshness_eta"]
        self.assertTrue(op.time_decay(0.0, eta).is_close(op, 1e-15))
        self.assertAlmostEqual(freshness(100, eta), math.exp(-eta * 100), places=15)
        self.assertTrue(op.time_decay(100, eta).is_close(op.trust_discount(math.exp(-eta * 100)), 1e-15))
        us = [op.time_decay(dt, eta).uncertainty for dt in (0, 10, 50, 168, 500, 2000)]
        self.assertTrue(all(a < b for a, b in zip(us, us[1:])))
        # pi (x) (psi (x) w) == (pi psi) (x) w
        a = op.time_decay(70, eta).trust_discount(0.7)
        b = op.trust_discount(0.7 * freshness(70, eta))
        self.assertTrue(a.is_close(b, 1e-14))
        self.assertGreater(op.time_decay(10_000, eta).uncertainty, 0.999)

    def test_cumulative_consensus_hand_computed(self) -> None:
        a = SubjectiveOpinion(0.6, 0.2, 0.2, 0.5)
        b = SubjectiveOpinion(0.3, 0.3, 0.4, 0.5)
        f = a.consensus_fuse(b)
        kappa = 0.2 + 0.4 - 0.08
        self.assertAlmostEqual(f.belief, (0.6 * 0.4 + 0.3 * 0.2) / kappa, places=14)
        self.assertAlmostEqual(f.disbelief, (0.2 * 0.4 + 0.3 * 0.2) / kappa, places=14)
        self.assertAlmostEqual(f.uncertainty, 0.08 / kappa, places=14)
        self.assertAlmostEqual(f.base_rate, 0.5, places=14)
        # Different base rates follow Josang's base-rate formula.
        c = SubjectiveOpinion(0.3, 0.3, 0.4, 0.2)
        g = a.consensus_fuse(c)
        expect_a = (0.5 * 0.4 + 0.2 * 0.2 - 0.7 * 0.08) / (0.6 - 0.16)
        self.assertAlmostEqual(g.base_rate, expect_a, places=14)
        self.assertTrue((a + b).is_close(f, 0.0))

    def test_consensus_algebra_and_corroboration(self) -> None:
        rng = np.random.default_rng(SEED + 1)
        vac = SubjectiveOpinion.vacuous()
        for _ in range(300):
            x = SubjectiveOpinion.from_evidence(*rng.gamma(1.0, 30.0, size=2))
            y = SubjectiveOpinion.from_evidence(*rng.gamma(1.0, 30.0, size=2))
            z = SubjectiveOpinion.from_evidence(*rng.gamma(1.0, 30.0, size=2))
            self.assertTrue((x + y).is_close(y + x, 1e-12))                       # commutative
            self.assertTrue(((x + y) + z).is_close(x + (y + z), 1e-10))           # associative
            self.assertTrue((x + vac).is_close(x, 1e-12))                         # neutral element
            fused = x + y
            self.assertLessEqual(fused.uncertainty, min(x.uncertainty, y.uncertainty) + 1e-15)
            rx, sx = x.evidence()
            ry, sy = y.evidence()
            self.assertTrue(fused.is_close(SubjectiveOpinion.from_evidence(rx + ry, sx + sy), 1e-10))
        # Corroboration: k agreeing independent sources raise belief monotonically.
        src = SubjectiveOpinion.from_evidence(40, 4).trust_discount(0.8)
        beliefs = [cumulative_fuse([src] * k).belief for k in range(1, 7)]
        self.assertTrue(all(a < b for a, b in zip(beliefs, beliefs[1:])))
        # n-ary fusion equals sequential pairwise fusion.
        ops = [SubjectiveOpinion.from_evidence(*rng.gamma(1.0, 20.0, size=2)) for _ in range(6)]
        seq = ops[0]
        for o in ops[1:]:
            seq = seq + o
        self.assertTrue(cumulative_fuse(ops).is_close(seq, 1e-10))

    def test_dogmatic_and_vacuous_limits(self) -> None:
        d1, d2 = SubjectiveOpinion.dogmatic(0.9), SubjectiveOpinion.dogmatic(0.5)
        f = d1 + d2
        self.assertAlmostEqual(f.belief, 0.7, places=14)
        self.assertEqual(f.uncertainty, 0.0)
        reg = SubjectiveOpinion.from_evidence(5, 5)
        self.assertTrue((d1 + reg).is_close(d1, 1e-12))   # dogmatic dominates
        both_vac = SubjectiveOpinion.vacuous(0.2) + SubjectiveOpinion.vacuous(0.6)
        self.assertTrue(both_vac.is_vacuous)
        self.assertAlmostEqual(both_vac.base_rate, 0.4, places=14)
        self.assertEqual(SubjectiveOpinion.dogmatic(0.3).variance(), 0.0)

    def test_averaging_fusion_is_idempotent(self) -> None:
        op = SubjectiveOpinion.from_evidence(95, 5).trust_discount(0.2)
        self.assertTrue(averaging_fuse([op] * 7).is_close(op, 1e-12))
        self.assertTrue(op.average_fuse(op).is_close(op, 1e-12))
        self.assertGreater(cumulative_fuse([op] * 7).belief, op.belief + 0.2)
        a, b = SubjectiveOpinion.from_evidence(10, 2), SubjectiveOpinion.from_evidence(4, 8)
        ra, sa = a.evidence()
        rb, sb = b.evidence()
        self.assertTrue(averaging_fuse([a, b]).is_close(
            SubjectiveOpinion.from_evidence((ra + rb) / 2, (sa + sb) / 2), 1e-12))

    def test_serialisation_round_trip(self) -> None:
        op = SubjectiveOpinion.from_evidence(12, 3, 0.4).trust_discount(0.7)
        self.assertTrue(SubjectiveOpinion.from_dict(op.to_dict()).is_close(op, 0.0))
        self.assertTrue(SubjectiveOpinion.from_tuple(op.as_tuple()).is_close(op, 0.0))
        self.assertIn("SubjectiveOpinion(", repr(op))

    def test_fusion_engine_separates_feed_categories(self) -> None:
        feed = build_feed(0.20)
        engine = EpistemicFusionEngine.from_config(HYPERPARAMS)
        stats = {}
        for r in feed.rules:
            op = engine.fuse_rule(r, feed.current_step).opinion
            stats.setdefault(r.category, []).append((op.belief, op.uncertainty))
        mean = {c: np.mean(v, axis=0) for c, v in stats.items()}
        for bad in ("poisoned", "sybil", "stale"):
            self.assertLess(mean[bad][0], mean["genuine"][0] - 0.25, bad)       # lower belief
            self.assertGreater(mean[bad][1], mean["genuine"][1] + 0.3, bad)     # higher uncertainty
        # Stale evidence is decayed towards vacuity.
        self.assertGreater(mean["stale"][1], 0.9)

    def test_cluster_aware_fusion_counters_sybil_amplification(self) -> None:
        feed = build_feed(0.20)
        naive = EpistemicFusionEngine.from_config(HYPERPARAMS, cluster_aware=False)
        aware = EpistemicFusionEngine.from_config(HYPERPARAMS, cluster_aware=True)
        sybils = feed.by_category("sybil")
        self.assertGreater(len(sybils), 0)
        naive_b = np.array([naive.fuse_rule(r, feed.current_step).opinion.belief for r in sybils])
        aware_b = np.array([aware.fuse_rule(r, feed.current_step).opinion.belief for r in sybils])
        self.assertTrue(np.all(aware_b < naive_b))
        naive_pass = np.mean([naive.fuse_rule(r, feed.current_step).decision == "quarantine" for r in sybils])
        aware_pass = np.mean([aware.fuse_rule(r, feed.current_step).decision == "quarantine" for r in sybils])
        self.assertGreater(naive_pass, 0.3)   # amplification is real without dependence modelling
        self.assertEqual(aware_pass, 0.0)
        res = aware.fuse_rule(sybils[0], feed.current_step)
        self.assertEqual(res.n_groups, 1)
        self.assertEqual(len(res.to_dict()["sources"]), len(sybils[0].evidence))


# --------------------------------------------------------------------------- #
# 2-4. Martingale e-processes                                                 #
# --------------------------------------------------------------------------- #
class TestEProcessMathematics(unittest.TestCase):

    def test_wealth_equals_synopsis_product(self) -> None:
        cfg = GATE_CFG
        rng = np.random.default_rng(SEED)
        y = (rng.random(60) < 0.7).astype(int)
        adm = admission_process(cfg.p0, cfg.p1, 1e-9)
        roll = rollback_process(cfg.q0, cfg.q1, 1e-9)
        for v in y:
            adm.update(int(v))
            roll.update(int(1 - v))
        m_admit = np.prod([v * cfg.p1 / cfg.p0 + (1 - v) * (1 - cfg.p1) / (1 - cfg.p0) for v in y])
        m_roll = np.prod([(1 - v) * cfg.q1 / cfg.q0 + v * (1 - cfg.q1) / (1 - cfg.q0) for v in y])
        self.assertAlmostEqual(adm.log_wealth, math.log(m_admit), places=10)
        self.assertAlmostEqual(roll.log_wealth, math.log(m_roll), places=10)
        self.assertEqual(betting_factor(1, cfg.p0, cfg.p1), cfg.p1 / cfg.p0)
        self.assertAlmostEqual(adm.anytime_p_value, min(1.0, math.exp(-adm.max_log_wealth)), places=15)

    def test_supermartingale_under_composite_null(self) -> None:
        for theta0, theta1 in ((GATE_CFG.p0, GATE_CFG.p1), (GATE_CFG.q0, GATE_CFG.q1)):
            for theta in np.linspace(0.0, theta0, 51):
                self.assertLessEqual(expected_factor(theta, theta0, theta1), 1.0 + 1e-12)
            self.assertAlmostEqual(expected_factor(theta0, theta0, theta1), 1.0, places=12)
            self.assertGreater(expected_factor(theta1, theta0, theta1), 1.0)
            self.assertGreater(expected_log_growth(theta1, theta0, theta1), 0.0)

    def test_minimum_evidence_to_cross(self) -> None:
        k_admit = min_events_to_cross(GATE_CFG.p0, GATE_CFG.p1, GATE_CFG.alpha_admit)
        self.assertEqual(k_admit, math.ceil(math.log(20.0) / math.log(1.6)))
        proc = admission_process(GATE_CFG.p0, GATE_CFG.p1, GATE_CFG.alpha_admit)
        self.assertIsNone(proc.update_many([1] * (k_admit - 1)))
        self.assertTrue(proc.update(1))
        self.assertGreaterEqual(proc.wealth, 1.0 / GATE_CFG.alpha_admit)
        with self.assertRaises(RuntimeError):
            proc.update(1)
        with self.assertRaises(ValueError):
            BettingEProcess(0.5, 0.4, 0.05)
        with self.assertRaises(ValueError):
            BettingEProcess(0.5, 0.8, 0.05).update(2)

    def test_exact_crossing_probability_matches_monte_carlo(self) -> None:
        for theta in (0.2, 0.35, 0.6):
            exact = crossing_probability(200, theta, GATE_CFG.q0, GATE_CFG.q1, GATE_CFG.alpha_roll)
            mc = simulate_crossing_rate(theta, GATE_CFG.q0, GATE_CFG.q1, GATE_CFG.alpha_roll,
                                        200, 20_000, seed=SEED)["rate"]
            se = math.sqrt(max(exact * (1 - exact), 1e-6) / 20_000)
            self.assertLess(abs(exact - mc), 5 * se + 1e-3, (theta, exact, mc))


class TestAdmissionGate(unittest.TestCase):

    def test_promotes_true_high_precision_rules(self) -> None:
        rng = np.random.default_rng(SEED)
        cfg = GATE_CFG
        times = []
        for precision in (0.85, 0.9, 0.95):
            for _ in range(300):
                proc = admission_process(cfg.p0, cfg.p1, cfg.alpha_admit)
                idx = proc.update_many((rng.random(500) < precision).astype(int))
                self.assertIsNotNone(idx, precision)
                self.assertGreaterEqual(proc.log_wealth, math.log(1.0 / cfg.alpha_admit) - 1e-12)
                times.append(proc.crossed_at)
        self.assertGreaterEqual(min(times), cfg.admission_min_hits())
        self.assertLess(np.median(times), 40)

    def test_rarely_promotes_useless_rules(self) -> None:
        rng = np.random.default_rng(SEED + 7)
        cfg = GATE_CFG
        reps = 1000
        for precision in (0.05, 0.2, 0.35):
            promoted = 0
            for _ in range(reps):
                proc = admission_process(cfg.p0, cfg.p1, cfg.alpha_admit)
                promoted += proc.update_many((rng.random(1000) < precision).astype(int)) is not None
            exact = crossing_probability(1000, precision, cfg.p0, cfg.p1, cfg.alpha_admit)
            self.assertLess(exact, cfg.alpha_admit / 5, precision)     # far inside the null
            se = math.sqrt(max(exact * (1 - exact), 1e-6) / reps)
            self.assertLessEqual(promoted / reps, exact + 4 * se, precision)
            self.assertLessEqual(promoted / reps, cfg.alpha_admit, precision)

    def test_plugin_betting_is_valid_and_powerful(self) -> None:
        rng = np.random.default_rng(SEED + 3)
        cfg = GATE_CFG
        null_hits = power_hits = 0
        reps = 2000
        for _ in range(reps):
            proc = admission_process(cfg.p0, cfg.p1, cfg.alpha_admit, betting="plugin", prior_mean=0.9)
            null_hits += proc.update_many((rng.random(1000) < cfg.p0).astype(int)) is not None
            proc = admission_process(cfg.p0, cfg.p1, cfg.alpha_admit, betting="plugin", prior_mean=0.9)
            power_hits += proc.update_many((rng.random(300) < 0.7).astype(int)) is not None
        self.assertLessEqual(null_hits / reps, cfg.alpha_admit)
        self.assertGreater(power_hits / reps, 0.9)  # fixed-bet LR is weak at p = 0.7 < p1

    def test_gate_promotes_into_active_kb_with_certificate(self) -> None:
        gate = MartingaleEProcessGate.from_config(HYPERPARAMS)
        good, bad = make_rule("G1", 0.92), make_rule("B1", 0.10)
        gate.submit_candidate(good, 632)
        gate.submit_candidate(bad, 632)
        self.assertEqual(gate.quarantine_buffer, ["B1", "G1"])
        self.assertEqual(gate.store.get_active_rules(), [])
        reports = simulate_cycles(gate, {"G1": 0.92, "B1": 0.10}.get, 10, np.random.default_rng(SEED))
        self.assertIn("G1", [r for rep in reports for r in rep.promoted])
        self.assertEqual([r.rule_id for r in gate.store.get_active_rules()], ["G1"])
        self.assertEqual(gate.status("B1"), "quarantined")
        cert = gate.store.certificates("G1")[0]
        self.assertEqual(cert["test"], "admission")
        self.assertGreaterEqual(cert["payload"]["log_wealth"], math.log(1.0 / GATE_CFG.alpha_admit) - 1e-12)
        self.assertLessEqual(cert["payload"]["anytime_p_value"], GATE_CFG.alpha_admit)
        rep = gate.ville_report("G1")
        self.assertEqual(rep["ville_bound"], GATE_CFG.alpha_roll)
        self.assertGreater(gate.store.active_weights()["G1"], 0.85)  # opinion updated by local evidence

    def test_firing_budget_rejects_undecided_candidates(self) -> None:
        gate = MartingaleEProcessGate.from_config(HYPERPARAMS, max_firings_per_candidate=40)
        gate.submit_candidate(make_rule("U1", 0.5), 632)
        simulate_cycles(gate, lambda _: 0.5, 10, np.random.default_rng(SEED))
        self.assertEqual(gate.status("U1"), "rejected")
        self.assertEqual(gate.store.status_of("U1"), "rejected")

    def test_end_to_end_feed_false_admission_and_accuracy(self) -> None:
        for rate in (0.05, 0.20, 0.25):
            feed = build_feed(rate)
            gate = MartingaleEProcessGate.from_config(HYPERPARAMS)
            for r in feed.rules:
                gate.submit_candidate(r, feed.current_step)
            truth = {r.rule_id: r.true_precision for r in feed.rules}
            reports = simulate_cycles(gate, truth.get, 40, np.random.default_rng(SEED))
            ever = {rid for rep in reports for rid in rep.promoted}
            harmful = [r for r in feed.rules if r.is_harmful]
            adversarial = [r for r in feed.rules if r.is_adversarial]
            far = sum(r.rule_id in ever for r in harmful) / len(harmful)
            adv_far = sum(r.rule_id in ever for r in adversarial) / len(adversarial)
            accuracy = np.mean([(gate.status(r.rule_id) == "active") != r.is_harmful for r in feed.rules])
            self.assertLessEqual(far, GATE_CFG.alpha_admit, rate)
            self.assertEqual(adv_far, 0.0, rate)
            self.assertGreaterEqual(accuracy, 0.90, rate)
            self.assertTrue(gate.store.verify_integrity()["ok"])
            print(f"\n    poison={rate:.2f}: false-admission={far:.3f} admission-accuracy={accuracy:.3f} "
                  f"active={len(gate.active_rule_ids)} snapshots={gate.store.head_snapshot_id}",
                  file=sys.stderr)


class TestRollbackGate(unittest.TestCase):

    def test_certified_firing_volume_for_three_cycles(self) -> None:
        cfg = GATE_CFG
        self.assertEqual(cfg.max_rollback_cycles, 3)
        k = cfg.rollback_min_false_alarms()
        self.assertEqual(k, math.ceil(math.log(1 / cfg.alpha_roll) / math.log(cfg.q1 / cfg.q0)))
        n = cfg.max_rollback_cycles * FIRINGS_PER_CYCLE
        for fpr in (0.75, 0.9, 0.98):  # poisoned rules: precision <= 0.25
            self.assertGreater(cfg.rollback_crossing_probability(n, fpr), 0.999, fpr)
            self.assertLessEqual(cfg.firings_per_cycle_for_rollback(fpr, 0.99), FIRINGS_PER_CYCLE)

    def test_injected_poisoned_rules_rolled_back_within_three_cycles(self) -> None:
        rng = np.random.default_rng(SEED)
        gate = MartingaleEProcessGate.from_config(HYPERPARAMS)
        precisions = {}
        for i in range(150):
            rid = f"P{i:03d}"
            precisions[rid] = float(rng.uniform(0.02, 0.25))
            gate.inject_active(make_rule(rid, precisions[rid], category="poisoned"),
                               SubjectiveOpinion.from_evidence(380, 20).trust_discount(0.2))
        for i in range(30):
            rid = f"G{i:03d}"
            precisions[rid] = 0.97
            gate.inject_active(make_rule(rid, 0.97))
        self.assertEqual(len(gate.store.get_active_rules()), 180)
        reports = simulate_cycles(gate, precisions.get, GATE_CFG.max_rollback_cycles, rng)
        rolled = {rid: rep.cycle for rep in reports for rid in rep.rolled_back}
        poisoned = [r for r in precisions if r.startswith("P")]
        self.assertEqual(set(poisoned) - set(rolled), set())
        self.assertLessEqual(max(rolled[r] for r in poisoned), GATE_CFG.max_rollback_cycles)
        active = {r.rule_id for r in gate.store.get_active_rules()}
        self.assertTrue(active.isdisjoint(poisoned))
        self.assertEqual(sum(r.startswith("G") for r in rolled), 0)
        for rid in poisoned[:5]:
            self.assertEqual(gate.status(rid), "quarantined")
            self.assertEqual(gate.store.certificates(rid)[-1]["test"], "rollback")
        print(f"\n    injected poisoned rules: {len(poisoned)}/{len(poisoned)} rolled back, "
              f"mean time-to-rollback={np.mean([rolled[r] for r in poisoned]):.2f} cycles "
              f"(max {max(rolled[r] for r in poisoned)})", file=sys.stderr)

    def test_degraded_admitted_rule_rolled_back(self) -> None:
        """Genuine rule is admitted by the e-process, then degrades (concept drift / poisoning)."""
        rng = np.random.default_rng(SEED + 11)
        delays = {}
        for monitor, good_cycles in (("martingale", 2), ("e_detector", 2), ("e_detector", 25)):
            gate = MartingaleEProcessGate.from_config(HYPERPARAMS, rollback_monitor=monitor)
            gate.submit_candidate(make_rule("D1", 0.95), 632)
            precision = {"D1": 0.95}
            while gate.status("D1") != "active":
                simulate_cycles(gate, precision.get, 1, rng)
            simulate_cycles(gate, precision.get, good_cycles, rng)
            self.assertEqual(gate.status("D1"), "active")
            precision["D1"] = 0.10
            for c in range(1, 50):
                if simulate_cycles(gate, precision.get, 1, rng)[0].rolled_back:
                    delays[(monitor, good_cycles)] = c
                    break
            self.assertEqual(gate.status("D1"), "quarantined")
        self.assertLessEqual(delays[("martingale", 2)], GATE_CFG.max_rollback_cycles)
        self.assertLessEqual(delays[("e_detector", 2)], GATE_CFG.max_rollback_cycles)
        # The CUSUM e-detector's delay does not grow with the rule's good history.
        self.assertLessEqual(delays[("e_detector", 25)], GATE_CFG.max_rollback_cycles)

    def test_martingale_delay_grows_with_good_history(self) -> None:
        """Documented property: the product martingale must first repay accumulated losses."""
        cfg = GATE_CFG
        rng = np.random.default_rng(SEED)
        good = (rng.random(400) < 0.95).astype(int)
        mart = rollback_process(cfg.q0, cfg.q1, cfg.alpha_roll, monitor="martingale")
        det = rollback_process(cfg.q0, cfg.q1, cfg.alpha_roll, monitor="e_detector")
        mart.update_many(1 - good)
        det.update_many(1 - good)
        self.assertLess(mart.log_wealth, -50.0)
        self.assertLessEqual(det.log_wealth, 0.0)
        self.assertGreaterEqual(det.log_wealth, math.log((1 - cfg.q1) / (1 - cfg.q0)) - 1e-12)
        bad = 1 - (rng.random(400) < 0.10).astype(int)
        t_mart = mart.update_many(bad)
        det_idx = det.update_many(bad)
        self.assertIsNotNone(det_idx)
        self.assertLess(det_idx, 3 * FIRINGS_PER_CYCLE)
        self.assertTrue(t_mart is None or t_mart > det_idx)

    def test_rollback_restores_snapshot_v_j(self) -> None:
        for policy in ("targeted", "snapshot"):
            gate = MartingaleEProcessGate.from_config(HYPERPARAMS, rollback_policy=policy)
            gate.inject_active(make_rule("A1", 0.97))
            v_j = gate.store.head_snapshot_id
            before = gate.store.snapshot_members(v_j)
            gate.inject_active(make_rule("X1", 0.05, category="poisoned"))
            gate.inject_active(make_rule("A2", 0.97))
            pre_x1 = gate.records["X1"].pre_admission_snapshot
            self.assertEqual(gate.store.pre_admission_snapshot("X1"), pre_x1)
            rng = np.random.default_rng(SEED)
            truth = {"A1": 0.97, "X1": 0.05, "A2": 0.97}
            for _ in range(GATE_CFG.max_rollback_cycles):   # stop at the rollback cycle
                if simulate_cycles(gate, truth.get, 1, rng)[0].rolled_back:
                    break
            self.assertEqual(gate.status("X1"), "quarantined")
            active = sorted(r.rule_id for r in gate.store.get_active_rules())
            snaps = gate.store.list_snapshots()
            rb = [s for s in snaps if s["operation"] in ("rollback", "rollback_rule")]
            self.assertEqual(len(rb), 1)
            self.assertEqual(rb[0]["restored_from"], pre_x1)
            if policy == "targeted":
                self.assertEqual(active, ["A1", "A2"])
            else:
                # Exact restore of V_j: A2 (admitted after V_j) is demoted to quarantine as well.
                self.assertEqual(active, ["A1"])
                self.assertEqual(gate.status("A2"), "quarantined")
                head = gate.store.snapshot_members()
                self.assertEqual({k: v for k, v in head.items() if k == "A1"}, before)
                self.assertEqual(head["A2"][1], "quarantined")            # not orphaned
                # A2 is genuinely good, so its fresh admission e-process re-promotes it.
                simulate_cycles(gate, truth.get, 4, rng)
                self.assertEqual(gate.status("A2"), "active")
                self.assertEqual(gate.records["A2"].n_admissions, 2)
            self.assertEqual(gate.store.status_of("X1"), "quarantined")
            self.assertTrue(gate.store.verify_integrity()["ok"])


class TestVilleInequality(unittest.TestCase):

    def test_false_admission_frequency_bounded(self) -> None:
        cfg = GATE_CFG
        exact = crossing_probability(2000, cfg.p0, cfg.p0, cfg.p1, cfg.alpha_admit)
        self.assertLessEqual(exact, cfg.alpha_admit)
        rates = []
        for theta, seed in ((cfg.p0, SEED), (cfg.p0 - 0.1, SEED + 1)):
            sim = simulate_crossing_rate(theta, cfg.p0, cfg.p1, cfg.alpha_admit, 2000, 10_000, seed=seed)
            self.assertLessEqual(sim["rate"], cfg.alpha_admit, sim)
            rates.append(sim["rate"])
        self.assertLess(abs(exact - rates[0]), 0.01)
        print(f"\n    Ville (admission, p=p0): empirical={rates[0]:.4f} exact(2000)={exact:.4f} "
              f"bound={cfg.alpha_admit}", file=sys.stderr)

    def test_false_rollback_frequency_bounded(self) -> None:
        cfg = GATE_CFG
        rates = []
        for theta, seed in ((cfg.q0, SEED), (cfg.q0 * 0.5, SEED + 1), (0.03, SEED + 2)):
            sim = simulate_crossing_rate(theta, cfg.q0, cfg.q1, cfg.alpha_roll, 2000, 10_000, seed=seed)
            self.assertLessEqual(sim["rate"], cfg.alpha_roll, sim)
            rates.append(sim["rate"])
        exact = crossing_probability(5000, cfg.q0, cfg.q0, cfg.q1, cfg.alpha_roll)
        self.assertLessEqual(exact, cfg.alpha_roll)
        self.assertLess(abs(exact - rates[0]), 0.01)
        print(f"\n    Ville (rollback, FPR=q0): empirical={rates[0]:.4f} exact(5000)={exact:.4f} "
              f"bound={cfg.alpha_roll}", file=sys.stderr)

    def test_bound_holds_across_sequential_gate_tests(self) -> None:
        """Many good rules (FPR = q0) monitored by the full gate for 60 cycles."""
        cfg = GATE_CFG
        gate = MartingaleEProcessGate.from_config(HYPERPARAMS)
        n = 400
        for i in range(n):
            gate.inject_active(make_rule(f"N{i:03d}", 1.0 - cfg.q0))
        simulate_cycles(gate, lambda _: 1.0 - cfg.q0, 60, np.random.default_rng(SEED))
        false_rollbacks = sum(rec.n_rollbacks > 0 for rec in gate.records.values())
        self.assertLessEqual(false_rollbacks / n, cfg.alpha_roll)
        for rec in gate.records.values():
            if rec.status == "active":
                self.assertGreaterEqual(rec.rollback.anytime_p_value, cfg.alpha_roll)

    def test_e_detector_average_run_length(self) -> None:
        cfg = GATE_CFG
        sim = simulate_crossing_rate(cfg.q0, cfg.q0, cfg.q1, cfg.alpha_roll, 3000, 4000,
                                     seed=SEED, monitor="e_detector")
        self.assertGreaterEqual(sim["censored_mean_run_length"], 1.0 / cfg.alpha_roll)


# --------------------------------------------------------------------------- #
# 5. Knowledge store                                                          #
# --------------------------------------------------------------------------- #
class TestKnowledgeStore(unittest.TestCase):

    def setUp(self) -> None:
        self.store = KnowledgeStore()
        self.rules = {rid: make_rule(rid, 0.9, (Condition("log1p_amount", ">", float(8 + i)),))
                      for i, rid in enumerate(("K1", "K2", "K3"))}

    def tearDown(self) -> None:
        self.store.close()

    def _quarantine(self, rid: str) -> int:
        return self.store.quarantine_candidate(rid, rule_to_logic(self.rules[rid]),
                                               SubjectiveOpinion.from_evidence(30, 3))

    def test_schema(self) -> None:
        self.assertEqual(self.store.table_columns("rule_versions"), RULE_VERSION_COLUMNS)
        self.assertEqual(RULE_VERSION_COLUMNS, ("rule_id", "version_id", "snapshot_id", "rule_logic",
                                                "opinion_tuple", "status", "timestamp"))
        self.assertIn(self.store.storage_format, ("jsonb", "json"))
        self.assertEqual(self.store.head_snapshot_id, 0)
        self.assertEqual(self.store.list_snapshots()[0]["operation"], "genesis")

    def test_snapshots_and_commits(self) -> None:
        s1 = self._quarantine("K1")
        s2 = self.store.admit_candidate("K1")
        s3 = self._quarantine("K2")
        s4 = self.store.admit_candidate("K2", SubjectiveOpinion.from_evidence(90, 5))
        self.assertEqual([s1, s2, s3, s4], [1, 2, 3, 4])
        self.assertEqual(self.store.head_snapshot_id, 4)
        self.assertEqual([r.rule_id for r in self.store.get_active_rules()], ["K1", "K2"])
        self.assertEqual([r.rule_id for r in self.store.get_active_rules(2)], ["K1"])
        self.assertEqual(self.store.get_active_rules(1), [])
        hist = self.store.rule_history("K1")
        self.assertEqual([h["status"] for h in hist], ["quarantined", "active"])
        self.assertEqual(hist[1]["opinion_tuple"]["b"], hist[0]["opinion_tuple"]["b"])  # opinion kept
        self.assertEqual(self.store.pre_admission_snapshot("K2"), 3)
        with self.assertRaises(KnowledgeStoreError):
            self.store.admit_candidate("K1")                 # already active
        with self.assertRaises(KnowledgeStoreError):
            self.store.admit_candidate("UNKNOWN")
        with self.assertRaises(KnowledgeStoreError):
            self.store.quarantine_candidate("NEW")           # new candidate without logic
        self.assertTrue(self.store.verify_integrity()["ok"])

    def test_atomic_rollback_to_snapshot(self) -> None:
        self._quarantine("K1")
        self.store.admit_candidate("K1")
        v_j = self.store.head_snapshot_id
        members_vj = self.store.snapshot_members(v_j)
        self._quarantine("K2")
        self.store.admit_candidate("K2")
        self._quarantine("K3")
        self.store.admit_candidate("K3")
        self.assertEqual(len(self.store.get_active_rules()), 3)
        n_before = len(self.store.list_snapshots())
        sid = self.store.rollback_to_snapshot(v_j, quarantine_rule_ids=("K1",), reason="test")
        self.assertEqual(sid, n_before)                       # new snapshot appended
        self.assertEqual(self.store.head_snapshot_id, sid)
        head = self.store.snapshot_members()
        self.assertEqual(set(head), set(members_vj))          # exact V_j membership
        self.assertEqual(head["K1"][1], "quarantined")
        self.assertEqual(self.store.get_active_rules(), [])
        # History is preserved and the previous head is still queryable.
        self.assertEqual(len(self.store.get_active_rules(sid - 1)), 3)
        self.assertEqual(self.store.list_snapshots()[-1]["restored_from"], v_j)
        # Plain restore without quarantine returns to V_j exactly, and K2 can be re-admitted.
        self.store.rollback_to_snapshot(v_j)
        self.assertEqual(self.store.snapshot_members(), members_vj)
        self._quarantine("K2")
        self.store.admit_candidate("K2")
        self.assertEqual([r.rule_id for r in self.store.get_active_rules()], ["K1", "K2"])
        self.assertEqual(self.store.rule_history("K2")[-1]["version_id"], 4)
        with self.assertRaises(KnowledgeStoreError):
            self.store.rollback_to_snapshot(999)
        self.assertTrue(self.store.verify_integrity()["ok"])

    def test_fault_injection_leaves_no_partial_state(self) -> None:
        for rid in ("K1", "K2"):
            self._quarantine(rid)
            self.store.admit_candidate(rid)
        snap_count = len(self.store.list_snapshots())
        head, members = self.store.head_snapshot_id, self.store.snapshot_members()
        hist = {rid: len(self.store.rule_history(rid)) for rid in ("K1", "K2")}

        for stage in ("after_version", "before_commit"):
            def boom(where: str, stage=stage) -> None:
                if where == stage:
                    raise RuntimeError(f"injected failure at {where}")
            self.store._fault_injector = boom
            with self.assertRaises(RuntimeError):
                self.store.rollback_to_snapshot(2, quarantine_rule_ids=("K1", "K2"))
            with self.assertRaises(RuntimeError):
                self.store.revert_rule("K2")
            self.store._fault_injector = None
            self.assertEqual(len(self.store.list_snapshots()), snap_count)
            self.assertEqual(self.store.head_snapshot_id, head)
            self.assertEqual(self.store.snapshot_members(), members)
            self.assertEqual({rid: len(self.store.rule_history(rid)) for rid in hist}, hist)
        self.assertTrue(self.store.verify_integrity()["ok"])
        self.store.revert_rule("K2")  # store remains fully usable afterwards
        self.assertEqual([r.rule_id for r in self.store.get_active_rules()], ["K1"])

    def test_json_constraints_enforced(self) -> None:
        with self.assertRaises(sqlite3.Error):
            with self.store._transaction() as cur:
                cur.execute("INSERT INTO rule_versions VALUES ('BAD', 1, 0, '{not json', NULL, 'active', 't')")
        with self.assertRaises(sqlite3.Error):
            with self.store._transaction() as cur:
                cur.execute("INSERT INTO rule_versions VALUES ('BAD', 1, 0, '{}', NULL, 'bogus', 't')")
        self.assertEqual(self.store.rule_history("BAD"), [])

    def test_thread_safe_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(Path(tmp) / "kb.sqlite")
            errors = []

            def worker(t: int) -> None:
                try:
                    for i in range(15):
                        rid = f"T{t}_{i:02d}"
                        store.quarantine_candidate(rid, rule_to_logic(make_rule(rid, 0.9)))
                        store.admit_candidate(rid)
                        store.record_opinions({rid: SubjectiveOpinion.from_evidence(i, 1)})
                except Exception as exc:  # pragma: no cover - surfaced by the assertion below
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.assertEqual(errors, [])
            self.assertEqual(len(store.get_active_rules()), 120)
            self.assertEqual(store.head_snapshot_id, 240)
            integrity = store.verify_integrity()
            self.assertTrue(integrity["ok"], integrity)
            store.close()
            # Durable: reopening the file yields the same committed state.
            with KnowledgeStore(Path(tmp) / "kb.sqlite") as reopened:
                self.assertEqual(reopened.head_snapshot_id, 240)
                self.assertEqual(len(reopened.get_active_rules()), 120)
                self.assertAlmostEqual(reopened.active_weights()["T3_14"],
                                       SubjectiveOpinion.from_evidence(14, 1).expected_probability, places=12)

    def test_active_rules_feed_phase3_router(self) -> None:
        rule = self.rules["K1"]
        self._quarantine("K1")
        self.store.admit_candidate("K1")
        router = PreTransactionRouter.from_config(HYPERPARAMS)
        hit, miss = firing_event(rule), {"log1p_amount": 1.0, "tx_type": "PAYMENT"}
        stored = self.store.get_active_rules()
        self.assertTrue(np.array_equal(stored[0].evaluate(pd.DataFrame([hit, miss])),
                                       rule.evaluate(pd.DataFrame([hit, miss]))))
        ids = self.store.firing_rule_ids(hit)
        self.assertEqual(ids, ("K1",))
        self.assertEqual(self.store.firing_rule_ids(miss), ())
        decision = router.route(0.001, 0.0, 50.0, active_rules=ids)
        self.assertEqual(decision.reason, REASON_RULE_FLOOR)
        self.assertGreaterEqual(int(decision.action), 1)
        self.store.revert_rule("K1")
        ids_after = self.store.firing_rule_ids(hit)
        self.assertEqual(ids_after, ())
        self.assertNotEqual(router.route(0.001, 0.0, 50.0, active_rules=ids_after).reason, REASON_RULE_FLOOR)

    def test_active_rules_feed_phase2_fusion_encoder(self) -> None:
        try:
            from src.detection.fusion_classifier import KnowledgeBaseTriggerEncoder
        except ImportError as exc:  # torch / xgboost not installed
            self.skipTest(f"Phase 2 stack unavailable: {exc}")
        for rid in ("K1", "K2"):
            self._quarantine(rid)
            self.store.admit_candidate(rid)
        self.store.record_opinions({"K1": SubjectiveOpinion.from_evidence(80, 2)})
        encoder = KnowledgeBaseTriggerEncoder()
        self.assertEqual(self.store.sync_encoder(encoder), 2)
        self.assertAlmostEqual(encoder.weights["K1"],
                               SubjectiveOpinion.from_evidence(80, 2).expected_probability, places=12)
        event = firing_event(self.rules["K2"])  # log1p_amount > 9 fires K1 and K2
        vec = encoder.transform_event(event)
        frame_vec = encoder.transform_frame(pd.DataFrame([event]))[0]
        np.testing.assert_allclose(vec, frame_vec)
        self.assertEqual(vec[0], 2.0)
        self.store.revert_rule("K1")
        self.store.sync_encoder(encoder)
        self.assertEqual(encoder.transform_event(event)[0], 1.0)


class TestConfigAndDeterminism(unittest.TestCase):

    def test_from_config(self) -> None:
        mp = HYPERPARAMS["martingale_eprocess"]
        self.assertEqual(GATE_CFG.p0, mp["admission"]["p0"])
        self.assertEqual(GATE_CFG.q1, mp["rollback"]["q1"])
        self.assertEqual(GATE_CFG.alpha_roll, mp["rollback"]["alpha_roll"])
        self.assertEqual(GATE_CFG.betting, mp["betting"])
        # Rollback null is the dual of the admission alternative: admitted rules are H0^good.
        self.assertLessEqual(GATE_CFG.q0, 1.0 - GATE_CFG.p1 + 1e-12)
        self.assertGreaterEqual(GATE_CFG.q1, 1.0 - GATE_CFG.p0 - 1e-12)
        eng = EpistemicFusionEngine.from_config(HYPERPARAMS)
        self.assertEqual(eng.eta, HYPERPARAMS["subjective_logic"]["freshness_eta"])
        with self.assertRaises(ValueError):
            GateConfig(rollback_policy="nope")

    def test_deterministic_replay(self) -> None:
        def run():
            feed = build_feed(0.15)
            gate = MartingaleEProcessGate.from_config(HYPERPARAMS)
            for r in feed.rules:
                gate.submit_candidate(r, feed.current_step)
            truth = {r.rule_id: r.true_precision for r in feed.rules}
            reps = simulate_cycles(gate, truth.get, 15, np.random.default_rng(SEED))
            return [rep.to_dict() for rep in reps], sorted(r.rule_id for r in gate.store.get_active_rules())
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
