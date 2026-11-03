"""
tests/test_phase3_conformal.py
==============================

Phase 3 verification: Stage 1 of MG-ECFM, Cost-Sensitive Online Adaptive
Conformal Risk Control (CS-ACRC) and pre-transaction prevention routing.

1. the asymmetric loss is exact on all 8 (action x outcome) cells, scalar and
   vectorised;
2. Hold-and-Verify on a fraud saves ``A_t`` and incurs only ``c_hold``;
3. the online ACI update follows ``theta_{t+1} = theta_t - gamma (alpha - L)``
   and the thresholds tighten / relax as the fraud rate drifts;
4. the empirical loss stays within ``alpha +/- eps`` (exact telescoping
   certificate, one-sided risk bound, rolling-window bound);
5. ``lambda^low < lambda^mid < lambda^high`` is strictly preserved by an exact
   Euclidean projection;
6. routing takes < 1 ms per transaction.

Run from the repository root:

    python tests/test_phase3_conformal.py          # unittest runner
    python -m pytest tests/test_phase3_conformal.py -q
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal.adaptive_crc import (  # noqa: E402
    OnlineAdaptiveCRC,
    project_ordered_thresholds,
)
from src.conformal.loss_functions import (  # noqa: E402
    Action,
    AsymmetricLoss,
    CostParameters,
    asymmetric_loss,
    batch_asymmetric_loss,
)
from src.conformal.router import (  # noqa: E402
    REASON_AMOUNT,
    REASON_RULE_FLOOR,
    REASON_THRESHOLD,
    REASON_UNCERTAIN_BLOCK,
    PreTransactionRouter,
    RoutingDecision,
)

SEED = 42
COSTS = CostParameters(c_step=5.0, c_hold=25.0, c_block_churn=150.0, beta_bypass=0.10,
                       loss_normalizer=10000.0, loss_cap=1.0)
with open(ROOT / "configs" / "project_config.json", "r", encoding="utf-8") as _fh:
    PROJECT_CONFIG = json.load(_fh)
with open(ROOT / "configs" / "hyperparams.json", "r", encoding="utf-8") as _fh:
    HYPERPARAMS = json.load(_fh)


# --------------------------------------------------------------------------- #
# Synthetic drifting stream                                                   #
# --------------------------------------------------------------------------- #
def drift_stream(n: int, rates, seed: int = SEED):
    """Piecewise-constant fraud rate ``pi_t``; calibrated-ish scores; heavy fraud amounts."""
    rng = np.random.default_rng(seed)
    seg = n // len(rates)
    pi = np.concatenate([np.full(seg, r) for r in rates])
    y = (rng.random(pi.size) < pi).astype(np.int64)
    p = np.where(y == 1, rng.beta(3, 3, pi.size), rng.beta(1, 12, pi.size))
    amount = np.where(y == 1, rng.lognormal(9.5, 1.0, pi.size), rng.lognormal(7.0, 1.2, pi.size))
    return y, p, amount, seg


def run_stream(router: PreTransactionRouter, y, p, amount, return_projected: bool = False):
    """Route each transaction, then feed back its label (online, delay-free)."""
    lam = np.empty((y.size, 3))
    ell = np.empty(y.size)
    projected = np.zeros(y.size, dtype=bool)
    for i in range(y.size):
        d = router.route(p[i], 0.0, amount[i])
        step = router.feedback(d, int(y[i]))
        ell[i], projected[i] = step.loss_normalized, step.projected
        lam[i] = router.controller.thresholds
    return (lam, ell, projected) if return_projected else (lam, ell)


def make_router(**ctrl_kwargs) -> PreTransactionRouter:
    kw = dict(alpha=0.02, gamma=0.005, lambda_init=(0.20, 0.50, 0.85), min_gap=0.01,
              loss=AsymmetricLoss(COSTS))
    kw.update(ctrl_kwargs)
    return PreTransactionRouter(OnlineAdaptiveCRC(**kw))


# --------------------------------------------------------------------------- #
# 1. Loss function                                                            #
# --------------------------------------------------------------------------- #
class TestLossFunction(unittest.TestCase):
    A = 1234.5

    def expected_cells(self, a):
        c = COSTS
        return {
            (Action.APPROVE, 0): 0.0,
            (Action.APPROVE, 1): a,
            (Action.STEP_UP, 0): c.c_step,
            (Action.STEP_UP, 1): c.c_step + c.beta_bypass * a,
            (Action.HOLD, 0): c.c_hold,
            (Action.HOLD, 1): c.c_hold,
            (Action.BLOCK, 0): c.c_block_churn,
            (Action.BLOCK, 1): 0.0,
        }

    def test_all_eight_cells(self) -> None:
        loss = AsymmetricLoss(COSTS)
        cells = self.expected_cells(self.A)
        self.assertEqual(len(cells), 8)
        for (a, y), want in cells.items():
            fn = asymmetric_loss(a, y, self.A, COSTS.c_step, COSTS.c_hold,
                                 COSTS.c_block_churn, COSTS.beta_bypass)
            self.assertAlmostEqual(fn, want, places=12, msg=f"function {a.name}, y={y}")
            self.assertAlmostEqual(loss(a, y, self.A), want, places=12, msg=f"{a.name}, y={y}")
            self.assertAlmostEqual(loss(a.name.lower(), y, self.A), want, places=12)
            self.assertAlmostEqual(float(loss.batch([int(a)], [y], [self.A])[0]), want, places=12)
        # hand-written numbers for the default costs
        self.assertAlmostEqual(loss(Action.STEP_UP, 1, 1000.0), 5.0 + 100.0)
        self.assertAlmostEqual(loss(Action.BLOCK, 0, 1000.0), 150.0)

    def test_loss_matrix(self) -> None:
        m = AsymmetricLoss(COSTS).matrix(self.A)
        self.assertEqual(m.shape, (4, 2))
        for (a, y), want in self.expected_cells(self.A).items():
            self.assertAlmostEqual(m[int(a), y], want)

    def test_vectorised_matches_scalar(self) -> None:
        rng = np.random.default_rng(SEED)
        n = 5000
        a = rng.integers(0, 4, n)
        y = rng.integers(0, 2, n)
        amt = rng.lognormal(7.0, 2.0, n)
        loss = AsymmetricLoss(COSTS)
        vec = loss.batch(a, y, amt)
        ref = np.array([loss(int(a[i]), int(y[i]), float(amt[i])) for i in range(n)])
        np.testing.assert_allclose(vec, ref, rtol=0, atol=1e-9)
        np.testing.assert_allclose(batch_asymmetric_loss(a, y, amt, COSTS), ref, atol=1e-9)
        names = np.array([Action(int(v)).name.lower() for v in a])
        np.testing.assert_allclose(loss.batch(names, y, amt), ref, atol=1e-9)
        norm = loss.batch_normalized(a, y, amt)
        np.testing.assert_allclose(norm, np.minimum(ref / COSTS.loss_normalizer, COSTS.loss_cap))
        self.assertLessEqual(norm.max(), COSTS.loss_cap)

    def test_amount_dependence(self) -> None:
        loss = AsymmetricLoss(COSTS)
        for a in Action:
            self.assertEqual(loss(a, 0, 10.0), loss(a, 0, 1e7), "legit loss must ignore A_t")
        self.assertAlmostEqual(loss(Action.APPROVE, 1, 2e6) / loss(Action.APPROVE, 1, 1e6), 2.0)
        slope = loss(Action.STEP_UP, 1, 2000.0) - loss(Action.STEP_UP, 1, 1000.0)
        self.assertAlmostEqual(slope, COSTS.beta_bypass * 1000.0)

    def test_validation(self) -> None:
        loss = AsymmetricLoss(COSTS)
        with self.assertRaises(ValueError):
            loss(Action.APPROVE, 2, 10.0)
        with self.assertRaises(ValueError):
            loss(Action.APPROVE, 1, -1.0)
        with self.assertRaises(ValueError):
            loss(Action.APPROVE, 1, float("nan"))
        with self.assertRaises(ValueError):
            loss("teleport", 1, 1.0)
        with self.assertRaises(ValueError):
            loss.batch([4], [0], [1.0])
        with self.assertRaises(ValueError):
            CostParameters(beta_bypass=1.5)
        with self.assertRaises(ValueError):
            CostParameters(c_hold=-1.0)

    def test_from_project_config(self) -> None:
        c = CostParameters.from_config(PROJECT_CONFIG)
        self.assertEqual(c.c_step, PROJECT_CONFIG["costs"]["c_step"])
        self.assertEqual(c.c_hold, PROJECT_CONFIG["costs"]["c_hold"])
        self.assertEqual(c.c_block_churn, PROJECT_CONFIG["costs"]["c_block_churn"])
        self.assertEqual(c.beta_bypass, PROJECT_CONFIG["costs"]["beta_bypass"])
        self.assertEqual(CostParameters.from_config().to_dict(), c.to_dict())

    def test_expected_loss_and_bayes_action(self) -> None:
        loss = AsymmetricLoss(COSTS)
        e = loss.expected_losses(0.3, 1000.0)
        m = loss.matrix(1000.0)
        np.testing.assert_allclose(e, 0.7 * m[:, 0] + 0.3 * m[:, 1])
        self.assertEqual(loss.bayes_action(0.0, 1000.0), Action.APPROVE)
        self.assertEqual(loss.bayes_action(1.0, 1000.0), Action.BLOCK)


# --------------------------------------------------------------------------- #
# 2. Hold-and-Verify                                                          #
# --------------------------------------------------------------------------- #
class TestHoldAndVerify(unittest.TestCase):
    def test_fraud_held_costs_only_review(self) -> None:
        loss = AsymmetricLoss(COSTS)
        for amount in np.logspace(0, 7, 15):
            self.assertEqual(loss(Action.HOLD, 1, amount), COSTS.c_hold)
            self.assertEqual(loss(Action.HOLD, 1, amount), loss(Action.HOLD, 0, amount))
            self.assertAlmostEqual(loss.hold_savings(amount), amount - COSTS.c_hold, places=6)

    def test_portfolio_of_held_frauds(self) -> None:
        rng = np.random.default_rng(SEED)
        amounts = rng.lognormal(10.0, 1.5, 1000)
        loss = AsymmetricLoss(COSTS)
        held = loss.batch(np.full(1000, int(Action.HOLD)), np.ones(1000, int), amounts)
        approved = loss.batch(np.zeros(1000, int), np.ones(1000, int), amounts)
        self.assertTrue(np.all(held == COSTS.c_hold))
        self.assertAlmostEqual(held.sum(), 1000 * COSTS.c_hold)
        self.assertAlmostEqual(approved.sum() - held.sum(), amounts.sum() - 1000 * COSTS.c_hold,
                               places=3)

    def test_hold_beats_step_up_on_large_fraud(self) -> None:
        loss = AsymmetricLoss(COSTS)
        crossover = (COSTS.c_hold - COSTS.c_step) / COSTS.beta_bypass  # beta A = c_hold - c_step
        self.assertLess(loss(Action.HOLD, 1, crossover * 2), loss(Action.STEP_UP, 1, crossover * 2))
        self.assertGreater(loss(Action.HOLD, 1, crossover / 2), loss(Action.STEP_UP, 1, crossover / 2))


# --------------------------------------------------------------------------- #
# 3. Online adaptation under drift                                            #
# --------------------------------------------------------------------------- #
class TestAdaptiveDrift(unittest.TestCase):
    def test_single_update_matches_synopsis_equation(self) -> None:
        ctrl = OnlineAdaptiveCRC(alpha=0.02, gamma=0.005, lambda_init=(0.2, 0.5, 0.85),
                                 loss=AsymmetricLoss(COSTS))
        theta0 = np.array(ctrl.protection_levels)
        lam0 = np.array(ctrl.thresholds)
        step = ctrl.update(Action.APPROVE, 1, 3000.0)  # ell = 0.3 > alpha
        ell = 3000.0 / COSTS.loss_normalizer
        self.assertAlmostEqual(step.loss_normalized, ell)
        theta1 = np.array(ctrl.protection_levels)
        np.testing.assert_allclose(theta1, theta0 - 0.005 * (0.02 - ell), atol=1e-15)
        np.testing.assert_allclose(ctrl.thresholds, lam0 + 0.005 * (0.02 - ell), atol=1e-15)
        self.assertFalse(step.projected)
        lam1 = np.array(ctrl.thresholds)
        ctrl.update(Action.APPROVE, 0, 50.0)  # ell = 0 < alpha -> relax
        np.testing.assert_allclose(ctrl.thresholds, lam1 + 0.005 * 0.02, atol=1e-15)

    def test_thresholds_track_fraud_rate_drift(self) -> None:
        y, p, amount, seg = drift_stream(30000, [0.03, 0.08, 0.03])
        router = make_router()
        lam, _ = run_stream(router, y, p, amount)
        settle = seg // 3
        before = lam[settle:seg].mean(axis=0)
        during = lam[seg + settle:2 * seg].mean(axis=0)
        after = lam[2 * seg + settle:].mean(axis=0)
        # fraud surge -> every threshold tightens; fraud recedes -> they relax again
        self.assertTrue(np.all(during < before - 0.05), (before, during))
        self.assertTrue(np.all(after > during + 0.05), (during, after))
        # and the tighter thresholds route a larger share of genuine customers to friction
        legit = y == 0
        fr_before = (p[:seg][legit[:seg]] >= lam[:seg, 0][legit[:seg]]).mean()
        fr_during = (p[seg:2 * seg][legit[seg:2 * seg]] >= lam[seg:2 * seg, 0][legit[seg:2 * seg]]).mean()
        self.assertGreater(fr_during, fr_before)

    def test_low_base_rate_relaxes_to_permissive_extreme(self) -> None:
        y, p, amount, seg = drift_stream(30000, [0.005, 0.06, 0.005])
        router = make_router()
        lam, _ = run_stream(router, y, p, amount)
        self.assertGreater(lam[seg - 1, 0], lam[2 * seg - 1, 0] + 0.2)
        self.assertGreater(lam[-1, 0], lam[2 * seg - 1, 0] + 0.2)

    def test_rolling_fraud_rate_follows_drift(self) -> None:
        y, p, amount, seg = drift_stream(30000, [0.005, 0.06, 0.005])
        router = make_router(rolling_window=2000)
        rates = []
        for i in range(y.size):
            router.feedback(router.route(p[i], 0.0, amount[i]), int(y[i]))
            if i in (seg - 1, 2 * seg - 1):
                rates.append(router.controller.rolling_fraud_rate)
        self.assertLess(rates[0], 0.015)
        self.assertGreater(rates[1], 0.04)

    def test_static_mode_is_frozen(self) -> None:
        y, p, amount, _ = drift_stream(6000, [0.03, 0.08])
        router = make_router(mode="static")
        lam, _ = run_stream(router, y, p, amount)
        self.assertTrue(np.all(lam == np.array([0.20, 0.50, 0.85])))

    def test_warmup_freezes_thresholds(self) -> None:
        ctrl = OnlineAdaptiveCRC(warmup=10, loss=AsymmetricLoss(COSTS))
        init = ctrl.thresholds
        for _ in range(10):
            self.assertFalse(ctrl.update(Action.APPROVE, 1, 5000.0).adapted)
        self.assertEqual(ctrl.thresholds, init)
        self.assertTrue(ctrl.update(Action.APPROVE, 1, 5000.0).adapted)
        self.assertNotEqual(ctrl.thresholds, init)
        self.assertEqual(ctrl.n_adaptive, 1)

    def test_from_config(self) -> None:
        ctrl = OnlineAdaptiveCRC.from_config(HYPERPARAMS, PROJECT_CONFIG)
        crc = HYPERPARAMS["conformal_crc"]
        self.assertEqual(ctrl.alpha, crc["target_risk_alpha"])
        self.assertEqual(ctrl.gamma, (crc["gamma_step_size"],) * 3)
        self.assertEqual(ctrl.warmup, crc["warmup_transactions"])
        np.testing.assert_allclose(ctrl.thresholds, [crc["lambda_init"][k] for k in ("low", "mid", "high")])
        self.assertTrue(OnlineAdaptiveCRC.from_config().feasibility()["feasible"])


# --------------------------------------------------------------------------- #
# 4. Loss stays within alpha +/- eps                                          #
# --------------------------------------------------------------------------- #
class TestLossBound(unittest.TestCase):
    EPS = 0.01

    def test_cumulative_loss_within_alpha_pm_eps(self) -> None:
        y, p, amount, _ = drift_stream(30000, [0.03, 0.08, 0.03])
        router = make_router()
        _, ell = run_stream(router, y, p, amount)
        g = router.controller.guarantee()
        self.assertLess(g["identity_residual"], 1e-9)
        self.assertTrue(g["within_bound"])
        self.assertLessEqual(abs(ell.mean() - 0.02), g["epsilon_T"] + 1e-12)
        self.assertLessEqual(g["epsilon_T"], self.EPS)
        self.assertLessEqual(abs(g["mean_loss"] - 0.02), self.EPS)
        self.assertAlmostEqual(g["mean_loss"], ell.mean(), places=12)

    def test_one_sided_risk_bound_with_rare_fraud(self) -> None:
        y, p, amount, _ = drift_stream(30000, [0.005, 0.06, 0.005])
        router = make_router()
        _, ell = run_stream(router, y, p, amount)
        g = router.controller.guarantee()
        self.assertTrue(g["within_risk_bound"])
        self.assertLessEqual(ell.mean(), g["risk_upper_bound"])
        self.assertLessEqual(g["risk_upper_bound"], 0.02 + self.EPS)
        self.assertEqual(g["protective_displacement"], 0.0)
        self.assertLess(g["identity_residual"], 1e-9)

    def test_rolling_window_within_bound_after_drift(self) -> None:
        y, p, amount, seg = drift_stream(30000, [0.03, 0.08, 0.03])
        router = make_router()
        lam, ell, projected = run_stream(router, y, p, amount, return_projected=True)
        w = 5000
        window = slice(2 * seg - w, 2 * seg)  # steady state inside the fraud surge
        self.assertFalse(projected[window].any(), "window must be projection-free")
        bound = router.controller.window_bound(w)
        # exact window identity: gamma * sum(alpha - ell) = lambda_end - lambda_start
        lhs = 0.005 * (w * 0.02 - ell[window].sum())
        self.assertAlmostEqual(lhs, lam[2 * seg - 1, 0] - lam[2 * seg - w - 1, 0], places=9)
        self.assertLessEqual(abs(ell[window].mean() - 0.02), bound)

    def test_static_thresholds_break_under_drift(self) -> None:
        y, p, amount, seg = drift_stream(30000, [0.03, 0.08, 0.03])
        permissive = (0.60, 0.80, 0.95)
        _, ell_static = run_stream(make_router(mode="static", lambda_init=permissive), y, p, amount)
        adaptive = make_router(lambda_init=permissive)
        _, ell_adapt = run_stream(adaptive, y, p, amount)
        surge = slice(seg, 2 * seg)
        self.assertGreater(ell_static[surge].mean(), 0.02 + self.EPS)
        self.assertLess(ell_adapt[surge].mean(), ell_static[surge].mean())
        self.assertLessEqual(ell_adapt.mean(), adaptive.controller.risk_upper_bound())

    def test_certificate_holds_for_adversarial_sequence(self) -> None:
        rng = random.Random(SEED)
        ctrl = OnlineAdaptiveCRC(alpha=0.02, gamma=0.05, loss=AsymmetricLoss(COSTS))
        for t in range(20000):
            phase = (t // 1500) % 3
            if phase == 0:
                ctrl.update(Action.APPROVE, 1, rng.uniform(5e3, 5e4))
            elif phase == 1:
                ctrl.update(Action.BLOCK, rng.random() < 0.5, 100.0)
            else:
                ctrl.update(rng.randrange(4), rng.random() < 0.1, rng.uniform(0, 2e4))
        g = ctrl.guarantee()
        self.assertLess(g["identity_residual"], 1e-8)
        self.assertTrue(g["within_bound"])
        self.assertTrue(g["within_risk_bound"])

    def test_delayed_feedback_preserves_certificate(self) -> None:
        y, p, amount, _ = drift_stream(12000, [0.03, 0.08])
        router = make_router()
        rng = np.random.default_rng(SEED)
        pending = []
        for i in range(y.size):
            pending.append((router.route(p[i], 0.0, amount[i]), int(y[i]),
                            i + int(rng.integers(1, 300))))
            due = [x for x in pending if x[2] <= i]
            pending = [x for x in pending if x[2] > i]
            for d, label, _ in due:
                router.feedback(d, label)
        for d, label, _ in pending:
            router.feedback(d, label)
        g = router.controller.guarantee()
        self.assertEqual(router.controller.t, y.size)
        self.assertLess(g["identity_residual"], 1e-9)
        self.assertLessEqual(g["mean_loss"], g["risk_upper_bound"])
        self.assertLessEqual(abs(g["mean_loss"] - 0.02), self.EPS + g["epsilon_T"])

    def test_feasibility_report(self) -> None:
        ok = OnlineAdaptiveCRC(alpha=0.02, loss=AsymmetricLoss(COSTS)).feasibility()
        self.assertTrue(ok["feasible"])
        bad = OnlineAdaptiveCRC(alpha=0.01, loss=AsymmetricLoss(COSTS)).feasibility()
        self.assertFalse(bad["feasible"])  # 150 / 1e4 = 0.015 >= 0.01


# --------------------------------------------------------------------------- #
# 5. Monotonicity                                                             #
# --------------------------------------------------------------------------- #
class TestMonotonicity(unittest.TestCase):
    def assert_feasible(self, lam, lo=0.0, hi=1.0, gap=0.01) -> None:
        self.assertGreaterEqual(lam[0], lo - 1e-12)
        self.assertLessEqual(lam[2], hi + 1e-12)
        self.assertGreaterEqual(lam[1] - lam[0], gap - 1e-12)
        self.assertGreaterEqual(lam[2] - lam[1], gap - 1e-12)
        self.assertTrue(lam[0] < lam[1] < lam[2])

    def test_strict_order_under_aggressive_updates(self) -> None:
        rng = np.random.default_rng(SEED)
        ctrl = OnlineAdaptiveCRC(alpha=0.02, gamma=(0.9, 0.3, 0.05), min_gap=0.01,
                                 loss=AsymmetricLoss(COSTS))
        n_proj = 0
        for _ in range(20000):
            step = ctrl.update(int(rng.integers(0, 4)), int(rng.random() < 0.3),
                               float(rng.lognormal(8, 2)))
            n_proj += step.projected
            self.assert_feasible(ctrl.thresholds)
            self.assertTrue(ctrl.is_strictly_monotone())
        self.assertGreater(n_proj, 0, "the projection must have been exercised")
        lam = ctrl.history()["lambda"]
        self.assertTrue(np.all(np.diff(lam, axis=1) >= 0.01 - 1e-12))

    def test_projection_is_exact_euclidean_projection(self) -> None:
        from scipy.optimize import minimize

        rng = np.random.default_rng(SEED)
        gap = 0.05
        cons = [{"type": "ineq", "fun": lambda z: z[0]},
                {"type": "ineq", "fun": lambda z: z[1] - z[0] - gap},
                {"type": "ineq", "fun": lambda z: z[2] - z[1] - gap},
                {"type": "ineq", "fun": lambda z: 1.0 - z[2]}]
        for _ in range(150):
            x = rng.uniform(-0.6, 1.6, 3)
            ours = np.array(project_ordered_thresholds(x, 0.0, 1.0, gap))
            self.assert_feasible(ours, gap=gap)
            ref = minimize(lambda z: np.sum((z - x) ** 2), x0=np.array([0.2, 0.5, 0.8]),
                           jac=lambda z: 2 * (z - x), constraints=cons, method="SLSQP",
                           options={"ftol": 1e-14, "maxiter": 500}).x
            self.assertLessEqual(np.sum((ours - x) ** 2), np.sum((ref - x) ** 2) + 1e-9)
            np.testing.assert_allclose(ours, ref, atol=1e-5)
            # idempotent
            np.testing.assert_allclose(project_ordered_thresholds(ours, 0.0, 1.0, gap), ours,
                                       atol=1e-12)

    def test_unordered_init_is_projected(self) -> None:
        ctrl = OnlineAdaptiveCRC(lambda_init=(0.9, 0.2, 0.5), loss=AsymmetricLoss(COSTS))
        self.assert_feasible(ctrl.thresholds)

    def test_invalid_constraints_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OnlineAdaptiveCRC(min_gap=0.0)
        with self.assertRaises(ValueError):
            OnlineAdaptiveCRC(lambda_bounds=(0.0, 0.01), min_gap=0.01)
        with self.assertRaises(ValueError):
            OnlineAdaptiveCRC(gamma=-0.1)
        with self.assertRaises(ValueError):
            project_ordered_thresholds([0.1, 0.2, 0.3], 0.0, 1.0, 0.0)

    def test_state_round_trip_and_rollback(self) -> None:
        y, p, amount, _ = drift_stream(4000, [0.03, 0.08])
        router = make_router()
        run_stream(router, y[:2000], p[:2000], amount[:2000])
        snap = router.controller.state_dict()
        thresholds = router.controller.thresholds
        run_stream(router, y[2000:], p[2000:], amount[2000:])
        self.assertNotEqual(router.controller.thresholds, thresholds)
        router.controller.load_state_dict(snap)
        self.assertEqual(router.controller.thresholds, thresholds)
        bad = dict(snap, **{"lambda": [0.5, 0.4, 0.9]})
        with self.assertRaises(ValueError):
            router.controller.load_state_dict(bad)


# --------------------------------------------------------------------------- #
# 6. Router behaviour and latency                                             #
# --------------------------------------------------------------------------- #
class TestRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.router = make_router()  # thresholds (0.20, 0.50, 0.85)

    def test_threshold_regions(self) -> None:
        r = self.router
        for p, want in [(0.0, Action.APPROVE), (0.19, Action.APPROVE), (0.20, Action.STEP_UP),
                        (0.49, Action.STEP_UP), (0.50, Action.HOLD), (0.84, Action.HOLD),
                        (0.85, Action.BLOCK), (1.0, Action.BLOCK)]:
            d = r.route(p, 0.0, 100.0)
            self.assertIsInstance(d, RoutingDecision)
            self.assertEqual(d.action, want, p)
            self.assertEqual(d.reason, REASON_THRESHOLD)
            self.assertEqual(d.thresholds, (0.20, 0.50, 0.85))
            self.assertEqual(r.controller.action_for_score(p), want)

    def test_epistemic_uncertainty(self) -> None:
        r = self.router
        self.assertEqual(r.route(0.15, 0.2, 10.0).action, Action.STEP_UP)  # 0.15 + 0.5*0.2
        d = r.route(0.95, 0.30, 10.0)  # would block, but calibration too wide
        self.assertEqual(d.action, Action.HOLD)
        self.assertEqual(d.reason, REASON_UNCERTAIN_BLOCK)
        self.assertEqual(r.route(0.95, 0.05, 10.0).action, Action.BLOCK)

    def test_kb_rule_floor(self) -> None:
        r = self.router
        for rules in (["R17"], {"R3": 0.9, "R8": 0.7}, 2):
            d = r.route(0.01, 0.0, 10.0, rules)
            self.assertEqual(d.action, Action.STEP_UP)
            self.assertEqual(d.reason, REASON_RULE_FLOOR)
        self.assertEqual(r.route(0.01, 0.0, 10.0, []).action, Action.APPROVE)
        self.assertEqual(r.route(0.9, 0.0, 10.0, ["R1"]).action, Action.BLOCK)
        no_floor = PreTransactionRouter(r.controller, rule_floor_action=None)
        self.assertEqual(no_floor.route(0.01, 0.0, 10.0, ["R1"]).action, Action.APPROVE)

    def test_amount_aware_escalation(self) -> None:
        r = PreTransactionRouter(self.router.controller, amount_aware_escalation=True)
        # 0.1 * 1e4 * (1 - 0.1) = 900 > c_step = 5 -> step-up
        d = r.route(0.1, 0.0, 1e4)
        self.assertEqual((d.action, d.reason), (Action.STEP_UP, REASON_AMOUNT))
        self.assertEqual(r.route(0.1, 0.0, 20.0).action, Action.APPROVE)  # 1.8 < 5

    def test_tuple_and_scored_event_inputs(self) -> None:
        r = self.router
        self.assertEqual(r.route_tuple((0.6, 0.0, 500.0, ())).action, Action.HOLD)
        d = r.route_scored_event({"p_hat": 0.1, "margin": -2.0, "p_interval": (0.05, 0.45)}, 300.0)
        self.assertAlmostEqual(d.delta_p_hat, 0.40)
        self.assertAlmostEqual(d.score, 0.30)  # 0.1 + 0.5 * 0.4
        self.assertEqual(d.action, Action.STEP_UP)
        acts = r.route_batch([0.1, 0.3, 0.6, 0.9])
        np.testing.assert_array_equal(acts, [0, 1, 2, 3])
        self.assertEqual(sum(r.action_counts), r.n_routed)

    def test_input_validation(self) -> None:
        r = self.router
        for bad in [(-0.1, 0.0, 1.0), (1.1, 0.0, 1.0), (0.5, -0.1, 1.0), (0.5, 0.0, -1.0),
                    (0.5, 0.0, math.inf), (float("nan"), 0.0, 1.0)]:
            with self.assertRaises(ValueError):
                r.route(*bad)

    def test_feedback_updates_controller(self) -> None:
        r = self.router
        d = r.route(0.1, 0.0, 8000.0)
        step = r.feedback(d, 1)
        self.assertEqual(step.action, int(Action.APPROVE))
        self.assertAlmostEqual(step.loss, 8000.0)
        self.assertLess(r.controller.lambda_low, 0.20)

    def test_from_config(self) -> None:
        r = PreTransactionRouter.from_config(HYPERPARAMS, PROJECT_CONFIG)
        cfg = HYPERPARAMS["conformal_crc"]["router"]
        self.assertEqual(r.kappa, cfg["kappa_uncertainty"])
        self.assertEqual(r.rule_floor, Action.STEP_UP)
        self.assertEqual(r.latency_budget_ms, 1.0)

    def test_sub_millisecond_routing(self) -> None:
        rng = np.random.default_rng(SEED)
        n = 20000
        tuples = list(zip(rng.random(n), rng.random(n) * 0.1, rng.lognormal(8, 2, n),
                          [() if rng.random() < 0.9 else ("R1", "R2") for _ in range(n)]))
        prof = self.router.latency_profile(tuples)
        self.assertTrue(prof["within_budget"], prof)
        self.assertLess(prof["p99_ms"], 1.0, prof)
        self.assertLess(prof["p50_ms"], 0.1, prof)
        self.assertLess(prof["max_ms"], 50.0, prof)  # tolerate a rare OS scheduling hiccup
        print(f"\n    router latency: p50={prof['p50_ms'] * 1e3:.2f} us  "
              f"p95={prof['p95_ms'] * 1e3:.2f} us  p99={prof['p99_ms'] * 1e3:.2f} us",
              file=sys.stderr)

    def test_route_plus_feedback_under_budget(self) -> None:
        y, p, amount, _ = drift_stream(10000, [0.03, 0.08])
        lat = np.empty(y.size)
        r = self.router
        for i in range(y.size):
            t0 = time.perf_counter()
            r.feedback(r.route(p[i], 0.0, amount[i]), int(y[i]))
            lat[i] = (time.perf_counter() - t0) * 1000.0
        self.assertLess(np.percentile(lat, 99), 1.0)

    def test_deterministic_replay(self) -> None:
        y, p, amount, _ = drift_stream(5000, [0.03, 0.08])
        lam1, ell1 = run_stream(make_router(), y, p, amount)
        lam2, ell2 = run_stream(make_router(), y, p, amount)
        np.testing.assert_array_equal(lam1, lam2)
        np.testing.assert_array_equal(ell1, ell2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
