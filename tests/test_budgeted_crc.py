"""
tests/test_budgeted_crc.py
==========================

Unit tests of the budgeted cost-sensitive Bayes controller
(:mod:`src.conformal.budgeted_crc`) and its integration with the router.
"""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal.adaptive_crc import OnlineAdaptiveCRC  # noqa: E402
from src.conformal.budgeted_crc import BudgetedBayesCRC, build_controller  # noqa: E402
from src.conformal.loss_functions import Action, AsymmetricLoss, CostParameters  # noqa: E402
from src.conformal.router import PreTransactionRouter  # noqa: E402

SEED = 42
PROJECT_CONFIG = json.loads((ROOT / "configs" / "project_config.json").read_text())
HYPERPARAMS = json.loads((ROOT / "configs" / "hyperparams.json").read_text())
LOSS = AsymmetricLoss(CostParameters.from_config(PROJECT_CONFIG))


def drift_stream(n: int, rates=(0.005, 0.02), seed: int = SEED):
    """Fraud rate rises from rates[0] to rates[1]; scores are informative but stale."""
    rng = np.random.default_rng(seed)
    base = np.linspace(rates[0], rates[1], n)
    y = (rng.random(n) < base).astype(int)
    p = np.clip(np.where(y == 1, rng.beta(2, 3, n), rng.beta(1, 60, n)), 1e-6, 1 - 1e-6)
    amount = rng.lognormal(10, 1.5, n)
    return y, p, amount


def run(ctrl: BudgetedBayesCRC, y, p, amount):
    acts = np.empty(y.size, dtype=int)
    for i in range(y.size):
        acts[i] = ctrl.decide(p[i], amount[i])
        ctrl.update(acts[i], int(y[i]), float(amount[i]), float(p[i]))
    return acts


class TestDecisionRule(unittest.TestCase):
    def test_bayes_rule_at_zero_price(self) -> None:
        c = BudgetedBayesCRC(loss=LOSS, calibration_shift=False)
        rng = np.random.default_rng(SEED)
        cs = LOSS.costs
        for p, a in zip(rng.random(500), rng.lognormal(8, 2, 500)):
            e = [p * a, cs.c_step + cs.beta_bypass * p * a, cs.c_hold, (1 - p) * cs.c_block_churn]
            self.assertEqual(c.decide(p, a), int(np.argmin(e)))

    def test_monotone_in_probability_and_price(self) -> None:
        c = BudgetedBayesCRC(loss=LOSS)
        grid = np.linspace(0, 1, 501)
        for amount in (10.0, 1e3, 1e5, 5e6):
            for eta in (0.0, 3.0, 8.0, 15.0):
                c._eta = eta
                acts = c.decide_batch(grid, np.full_like(grid, amount))
                self.assertTrue(np.all(np.diff(acts) >= 0), (amount, eta))
        p = np.random.default_rng(SEED).random(2000)
        a = np.random.default_rng(SEED + 1).lognormal(10, 2, 2000)
        rates = []
        for eta in (0.0, 2.0, 5.0, 10.0, 15.0):
            c._eta = eta
            rates.append((c.decide_batch(p, a) >= 1).mean())
        self.assertTrue(all(x >= y for x, y in zip(rates, rates[1:])), rates)

    def test_batch_matches_scalar(self) -> None:
        c = BudgetedBayesCRC(loss=LOSS)
        c._eta, c._theta = 6.0, 0.7
        y, p, a = drift_stream(3000)
        np.testing.assert_array_equal(c.decide_batch(p, a), [c.decide(pi, ai) for pi, ai in zip(p, a)])

    def test_thresholds_strictly_increasing(self) -> None:
        c = BudgetedBayesCRC(loss=LOSS)
        for eta in (0.0, 5.0, 12.0, 20.0):
            c._eta = eta
            lo, mid, hi = c.thresholds
            self.assertLess(lo, mid)
            self.assertLess(mid, hi)


class TestGuarantees(unittest.TestCase):
    def test_intervention_budget_bound(self) -> None:
        y, p, a = drift_stream(20000)
        cold = BudgetedBayesCRC(intervention_budget=0.03, gamma_budget=0.01, loss=LOSS)
        run(cold, y, p, a)
        self.assertTrue(cold.guarantee()["within_budget_bound"], cold.guarantee())
        c = BudgetedBayesCRC(intervention_budget=0.03, gamma_budget=0.01, loss=LOSS)
        c.warm_start(p[:2000], a[:2000])                     # label-free warm start
        acts = run(c, y, p, a)
        g = c.guarantee()
        self.assertTrue(g["within_budget_bound"], g)
        self.assertLess(g["budget_identity_residual"], 1e-8)
        self.assertLess(abs((acts >= 1).mean() - 0.03), 0.01)

    def test_calibration_bound_under_drift(self) -> None:
        y, p, a = drift_stream(20000, rates=(0.002, 0.03))
        c = BudgetedBayesCRC(intervention_budget=0.05, gamma_calibration=0.02, loss=LOSS)
        run(c, y, p, a)
        g = c.guarantee()
        self.assertTrue(g["within_calibration_bound"], g)

    def test_static_mode_is_frozen(self) -> None:
        y, p, a = drift_stream(2000)
        c = BudgetedBayesCRC(mode="static", loss=LOSS)
        c._eta = 4.0
        run(c, y, p, a)
        self.assertEqual(c._eta, 4.0)
        self.assertEqual(c.theta, 0.0)
        self.assertEqual(c.n_adaptive, 0)

    def test_guarantee_after_mode_switch_and_reload(self) -> None:
        y, p, a = drift_stream(6000)
        c = BudgetedBayesCRC(intervention_budget=0.03, gamma_budget=0.01, loss=LOSS)
        run(c, y[:4000], p[:4000], a[:4000])
        before = c.guarantee()
        c.mode = "static"                                   # as in the ablation's B3 arm
        run(c, y[4000:], p[4000:], a[4000:])
        after = c.guarantee()
        self.assertLess(after["budget_identity_residual"], 1e-8)
        self.assertEqual(after["intervention_rate"], before["intervention_rate"])
        fresh = BudgetedBayesCRC(intervention_budget=0.03, gamma_budget=0.01, loss=LOSS)
        fresh.load_state_dict(c.state_dict())
        g = fresh.guarantee()
        self.assertAlmostEqual(g["intervention_rate"], after["intervention_rate"])
        self.assertLess(g["budget_identity_residual"], 1e-8)

    def test_router_without_bayes_flag_attribute(self) -> None:
        r = PreTransactionRouter.from_config(HYPERPARAMS, PROJECT_CONFIG)
        r.__dict__.pop("_bayes", None)                      # router unpickled from an old run
        self.assertTrue(r._bayes)
        r.feedback(r.route(0.2, 0.0, 1000.0), 1)

    def test_warm_start_meets_budget(self) -> None:
        _, p, a = drift_stream(10000)
        c = BudgetedBayesCRC(intervention_budget=0.02, loss=LOSS)
        rate = c.warm_start(p, a)
        self.assertLessEqual(rate, 0.02 + 1e-9)
        self.assertGreater(rate, 0.015)

    def test_budget_reduces_loss_versus_legacy_bands(self) -> None:
        y, p, a = drift_stream(20000)
        c = BudgetedBayesCRC(intervention_budget=0.05, gamma_budget=0.01, loss=LOSS)
        c.warm_start(p[:2000], a[:2000])
        acts = run(c, y, p, a)
        legacy = PreTransactionRouter(OnlineAdaptiveCRC(loss=LOSS))
        acts_l = np.empty(y.size, dtype=int)
        for i in range(y.size):
            d = legacy.route(p[i], 0.0, a[i])
            acts_l[i] = int(d.action)
            legacy.feedback(d, int(y[i]))
        self.assertLess(LOSS.batch(acts, y, a).sum(), LOSS.batch(acts_l, y, a).sum())


class TestRouterIntegration(unittest.TestCase):
    def test_factory_follows_config(self) -> None:
        self.assertIsInstance(build_controller(HYPERPARAMS, PROJECT_CONFIG), BudgetedBayesCRC)
        legacy = json.loads(json.dumps(HYPERPARAMS))
        legacy["conformal_crc"]["decision_rule"] = "threshold"
        self.assertIsInstance(build_controller(legacy, PROJECT_CONFIG), OnlineAdaptiveCRC)
        bad = json.loads(json.dumps(HYPERPARAMS))
        bad["conformal_crc"]["decision_rule"] = "nope"
        with self.assertRaises(ValueError):
            build_controller(bad, PROJECT_CONFIG)

    def test_router_uses_amount(self) -> None:
        r = PreTransactionRouter.from_config(HYPERPARAMS, PROJECT_CONFIG)
        self.assertIsInstance(r.controller, BudgetedBayesCRC)
        r.controller._eta = 0.0
        self.assertEqual(r.route(0.01, 0.0, 10.0).action, Action.APPROVE)
        self.assertGreaterEqual(int(r.route(0.01, 0.0, 1e6).action), int(Action.STEP_UP))

    def test_rule_floor_and_feedback(self) -> None:
        r = PreTransactionRouter.from_config(HYPERPARAMS, PROJECT_CONFIG)
        d = r.route(0.0001, 0.0, 5.0, ["R1"])
        self.assertEqual(d.action, Action.STEP_UP)
        t0 = r.controller.t
        r.feedback(d, 0)
        self.assertEqual(r.controller.t, t0 + 1)

    def test_sub_millisecond_routing(self) -> None:
        r = PreTransactionRouter.from_config(HYPERPARAMS, PROJECT_CONFIG)
        y, p, a = drift_stream(20000)
        lat = np.empty(y.size)
        for i in range(y.size):
            t0 = time.perf_counter()
            r.route(p[i], 0.0, a[i])
            lat[i] = (time.perf_counter() - t0) * 1000.0
        self.assertLess(np.percentile(lat, 99), 1.0)

    def test_state_roundtrip(self) -> None:
        c = BudgetedBayesCRC(loss=LOSS)
        y, p, a = drift_stream(3000)
        run(c, y, p, a)
        snap = c.state_dict()
        thr = c.thresholds
        c.update(3, 0, 100.0, 0.9)
        c.load_state_dict(snap)
        self.assertEqual(c.thresholds, thr)
        with self.assertRaises(ValueError):
            c.load_state_dict({"kind": "threshold"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
