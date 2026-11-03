"""
tests/test_phase7_experiments.py
================================

Phase 7 verification: the experimental evaluation suite.

1. Drift evaluator - chronological, step-aligned windows; every model steps
   through the windows in order and logs every metric; no information from a
   window (features or labels) can influence the scores of an earlier window;
   labels reach the online components only after the configured delay.
2. Ablation runner - the five ablation points (seven arms) activate exactly
   the intended components; the branched arms share one fitted pipeline; the
   Analyst-Assist layer never changes a decision.
3. Adversarial stress test - 5% ... 25% poisoning sweep; harmful rules are
   rolled back within the certified number of update cycles; Ville's maximal
   inequality ``P(exists t: M_t^roll >= 1/alpha_roll) <= alpha_roll`` holds.
4. Statistical testing - McNemar chi-squared values / p-values / odds ratios,
   Wilcoxon signed-rank (exact and normal), Holm adjustment and block-bootstrap
   confidence intervals are mathematically valid (cross-checked against SciPy
   when it is installed).

Run from the repository root::

    python tests/test_phase7_experiments.py
    python -m pytest tests/test_phase7_experiments.py -q
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_adversarial_stress_test import (  # noqa: E402
    POISON_RATES,
    POLICIES,
    binomial_cdf,
    binomial_sf,
    clopper_pearson,
    run_adversarial_stress_test,
)
from experiments.run_drift_eval import (  # noqa: E402
    FAST_BASELINE_CONFIGS,
    METRIC_COLUMNS,
    RESULT_COLUMNS,
    SEED,
    DriftEvaluator,
    WindowOutput,
    build_synthetic_stream,
    build_systems,
    chronological_splits,
    classification_metrics,
    clone_system,
    f_beta,
    fast_hyperparams,
    inject_camouflage_drift,
    load_configs,
    make_loss,
    partition_windows,
    prepare_splits,
    resolve_source,
    run_drift_evaluation,
    select_operating_threshold,
)
from experiments.statistical_significance import (  # noqa: E402
    block_bootstrap_ci,
    chi2_sf_df1,
    holm_bonferroni,
    mcnemar_test,
    pooled_metric,
    rankdata_average,
    run_statistical_tests,
    signed_rank_null_counts,
    wilcoxon_signed_rank,
)
from experiments.statistical_significance import main as stats_main  # noqa: E402
from src.conformal.loss_functions import Action  # noqa: E402
from src.data.causal_preprocessor import set_global_seed  # noqa: E402
from src.data.dataset_loaders import make_synthetic_paysim_frame  # noqa: E402
from src.epistemic_gate.martingale_eprocess import GateConfig  # noqa: E402

HAS_HEAVY = (importlib.util.find_spec("torch") is not None
             and importlib.util.find_spec("xgboost") is not None)
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HEAVY_REASON = "PyTorch and XGBoost are required for detector-level Phase-7 tests"

HYPERPARAMS, PROJECT_CONFIG = load_configs()
GATE_CFG = GateConfig.from_config(HYPERPARAMS)
N_TEST_WINDOWS = 10


def tiny_hyperparams() -> dict:
    """Fast-profile MG-ECFM detector further reduced for unit-test speed."""
    hp = fast_hyperparams(HYPERPARAMS)
    hp["graphsage"]["epochs"] = 1
    hp["autoencoder"]["epochs"] = 2
    hp["xgboost"].update({"n_estimators": 60, "early_stopping_rounds": 20})
    return hp


def tiny_baseline_configs() -> dict:
    cfg = copy.deepcopy(FAST_BASELINE_CONFIGS)
    cfg["bp3"]["epochs"] = 3
    return cfg


def small_splits(n_windows: int = N_TEST_WINDOWS):
    frame, names, canon = build_synthetic_stream(n_steps=120, tx_per_step=(25, 40), n_customers=300,
                                                 n_merchants=40, drift_strength=0.5, seed=SEED)
    return chronological_splits(frame, names, n_windows, (0.45, 0.10, 0.10),
                                source="synthetic_test", canonical=canon)


# --------------------------------------------------------------------------- #
# 1a. Data, windows and metrics (no PyTorch / XGBoost needed)                 #
# --------------------------------------------------------------------------- #
class TestSourceResolution(unittest.TestCase):
    """``auto`` must never fall back to synthetic data when a raw file is missing."""

    MISSING = {"datasets": {"paysim": {"transaction_path": "data/raw/paysim/__missing__.csv"},
                            "ieee_cis": {"transaction_path": "data/raw/ieee_cis/__missing__.csv"}}}

    def test_auto_raises_when_paysim_missing(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_source("auto", self.MISSING)
        self.assertIn("--source synthetic", str(ctx.exception))

    def test_explicit_real_source_raises_when_missing(self) -> None:
        for source in ("paysim", "ieee_cis"):
            with self.assertRaises(FileNotFoundError):
                resolve_source(source, self.MISSING)

    def test_prepare_splits_auto_does_not_return_synthetic(self) -> None:
        with self.assertRaises(FileNotFoundError):
            prepare_splits("auto", 5, self.MISSING)

    def test_explicit_synthetic_is_allowed(self) -> None:
        self.assertEqual(resolve_source("synthetic", self.MISSING), "synthetic")


class TestDriftWindowsAndMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        set_global_seed(SEED)
        cls.splits = small_splits()

    def test_fifty_step_aligned_windows_cover_test_split(self) -> None:
        rng = np.random.default_rng(SEED)
        steps = np.repeat(np.arange(261, 401), rng.integers(50, 91, size=140))
        windows = partition_windows(steps, 50)
        self.assertEqual(len(windows), 50)
        self.assertEqual([w.index for w in windows], list(range(1, 51)))
        self.assertEqual(windows[0].row_start, 0)
        self.assertEqual(windows[-1].row_end, steps.size)
        for a, b in zip(windows, windows[1:]):
            self.assertEqual(a.row_end, b.row_start)                 # contiguous, no overlap
            self.assertLess(a.end_step, b.start_step)                # chronological, step-aligned
        for w in windows:
            self.assertGreater(w.n_rows, 0)
            self.assertTrue(np.all(steps[w.row_start:w.row_end] >= w.start_step))
            self.assertTrue(np.all(steps[w.row_start:w.row_end] <= w.end_step))
        sizes = np.array([w.n_rows for w in windows])
        self.assertLess(sizes.max() / sizes.min(), 3.0)              # approximately equal size

    def test_partition_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            partition_windows(np.repeat(np.arange(10), 5), 50)       # fewer steps than windows
        with self.assertRaises(ValueError):
            partition_windows(np.array([3, 2, 1]), 2)                # not chronological
        with self.assertRaises(ValueError):
            partition_windows(np.array([], dtype=int), 1)

    def test_chronological_split_order_and_classes(self) -> None:
        s = self.splits
        parts = [s.train, s.calibration, s.operating, s.test]
        for a, b in zip(parts, parts[1:]):
            self.assertLess(a["step"].max(), b["step"].min())
        for p in parts:
            self.assertTrue(np.all(np.diff(p["step"].to_numpy()) >= 0))
        self.assertEqual(len(s.windows), N_TEST_WINDOWS)
        self.assertEqual(sum(w.n_rows for w in s.windows), len(s.test))
        for name in ("train", "calibration"):
            y = getattr(s, name)["label"]
            self.assertTrue(0 < y.sum() < len(y))
        forbidden = {"oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "label"}
        self.assertTrue(forbidden.isdisjoint(s.feature_names))

    def test_camouflage_drift_only_moves_later_fraud(self) -> None:
        raw = make_synthetic_paysim_frame(n_steps=60, seed=SEED)
        drifted = inject_camouflage_drift(raw, start_step=31, strength=0.8, seed=SEED)
        legit = raw["isFraud"] == 0
        early = raw["step"] < 31
        pd.testing.assert_series_equal(raw.loc[legit, "amount"], drifted.loc[legit, "amount"])
        pd.testing.assert_series_equal(raw.loc[early, "amount"], drifted.loc[early, "amount"])
        late_fraud = (~legit) & (~early)
        self.assertTrue(late_fraud.any())
        gap_before = abs(np.log(raw.loc[late_fraud, "amount"]).median() - 9.0)
        gap_after = abs(np.log(drifted.loc[late_fraud, "amount"]).median() - 9.0)
        self.assertLess(gap_after, gap_before)
        pd.testing.assert_frame_equal(raw, inject_camouflage_drift(raw, 31, 0.0))
        with self.assertRaises(ValueError):
            inject_camouflage_drift(raw, 31, 1.5)

    def test_classification_metrics_hand_computed(self) -> None:
        y = np.array([1, 1, 0, 0, 0, 1, 0, 0])
        p = np.array([0.9, 0.4, 0.35, 0.1, 0.2, 0.8, 0.6, 0.05])
        yhat = (p >= 0.5).astype(int)                 # TP=2, FP=1, FN=1
        m = classification_metrics(y, p, yhat)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)
        self.assertAlmostEqual(m["f2"], f_beta(2 / 3, 2 / 3, 2.0))
        self.assertAlmostEqual(m["brier"], float(np.mean((p - y) ** 2)))
        self.assertTrue(0.0 <= m["pr_auc"] <= 1.0 and 0.0 <= m["roc_auc"] <= 1.0)
        self.assertAlmostEqual(f_beta(0.5, 1.0, 2.0), 5 * 0.5 / (4 * 0.5 + 1.0))
        no_fraud = classification_metrics(np.zeros(5, int), np.full(5, 0.1), np.zeros(5, int))
        self.assertTrue(math.isnan(no_fraud["pr_auc"]) and math.isnan(no_fraud["roc_auc"]))

    def test_operating_threshold_maximises_f1(self) -> None:
        rng = np.random.default_rng(SEED)
        y = (rng.random(500) < 0.1).astype(int)
        p = np.clip(0.6 * y + rng.normal(0.2, 0.15, 500), 0, 1)
        tau = select_operating_threshold(y, p)

        def f1_at(t):
            yh = (p >= t).astype(int)
            return classification_metrics(y, p, yh)["f1"]
        grid = np.unique(p)
        self.assertGreaterEqual(f1_at(tau) + 1e-12, max(f1_at(t) for t in grid))
        self.assertEqual(select_operating_threshold(np.zeros(4, int), np.zeros(4)), 0.5)


# --------------------------------------------------------------------------- #
# 1b. Drift evaluator with real detectors                                     #
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_HEAVY, HEAVY_REASON)
class TestDriftEvaluator(unittest.TestCase):
    K = 4  # windows kept untouched in the leakage audits

    @classmethod
    def setUpClass(cls) -> None:
        set_global_seed(SEED)
        cls.splits = small_splits()
        cls.loss_fn = make_loss(PROJECT_CONFIG)
        systems = build_systems(cls.splits, tiny_hyperparams(), PROJECT_CONFIG, tiny_baseline_configs(),
                                models=("mg_ecfm", "bp3"), poison_rate=0.10, label_delay_windows=1)
        cls.evaluator = DriftEvaluator(cls.splits, systems, cls.loss_fn)
        cls.evaluator.fit_all(verbose=False)
        cls.pristine = {k: clone_system(v) for k, v in systems.items()}   # fitted, never evaluated
        cls.metrics, cls.predictions = cls.evaluator.evaluate(verbose=False)

    def _fresh(self):
        return {k: clone_system(v) for k, v in self.pristine.items()}

    def test_steps_through_every_window_in_order(self) -> None:
        for model in ("mg_ecfm", "bp3"):
            m = self.metrics[self.metrics["model"] == model]
            self.assertEqual(m["window"].tolist(), list(range(1, N_TEST_WINDOWS + 1)))
            self.assertEqual(int(m["n_transactions"].sum()), len(self.splits.test))
            self.assertTrue((np.diff(m["start_step"].to_numpy()) > 0).all())
        self.assertEqual(list(self.metrics.columns), [c for c in RESULT_COLUMNS if c in self.metrics.columns])
        for col in METRIC_COLUMNS:
            self.assertIn(col, self.metrics.columns)

    def test_logged_metrics_are_valid(self) -> None:
        m = self.metrics
        for col in ("pr_auc", "roc_auc", "precision", "recall", "f1", "f2", "brier", "ece"):
            vals = m[col].dropna().to_numpy()
            self.assertTrue(np.all((vals >= 0.0) & (vals <= 1.0)), col)
        self.assertGreater(m["pr_auc"].notna().mean(), 0.5)
        self.assertTrue((m["expected_monetary_loss"] >= 0).all())
        self.assertTrue((m["eml_binary_policy"] >= 0).all())
        np.testing.assert_allclose(m["total_monetary_loss"], m["expected_monetary_loss"] * m["n_transactions"],
                                   rtol=1e-9)
        mg = m[m["model"] == "mg_ecfm"]
        self.assertTrue(mg["lambda_low"].notna().all())
        self.assertTrue((mg["lambda_low"] < mg["lambda_mid"]).all() and (mg["lambda_mid"] < mg["lambda_high"]).all())
        bp = m[m["model"] == "bp3"]
        self.assertTrue(bp["lambda_low"].isna().all())
        self.assertTrue(set(bp["threshold"].round(12)) == {round(bp["threshold"].iloc[0], 12)})

    def test_predictions_cover_each_test_transaction_once(self) -> None:
        for model in ("mg_ecfm", "bp3"):
            pr = self.predictions[self.predictions["model"] == model]
            self.assertEqual(len(pr), len(self.splits.test))
            np.testing.assert_array_equal(pr["tx_id"].to_numpy(), self.splits.test["tx_id"].to_numpy())
            np.testing.assert_array_equal(pr["y"].to_numpy(), self.splits.test["label"].to_numpy())
            self.assertTrue(np.all((pr["p"] >= 0) & (pr["p"] <= 1)))
            self.assertTrue(set(pr["action"].unique()) <= {int(a) for a in Action})

    def test_no_leakage_truncated_stream(self) -> None:
        """Scores of windows 1..K do not depend on whether later windows exist."""
        ev = DriftEvaluator(self.splits, self._fresh(), self.loss_fn)
        _, preds = ev.evaluate(self.splits.windows[:self.K], verbose=False)
        full = self.predictions[self.predictions["window"] <= self.K].reset_index(drop=True)
        for model in ("mg_ecfm", "bp3"):
            a = preds[preds["model"] == model].reset_index(drop=True)
            b = full[full["model"] == model].reset_index(drop=True)
            np.testing.assert_allclose(a["p"], b["p"], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(a["action"], b["action"])

    def test_no_leakage_from_future_features_and_labels(self) -> None:
        """Corrupting features and labels after window K leaves windows <= K + 1 + delay unchanged."""
        test = self.splits.test.copy()
        cut = self.splits.windows[self.K - 1].row_end
        rows = np.arange(len(test)) >= cut
        test.loc[rows, "label"] = 1 - test.loc[rows, "label"]
        for col in self.splits.feature_names:
            if pd.api.types.is_numeric_dtype(test[col]):
                test.loc[rows, col] = test.loc[rows, col] * 3.0 + 7.0
        test.loc[rows, "amount"] = test.loc[rows, "amount"] * 50.0
        corrupted = dataclasses.replace(self.splits, test=test)
        ev = DriftEvaluator(corrupted, self._fresh(), self.loss_fn)
        _, preds = ev.evaluate(verbose=False)
        unaffected = self.K + 1 + 1          # labels of window K are first used for window K + 2
        for model in ("mg_ecfm", "bp3"):
            a = preds[(preds["model"] == model) & (preds["window"] <= self.K)].reset_index(drop=True)
            b = self.predictions[(self.predictions["model"] == model)
                                 & (self.predictions["window"] <= self.K)].reset_index(drop=True)
            np.testing.assert_allclose(a["p"], b["p"], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(a["action"], b["action"])
        # label-only corruption: online components see flipped labels only after the delay
        test_l = self.splits.test.copy()
        test_l.loc[rows, "label"] = 1 - test_l.loc[rows, "label"]
        ev = DriftEvaluator(dataclasses.replace(self.splits, test=test_l), self._fresh(), self.loss_fn)
        _, preds_l = ev.evaluate(verbose=False)
        a = preds_l[(preds_l["model"] == "mg_ecfm") & (preds_l["window"] <= unaffected)]
        b = self.predictions[(self.predictions["model"] == "mg_ecfm") & (self.predictions["window"] <= unaffected)]
        np.testing.assert_allclose(a["p"].to_numpy(), b["p"].to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(a["action"].to_numpy(), b["action"].to_numpy())

    def test_label_release_schedule(self) -> None:
        mg = self.evaluator.systems["mg_ecfm"]
        delay = mg.label_delay_windows
        pending = sorted(w for w, _, _ in mg._pending)
        # windows W - delay .. W are still unverified after the last window
        self.assertEqual(len(pending), delay + 1)
        self.assertEqual(pending, [N_TEST_WINDOWS - delay + 1 + delay + i for i in range(delay + 1)])
        self.assertTrue(0.0 < mg.threshold_ <= 1.0)
        self.assertGreater(len(mg.gate_log), 0)
        self.assertTrue(mg.gate.store.verify_integrity()["ok"])

    def test_run_drift_evaluation_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = Path(tmp) / "drift_evaluation_results.csv"
            pred = Path(tmp) / "drift_evaluation_predictions.csv"
            man = Path(tmp) / "drift_evaluation_manifest.json"
            metrics, predictions, manifest = run_drift_evaluation(
                source="synthetic", profile="fast", n_windows=5, models=("bp3",),
                synthetic_kwargs={"n_steps": 80, "tx_per_step": (20, 30), "n_customers": 200,
                                  "n_merchants": 30, "seed": SEED},
                baseline_configs_override=tiny_baseline_configs(),
                results_path=res, predictions_path=pred, manifest_path=man, verbose=False)
            self.assertTrue(res.is_file() and pred.is_file() and man.is_file())
            loaded = pd.read_csv(res)
            self.assertEqual(len(loaded), 5)
            self.assertEqual(list(loaded.columns), list(RESULT_COLUMNS))
            meta = json.loads(man.read_text(encoding="utf-8"))
            self.assertEqual(meta["n_windows"], 5)
            self.assertIn("Menezes", meta["protocol"])


# --------------------------------------------------------------------------- #
# 2. Ablation runner                                                          #
# --------------------------------------------------------------------------- #
class TestAblationComponents(unittest.TestCase):
    """Component-level checks that need no detector training."""

    def test_assist_layer_is_causal_and_read_only(self) -> None:
        from experiments.run_ablation_study import AnalystAssistLayer
        splits = small_splits()
        names = splits.feature_names
        assist = AnalystAssistLayer(names, k=5).fit(splits.train)
        self.assertEqual(assist.index_size, len(splits.train))
        frame = splits.window_frame(splits.windows[0])
        n = len(frame)
        actions = np.full(n, int(Action.APPROVE))
        actions[:7] = int(Action.HOLD)
        out = WindowOutput(np.linspace(0, 1, n), np.zeros(n, int), actions.copy(), 0.5, 0.01)
        before = (out.p.copy(), out.yhat.copy(), out.actions.copy())
        expl = assist.explain(frame, out)
        self.assertEqual(len(expl), 7)                               # only HOLD rows explained
        for e in expl:
            self.assertTrue(0.0 < e.neighbour_fraud_share < 1.0)
            self.assertEqual(len(e.neighbour_ids), 5)
            self.assertTrue(set(e.neighbour_ids) <= set(splits.train["tx_id"].tolist()))  # causal index
        np.testing.assert_array_equal(out.p, before[0])
        np.testing.assert_array_equal(out.actions, before[2])
        assist.ingest(frame)
        self.assertEqual(assist.index_size, len(splits.train) + n)

    def test_recovery_accounting_rules(self) -> None:
        from experiments.run_ablation_study import recovery_accounting
        canon = pd.DataFrame({
            "tx_id": [0, 1, 2, 3, 4], "step": [1, 1, 2, 3, 3],
            "event_time": [3600.0, 3600.0, 7200.0, 10800.0, 10800.0],
            "src": ["C1", "C9", "C2", "C3", "C5"], "dst": ["C2", "C8", "C3", "C4", "C6"],
            "amount": [1000.0, 50.0, 400.0, 300.0, 200.0],
            "tx_type": ["TRANSFER", "PAYMENT", "TRANSFER", "CASH_OUT", "TRANSFER"],
            "label": [1, 0, 0, 1, 1]})
        preds = pd.DataFrame({"tx_id": [0, 1, 3, 4], "y": [1, 0, 1, 1],
                              "amount": [1000.0, 50.0, 300.0, 200.0],
                              "action": [int(Action.STEP_UP), int(Action.APPROVE),
                                         int(Action.APPROVE), int(Action.BLOCK)]})
        loss_fn = make_loss(PROJECT_CONFIG)
        beta = loss_fn.costs.beta_bypass
        rec = recovery_accounting(preds, canon, HYPERPARAMS, PROJECT_CONFIG, loss_fn, seed=SEED)
        self.assertAlmostEqual(rec["fraud_value"], 1500.0)
        self.assertAlmostEqual(rec["intercepted_value"], 200.0 + (1 - beta) * 1000.0)
        self.assertEqual(rec["n_recovery_cases"], 2.0)                # step-up fraud + approved cash-out
        self.assertEqual(rec["n_untraceable"], 1.0)                   # CASH_OUT leaves the ledger
        self.assertGreaterEqual(rec["recovered_value"], 0.0)
        self.assertLessEqual(rec["recovered_value"], beta * 1000.0 + 1e-9)
        self.assertTrue(0.0 <= rec["prevented_loss_ratio_pct"] <= 100.0)


@unittest.skipUnless(HAS_HEAVY, HEAVY_REASON)
class TestAblationRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from experiments.run_ablation_study import ARM_ORDER, run_ablation_study
        set_global_seed(SEED)
        cls.arm_order = ARM_ORDER
        cls.splits = small_splits()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "ablation_study_results.csv"
        cls.summary, cls.runs = run_ablation_study(
            splits=cls.splits, hyperparams_override=tiny_hyperparams(), poison_rate=0.10,
            results_path=cls.path, windows_path=Path(cls.tmp.name) / "ablation_study_windows.csv",
            verbose=False)
        cls.rows = cls.summary.set_index("arm")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_five_point_matrix_written(self) -> None:
        self.assertEqual(self.summary["arm"].tolist(), list(self.arm_order))
        self.assertEqual({a.split("_")[0] for a in self.arm_order}, {"B1", "B2", "B3", "B4", "B5"})
        saved = pd.read_csv(self.path)
        self.assertEqual(saved["arm"].tolist(), list(self.arm_order))
        for col in ("pr_auc", "roc_auc", "f1", "f2", "brier", "expected_monetary_loss",
                    "net_monetary_loss", "prevented_loss_ratio_pct"):
            self.assertIn(col, saved.columns)
            self.assertTrue(saved[col].notna().all(), col)

    def test_component_switches(self) -> None:
        r = self.rows
        b1 = self.runs["B1_xgboost_only"].system
        self.assertFalse(b1.classifier.use_relational or b1.classifier.use_ae or b1.classifier.use_kb)
        self.assertIsNone(b1.gate)
        self.assertEqual(type(b1.classifier.tabular.calibrator).__name__, "PlattCalibrator")
        self.assertEqual(r.loc["B1_xgboost_only", "step_up_rate"], 0.0)          # approve / block only
        self.assertTrue(math.isnan(r.loc["B1_xgboost_only", "kb_rules_active_end"]))
        b2 = self.runs["B2_graph_gate_off"].system
        self.assertTrue(b2.classifier.use_relational and b2.classifier.use_ae and b2.classifier.use_kb)
        self.assertIsNone(b2.gate)
        self.assertEqual(r.loc["B2_graph_gate_off", "kb_rules_active_end"], len(b2.candidates))
        full = self.runs["B4_full_mg_ecfm"].system
        self.assertIsNotNone(full.gate)
        self.assertLessEqual(r.loc["B4_full_mg_ecfm", "kb_rules_active_end"], len(full.candidates))
        for arm in ("B1_xgboost_only", "B2_graph_gate_off", "B3_static_crc", "B3_adaptive_crc"):
            self.assertEqual(r.loc[arm, "recovered_value"], 0.0)
            self.assertFalse(bool(r.loc[arm, "recovery"]))
        self.assertTrue(bool(r.loc["B4_full_mg_ecfm", "recovery"]))

    def test_static_vs_adaptive_conformal_control(self) -> None:
        static = self.runs["B3_static_crc"]
        adaptive = self.runs["B3_adaptive_crc"]
        self.assertEqual(static.system.router.controller.mode, "static")
        self.assertEqual(adaptive.system.router.controller.mode, "adaptive")
        for col in ("lambda_low", "lambda_mid", "lambda_high"):
            self.assertEqual(static.metrics[col].nunique(), 1, col)               # frozen thresholds
        self.assertGreater(adaptive.system.router.controller.n_adaptive,
                           static.system.router.controller.n_adaptive)             # online updates
        # identical detector (branched from one fit): ranking metrics are equal
        self.assertAlmostEqual(self.rows.loc["B3_static_crc", "pr_auc"], self.rows.loc["B3_adaptive_crc", "pr_auc"])

    def test_full_pipeline_recovery(self) -> None:
        r = self.rows
        self.assertAlmostEqual(r.loc["B4_full_mg_ecfm", "pr_auc"], r.loc["B3_adaptive_crc", "pr_auc"])
        self.assertAlmostEqual(r.loc["B4_full_mg_ecfm", "total_monetary_loss"],
                               r.loc["B3_adaptive_crc", "total_monetary_loss"])
        self.assertGreaterEqual(r.loc["B4_full_mg_ecfm", "recovered_value"], 0.0)
        self.assertGreaterEqual(r.loc["B4_full_mg_ecfm", "prevented_loss_ratio_pct"],
                                r.loc["B3_adaptive_crc", "prevented_loss_ratio_pct"] - 1e-9)
        self.assertTrue(0.0 <= r.loc["B4_full_mg_ecfm", "prevented_loss_ratio_pct"] <= 100.0)

    def test_assist_layer_is_decision_independent(self) -> None:
        r = self.rows
        self.assertEqual(r.loc["B5_assist_on", "decision_agreement_with_assist_off"], 1.0)
        self.assertEqual(r.loc["B5_assist_on", "max_abs_score_diff_vs_assist_off"], 0.0)
        for col in ("pr_auc", "f1", "expected_monetary_loss", "net_monetary_loss"):
            self.assertAlmostEqual(r.loc["B5_assist_on", col], r.loc["B5_assist_off", col])
        self.assertEqual(r.loc["B5_assist_on", "assist_explanations"], r.loc["B5_assist_on", "assist_hold_queue_size"])
        if r.loc["B5_assist_on", "assist_hold_queue_size"] > 0:
            for col in ("triage_accuracy_score_only", "triage_accuracy_assisted"):
                self.assertTrue(0.0 <= r.loc["B5_assist_on", col] <= 1.0)
        self.assertTrue(math.isnan(r.loc["B5_assist_off", "assist_explanations"]))


# --------------------------------------------------------------------------- #
# 3. Adversarial stress test                                                  #
# --------------------------------------------------------------------------- #
class TestAdversarialStressTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "adversarial_robustness_results.csv"
        cls.results, cls.ttr = run_adversarial_stress_test(
            n_candidates=80, n_cycles=12, firings_per_cycle=8, ville_horizon=600, n_ville_reps=4000,
            results_path=cls.path, ttr_path=Path(cls.tmp.name) / "ttr.csv", verbose=False)
        cls.gate = cls.results[cls.results["policy"] == "mg_ecfm_gate"].set_index("poison_rate")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_sweep_covers_all_rates_and_policies(self) -> None:
        self.assertEqual(len(self.results), len(POISON_RATES) * len(POLICIES))
        self.assertEqual(sorted(self.results["poison_rate"].unique()), list(POISON_RATES))
        self.assertEqual(set(self.results["policy"]), set(POLICIES))
        for rate in POISON_RATES:
            realised = self.gate.loc[rate, "realized_poison_rate"]
            self.assertAlmostEqual(realised, rate, delta=1.0 / 80 + 1e-9)
        self.assertTrue(pd.read_csv(self.path).shape == self.results.shape)

    def test_false_admission(self) -> None:
        auto = self.results[self.results["policy"] == "auto_admit"]
        self.assertTrue((auto["false_admission_rate"] == 1.0).all())
        for rate in POISON_RATES:
            row = self.gate.loc[rate]
            self.assertEqual(row["adversarial_false_admission_rate"], 0.0)
            self.assertLessEqual(row["false_admission_rate"], 0.15)
            self.assertGreater(row["genuine_admission_rate"], 0.5)
            self.assertTrue(bool(row["store_integrity_ok"]))

    def test_rollback_cycles_measured(self) -> None:
        k = GATE_CFG.max_rollback_cycles
        for rate in POISON_RATES:
            row = self.gate.loc[rate]
            self.assertEqual(row["adversarial_share_within_max_cycles"], 1.0)
            self.assertLessEqual(row["adversarial_time_to_rollback_max"], k)
            self.assertGreaterEqual(row["time_to_rollback_mean"], 1.0)
            self.assertEqual(row["harmful_active_after_injection_end"], 0)
        adv = self.ttr[self.ttr["category"].isin(["poisoned", "sybil"])]
        self.assertTrue(adv["rolled_back"].all())
        self.assertTrue((adv["time_to_rollback_cycles"] >= 1).all())
        np.testing.assert_array_equal(adv["firings_to_rollback"], adv["time_to_rollback_cycles"] * 8)
        others = self.results[self.results["policy"] != "mg_ecfm_gate"]
        self.assertTrue((others["rollback_rate_harmful"] == 0.0).all())

    def test_ville_maximal_inequality(self) -> None:
        a = GATE_CFG.alpha_roll
        for rate in POISON_RATES:
            row = self.gate.loc[rate]
            self.assertTrue(bool(row["ville_bound_satisfied"]), rate)
            self.assertLessEqual(row["ville_exact_crossing_prob"], a)
            self.assertLessEqual(row["ville_mc_ci_low"], a)
            self.assertLessEqual(row["ville_mc_ci_low"], row["ville_mc_rate"])
            self.assertLessEqual(row["ville_mc_rate"], row["ville_mc_ci_high"])
            self.assertAlmostEqual(row["ville_threshold"], 1.0 / a)
            self.assertGreater(row["ville_mc_pvalue_exceeds_alpha"], 0.05)

    def test_exact_binomial_helpers(self) -> None:
        from math import comb
        n, p = 12, 0.3
        for k in range(n + 1):
            brute = sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
            self.assertAlmostEqual(binomial_sf(k, n, p), brute, places=12)
            self.assertAlmostEqual(binomial_cdf(k, n, p) + binomial_sf(k + 1, n, p), 1.0, places=12)
        lo, hi = clopper_pearson(7, 40)
        self.assertLess(lo, 7 / 40)
        self.assertGreater(hi, 7 / 40)
        self.assertEqual(clopper_pearson(0, 10)[0], 0.0)
        self.assertEqual(clopper_pearson(10, 10)[1], 1.0)
        if HAS_SCIPY:
            from scipy.stats import beta
            self.assertAlmostEqual(lo, beta.ppf(0.025, 7, 34), places=6)
            self.assertAlmostEqual(hi, beta.ppf(0.975, 8, 33), places=6)


# --------------------------------------------------------------------------- #
# 4. Statistical significance                                                 #
# --------------------------------------------------------------------------- #
class TestStatisticalSignificance(unittest.TestCase):

    def test_mcnemar_hand_computed(self) -> None:
        # 10 transactions only A gets right, 2 only B gets right, 5 both right, 3 both wrong
        a = [1] * 10 + [0] * 2 + [1] * 5 + [0] * 3
        b = [0] * 10 + [1] * 2 + [1] * 5 + [0] * 3
        r = mcnemar_test(a, b)
        self.assertEqual(r["contingency_table"], [[5, 10], [2, 3]])
        self.assertEqual((r["b_discordant_a_correct"], r["c_discordant_b_correct"]), (10, 2))
        self.assertAlmostEqual(r["chi2"], (abs(10 - 2) - 1) ** 2 / 12)
        self.assertAlmostEqual(r["p_value"], math.erfc(math.sqrt(r["chi2"] / 2)))
        self.assertAlmostEqual(r["p_value"], 0.0433081428, places=8)
        self.assertAlmostEqual(r["p_value_exact_binomial"], 2 * sum(math.comb(12, i) for i in range(3)) / 2 ** 12)
        self.assertAlmostEqual(r["odds_ratio"], 5.0)
        self.assertLess(r["odds_ratio_ci_low"], 5.0)
        self.assertGreater(r["odds_ratio_ci_high"], 5.0)
        self.assertEqual(r["favours"], "A")
        swapped = mcnemar_test(b, a)
        self.assertAlmostEqual(swapped["chi2"], r["chi2"])
        self.assertAlmostEqual(swapped["odds_ratio"], 1 / 5.0)
        no_cc = mcnemar_test(a, b, continuity=False)
        self.assertAlmostEqual(no_cc["chi2"], 64 / 12)

    def test_mcnemar_degenerate_and_haldane(self) -> None:
        same = mcnemar_test([1, 0, 1, 1], [1, 0, 1, 1])
        self.assertEqual(same["chi2"], 0.0)
        self.assertEqual(same["p_value"], 1.0)
        self.assertEqual(same["p_value_exact_binomial"], 1.0)
        one_sided = mcnemar_test([1, 1, 1, 0], [0, 0, 1, 0])
        self.assertTrue(one_sided["haldane_correction"])
        self.assertAlmostEqual(one_sided["odds_ratio"], 2.5 / 0.5)
        with self.assertRaises(ValueError):
            mcnemar_test([1, 0], [1])

    def test_chi2_survival_function(self) -> None:
        self.assertAlmostEqual(chi2_sf_df1(3.841458820694124), 0.05, places=10)
        self.assertEqual(chi2_sf_df1(0.0), 1.0)
        self.assertEqual(chi2_sf_df1(float("inf")), 0.0)
        xs = np.linspace(0.01, 30, 60)
        vals = np.array([chi2_sf_df1(x) for x in xs])
        self.assertTrue(np.all((vals >= 0) & (vals <= 1)) and np.all(np.diff(vals) < 0))
        if HAS_SCIPY:
            from scipy.stats import chi2
            np.testing.assert_allclose(vals, chi2.sf(xs, 1), rtol=1e-10, atol=1e-15)

    def test_wilcoxon_exact_matches_reference(self) -> None:
        rng = np.random.default_rng(SEED)
        for n in (6, 15, 50):
            x = rng.normal(0.25, 1.0, n)
            y = rng.normal(0.0, 1.0, n)
            for alt in ("two-sided", "greater", "less"):
                r = wilcoxon_signed_rank(x, y, alt)
                self.assertEqual(r["method"], "exact")
                self.assertTrue(0.0 <= r["p_value"] <= 1.0)
                self.assertAlmostEqual(r["w_plus"] + r["w_minus"], n * (n + 1) / 2)
                if HAS_SCIPY:
                    from scipy.stats import wilcoxon
                    ref = wilcoxon(x, y, alternative=alt, method="exact")
                    self.assertAlmostEqual(r["p_value"], float(ref.pvalue), places=10)
                    self.assertAlmostEqual(r["statistic"], float(ref.statistic))
        # brute-force exact p-value for a small sample
        d = np.array([1.5, -0.4, 2.2, 0.9, -1.1, 3.0])
        r = wilcoxon_signed_rank(d, np.zeros(6), "greater")
        ranks = rankdata_average(np.abs(d))
        obs = ranks[d > 0].sum()
        count = sum(1 for m in range(64) if sum(ranks[i] for i in range(6) if m >> i & 1) >= obs)
        self.assertAlmostEqual(r["p_value"], count / 64)

    def test_wilcoxon_ties_zeros_and_normal_approximation(self) -> None:
        rng = np.random.default_rng(SEED + 1)
        x = np.round(rng.normal(0.3, 1.0, 70), 1)
        y = np.round(rng.normal(0.0, 1.0, 70), 1)
        r = wilcoxon_signed_rank(x, y)
        self.assertEqual(r["method"], "normal_approximation")
        self.assertTrue(0.0 <= r["p_value"] <= 1.0)
        self.assertEqual(r["n_nonzero"] + r["n_zero_differences"], 70)
        if HAS_SCIPY:
            from scipy.stats import wilcoxon
            ref = wilcoxon(x, y, method="approx")
            self.assertAlmostEqual(r["p_value"], float(ref.pvalue), places=10)
        nan_pairs = wilcoxon_signed_rank([0.5, np.nan, 0.7, 0.2], [0.1, 0.3, np.nan, 0.1])
        self.assertEqual(nan_pairs["n_pairs"], 2)
        zero = wilcoxon_signed_rank([0.4, 0.4], [0.4, 0.4])
        self.assertEqual(zero["p_value"], 1.0)
        self.assertEqual(rankdata_average(np.array([3.0, 1.0, 3.0, 2.0])).tolist(), [3.5, 1.0, 3.5, 2.0])
        counts = signed_rank_null_counts(10)
        self.assertEqual(counts.sum(), 2 ** 10)
        np.testing.assert_array_equal(counts, counts[::-1])

    def test_holm_bonferroni(self) -> None:
        raw = {"bp1": 0.01, "bp2": 0.04, "bp3": 0.03, "bp4": 0.5}
        adj = holm_bonferroni(raw)
        self.assertAlmostEqual(adj["bp1"], 0.04)
        self.assertAlmostEqual(adj["bp3"], 0.09)
        self.assertAlmostEqual(adj["bp2"], 0.09)
        self.assertAlmostEqual(adj["bp4"], 0.5)
        for k in raw:
            self.assertGreaterEqual(adj[k], raw[k])
            self.assertLessEqual(adj[k], 1.0)

    def test_block_bootstrap_ci(self) -> None:
        rng = np.random.default_rng(SEED)
        blocks = rng.normal(2.0, 1.0, 50)
        ci = block_bootstrap_ci(lambda idx: float(blocks[idx].mean()), 50, 1000, seed=SEED)
        self.assertEqual(ci["n_resamples"], 1000)
        self.assertLessEqual(ci["ci_low"], ci["estimate"])
        self.assertLessEqual(ci["estimate"], ci["ci_high"])
        self.assertLess(ci["ci_low"], 2.0)
        self.assertGreater(ci["ci_high"], 2.0)
        self.assertAlmostEqual(ci["bootstrap_se"], blocks.std(ddof=1) / math.sqrt(50), delta=0.05)
        again = block_bootstrap_ci(lambda idx: float(blocks[idx].mean()), 50, 1000, seed=SEED)
        self.assertEqual(ci, again)                                   # reproducible
        y = np.array([0, 1, 0, 1, 1, 0])
        p = np.array([0.1, 0.8, 0.3, 0.6, 0.9, 0.2])
        self.assertAlmostEqual(pooled_metric("pr_auc", y, p, (p >= .5).astype(int)), 1.0)
        self.assertAlmostEqual(pooled_metric("f1", y, p, (p >= .5).astype(int)), 1.0)

    def _synthetic_evaluation(self, n_windows: int = 20, per_window: int = 60):
        rng = np.random.default_rng(SEED)
        rows, metrics = [], []
        quality = {"mg_ecfm": 2.0, "bp1": 1.0, "bp2": 0.9, "bp3": 1.2, "bp4": 0.8}
        tx = 0
        for w in range(1, n_windows + 1):
            y = (rng.random(per_window) < 0.15).astype(int)
            y[0] = 1
            amount = rng.lognormal(8, 1, per_window)
            ids = np.arange(tx, tx + per_window)
            tx += per_window
            for m, q in quality.items():
                s = 1 / (1 + np.exp(-(q * (2 * y - 1) + rng.normal(0, 1.0, per_window))))
                yhat = (s >= 0.5).astype(int)
                act = np.where(yhat == 1, int(Action.BLOCK), int(Action.APPROVE))
                rows.append(pd.DataFrame({"model": m, "window": w, "tx_id": ids, "step": w, "y": y,
                                          "amount": amount, "p": s, "yhat": yhat, "action": act}))
                mm = classification_metrics(y, s, yhat)
                mm.update({"model": m, "window": w, "expected_monetary_loss": float(rng.random()),
                           "data_source": "unit-test"})
                metrics.append(mm)
        return pd.DataFrame(metrics), pd.concat(rows, ignore_index=True)

    def test_full_significance_report(self) -> None:
        results, preds = self._synthetic_evaluation()
        report = run_statistical_tests(results, preds, n_resamples=200)
        baselines = ["bp1", "bp2", "bp3", "bp4"]
        self.assertEqual(sorted(report["mcnemar"]), baselines)
        for m in baselines:
            mc = report["mcnemar"][m]
            self.assertGreaterEqual(mc["chi2"], 0.0)
            self.assertTrue(0.0 <= mc["p_value"] <= 1.0)
            self.assertTrue(0.0 <= mc["p_value_exact_binomial"] <= 1.0)
            self.assertGreaterEqual(mc["p_value_holm"], mc["p_value"])
            self.assertEqual(sum(map(sum, mc["contingency_table"])), len(preds) // 5)
            self.assertGreater(mc["odds_ratio"], 1.0)                 # proposed model is stronger
            wx = report["wilcoxon"]["pr_auc"][m]
            self.assertTrue(0.0 <= wx["p_value"] <= 1.0 and 0.0 <= wx["p_value_one_sided"] <= 1.0)
            self.assertLess(wx["p_value_one_sided"], 0.05)
            self.assertEqual(wx["n_pairs"], 20)
            for metric, ci in report["paired_difference_ci"][m].items():
                self.assertLessEqual(ci["ci_low"], ci["ci_high"], metric)
            self.assertGreater(report["paired_difference_ci"][m]["pr_auc"]["ci_low"], 0.0)
        for model, cis in report["bootstrap_ci"].items():
            for metric, ci in cis.items():
                self.assertLessEqual(ci["ci_low"], ci["ci_high"], (model, metric))
                self.assertEqual(ci["n_resamples"], 200)
        with tempfile.TemporaryDirectory() as tmp:
            rp, pp, out = Path(tmp) / "r.csv", Path(tmp) / "p.csv", Path(tmp) / "s.json"
            results.to_csv(rp, index=False)
            preds.to_csv(pp, index=False)
            self.assertEqual(stats_main(["--results", str(rp), "--predictions", str(pp), "--output", str(out),
                                         "--resamples", "100"]), 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("mcnemar", payload)
            self.assertIn("wilcoxon", payload)
            self.assertIn("bootstrap_ci", payload)
            self.assertEqual(stats_main(["--results", str(Path(tmp) / "missing.csv")]), 2)


if __name__ == "__main__":
    set_global_seed(SEED)
    unittest.main(verbosity=2)
