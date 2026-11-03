"""
tests/test_phase6_baselines.py
==============================

Phase 6 verification: the four re-implemented state-of-the-art baselines
(BP1 EARN+RXT-J, BP2 Lambda GNN+AE, BP3 CARE-GNN, BP4 ARGUS) share one
interface, train on synthetic and Phase-1 causal features, emit valid
probability distributions, report latency percentiles, and their key
mathematical components (SMOTE, Jaya, CARE top-p + RL thresholds, SSM
kernel, split-CRC FNR control, MC dropout) behave as specified.

Run from the repository root:

    python tests/test_phase6_baselines.py          # unittest runner
    python -m pytest tests/test_phase6_baselines.py -q
"""
from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from src.baselines import (  # noqa: E402
    BASELINE_REGISTRY,
    CAREGNN,
    EARNRXTJ,
    ARGUSConformal,
    BaselineModel,
    LambdaGNNAutoencoder,
    crc_fnr_threshold,
    get_baseline_model,
    jaya_minimize,
    resolve_baseline_name,
    smote_oversample,
)
from src.baselines.argus_conformal import (  # noqa: E402
    DiagonalSSMKernel,
    causal_toeplitz_conv,
)
from src.baselines.base import (  # noqa: E402
    KNNNeighborIndex,
    SignedLogStandardScaler,
    chronological_holdout,
    cost_sensitive_class_weights,
)
from src.baselines.care_gnn import CARELayer, RelationThreshold  # noqa: E402
from src.data.causal_preprocessor import SEED, CausalFeatureEngineer, set_global_seed  # noqa: E402
from src.data.dataset_loaders import make_synthetic_paysim_frame  # noqa: E402
from src.detection.fusion_classifier import StackedFusionClassifier, load_hyperparams  # noqa: E402

import pandas as pd  # noqa: E402

# Small but non-trivial configurations so the full suite runs in about a minute on CPU.
FAST_CONFIGS = {
    "bp1": {"earn_epochs": 4, "rxt_epochs": 4, "batch_size": 128, "ae_hidden": 32,
            "resnet_width": 32, "resnet_embedding": 16, "rxt_tokens": 4, "rxt_channels": 16,
            "cardinality": 4, "bottleneck_width": 4, "gru_hidden": 32, "smote_ratio": 0.5,
            "jaya_population": 6, "jaya_iterations": 5},
    "bp2": {"k_neighbors": 5, "gnn_hidden": 32, "gnn_epochs": 20, "gnn_batch_size": 128,
            "ae_epochs": 10, "ae_batch_size": 256, "patience": 8},
    "bp3": {"k_neighbors": 5, "embed_dim": 16, "epochs": 6, "batch_size": 256,
            "learning_rate": 0.01},
    "bp4": {"d_model": 16, "n_heads": 2, "n_blocks": 1, "ssm_state": 8, "epochs": 25,
            "batch_size": 64, "mc_samples": 8, "patience": 8},
}
MODEL_CLASSES = {"bp1": EARNRXTJ, "bp2": LambdaGNNAutoencoder, "bp3": CAREGNN,
                 "bp4": ARGUSConformal}


def make_synthetic(n: int = 1500, d: int = 12, fraud_rate: float = 0.1, seed: int = SEED):
    """Time-ordered tabular stream with a learnable non-linear fraud signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[:, 0] = np.exp(rng.normal(3.0, 1.2, size=n))          # heavy-tailed "amount"
    logit = 1.6 * X[:, 1] - 1.2 * X[:, 2] + 0.9 * X[:, 3] * X[:, 4] + 0.4 * np.log(X[:, 0])
    shift = np.quantile(logit, 1 - fraud_rate)
    y = (logit + rng.normal(scale=0.6, size=n) > shift).astype(np.int64)
    return X, y


def phase1_features(n_steps: int = 50, seed: int = SEED):
    """Phase-1 leakage-free causal features for a synthetic PaySim stream."""
    raw = make_synthetic_paysim_frame(n_steps=n_steps, seed=seed, tx_per_step=(25, 45),
                                      n_customers=250, n_merchants=40)
    canon = pd.DataFrame({
        "tx_id": np.arange(len(raw), dtype=np.int64),
        "step": raw["step"].to_numpy(dtype=np.int64),
        "src": raw["nameOrig"].astype(str).to_numpy(),
        "dst": raw["nameDest"].astype(str).to_numpy(),
        "amount": raw["amount"].to_numpy(dtype=np.float64),
        "tx_type": raw["type"].astype(str).to_numpy(),
        "label": raw["isFraud"].to_numpy(dtype=np.int8),
    })
    engineer = CausalFeatureEngineer()
    frame = engineer.transform(canon)
    return frame, engineer.feature_names


def small_phase2_hyperparams() -> dict:
    hp = copy.deepcopy(load_hyperparams())
    hp["graphsage"].update({"hidden_dim": 32, "out_dim": 16})
    return hp


# --------------------------------------------------------------------------- #
# 1. Training, probability validity and predictive skill                      #
# --------------------------------------------------------------------------- #
class TestBaselinesTrainAndPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_global_seed(SEED)
        X, y = make_synthetic()
        cls.X_train, cls.y_train = X[:1100], y[:1100]
        cls.X_test, cls.y_test = X[1100:], y[1100:]
        cls.models = {}
        cls.probas = {}
        for key in sorted(BASELINE_REGISTRY):
            model = get_baseline_model(key, FAST_CONFIGS[key])
            returned = model.fit(cls.X_train, cls.y_train)
            cls.models[key] = (model, returned)
            cls.probas[key] = model.predict_proba(cls.X_test)

    def test_fit_returns_self(self) -> None:
        for key, (model, returned) in self.models.items():
            with self.subTest(model=key):
                self.assertIs(returned, model)
                self.assertTrue(model.is_fitted_)
                self.assertEqual(model.n_features_in_, self.X_train.shape[1])
                self.assertGreater(model.count_parameters(), 0)

    def test_predict_proba_is_valid_distribution(self) -> None:
        for key, proba in self.probas.items():
            with self.subTest(model=key):
                self.assertIsInstance(proba, np.ndarray)
                self.assertEqual(proba.shape, (self.X_test.shape[0], 2))
                self.assertTrue(np.all(np.isfinite(proba)))
                self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_predict_is_thresholded_proba(self) -> None:
        for key, (model, _) in self.models.items():
            with self.subTest(model=key):
                pred = model.predict(self.X_test)
                self.assertEqual(pred.shape, (self.X_test.shape[0],))
                self.assertTrue(np.issubdtype(pred.dtype, np.integer))
                self.assertTrue(set(np.unique(pred)).issubset({0, 1}))
                expected = (self.probas[key][:, 1] >= model.threshold_).astype(np.int64)
                np.testing.assert_array_equal(pred, expected)

    def test_models_learn_the_fraud_signal(self) -> None:
        for key, proba in self.probas.items():
            with self.subTest(model=key):
                auc = roc_auc_score(self.y_test, proba[:, 1])
                self.assertGreater(auc, 0.75, f"{key} ROC-AUC {auc:.3f}")

    def test_inference_is_deterministic(self) -> None:
        for key, (model, _) in self.models.items():
            with self.subTest(model=key):
                np.testing.assert_allclose(model.predict_proba(self.X_test), self.probas[key])

    def test_training_is_reproducible_with_seed(self) -> None:
        for key in ("bp1", "bp4"):
            with self.subTest(model=key):
                twin = get_baseline_model(key, FAST_CONFIGS[key]).fit(self.X_train, self.y_train)
                np.testing.assert_allclose(twin.predict_proba(self.X_test), self.probas[key],
                                           atol=1e-6)

    def test_single_row_scoring(self) -> None:
        for key, (model, _) in self.models.items():
            with self.subTest(model=key):
                p_row = model.predict_proba(self.X_test[3])
                self.assertEqual(p_row.shape, (1, 2))
                np.testing.assert_allclose(p_row.sum(), 1.0, atol=1e-9)
                if key != "bp4":   # MC-dropout masks depend on the batch composition
                    np.testing.assert_allclose(p_row[0], self.probas[key][3], atol=1e-6)

    def test_fit_reports(self) -> None:
        for key, (model, _) in self.models.items():
            with self.subTest(model=key):
                rep = model.fit_report_
                self.assertEqual(rep["n_train"], self.X_train.shape[0])
                self.assertGreater(rep["fit_seconds"], 0.0)
        bp1 = self.models["bp1"][0].fit_report_
        self.assertGreater(bp1["n_synthetic"], 0)
        self.assertLessEqual(bp1["jaya_best_loss"], bp1["jaya_start_loss"] + 1e-12)
        bp2 = self.models["bp2"][0].fit_report_
        self.assertEqual(bp2["combiner_fit_on"], "holdout_tail")
        bp3 = self.models["bp3"][0]
        for row in bp3.relation_thresholds:
            for p in row:
                self.assertTrue(0.0 < p <= 1.0)
        bp4 = self.models["bp4"][0].fit_report_
        self.assertTrue(bp4["crc"]["feasible"])
        self.assertEqual(self.models["bp4"][0].threshold_, bp4["crc"]["t_hat"])


# --------------------------------------------------------------------------- #
# 2. Latency harness                                                          #
# --------------------------------------------------------------------------- #
class TestLatency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        X, y = make_synthetic(n=900)
        cls.X = X
        cls.models = {k: get_baseline_model(k, FAST_CONFIGS[k]).fit(X, y)
                      for k in sorted(BASELINE_REGISTRY)}

    def test_latency_percentiles(self) -> None:
        for key, model in self.models.items():
            with self.subTest(model=key):
                lat = model.measure_latency(self.X[:50], n_runs=40)
                for q in ("p50", "p95", "p99", "mean", "max"):
                    self.assertIn(q, lat)
                    self.assertTrue(math.isfinite(lat[q]) and lat[q] > 0.0)
                self.assertLessEqual(lat["p50"], lat["p95"])
                self.assertLessEqual(lat["p95"], lat["p99"])
                self.assertLessEqual(lat["p99"], lat["max"] + 1e-12)
                self.assertEqual(lat["n_runs"], 40)
                self.assertEqual(lat["batch_size"], 1)
                self.assertLess(lat["p50"], 2000.0)          # sanity: < 2 s per transaction

    def test_default_n_runs_and_batched_mode(self) -> None:
        model = self.models["bp3"]
        lat = model.measure_latency(self.X[:20])
        self.assertEqual(lat["n_runs"], 100)
        batched = model.measure_latency(self.X, n_runs=5, batch_size=64)
        self.assertEqual(batched["batch_size"], 64)

    def test_latency_argument_validation(self) -> None:
        model = self.models["bp1"]
        with self.assertRaises(ValueError):
            model.measure_latency(self.X, n_runs=0)
        with self.assertRaises(ValueError):
            model.measure_latency(self.X, batch_size=0)
        with self.assertRaises(RuntimeError):
            get_baseline_model("bp1").measure_latency(self.X)


# --------------------------------------------------------------------------- #
# 3. Feature dimensionality vs Phase 1 / Phase 2 outputs                      #
# --------------------------------------------------------------------------- #
class TestPhaseFeatureCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_global_seed(SEED)
        cls.frame, cls.names = phase1_features()
        steps = cls.frame["step"].to_numpy()
        cls.train_mask = steps <= 38
        cls.X = cls.frame[cls.names].to_numpy(dtype=np.float64)
        cls.y = cls.frame["label"].to_numpy(dtype=np.int64)
        fusion = StackedFusionClassifier(cls.names, hyperparams=small_phase2_hyperparams())
        cls.fusion_names = fusion.fusion_feature_names

    def test_phase1_frame_has_both_classes_in_training_window(self) -> None:
        y_tr = self.y[self.train_mask]
        self.assertGreater(int(y_tr.sum()), 5)
        self.assertGreater(int((y_tr == 0).sum()), 100)

    def test_baselines_on_phase1_causal_features(self) -> None:
        X_tr, y_tr = self.X[self.train_mask], self.y[self.train_mask]
        X_te = self.X[~self.train_mask]
        for key in sorted(BASELINE_REGISTRY):
            with self.subTest(model=key):
                model = get_baseline_model(key, FAST_CONFIGS[key]).fit(X_tr, y_tr)
                self.assertEqual(model.n_features_in_, len(self.names))
                proba = model.predict_proba(X_te)
                self.assertEqual(proba.shape, (X_te.shape[0], 2))
                self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
                with self.assertRaises(ValueError):
                    model.predict_proba(X_te[:, :-1])
                with self.assertRaises(ValueError):
                    model.predict_proba(np.hstack([X_te, X_te[:, :1]]))

    def test_baselines_on_phase2_fusion_width(self) -> None:
        d_fusion = len(self.fusion_names)
        k = small_phase2_hyperparams()["graphsage"]["out_dim"]
        self.assertEqual(d_fusion, len(self.names) + 2 * k + 1 + 5)
        rng = np.random.default_rng(SEED)
        extra = rng.normal(size=(self.X.shape[0], d_fusion - len(self.names)))
        Z = np.hstack([self.X, extra])
        X_tr, y_tr = Z[self.train_mask], self.y[self.train_mask]
        for key in sorted(BASELINE_REGISTRY):
            with self.subTest(model=key):
                cfg = dict(FAST_CONFIGS[key])
                if key == "bp3":
                    cfg["n_relations"] = 4
                model = get_baseline_model(key, cfg).fit(X_tr, y_tr)
                self.assertEqual(model.n_features_in_, d_fusion)
                proba = model.predict_proba(Z[~self.train_mask][:64])
                self.assertEqual(proba.shape, (64, 2))
                np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-9)

    def test_care_gnn_relation_views_cover_all_columns(self) -> None:
        model = get_baseline_model("bp3", FAST_CONFIGS["bp3"])
        views = model._resolve_views(len(self.names))
        self.assertEqual(len(views), 3)
        np.testing.assert_array_equal(np.sort(np.concatenate(views)), np.arange(len(self.names)))
        explicit = CAREGNN({"relation_views": [[0, 1], [2, 3, 4]]})
        self.assertEqual([v.tolist() for v in explicit._resolve_views(6)], [[0, 1], [2, 3, 4]])
        with self.assertRaises(ValueError):
            CAREGNN({"relation_views": [[0, 99]]})._resolve_views(6)


# --------------------------------------------------------------------------- #
# 4. Unified interface contract                                               #
# --------------------------------------------------------------------------- #
class TestUnifiedInterface(unittest.TestCase):
    def test_registry_and_signatures(self) -> None:
        import inspect

        self.assertEqual(sorted(BASELINE_REGISTRY), ["bp1", "bp2", "bp3", "bp4"])
        for key, cls in BASELINE_REGISTRY.items():
            with self.subTest(model=key):
                self.assertIs(cls, MODEL_CLASSES[key])
                self.assertTrue(issubclass(cls, BaselineModel))
                self.assertEqual(list(inspect.signature(cls.fit).parameters), ["self", "X", "y"])
                self.assertEqual(list(inspect.signature(cls.predict_proba).parameters),
                                 ["self", "X"])
                self.assertEqual(list(inspect.signature(cls.predict).parameters), ["self", "X"])
                lat = inspect.signature(cls.measure_latency).parameters
                self.assertEqual(lat["n_runs"].default, 100)
                self.assertTrue(cls.reference)

    def test_factory_aliases_and_config_routing(self) -> None:
        self.assertEqual(resolve_baseline_name("CARE-GNN"), "bp3")
        self.assertEqual(resolve_baseline_name("argus"), "bp4")
        self.assertEqual(resolve_baseline_name("earn_rxt_j"), "bp1")
        self.assertEqual(resolve_baseline_name("lambda_gnn_ae"), "bp2")
        self.assertIsInstance(get_baseline_model("bp2"), LambdaGNNAutoencoder)
        model = get_baseline_model("bp3", {"bp3": {"epochs": 7}, "bp4": {"epochs": 1}})
        self.assertEqual(model.config.epochs, 7)
        with self.assertRaises(KeyError):
            get_baseline_model("xgboost")
        with self.assertRaises(KeyError):
            get_baseline_model("bp1", {"not_a_field": 1})

    def test_unfitted_and_invalid_inputs(self) -> None:
        X, y = make_synthetic(n=300)
        for key in sorted(BASELINE_REGISTRY):
            with self.subTest(model=key):
                model = get_baseline_model(key, FAST_CONFIGS[key])
                with self.assertRaises(RuntimeError):
                    model.predict_proba(X)
                with self.assertRaises(RuntimeError):
                    model.predict(X)
                with self.assertRaises(ValueError):
                    model.fit(X, np.zeros(300, dtype=int))           # one class only
                with self.assertRaises(ValueError):
                    model.fit(X, np.full(300, 2))                     # non-binary
                with self.assertRaises(ValueError):
                    model.fit(X, y[:-1])                              # length mismatch
                bad = X.copy()
                bad[0, 0] = np.nan
                with self.assertRaises(ValueError):
                    model.fit(bad, y)

    def test_accepts_lists_and_pandas(self) -> None:
        X, y = make_synthetic(n=600)
        model = get_baseline_model("bp3", FAST_CONFIGS["bp3"]).fit(pd.DataFrame(X), list(y))
        p_list = model.predict_proba(X[:5].tolist())
        p_np = model.predict_proba(X[:5])
        np.testing.assert_allclose(p_list, p_np)


# --------------------------------------------------------------------------- #
# 5. Component mathematics                                                    #
# --------------------------------------------------------------------------- #
class TestComponents(unittest.TestCase):
    def test_smote_interpolates_between_fraud_neighbours(self) -> None:
        rng = np.random.default_rng(SEED)
        X = np.vstack([rng.normal(size=(200, 2)), np.array([[5.0, 5.0], [6.0, 5.0], [5.0, 6.0]])])
        y = np.r_[np.zeros(200, int), np.ones(3, int)]
        Xo, yo = smote_oversample(X, y, ratio=0.5, k=2, rng=rng, max_synthetic=10_000)
        self.assertEqual(int(yo.sum()), 100)
        np.testing.assert_array_equal(Xo[:203], X)
        synth = Xo[203:]
        # every synthetic point lies inside the convex hull (triangle) of the frauds
        self.assertTrue(np.all(synth[:, 0] >= 5.0 - 1e-9) and np.all(synth[:, 1] >= 5.0 - 1e-9))
        self.assertTrue(np.all(synth.sum(axis=1) <= 11.0 + 1e-9))
        Xs, ys = smote_oversample(X[:201], y[:201], 0.5, 2, rng, 100)  # one fraud: no-op
        self.assertEqual(Xs.shape[0], 201)

    def test_jaya_minimises_sphere(self) -> None:
        theta0 = np.full(6, 3.0)
        best, rep = jaya_minimize(lambda t: float(np.sum(t ** 2)), theta0, population=15,
                                  iterations=80, init_scale=0.5, rng=np.random.default_rng(SEED))
        self.assertLess(rep["jaya_best_loss"], 0.05 * rep["jaya_start_loss"])
        self.assertAlmostEqual(rep["jaya_best_loss"], float(np.sum(best ** 2)))

    def test_crc_threshold_hand_computed(self) -> None:
        scores = np.arange(1, 20) / 20.0              # 19 calibration frauds
        out = crc_fnr_threshold(scores, alpha=0.2)    # budget = 0.2*20 - 1 = 3 misses
        self.assertTrue(out["feasible"])
        self.assertEqual(out["allowed_misses"], 3)
        self.assertAlmostEqual(out["t_hat"], 0.20)    # scores 0.05, 0.10, 0.15 below
        self.assertEqual(int((scores < out["t_hat"]).sum()), 3)
        self.assertFalse(crc_fnr_threshold(scores[:3], alpha=0.1)["feasible"])
        self.assertEqual(crc_fnr_threshold(scores[:3], alpha=0.1)["t_hat"], 0.0)

    def test_crc_controls_expected_fnr(self) -> None:
        rng = np.random.default_rng(SEED)
        alpha, misses = 0.1, []
        for _ in range(400):
            cal = rng.beta(4, 2, size=60)
            test = rng.beta(4, 2, size=200)
            t_hat = crc_fnr_threshold(cal, alpha)["t_hat"]
            misses.append(float((test < t_hat).mean()))
        self.assertLessEqual(np.mean(misses), alpha + 0.01)
        self.assertGreater(np.mean(misses), alpha - 0.05)   # not trivially conservative

    def test_care_rl_threshold_policy(self) -> None:
        ctl = RelationThreshold(p=0.5, tau=0.02, window=10, stop_abs_sum=2)
        ctl.update(1.0)
        ctl.update(0.8)                 # distance decreased -> +tau
        self.assertAlmostEqual(ctl.p, 0.52)
        ctl.update(0.9)                 # increased -> -tau
        self.assertAlmostEqual(ctl.p, 0.50)
        for i in range(20):             # alternating rewards -> converges & freezes
            ctl.update(0.5 if i % 2 else 0.6)
        self.assertTrue(ctl.converged)
        frozen = ctl.p
        ctl.update(0.0)
        self.assertEqual(ctl.p, frozen)

    def test_care_top_p_keeps_most_similar_neighbours(self) -> None:
        layer = CARELayer(d_in=1, d_out=1, n_relations=1, dropout=0.0)
        with torch.no_grad():
            layer.intra[0].weight.fill_(1.0)
            layer.intra[0].bias.zero_()
        h_c = torch.zeros(1, 1)
        h_nb = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        s_nb = torch.tensor([[0.50, 0.90, 0.10, 0.55]])
        with torch.no_grad():
            s_c = layer.fraud_score(h_c)[1]
        s_nb = s_nb - 0.5 + s_c            # distances 0, 0.4, 0.4, 0.05 -> keep 1st and 4th
        _, _, dist = layer(h_c, [h_nb], p=[0.5], s_nbs=[s_nb])
        self.assertAlmostEqual(dist[0], 0.025, places=5)
        captured = {}

        def grab(module, inputs, output):
            captured["x"] = inputs[0].detach().clone()

        hook = layer.intra[0].register_forward_hook(grab)
        layer(h_c, [h_nb], p=[0.5], s_nbs=[s_nb])
        hook.remove()
        self.assertAlmostEqual(float(captured["x"]), 2.5, places=5)   # mean of 1 and 4

    def test_ssm_kernel_and_causal_conv(self) -> None:
        torch.manual_seed(SEED)
        kern = DiagonalSSMKernel(channels=3, state=4)
        K = kern(12)
        self.assertEqual(tuple(K.shape), (12, 3))
        with torch.no_grad():
            dt = torch.exp(kern.log_dt)[:, None]
            a_bar = torch.exp(-dt * torch.exp(kern.A_log))
            bound = (kern.C * kern.B * dt).abs().sum(-1)                   # |K_t| <= sum|C B_bar|
        self.assertTrue(torch.all((a_bar > 0) & (a_bar < 1)))           # stable poles
        self.assertTrue(torch.all(K.abs() <= bound[None] + 1e-6))
        u = torch.randn(2, 12, 3)
        y = causal_toeplitz_conv(u, K)
        naive = torch.zeros_like(u)
        for i in range(12):
            for j in range(i + 1):
                naive[:, i] += K[i - j] * u[:, j]
        torch.testing.assert_close(y, naive, rtol=1e-5, atol=1e-6)

    def test_argus_mc_dropout_and_selective_prediction(self) -> None:
        X, y = make_synthetic(n=900)
        model = get_baseline_model("bp4", FAST_CONFIGS["bp4"]).fit(X[:700], y[:700])
        out = model.predict_with_uncertainty(X[700:])
        self.assertTrue(np.all(out["std"] >= 0) and np.any(out["std"] > 0))
        self.assertTrue(np.all(out["mutual_information"] >= 0))
        self.assertTrue(np.all(out["entropy"] <= math.log(2) + 1e-9))
        np.testing.assert_allclose(out["mean"], model.predict_proba(X[700:])[:, 1])
        sel = model.predict_selective(X[700:])
        self.assertTrue(set(np.unique(sel)).issubset({-1, 0, 1}))
        self.assertEqual(int((sel == -1).sum()), int((out["std"] > model.abstain_std_).sum()))

    def test_lambda_speed_layer_flags_out_of_distribution(self) -> None:
        X, y = make_synthetic(n=900)
        model = get_baseline_model("bp2", FAST_CONFIGS["bp2"]).fit(X, y)
        normal = model.layer_scores(X[y == 0][:100])["ae_log_recon"]
        odd = X[:100].copy()
        odd[:, 5:] = 25.0
        self.assertGreater(np.median(model.layer_scores(odd)["ae_log_recon"]), np.median(normal))

    def test_knn_excludes_self_even_with_duplicates(self) -> None:
        Z = np.array([[0.0], [0.0], [1.0], [2.0], [3.0]])
        idx = KNNNeighborIndex(k=2).fit(Z)
        nb = idx.train_neighbors(Z)
        self.assertEqual(nb.shape, (5, 2))
        for i in range(5):
            self.assertNotIn(i, nb[i].tolist())
        self.assertEqual(KNNNeighborIndex(k=10).fit(Z).k, 4)

    def test_shared_preprocessing_helpers(self) -> None:
        X = np.array([[1.0, -1e6], [10.0, 0.0], [1e5, 1e6]])
        z = SignedLogStandardScaler().fit(X).transform(X)
        np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-6)
        head, tail = chronological_holdout(10, 0.2)
        np.testing.assert_array_equal(tail, [8, 9])
        np.testing.assert_array_equal(head, np.arange(8))
        w = cost_sensitive_class_weights(np.r_[np.zeros(90, int), np.ones(10, int)], 25.0, 1.0)
        self.assertAlmostEqual(float(w[1] / w[0]), 25.0 * 9.0, places=4)
        self.assertAlmostEqual(float((w[0] * 90 + w[1] * 10) / 100), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
