"""
tests/test_phase2_graph_detection.py
====================================

Phase 2 verification: temporal-memory decay and O(1) access, label-aware
camouflage edge pruning, filtered GraphSAGE message passing, stream
causality, denoising-autoencoder residuals, cost-sensitive calibrated XGBoost
(isotonic + exact inductive Venn-Abers), stacked fusion vector layout and the
<= 25 ms real-time latency budget.

Run from the repository root:

    python tests/test_phase2_graph_detection.py          # unittest runner
    python -m pytest tests/test_phase2_graph_detection.py -q
"""
from __future__ import annotations

import copy
import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.causal_preprocessor import (  # noqa: E402
    SEED,
    CausalFeatureEngineer,
    LeakageError,
    set_global_seed,
)
from src.data.dataset_loaders import make_synthetic_paysim_frame  # noqa: E402
from src.data.perturbation_generator import PerturbationConfig, PerturbationGenerator  # noqa: E402
from src.detection.autoencoder_residual import (  # noqa: E402
    AutoencoderConfig,
    AutoencoderResidualEngine,
)
from src.detection.fusion_classifier import (  # noqa: E402
    KnowledgeBaseTriggerEncoder,
    StackedFusionClassifier,
    frame_to_events,
    load_hyperparams,
)
from src.detection.tabular_engine import (  # noqa: E402
    CalibratedXGBoostEngine,
    IsotonicCalibrator,
    TabularEngineConfig,
    VennAbersCalibrator,
    brier_score,
)
from src.graph.camouflage_graphsage import (  # noqa: E402
    HAS_PYG,
    CamouflageResistantGraphSAGE,
    FilteredSAGEConv,
    GraphSAGEConfig,
    scatter_aggregate,
)
from src.graph.similarity_filter import (  # noqa: E402
    CamouflageSimilarityFilter,
    SimilarityFilterConfig,
)
from src.graph.temporal_memory import (  # noqa: E402
    SignedLogStandardizer,
    TemporalMemoryConfig,
    TemporalNodeMemory,
)

LATENCY_BUDGET_MS = 25.0
HYPERPARAMS = load_hyperparams()


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _processed_frame(n_steps: int = 60, seed: int = SEED, **kwargs) -> pd.DataFrame:
    """Synthetic PaySim stream -> canonical columns -> Phase-1 causal features."""
    raw = make_synthetic_paysim_frame(n_steps=n_steps, seed=seed, **kwargs)
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
    return engineer.transform(canon), engineer.feature_names


def _small_hyperparams() -> dict:
    hp = copy.deepcopy(HYPERPARAMS)
    hp["temporal_memory"]["memory_dim"] = 16
    hp["similarity_filter"]["hidden_dim"] = 0
    hp["similarity_filter"]["min_neighbors_kept"] = 0
    hp["graphsage"].update({"hidden_dim": 32, "out_dim": 16, "fanouts": [5, 3],
                            "epochs": 2, "batch_size": 256})
    hp["autoencoder"].update({"epochs": 5, "batch_size": 256})
    hp["xgboost"].update({"n_estimators": 120, "early_stopping_rounds": 30, "max_depth": 4})
    return hp


def _memory(feature_dim: int = 3, delta: float = 0.1, staleness=None, amount: bool = False,
            capacity: int = 4) -> TemporalNodeMemory:
    cfg = TemporalMemoryConfig(feature_dim=feature_dim, memory_dim=8, decay_delta=delta,
                               message_includes_amount=amount, max_staleness_steps=staleness,
                               initial_capacity=capacity)
    return TemporalNodeMemory(cfg, seed=SEED)


def _camouflage_states(d: int = 8):
    """Fraud centre 0, fraud-like neighbours 1-2, camouflage (benign) neighbours 3-5."""
    states = torch.zeros(6, d)
    states[1] = 0.1
    states[2] = -0.1
    states[3:] = 3.0
    edge_index = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]])
    labels = torch.tensor([1, 1, 1, 0, 0, 0])
    return states, edge_index, labels


def _handcrafted_filter(d: int = 8, p: float = 0.5) -> CamouflageSimilarityFilter:
    filt = CamouflageSimilarityFilter(SimilarityFilterConfig(memory_dim=d, threshold_p=p))
    with torch.no_grad():
        filt.net[0].weight.fill_(-1.0)
        filt.net[0].bias.fill_(2.0)
    return filt


# --------------------------------------------------------------------------- #
# 1. Temporal node memory                                                     #
# --------------------------------------------------------------------------- #
class TestTemporalMemory(unittest.TestCase):
    def setUp(self) -> None:
        set_global_seed(SEED)

    def test_exponential_decay_formula(self) -> None:
        mem = _memory(delta=0.1)
        idx = mem.register(["acct"])
        m = torch.tensor([[1.0, -2.0, 0.5]])
        s0 = mem.update(idx, 0.0, m).detach()[0]
        np.testing.assert_allclose(s0.numpy(), mem.W_mem(m).detach()[0].numpy(), rtol=1e-6)
        for dt in (1.0, 5.0, 10.0, 37.5):
            expected = s0 * math.exp(-0.1 * dt)
            np.testing.assert_allclose(mem.read(idx, dt)[0].numpy(), expected.numpy(), rtol=1e-5)
            np.testing.assert_allclose(mem.read_one(0, dt).numpy(), expected.numpy(), rtol=1e-5)
        s1 = mem.update(idx, 10.0, m).detach()[0]
        np.testing.assert_allclose(s1.numpy(), (s0 * math.exp(-1.0) + s0).numpy(), rtol=1e-5)
        self.assertEqual(int(mem.n_updates[0]), 2)
        self.assertEqual(float(mem.last_update[0]), 10.0)

    def test_unseen_and_stale_nodes_read_zero(self) -> None:
        mem = _memory(delta=0.01, staleness=5.0)
        idx = mem.register(["a", "b"])
        self.assertTrue(torch.all(mem.read(idx, 3.0) == 0))
        mem.update(idx[:1], 0.0, torch.ones(1, 3))
        self.assertTrue(torch.any(mem.read(idx[:1], 5.0) != 0))
        self.assertTrue(torch.all(mem.read(idx[:1], 5.5) == 0))
        self.assertTrue(torch.all(mem.read_one(0, 6.0) == 0))

    def test_batch_commit_equals_sequential_updates(self) -> None:
        rng = np.random.default_rng(SEED)
        n_events = 60
        nodes = rng.integers(0, 7, n_events)
        times = np.sort(rng.uniform(0, 30, n_events))
        msgs = torch.from_numpy(rng.normal(size=(n_events, 3)).astype(np.float32))
        batch_mem, seq_mem = _memory(delta=0.07), _memory(delta=0.07)
        keys = [f"n{i}" for i in range(7)]
        batch_mem.register(keys)
        seq_mem.register(keys)
        batch_mem.commit(torch.as_tensor(nodes), torch.as_tensor(times), msgs)
        for v, t, m in zip(nodes, times, msgs):
            seq_mem.update_one(int(v), float(t), m)
        np.testing.assert_allclose(batch_mem.memory[:7].numpy(), seq_mem.memory[:7].numpy(),
                                   rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(batch_mem.last_update[:7].numpy(), seq_mem.last_update[:7].numpy())
        np.testing.assert_array_equal(batch_mem.n_updates[:7].numpy(), seq_mem.n_updates[:7].numpy())

    def test_event_states_do_not_see_same_batch(self) -> None:
        mem = _memory(delta=0.0)
        idx = mem.register(["x"])
        mem.update(idx, 0.0, torch.ones(1, 3))
        prior = mem.read(idx, 1.0)
        msgs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        states = mem.event_states(torch.tensor([0, 0]), 1.0, msgs)
        expected = prior + mem.W_mem(msgs)
        np.testing.assert_allclose(states.detach().numpy(), expected.detach().numpy(), rtol=1e-6)

    def test_gradient_reaches_w_mem_but_not_memory(self) -> None:
        mem = _memory()
        idx = mem.register(["a", "b"])
        states = mem.update(idx, 1.0, torch.randn(2, 3))
        states.sum().backward()
        self.assertIsNotNone(mem.W_mem.weight.grad)
        self.assertGreater(float(mem.W_mem.weight.grad.abs().sum()), 0.0)
        self.assertFalse(mem.memory.requires_grad)

    def test_constant_time_read_and_update(self) -> None:
        def median_cost(n_nodes: int) -> float:
            mem = _memory(capacity=16)
            idx = mem.register([f"k{i}" for i in range(n_nodes)])
            self.assertGreaterEqual(mem.capacity, n_nodes)
            mem.commit(idx, 0.0, torch.randn(n_nodes, 3))
            msg = torch.randn(3)
            probe = [int(i) for i in np.random.default_rng(SEED).integers(0, n_nodes, 400)]
            samples = []
            for j, node in enumerate(probe):
                t0 = time.perf_counter()
                mem.read_one(node, float(j + 1))
                mem.update_one(node, float(j + 1), msg)
                samples.append(time.perf_counter() - t0)
            return float(np.median(samples))

        small, large = median_cost(100), median_cost(100_000)
        self.assertLess(large / small, 4.0, f"read/update cost grew: {small:.2e}s -> {large:.2e}s")
        self.assertLess(large * 1000.0, 1.0)

    def test_export_load_roundtrip_and_scaler(self) -> None:
        mem = _memory(amount=True)
        feats = np.random.default_rng(SEED).lognormal(size=(50, 3))
        amts = np.random.default_rng(1).lognormal(size=50)
        mem.fit_normalizer(feats, amts)
        msg = mem.build_message_input(feats[:5], amts[:5])
        self.assertEqual(tuple(msg.shape), (5, 4))
        idx = mem.register(["a", "b", "c", "d", "e"])
        mem.commit(idx, 2.0, msg)
        state = mem.export_state()
        clone = _memory(amount=True)
        clone.load_state(state)
        np.testing.assert_allclose(clone.read(idx, 4.0).numpy(), mem.read(idx, 4.0).numpy())
        scaler = SignedLogStandardizer().fit(feats)
        z = scaler.transform(feats)
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. Camouflage similarity filter                                             #
# --------------------------------------------------------------------------- #
class TestSimilarityFilter(unittest.TestCase):
    def setUp(self) -> None:
        set_global_seed(SEED)

    def test_similarity_formula(self) -> None:
        filt = CamouflageSimilarityFilter(SimilarityFilterConfig(memory_dim=8))
        s_u, s_w = torch.randn(10, 8), torch.randn(10, 8)
        lin = filt.net[0]
        expected = torch.sigmoid((s_u - s_w).abs() @ lin.weight.T + lin.bias).squeeze(1)
        np.testing.assert_allclose(filt(s_u, s_w).detach().numpy(), expected.detach().numpy(), rtol=1e-6)
        sim = filt(s_u, s_w)
        self.assertTrue(bool(torch.all((sim > 0) & (sim < 1))))
        np.testing.assert_allclose(filt(s_u, s_w).detach().numpy(), filt(s_w, s_u).detach().numpy())

    def test_threshold_pruning_removes_camouflaged_edges(self) -> None:
        states, edge_index, labels = _camouflage_states()
        filt = _handcrafted_filter()
        res = filt.filter_edges(states, edge_index)
        kept_neighbours = sorted(res.edge_index[0].tolist())
        self.assertEqual(kept_neighbours, [1, 2])
        report = filt.camouflage_report(states, edge_index, labels)
        self.assertEqual(report["camouflage_pruned_rate"], 1.0)
        self.assertEqual(report["homophilic_kept_rate"], 1.0)

    def test_label_aware_training_breaks_camouflage(self) -> None:
        rng = np.random.default_rng(SEED)
        d, n_fraud, n_benign = 8, 60, 240
        states = np.vstack([rng.normal(1.5, 1.0, (n_fraud, d)),
                            rng.normal(-1.5, 1.0, (n_benign, d))]).astype(np.float32)
        labels = np.r_[np.ones(n_fraud), np.zeros(n_benign)].astype(np.int64)
        src, dst = [], []
        for u in range(n_fraud):
            for w in rng.choice(n_fraud, 3, replace=False):
                if w != u:
                    src.append(w); dst.append(u)
            for w in rng.choice(np.arange(n_fraud, n_fraud + n_benign), 6, replace=False):
                src.append(w); dst.append(u)          # relation camouflage
        for u in range(n_fraud, n_fraud + n_benign):
            for w in rng.choice(np.arange(n_fraud, n_fraud + n_benign), 4, replace=False):
                if w != u:
                    src.append(w); dst.append(u)
        edge_index = torch.tensor([src, dst])
        st, y = torch.from_numpy(states), torch.from_numpy(labels)
        filt = CamouflageSimilarityFilter(SimilarityFilterConfig(memory_dim=d, threshold_p=0.5))
        before = filt.camouflage_report(st, edge_index, y)
        hist = filt.fit(st, edge_index, y, epochs=300, learning_rate=0.05)
        after = filt.camouflage_report(st, edge_index, y)
        self.assertLess(hist["final_loss"], hist["initial_loss"])
        self.assertGreater(after["camouflage_pruned_rate"], 0.95, after)
        self.assertGreater(after["homophilic_kept_rate"], 0.90, after)
        self.assertGreater(after["camouflage_pruned_rate"], before["camouflage_pruned_rate"])

    def test_top_p_and_min_neighbour_modes(self) -> None:
        states, edge_index, _ = _camouflage_states()
        topp = CamouflageSimilarityFilter(SimilarityFilterConfig(memory_dim=8, threshold_p=0.4,
                                                                 mode="top_p"))
        with torch.no_grad():
            topp.net[0].weight.fill_(-1.0)
            topp.net[0].bias.fill_(2.0)
        res = topp.filter_edges(states, edge_index)
        self.assertEqual(sorted(res.edge_index[0].tolist()), [1, 2])   # ceil(0.4 * 5) = 2
        strict = _handcrafted_filter(p=0.9999)
        self.assertEqual(strict.filter_edges(states, edge_index).edge_index.shape[1], 0)
        keep_one = CamouflageSimilarityFilter(SimilarityFilterConfig(memory_dim=8, threshold_p=0.9999,
                                                                     min_neighbors_kept=1))
        keep_one.load_state_dict(strict.state_dict())
        kept = keep_one.filter_edges(states, edge_index).edge_index
        self.assertEqual(kept.shape[1], 1)
        self.assertIn(int(kept[0, 0]), (1, 2))

    def test_rank_within_centre(self) -> None:
        centres = torch.tensor([2, 0, 2, 0, 2])
        scores = torch.tensor([0.1, 0.9, 0.7, 0.3, 0.5])
        ranks = CamouflageSimilarityFilter.rank_within_centre(centres, scores)
        self.assertEqual(ranks.tolist(), [2, 0, 0, 1, 1])


# --------------------------------------------------------------------------- #
# 3. GraphSAGE over filtered neighbourhoods                                   #
# --------------------------------------------------------------------------- #
class TestCamouflageGraphSAGE(unittest.TestCase):
    def setUp(self) -> None:
        set_global_seed(SEED)

    def test_sage_layer_formula(self) -> None:
        conv = FilteredSAGEConv(4, 3, aggr="mean", use_pyg=False)
        x = torch.randn(4, 4)
        ei = torch.tensor([[1, 2, 0], [0, 0, 3]])
        out = conv(x, ei)
        agg = torch.zeros(4, 4)
        agg[0] = (x[1] + x[2]) / 2
        agg[3] = x[0]
        expected = torch.relu(conv.lin(torch.cat([x, agg], dim=1)))
        np.testing.assert_allclose(out.detach().numpy(), expected.detach().numpy(), rtol=1e-6)
        mx = scatter_aggregate(x, ei, "max")
        np.testing.assert_allclose(mx[0].numpy(), torch.maximum(x[1], x[2]).numpy())
        self.assertTrue(torch.all(mx[1] == 0))

    @unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
    def test_pyg_and_fallback_agree(self) -> None:
        a = FilteredSAGEConv(6, 5, aggr="mean", use_pyg=True)
        b = FilteredSAGEConv(6, 5, aggr="mean", use_pyg=False)
        b.load_state_dict(a.state_dict(), strict=False)
        x = torch.randn(12, 6)
        ei = torch.randint(0, 12, (2, 40))
        np.testing.assert_allclose(a(x, ei).detach().numpy(), b(x, ei).detach().numpy(),
                                   rtol=1e-5, atol=1e-6)

    def test_filtered_message_passing_ignores_camouflage(self) -> None:
        cfg = GraphSAGEConfig(feature_dim=4, memory_dim=8, hidden_dim=16, out_dim=8,
                              num_layers=2, fanouts=(5, 5), dropout=0.0, use_pyg=False)
        model = CamouflageResistantGraphSAGE(cfg).eval()
        model.similarity.load_state_dict(_handcrafted_filter().state_dict())
        states, edge_index, _ = _camouflage_states()
        with torch.no_grad():
            h_filtered, filt = model.propagate(states, edge_index)
            pruned = edge_index[:, :2]
            h = states
            for conv in model.convs:
                h = conv(h, pruned)
            h_unfiltered = states
            for conv in model.convs:
                h_unfiltered = conv(h_unfiltered, edge_index)
        np.testing.assert_allclose(h_filtered[0].numpy(), h[0].numpy(), rtol=1e-6, atol=1e-7)
        self.assertGreater(float((h_filtered[0] - h_unfiltered[0]).abs().max()), 1e-4)
        self.assertEqual(int(filt.keep_mask.sum()), 2)

    def _stream_model(self, feature_dim: int) -> CamouflageResistantGraphSAGE:
        hp = _small_hyperparams()
        cfg = GraphSAGEConfig.from_hyperparams(hp, feature_dim, dropout=0.0)
        return CamouflageResistantGraphSAGE(cfg, seed=SEED)

    def test_stream_embeddings_are_causal(self) -> None:
        frame, names = _processed_frame(n_steps=30)
        x = frame[names].to_numpy(dtype=np.float64)
        args = lambda f: (f[names].to_numpy(dtype=np.float64), f["amount"].to_numpy(),  # noqa: E731
                          f["step"].to_numpy(), f["src"].tolist(), f["dst"].tolist())
        perturbed = frame.copy()
        future = perturbed["step"] > 20
        perturbed.loc[future, "amount"] *= 50.0
        perturbed.loc[future, names] = perturbed.loc[future, names] * 3.0 + 7.0
        outs = []
        for f in (frame, perturbed):
            model = self._stream_model(len(names))
            model.fit_normalizer(x, frame["amount"].to_numpy())
            outs.append(model.transform_stream(*args(f)))
        past = ~future.to_numpy()
        np.testing.assert_allclose(outs[0]["h_src"][past], outs[1]["h_src"][past], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(outs[0]["h_dst"][past], outs[1]["h_dst"][past], rtol=1e-5, atol=1e-6)
        self.assertGreater(np.abs(outs[0]["h_src"][~past] - outs[1]["h_src"][~past]).max(), 1e-4)

    def test_within_step_permutation_invariance(self) -> None:
        frame, names = _processed_frame(n_steps=20)
        rng = np.random.default_rng(SEED)
        shuffled = (frame.assign(_r=rng.random(len(frame)))
                    .sort_values(["step", "_r"], kind="mergesort").drop(columns="_r")
                    .reset_index(drop=True))
        results = []
        for f in (frame, shuffled):
            model = self._stream_model(len(names))
            model.fit_normalizer(frame[names].to_numpy(dtype=np.float64), frame["amount"].to_numpy())
            out = model.transform_stream(f[names].to_numpy(dtype=np.float64), f["amount"].to_numpy(),
                                         f["step"].to_numpy(), f["src"].tolist(), f["dst"].tolist())
            results.append(pd.DataFrame(out["h_src"], index=f["tx_id"].to_numpy()).sort_index())
        np.testing.assert_allclose(results[0].to_numpy(), results[1].to_numpy(), rtol=1e-4, atol=1e-5)

    def test_fit_stream_trains_all_components(self) -> None:
        frame, names = _processed_frame(n_steps=25)
        model = self._stream_model(len(names))
        before = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k.endswith("weight")}
        hist = model.fit_stream(frame[names].to_numpy(dtype=np.float64), frame["amount"].to_numpy(),
                                frame["step"].to_numpy(), frame["src"].tolist(), frame["dst"].tolist(),
                                frame["label"].to_numpy(), epochs=2)
        self.assertEqual(len(hist), 2)
        self.assertTrue(all(np.isfinite([h["cls_loss"], h["sim_loss"]]).all() for h in hist))
        after = model.state_dict()
        for key in ("memory.W_mem.weight", "similarity.net.0.weight", "convs.0.lin.weight", "head.weight"):
            self.assertGreater(float((after[key] - before[key]).abs().max()), 0.0, key)

    def test_single_event_latency_under_budget(self) -> None:
        """Full-size config (d_s=64, 128->64, fanouts 15/10) on a dense graph."""
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        n_feat = 30
        cfg = GraphSAGEConfig.from_hyperparams(HYPERPARAMS, n_feat)
        self.assertEqual(cfg.fanouts, (15, 10))
        model = CamouflageResistantGraphSAGE(cfg).eval()
        rng = np.random.default_rng(SEED)
        n_nodes, n_edges = 3000, 60000
        model.fit_normalizer(rng.lognormal(size=(500, n_feat)), rng.lognormal(8, 1, 500))
        keys = [f"C{i}" for i in range(n_nodes)]
        idx = model.memory.register(keys)
        msg = model.memory.build_message_input(rng.lognormal(size=(n_nodes, n_feat)),
                                               rng.lognormal(8, 1, n_nodes))
        model.memory.commit(idx, 0.0, msg)
        a = rng.integers(0, n_nodes, n_edges)
        b = rng.integers(0, n_nodes, n_edges)
        model.neighbors.insert_many(a.tolist(), b.tolist())
        lat, sizes = [], []
        for i in range(220):
            u, v = rng.integers(0, n_nodes, 2)
            out = model.embed_event(keys[u], keys[v], rng.lognormal(size=n_feat),
                                    float(rng.lognormal(8, 1)), 1.0 + i)
            if i >= 20:
                lat.append(out["latency_ms"])
                sizes.append(out["n_edges"])
            self.assertEqual(out["h_src"].shape, (cfg.out_dim,))
            self.assertTrue(0.0 <= out["gnn_proba"] <= 1.0)
        p95 = float(np.percentile(lat, 95))
        self.assertGreater(np.mean(sizes), 150)          # genuinely 2-hop neighbourhoods
        self.assertLessEqual(p95, LATENCY_BUDGET_MS, f"GraphSAGE p95 latency {p95:.2f} ms")
        print(f"\n[latency] GraphSAGE embed_event p50={np.median(lat):.2f} ms "
              f"p95={p95:.2f} ms (mean edges={np.mean(sizes):.0f})")


# --------------------------------------------------------------------------- #
# 4. Denoising autoencoder residual                                           #
# --------------------------------------------------------------------------- #
class TestAutoencoderResidual(unittest.TestCase):
    def setUp(self) -> None:
        set_global_seed(SEED)
        rng = np.random.default_rng(SEED)
        latent = rng.normal(size=(3000, 3))
        mix = rng.normal(size=(3, 12))
        self.normal = latent @ mix + 0.05 * rng.normal(size=(3000, 12))
        self.anomal = rng.normal(0, 2.5, size=(200, 12))
        cfg = AutoencoderConfig.from_hyperparams(HYPERPARAMS, 12, epochs=40, batch_size=256)
        self.engine = AutoencoderResidualEngine(cfg).fit(
            np.vstack([self.normal[:2500], self.anomal[:100]]),
            np.r_[np.zeros(2500), np.ones(100)])

    def test_residual_matches_definition(self) -> None:
        x = self.normal[2500:2510]
        r = self.engine.residual(x)
        xs = torch.from_numpy(self.engine.scaler.transform(x))
        with torch.no_grad():
            recon = self.engine.model.decode(self.engine.model.encode(xs))
        expected = torch.linalg.vector_norm(xs - recon, dim=1).numpy()
        np.testing.assert_allclose(r, expected, rtol=1e-5)
        self.assertTrue(np.all(r >= 0))
        self.assertAlmostEqual(self.engine.residual_one(x[0]), r[0], places=5)

    def test_anomalies_have_larger_residual(self) -> None:
        r_norm = self.engine.residual(self.normal[2500:])
        r_anom = self.engine.residual(self.anomal[100:])
        auc = roc_auc_score(np.r_[np.zeros(r_norm.size), np.ones(r_anom.size)], np.r_[r_norm, r_anom])
        self.assertGreater(auc, 0.95)
        self.assertGreater(r_anom.mean(), 2 * r_norm.mean())
        self.assertLess(self.engine.latency_ms(self.normal[0], repeats=50), 5.0)


# --------------------------------------------------------------------------- #
# 5. Cost-sensitive calibrated XGBoost                                        #
# --------------------------------------------------------------------------- #
def _tabular_data(n: int = 9000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    logit = -4.0 + 1.6 * x[:, 0] - 1.2 * x[:, 1] + 0.8 * x[:, 2] * x[:, 3]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    amounts = rng.lognormal(8, 1, n)
    return x, y, amounts


class TestTabularEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_global_seed(SEED)
        cls.x, cls.y, cls.amounts = _tabular_data()
        cls.tr, cls.ca, cls.te = slice(0, 5000), slice(5000, 7000), slice(7000, 9000)
        cfg = TabularEngineConfig.from_hyperparams(HYPERPARAMS, n_estimators=200,
                                                   early_stopping_rounds=30)
        cls.engine = CalibratedXGBoostEngine(cfg).fit(
            cls.x[cls.tr], cls.y[cls.tr], cls.x[cls.ca], cls.y[cls.ca],
            amounts_train=cls.amounts[cls.tr])

    def test_cost_sensitive_weighting(self) -> None:
        self.assertEqual(self.engine.resolved_scale_pos_weight, 25.0)
        w = self.engine.sample_weights(np.array([0, 1, 1]), np.array([5.0, 1.0, 3.0]))
        self.assertEqual(w[0], 1.0)
        np.testing.assert_allclose(w[1:], [0.5, 1.5])
        plain = CalibratedXGBoostEngine(TabularEngineConfig(cost_sensitive=False))
        self.assertAlmostEqual(plain.resolve_scale_pos_weight(np.array([0, 0, 0, 1])), 3.0)

    def test_calibrated_risk_is_bounded_and_monotone(self) -> None:
        extreme = np.vstack([self.x[self.te], np.full((3, 8), 1e6), np.full((3, 8), -1e6)])
        p = self.engine.predict_proba(extreme)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
        m = self.engine.predict_margin(self.x[self.te])
        order = np.argsort(m)
        self.assertTrue(np.all(np.diff(self.engine.predict_proba(self.x[self.te])[order]) >= -1e-12))
        self.assertAlmostEqual(self.engine.predict_one(self.x[7000]), p[0], places=10)

    def test_calibration_improves_brier(self) -> None:
        y_te = self.y[self.te]
        raw = self.engine.predict_uncalibrated(self.x[self.te])
        cal = self.engine.predict_proba(self.x[self.te])
        self.assertLess(brier_score(cal, y_te), brier_score(raw, y_te))
        self.assertGreater(roc_auc_score(y_te, cal), 0.8)
        self.assertAlmostEqual(cal.mean(), y_te.mean(), delta=0.03)

    def test_venn_abers_engine_interval(self) -> None:
        self.engine.calibrate(self.x[self.ca], self.y[self.ca], method="venn_abers")
        try:
            p = self.engine.predict_proba(self.x[self.te])
            p0, p1 = self.engine.predict_interval(self.x[self.te])
            self.assertTrue(np.all((p >= 0) & (p <= 1)))
            self.assertTrue(np.all(p0 <= p1 + 1e-12))
            self.assertTrue(np.all((p >= p0 - 1e-12) & (p <= p1 + 1e-12)))
        finally:
            self.engine.calibrate(self.x[self.ca], self.y[self.ca], method="isotonic")

    def test_venn_abers_exact_against_brute_force(self) -> None:
        rng = np.random.default_rng(3)
        s = np.round(rng.normal(size=250), 1)
        y = (rng.random(250) < 1 / (1 + np.exp(-2 * s))).astype(int)
        va = VennAbersCalibrator(max_unique_scores=400).fit(s, y)
        queries = np.r_[s[:25], rng.normal(size=25), -10.0, 10.0]
        p0, p1 = va.predict_interval(queries)

        def brute(t: float, label: int) -> float:
            iso = IsotonicRegression(out_of_bounds="clip").fit(np.r_[s, t], np.r_[y, label])
            return float(iso.predict([t])[0])

        np.testing.assert_allclose(p0, [brute(t, 0) for t in queries], atol=1e-12)
        np.testing.assert_allclose(p1, [brute(t, 1) for t in queries], atol=1e-12)
        iso = IsotonicCalibrator().fit(s, y)
        self.assertTrue(np.all((iso.predict(queries) >= 0) & (iso.predict(queries) <= 1)))


# --------------------------------------------------------------------------- #
# 6. Stacked fusion classifier                                                #
# --------------------------------------------------------------------------- #
class TestFusionClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_global_seed(SEED)
        cls.frame, cls.names = _processed_frame(n_steps=60, tx_per_step=(20, 45),
                                                n_customers=250, n_merchants=40)
        steps = cls.frame["step"]
        cls.train = cls.frame[steps <= 40].reset_index(drop=True)
        cls.cal = cls.frame[(steps > 40) & (steps <= 50)].reset_index(drop=True)
        cls.test = cls.frame[steps > 50].reset_index(drop=True)
        gen = PerturbationGenerator(PerturbationConfig(n_candidates=40), seed=SEED)
        cls.rules = gen.generate_feed(0.20).rules
        cls.model = StackedFusionClassifier(cls.names, hyperparams=_small_hyperparams())
        cls.report = cls.model.fit(cls.train, cls.cal, rules=cls.rules)

    def test_fusion_vector_layout(self) -> None:
        names = self.model.fusion_feature_names
        k = self.model.graph.out_dim
        self.assertEqual(len(names), len(self.names) + 2 * k + 1 + 5)
        self.assertEqual(names[:len(self.names)], self.names)
        self.assertIn("r_ae", names)
        self.assertEqual(names[-5:], list(KnowledgeBaseTriggerEncoder.FEATURE_NAMES))
        z = self.model.assemble(np.zeros((2, len(self.names))), np.ones((2, k)), np.ones((2, k)) * 2,
                                np.array([3.0, 4.0]), np.zeros((2, 5)))
        self.assertEqual(z.shape, (2, len(names)))
        self.assertEqual(z[1, len(self.names) + 2 * k], 4.0)
        ablated = StackedFusionClassifier(self.names, hyperparams=_small_hyperparams(),
                                          use_relational_embeddings=False, use_ae_residual=False,
                                          use_kb_triggers=False)
        self.assertEqual(ablated.fusion_feature_names, self.names)
        self.assertIsNone(ablated.graph)

    def test_kb_encoder_frame_event_consistency(self) -> None:
        enc = KnowledgeBaseTriggerEncoder(self.rules)
        block = enc.transform_frame(self.frame)
        events = frame_to_events(self.frame)
        per_event = np.vstack([enc.transform_event(ev) for ev in events])
        np.testing.assert_allclose(block, per_event)
        self.assertGreater(block[:, 0].sum(), 0)
        np.testing.assert_array_equal(block[:, 1], (block[:, 0] > 0).astype(float))
        self.assertTrue(np.all(block[:, 3] >= block[:, 4] - 1e-12))

    def test_calibrated_risk_bounded_on_future_split(self) -> None:
        model = copy.deepcopy(self.model)
        p = model.predict_proba(self.test)
        self.assertEqual(p.shape, (len(self.test),))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
        self.assertGreater(self.report["n_fusion_features"], len(self.names))

    def test_end_to_end_latency_under_budget(self) -> None:
        model = copy.deepcopy(self.model)
        events = frame_to_events(self.test)[:200]
        prof = model.latency_profile(events, warmup=10)
        decision = model.score_event(events[0], commit=False)
        self.assertTrue(0.0 <= decision["p_hat"] <= 1.0)
        self.assertEqual(set(decision["timings"]), {"graph_ms", "ae_ms", "kb_ms", "tabular_ms"})
        self.assertLessEqual(prof["p95_ms"], LATENCY_BUDGET_MS, prof)
        print(f"\n[latency] fused score_event p50={prof['p50_ms']:.2f} ms "
              f"p95={prof['p95_ms']:.2f} ms p99={prof['p99_ms']:.2f} ms")

    def test_full_size_fusion_latency(self) -> None:
        """Default hyperparams (600 trees, d_s=64, 128->64 GraphSAGE)."""
        hp = copy.deepcopy(HYPERPARAMS)
        hp["graphsage"]["epochs"] = 1
        hp["autoencoder"]["epochs"] = 2
        model = StackedFusionClassifier(self.names, hyperparams=hp)
        model.fit(self.train, self.cal, rules=self.rules)
        prof = model.latency_profile(frame_to_events(self.test)[:150], warmup=10)
        self.assertLessEqual(prof["p95_ms"], LATENCY_BUDGET_MS, prof)
        print(f"\n[latency] full-size fused score_event p50={prof['p50_ms']:.2f} ms "
              f"p95={prof['p95_ms']:.2f} ms")

    def test_rejects_leakage_and_overlapping_splits(self) -> None:
        with self.assertRaises(LeakageError):
            StackedFusionClassifier(self.names + ["newbalanceOrig"], hyperparams=_small_hyperparams())
        model = StackedFusionClassifier(self.names, hyperparams=_small_hyperparams())
        with self.assertRaises(ValueError):
            model.fit(self.cal, self.train)


if __name__ == "__main__":
    set_global_seed(SEED)
    unittest.main(verbosity=2)
