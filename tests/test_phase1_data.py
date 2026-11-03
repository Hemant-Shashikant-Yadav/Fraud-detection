"""
tests/test_phase1_data.py
=========================

Phase 1 verification: scaffolding, configuration integrity, leakage-free
ingestion, strict temporal causality of engineered features, rolling velocity
correctness, chunked-stream exactness, loader iteration across PaySim /
IEEE-CIS / AMLSim schemas, and adversarial rule-feed generation.

Run from the repository root:

    python tests/test_phase1_data.py          # unittest runner
    python -m pytest tests/test_phase1_data.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.causal_preprocessor import (  # noqa: E402
    FORBIDDEN_FEATURE_COLUMNS,
    PAYSIM_LEAKAGE_COLUMNS,
    SEED,
    CausalFeatureConfig,
    CausalFeatureEngineer,
    LeakageError,
    SplitPartitionedWriter,
    TemporalOrderError,
    assert_no_leakage,
    assign_temporal_split,
    audit_temporal_causality,
    load_temporal_splits,
    preprocess_dataset,
    set_global_seed,
)
from src.data.dataset_loaders import (  # noqa: E402
    AMLSimLoader,
    IEEECISLoader,
    PaySimLoader,
    build_loader,
    make_synthetic_paysim_frame,
    write_synthetic_amlsim_csv,
    write_synthetic_ieee_cis_csv,
    write_synthetic_paysim_csv,
)
from src.data.perturbation_generator import (  # noqa: E402
    CandidateRule,
    Condition,
    PerturbationConfig,
    PerturbationGenerator,
)

PROJECT_CONFIG = json.loads((ROOT / "configs" / "project_config.json").read_text(encoding="utf-8"))
HYPERPARAMS = json.loads((ROOT / "configs" / "hyperparams.json").read_text(encoding="utf-8"))


def _paysim_canonical(seed: int = SEED, **kwargs) -> pd.DataFrame:
    """Synthetic PaySim frame mapped to canonical columns (leakage stripped)."""
    raw = make_synthetic_paysim_frame(seed=seed, **kwargs)
    return pd.DataFrame({
        "tx_id": np.arange(len(raw), dtype=np.int64),
        "step": raw["step"].to_numpy(dtype=np.int64),
        "src": raw["nameOrig"].astype(str).to_numpy(),
        "dst": raw["nameDest"].astype(str).to_numpy(),
        "amount": raw["amount"].to_numpy(dtype=np.float64),
        "tx_type": raw["type"].astype(str).to_numpy(),
        "label": raw["isFraud"].to_numpy(dtype=np.int8),
    })


def _brute_force_reference(frame: pd.DataFrame, windows=(1, 5, 20),
                           inflow_types=("CASH_IN",)) -> pd.DataFrame:
    """O(n^2) reference implementation using only rows with step < tau_t."""
    steps = frame["step"].to_numpy()
    src = frame["src"].to_numpy()
    dst = frame["dst"].to_numpy()
    amt = frame["amount"].to_numpy(dtype=np.float64)
    inflow = np.isin(frame["tx_type"].to_numpy(), list(inflow_types))
    src_flow = np.where(inflow, amt, -amt)
    out = {k: np.zeros(len(frame)) for k in (
        [f"src_vel_{w}" for w in windows] + [f"dst_vel_{w}" for w in windows]
        + ["src_hist_count", "src_hist_mean", "src_hist_std", "dst_hist_count",
           "dst_hist_mean", "dst_hist_std", "src_steps_since_last", "dst_steps_since_last",
           "src_ledger_pre", "dst_ledger_pre"])}
    for i in range(len(frame)):
        s = steps[i]
        past = steps < s
        for role, keys, ent in (("src", src, src[i]), ("dst", dst, dst[i])):
            mask = past & (keys == ent)
            for w in windows:
                out[f"{role}_vel_{w}"][i] = np.sum(mask & (steps >= s - w))
            vals = amt[mask]
            out[f"{role}_hist_count"][i] = vals.size
            out[f"{role}_hist_mean"][i] = vals.mean() if vals.size else 0.0
            out[f"{role}_hist_std"][i] = vals.std() if vals.size else 0.0
            out[f"{role}_steps_since_last"][i] = s - steps[mask].max() if vals.size else -1
        for role, ent in (("src", src[i]), ("dst", dst[i])):
            ledger = (np.sum(src_flow[past & (src == ent)])
                      - np.sum(src_flow[past & (dst == ent)]))
            out[f"{role}_ledger_pre"][i] = ledger
    return pd.DataFrame(out)


class TestConfigsAndScaffolding(unittest.TestCase):
    def test_seed_and_splits(self) -> None:
        self.assertEqual(PROJECT_CONFIG["project"]["seed"], 42)
        self.assertEqual(HYPERPARAMS["seed"], 42)
        splits = load_temporal_splits(PROJECT_CONFIG["temporal_splits"]["paysim"])
        self.assertEqual([s.name for s in splits], ["train", "calibration", "operating", "test"])
        self.assertEqual((splits[0].start_step, splits[-1].end_step), (1, 743))
        for prev, nxt in zip(splits, splits[1:]):
            self.assertEqual(prev.end_step + 1, nxt.start_step, "splits must be contiguous")
        expected = PROJECT_CONFIG["expected_split_sizes"]["paysim"]
        total = sum(v["transactions"] for v in expected.values())
        self.assertEqual(total, PROJECT_CONFIG["datasets"]["paysim"]["expected_rows"])

    def test_overlapping_splits_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_temporal_splits({"a": {"start_step": 1, "end_step": 10},
                                  "b": {"start_step": 10, "end_step": 20}})

    def test_assign_split(self) -> None:
        splits = load_temporal_splits(PROJECT_CONFIG["temporal_splits"]["paysim"])
        labels = assign_temporal_split(np.array([1, 520, 521, 575, 576, 631, 632, 743, 744]), splits)
        self.assertEqual(list(labels), ["train", "train", "calibration", "calibration", "operating",
                                        "operating", "test", "test", "out_of_range"])

    def test_hyperparams_sections(self) -> None:
        for key in ("xgboost", "temporal_memory", "graphsage", "conformal_crc",
                    "martingale_eprocess", "subjective_logic"):
            self.assertIn(key, HYPERPARAMS)
        adm = HYPERPARAMS["martingale_eprocess"]["admission"]
        self.assertLess(adm["p0"], adm["p1"])
        lam = HYPERPARAMS["conformal_crc"]["lambda_init"]
        self.assertLess(lam["low"], lam["mid"])
        self.assertLess(lam["mid"], lam["high"])

    def test_scaffolding_script(self) -> None:
        spec = importlib.util.spec_from_file_location("setup_scaffolding",
                                                      ROOT / "scripts" / "setup_scaffolding.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = module.build_scaffold(root)
            self.assertGreater(len(report["created_dirs"]), 10)
            for rel in ("configs", "data", "src/data", "src/graph", "src/detection",
                        "src/conformal", "src/epistemic_gate", "src/recovery",
                        "src/closed_loop", "src/baselines", "experiments", "tests", "reports"):
                self.assertTrue((root / rel).is_dir(), rel)
            self.assertTrue((root / "src" / "data" / "__init__.py").is_file())
            second = module.build_scaffold(root)
            self.assertEqual(second["created_dirs"], [])
            self.assertEqual(second["created_files"], [])
            missing = module.verify_scaffold(root)
            self.assertEqual(sorted(missing), ["configs/hyperparams.json", "configs/project_config.json"])


class TestLeakageStripping(unittest.TestCase):
    def test_engine_refuses_leakage_columns(self) -> None:
        raw = make_synthetic_paysim_frame(n_steps=3)
        frame = raw.rename(columns={"nameOrig": "src", "nameDest": "dst", "type": "tx_type"})
        with self.assertRaises(LeakageError):
            CausalFeatureEngineer().transform(frame)
        with self.assertRaises(LeakageError):
            assert_no_leakage(pd.DataFrame({"raw_oldbalanceOrg": [1.0]}))

    def test_loader_never_emits_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_synthetic_paysim_csv(Path(tmp) / "paysim.csv", n_steps=12)
            header = pd.read_csv(path, nrows=0).columns
            for col in PAYSIM_LEAKAGE_COLUMNS:
                self.assertIn(col, header)
            loader = PaySimLoader(path, chunksize=50)
            self.assertTrue(all(c not in loader._usecols() for c in FORBIDDEN_FEATURE_COLUMNS))
            for chunk in loader.iter_chunks():
                for col in FORBIDDEN_FEATURE_COLUMNS:
                    self.assertNotIn(col, chunk.columns)
            with self.assertRaises(ValueError):
                PaySimLoader(path, passthrough_columns=["newbalanceOrig"])

    def test_features_exclude_balance_information(self) -> None:
        engineer = CausalFeatureEngineer()
        for name in engineer.feature_names:
            self.assertFalse(any(name == c for c in FORBIDDEN_FEATURE_COLUMNS))
        out = engineer.transform(_paysim_canonical(n_steps=10))
        assert_no_leakage(out)


class TestTemporalCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = _paysim_canonical(n_steps=36, n_customers=120, n_merchants=20)

    def test_future_perturbation_invariance(self) -> None:
        for cutoff in (5, 17, 30):
            report = audit_temporal_causality(self.frame, cutoff_step=cutoff)
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["max_abs_diff"], 0.0)
            self.assertGreater(report["rows_checked"], 0)

    def test_within_step_permutation_invariance(self) -> None:
        base = CausalFeatureEngineer().transform(self.frame).sort_values("tx_id").reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        shuffled = self.frame.assign(_r=rng.random(len(self.frame)))
        shuffled = shuffled.sort_values(["step", "_r"], kind="mergesort").drop(columns="_r")
        alt = CausalFeatureEngineer().transform(shuffled).sort_values("tx_id").reset_index(drop=True)
        names = CausalFeatureEngineer().feature_names
        np.testing.assert_array_equal(base[names].to_numpy(), alt[names].to_numpy())

    def test_label_blindness(self) -> None:
        flipped = self.frame.assign(label=1 - self.frame["label"])
        a = CausalFeatureEngineer().transform(self.frame)
        b = CausalFeatureEngineer().transform(flipped)
        names = CausalFeatureEngineer().feature_names
        np.testing.assert_array_equal(a[names].to_numpy(), b[names].to_numpy())

    def test_matches_brute_force_reference(self) -> None:
        small = _paysim_canonical(n_steps=30, tx_per_step=(4, 10), n_customers=25,
                                  n_merchants=5, seed=7)
        out = CausalFeatureEngineer().transform(small)
        ref = _brute_force_reference(small)
        for col in ref.columns:
            np.testing.assert_allclose(out[col].to_numpy(dtype=float), ref[col].to_numpy(),
                                       rtol=1e-9, atol=1e-6, err_msg=col)

    def test_first_step_has_empty_history(self) -> None:
        out = CausalFeatureEngineer().transform(self.frame)
        first = out[out["step"] == out["step"].min()]
        self.assertTrue((first["src_hist_count"] == 0).all())
        self.assertTrue((first["dst_vel_20"] == 0).all())
        self.assertTrue((first["src_ledger_pre"] == 0.0).all())
        self.assertTrue((first["src_is_new"] == 1).all())

    def test_order_violations_raise(self) -> None:
        bad = self.frame.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(TemporalOrderError):
            CausalFeatureEngineer().transform(bad)
        eng = CausalFeatureEngineer()
        eng.transform(self.frame[self.frame["step"] <= 10])
        with self.assertRaises(TemporalOrderError):
            eng.transform_chunk(self.frame[self.frame["step"] == 10])


class TestRollingVelocity(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "tx_id": np.arange(6),
            "step": [1, 1, 2, 3, 6, 25],
            "src": ["A", "A", "A", "B", "A", "A"],
            "dst": ["X", "Y", "X", "X", "X", "X"],
            "amount": [100.0, 50.0, 30.0, 10.0, 20.0, 40.0],
            "tx_type": ["TRANSFER"] * 6,
            "label": [0] * 6,
        })
        self.out = CausalFeatureEngineer().transform(self.frame).set_index("tx_id")

    def test_hand_computed_velocities(self) -> None:
        o = self.out
        self.assertEqual(list(o.loc[[0, 1], "src_vel_1"]), [0, 0])
        self.assertEqual((o.loc[2, "src_vel_1"], o.loc[2, "src_vel_5"], o.loc[2, "src_hist_count"]), (2, 2, 2))
        self.assertEqual((o.loc[3, "src_vel_1"], o.loc[3, "dst_vel_1"], o.loc[3, "dst_vel_5"]), (0, 1, 2))
        self.assertEqual((o.loc[4, "src_vel_1"], o.loc[4, "src_vel_5"], o.loc[4, "src_vel_20"]), (0, 3, 3))
        self.assertEqual((o.loc[5, "src_vel_5"], o.loc[5, "src_vel_20"], o.loc[5, "src_hist_count"]), (0, 1, 4))
        self.assertEqual(o.loc[5, "src_steps_since_last"], 19)
        self.assertEqual(o.loc[4, "dst_vel_5"], 3)

    def test_hand_computed_moments_and_ledger(self) -> None:
        o = self.out
        self.assertAlmostEqual(o.loc[2, "src_hist_mean"], 75.0)
        self.assertAlmostEqual(o.loc[2, "src_hist_std"], 25.0)
        self.assertAlmostEqual(o.loc[2, "src_amount_to_mean"], 30.0 / 76.0)
        self.assertAlmostEqual(o.loc[2, "src_ledger_pre"], -150.0)
        self.assertAlmostEqual(o.loc[2, "src_balance_delta"], -180.0)
        self.assertAlmostEqual(o.loc[2, "dst_ledger_pre"], 100.0)
        self.assertAlmostEqual(o.loc[2, "dst_balance_delta"], 130.0)
        self.assertAlmostEqual(o.loc[0, "log1p_amount"], np.log1p(100.0))

    def test_cash_in_reverses_flow(self) -> None:
        frame = self.frame.copy()
        frame.loc[0, "tx_type"] = "CASH_IN"
        o = CausalFeatureEngineer().transform(frame).set_index("tx_id")
        self.assertAlmostEqual(o.loc[2, "src_ledger_pre"], 100.0 - 50.0)


class TestStreamingExactness(unittest.TestCase):
    def test_chunked_equals_full(self) -> None:
        frame = _paysim_canonical(n_steps=24)
        full = CausalFeatureEngineer().transform(frame)
        for size in (1, 7, 33, 500):
            eng = CausalFeatureEngineer()
            chunks = (frame.iloc[i:i + size] for i in range(0, len(frame), size))
            streamed = pd.concat(list(eng.transform_stream(chunks)), ignore_index=True)
            pd.testing.assert_frame_equal(full, streamed, check_dtype=False)
            self.assertEqual(eng.rows_emitted, len(frame))

    def test_pending_buffer_holds_last_step(self) -> None:
        frame = _paysim_canonical(n_steps=5)
        eng = CausalFeatureEngineer()
        out = eng.transform_chunk(frame)
        self.assertEqual(int(out["step"].max()), 4)
        tail = eng.flush()
        self.assertTrue((tail["step"] == 5).all())
        self.assertEqual(len(out) + len(tail), len(frame))


class TestStreamLoaders(unittest.TestCase):
    def test_paysim_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_synthetic_paysim_csv(Path(tmp) / "paysim.csv", n_steps=20)
            n_rows = len(pd.read_csv(path))
            loader = PaySimLoader(path, chunksize=37)
            chunks = list(loader.iter_chunks())
            self.assertGreater(len(chunks), 3)
            self.assertTrue(all(len(c) <= 37 for c in chunks))
            full = pd.concat(chunks, ignore_index=True)
            self.assertEqual(len(full), n_rows)
            np.testing.assert_array_equal(full["tx_id"].to_numpy(), np.arange(n_rows))
            self.assertTrue(np.all(np.diff(full["step"].to_numpy()) >= 0))
            self.assertTrue(loader.last_scan.is_sorted)
            events = list(PaySimLoader(path, chunksize=64).iter_events())
            self.assertEqual(len(events), n_rows)
            self.assertEqual(events[0].tx_id, 0)
            limited = PaySimLoader(path, chunksize=10, max_rows=25).read_all()
            self.assertEqual(len(limited), 25)

    def test_paysim_unsorted_file_is_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = make_synthetic_paysim_frame(n_steps=10).sample(frac=1.0, random_state=SEED)
            path = Path(tmp) / "shuffled.csv"
            raw.to_csv(path, index=False)
            loader = PaySimLoader(path, chunksize=30)
            full = loader.read_all()
            self.assertFalse(loader.last_scan.is_sorted)
            self.assertTrue(np.all(np.diff(full["step"].to_numpy()) >= 0))
            with self.assertRaises(TemporalOrderError):
                list(PaySimLoader(path, chunksize=30, ensure_sorted=False).iter_chunks())

    def test_ieee_cis_sorted_by_transaction_dt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ident = Path(tmp) / "train_identity.csv"
            path = write_synthetic_ieee_cis_csv(Path(tmp) / "train_transaction.csv", n_rows=400,
                                                shuffle=True, identity_path=ident)
            loader = IEEECISLoader(path, identity_path=ident, chunksize=64,
                                   passthrough_columns=["C1", "D1"])
            full = loader.read_all()
            self.assertFalse(loader.last_scan.is_sorted)
            self.assertEqual(len(full), 400)
            self.assertTrue(np.all(np.diff(full["event_time"].to_numpy()) >= 0))
            np.testing.assert_array_equal(full["step"].to_numpy(),
                                          (full["event_time"].to_numpy() // 3600).astype(np.int64))
            self.assertTrue(full["src"].str.startswith("card1=").all())
            self.assertIn("raw_DeviceType", full.columns)
            self.assertIn("raw_C1", full.columns)
            self.assertTrue((full["dst"] == "").all())
            cfg = CausalFeatureConfig(tx_types=("W", "H", "C", "S", "R"), inflow_types=(),
                                      include_destination=False)
            feats = CausalFeatureEngineer(cfg).transform(full)
            self.assertNotIn("dst_vel_1", feats.columns)
            self.assertEqual(len(feats), 400)

    def test_amlsim_v1_and_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            acc = Path(tmp) / "accounts.csv"
            v1 = write_synthetic_amlsim_csv(Path(tmp) / "tx.csv", schema="v1", accounts_path=acc)
            l1 = AMLSimLoader(v1, accounts_path=acc, chunksize=50)
            f1 = l1.read_all()
            self.assertEqual(l1.schema["src"], "SENDER_ACCOUNT_ID")
            self.assertEqual(int(f1["step"].min()), 0)
            self.assertTrue(set(f1["label"].unique()) <= {0, 1})
            self.assertEqual(len(l1.load_accounts()), 80)
            v2 = write_synthetic_amlsim_csv(Path(tmp) / "transactions.csv", schema="v2")
            f2 = AMLSimLoader(v2, chunksize=50).read_all()
            self.assertEqual(len(f1), len(f2))
            self.assertEqual(int(f2["step"].min()), 0)
            self.assertEqual(int(f2["step"].max()), 29)
            self.assertTrue(np.all(np.diff(f2["event_time"].to_numpy()) >= 0))
            out = CausalFeatureEngineer(CausalFeatureConfig(tx_types=("TRANSFER",))).transform(f2)
            self.assertEqual(len(out), len(f2))

    def test_build_loader_and_preprocess_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = json.loads(json.dumps(PROJECT_CONFIG))
            cfg["paths"]["processed_dir"] = "processed"
            cfg["datasets"]["paysim"]["transaction_path"] = "raw/paysim.csv"
            cfg["temporal_splits"]["paysim"] = {
                "train": {"start_step": 1, "end_step": 20},
                "calibration": {"start_step": 21, "end_step": 25},
                "operating": {"start_step": 26, "end_step": 30},
                "test": {"start_step": 31, "end_step": 40},
            }
            write_synthetic_paysim_csv(root / "raw" / "paysim.csv", n_steps=40)
            loader = build_loader("paysim", cfg, repo_root=root, chunksize=100)
            self.assertIsInstance(loader, PaySimLoader)
            manifest = preprocess_dataset("paysim", cfg, root, chunksize=100, fmt="csv")
            n_rows = len(pd.read_csv(root / "raw" / "paysim.csv"))
            self.assertEqual(manifest["rows_total"], n_rows)
            self.assertEqual(sum(manifest["rows_per_split"].values()), n_rows)
            self.assertEqual(set(manifest["rows_per_split"]), {"train", "calibration", "operating", "test"})
            test_df = pd.read_csv(manifest["outputs"]["test"])
            self.assertTrue(test_df["step"].between(31, 40).all())
            for col in FORBIDDEN_FEATURE_COLUMNS:
                self.assertNotIn(col, test_df.columns)
            self.assertTrue(Path(manifest["manifest_path"]).is_file())


class TestPerturbationGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PerturbationConfig.from_configs(PROJECT_CONFIG, HYPERPARAMS)
        self.gen = PerturbationGenerator(self.cfg, seed=SEED)

    def test_rates_honoured(self) -> None:
        n = self.cfg.n_candidates
        for rate, feed in self.gen.poison_rate_sweep().items():
            s = feed.summary()
            self.assertEqual(s["n_candidates"], n)
            self.assertAlmostEqual(s["realized_poison_rate"], round(rate * n) / n)
            self.assertAlmostEqual(s["realized_stale_rate"], round(self.cfg.stale_rate * n) / n)
            self.assertAlmostEqual(s["realized_conflict_rate"], round(self.cfg.conflict_rate * n) / n)
            self.assertEqual(sum(s["counts"].values()), n)
        self.assertEqual(sorted(self.gen.poison_rate_sweep()), [0.05, 0.10, 0.15, 0.20, 0.25])

    def test_bounds_validation(self) -> None:
        for bad in (0.01, 0.30, -0.1):
            with self.assertRaises(ValueError):
                self.gen.generate_feed(bad)
        clean = self.gen.generate_feed(0.0)
        self.assertEqual(clean.summary()["realized_poison_rate"], 0.0)

    def test_determinism(self) -> None:
        a = PerturbationGenerator(self.cfg, seed=SEED).generate_feed(0.15)
        b = PerturbationGenerator(self.cfg, seed=SEED).generate_feed(0.15)
        c = PerturbationGenerator(self.cfg, seed=SEED + 1).generate_feed(0.15)
        pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())
        self.assertFalse(a.to_frame().equals(c.to_frame()))

    def test_category_semantics(self) -> None:
        feed = self.gen.generate_feed(0.25)
        p0, horizon, now = self.cfg.p0, self.cfg.staleness_horizon_steps, self.cfg.current_step
        for r in feed.by_category("genuine"):
            self.assertGreater(r.true_precision, p0)
            self.assertFalse(r.is_harmful)
        for r in feed.by_category("stale"):
            self.assertTrue(r.is_harmful)
            self.assertGreater(r.historical_precision, p0)
            self.assertTrue(all(now - e.observed_at_step >= horizon for e in r.evidence))
        for r in feed.by_category("conflicting"):
            claims = [e.claimed_precision for e in r.evidence]
            self.assertGreater(max(claims), 0.6)
            self.assertLess(min(claims), 0.4)
            self.assertEqual(r.is_harmful, r.true_precision <= p0)
        for r in feed.by_category("poisoned"):
            self.assertLessEqual(r.true_precision, p0)
            self.assertEqual(len(r.evidence), 1)
            self.assertGreater(r.evidence[0].claimed_precision, 0.85)
            self.assertLessEqual(r.evidence[0].reliability, self.cfg.adversarial_reliability[1])
        sybil = feed.by_category("sybil")
        self.assertGreater(len(sybil), 0)
        for r in sybil:
            self.assertTrue(r.is_harmful)
            self.assertGreaterEqual(len(r.evidence), self.cfg.sybil_cluster_size[0])
            self.assertEqual(len({e.sybil_cluster_id for e in r.evidence}), 1)
            self.assertTrue(all(e.source_age_steps < 48 for e in r.evidence))

    def test_rules_execute_on_causal_features(self) -> None:
        features = CausalFeatureEngineer().transform(_paysim_canonical(n_steps=24))
        feed = self.gen.generate_feed(0.20)
        fired_any = False
        for rule in feed.rules:
            mask = rule.evaluate(features)
            self.assertEqual(mask.dtype, bool)
            self.assertEqual(mask.shape[0], len(features))
            fired_any |= bool(mask.any())
        self.assertTrue(fired_any)
        rule = CandidateRule("T", "genuine", (Condition("tx_type", "==", "TRANSFER"),
                                              Condition("log1p_amount", ">", 0.0)),
                             (), 0, 0.9, 0.9, False)
        np.testing.assert_array_equal(rule.evaluate(features),
                                      (features["tx_type"] == "TRANSFER").to_numpy())

    def test_outcome_simulation_and_serialisation(self) -> None:
        feed = self.gen.generate_feed(0.10)
        rng = np.random.default_rng(SEED)
        genuine = feed.by_category("genuine")[0]
        poisoned = feed.by_category("poisoned")[0]
        y_g = genuine.simulate_outcomes(4000, rng)
        y_p = poisoned.simulate_outcomes(4000, rng)
        self.assertAlmostEqual(y_g.mean(), genuine.true_precision, delta=0.03)
        self.assertAlmostEqual(y_p.mean(), poisoned.true_precision, delta=0.03)
        with tempfile.TemporaryDirectory() as tmp:
            path = feed.to_json(Path(tmp) / "feed.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["rules"]), len(feed))
            self.assertIn("predicate", payload["rules"][0])


if __name__ == "__main__":
    set_global_seed(SEED)
    unittest.main(verbosity=2)
