"""
tests/test_phase8_orchestrator.py
=================================

Phase 8 verification: central orchestrator, camera-ready artifacts and dashboard.

1. ``python main.py --mode quick-demo`` runs end to end as a subprocess, exits 0,
   and every internal check of its manifest passes (all five models produce
   predictions for every test transaction, significance / ablation / stress
   outputs exist, single-transaction authorisation latency, camouflage graph,
   batch scoring, TD-TPF trace, service save/load round trip, runtime budget).
2. Prediction files are well formed: probabilities in [0, 1], valid actions,
   one row per (model, test transaction), chronological windows.
3. Camera-ready tables: the LaTeX is structurally balanced and contains the
   four labelled tables; the Markdown holds the same tables; the tables are
   also generated from the published Phase-7 outputs in ``reports/``.
4. Figures: four PDF + PNG pairs, valid file signatures, PNG resolution
   300 DPI (pHYs chunk), auditable figure data written next to them.
5. Batch CLI: ``python main.py --predict sample.csv`` scores the upload,
   strips the PaySim leakage columns and writes decisions.
6. Live service: what-if scoring does not mutate the live state, presets
   cover the dashboard, KB inspector shows Dirichlet opinions with
   ``b + d + u = 1``.
7. Dashboard: ``app/app.py`` compiles and (when Streamlit is installed)
   renders headlessly through ``streamlit.testing`` without exceptions.

Run from the repository root::

    python tests/test_phase8_orchestrator.py
    python -m pytest tests/test_phase8_orchestrator.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import re
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUICK_DIR = ROOT / "reports" / "quick_demo"
QUICK_SERVICE = ROOT / "models" / "quick_demo_service.pkl"
MODELS = ("mg_ecfm", "bp1", "bp2", "bp3", "bp4")
DEMO_TIMEOUT_S = 900


def run_cli(*args: str, timeout: float = DEMO_TIMEOUT_S) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(ROOT / "main.py"), *args], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def png_dpi(path: Path) -> float:
    """Horizontal resolution from the PNG ``pHYs`` chunk (pixels per metre -> DPI)."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    while pos < len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        if ctype == b"pHYs":
            ppx, _ppy, unit = struct.unpack(">IIB", data[pos + 8:pos + 17])
            return ppx * 0.0254 if unit == 1 else float("nan")
        pos += 12 + length
    return float("nan")


def latex_balanced(text: str) -> bool:
    body = re.sub(r"\\[{}%$&_#]", "", text)
    body = "\n".join(line.split("%")[0] if not line.lstrip().startswith("%") else "" for line in body.splitlines())
    depth = 0
    for ch in body:
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            return False
    return depth == 0 and body.count("$") % 2 == 0


DEMO: dict = {}


def setUpModule() -> None:
    """Run ``python main.py --mode quick-demo`` once, before any test class (they reuse its outputs)."""
    manifest_path = QUICK_DIR / "quick_demo_manifest.json"
    if manifest_path.is_file():
        manifest_path.unlink()          # never validate a stale manifest from an earlier run
    t0 = time.perf_counter()
    proc = run_cli("--mode", "quick-demo")
    DEMO["wall"] = time.perf_counter() - t0
    DEMO["proc"] = proc
    sys.stdout.write("\n----- quick-demo stdout (tail) -----\n")
    sys.stdout.write("\n".join(proc.stdout.splitlines()[-60:]) + "\n")
    if proc.returncode != 0:
        sys.stdout.write("----- quick-demo stderr (tail) -----\n")
        sys.stdout.write("\n".join(proc.stderr.splitlines()[-60:]) + "\n")
    sys.stdout.flush()
    DEMO["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}


class TestQuickDemoEndToEnd(unittest.TestCase):
    """Checks of the quick-demo run executed in :func:`setUpModule`."""

    proc: subprocess.CompletedProcess
    manifest: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.proc = DEMO["proc"]
        cls.manifest = DEMO["manifest"]

    def test_exit_code_zero(self) -> None:
        self.assertEqual(self.proc.returncode, 0, msg=self.proc.stderr[-3000:])

    def test_all_internal_checks_pass(self) -> None:
        checks = self.manifest.get("checks", [])
        self.assertGreaterEqual(len(checks), 14)
        failed = [c for c in checks if not c["passed"]]
        self.assertEqual(failed, [], msg=json.dumps(failed, indent=2))
        self.assertTrue(self.manifest.get("all_passed"))

    def test_runtime_budget(self) -> None:
        self.assertLess(self.manifest["runtime_seconds"], 60.0,
                        msg=f"stages: {self.manifest.get('stages_seconds')}")

    def test_every_model_predicts_every_test_transaction(self) -> None:
        pred = pd.read_csv(QUICK_DIR / "drift_evaluation_predictions.csv")
        res = pd.read_csv(QUICK_DIR / "drift_evaluation_results.csv")
        self.assertEqual(set(pred["model"]), set(MODELS))
        counts = pred.groupby("model").size()
        self.assertEqual(counts.nunique(), 1, msg=str(counts))
        for m in MODELS:
            sub = pred[pred["model"] == m]
            self.assertTrue(np.isfinite(sub["p"]).all())
            self.assertTrue(((sub["p"] >= 0) & (sub["p"] <= 1)).all())
            self.assertTrue(set(sub["action"].unique()) <= {0, 1, 2, 3})
            self.assertTrue((np.diff(sub["window"].to_numpy()) >= 0).all(), "windows must be chronological")
            self.assertTrue(set(sub["yhat"].unique()) <= {0, 1})
        self.assertEqual(res["window"].nunique(), 10)
        self.assertFalse(res.loc[res["model"] == "mg_ecfm", ["lambda_low", "lambda_mid", "lambda_high"]]
                         .isna().any().any())

    def test_experiment_outputs_exist(self) -> None:
        for name in ("drift_evaluation_manifest.json", "statistical_significance_results.json",
                     "ablation_study_results.csv", "ablation_study_windows.csv",
                     "adversarial_robustness_results.csv", "adversarial_time_to_rollback.csv",
                     "sample_transactions.csv", "sample_transactions_scored.csv", "knowledge_base_snapshot.csv"):
            self.assertTrue((QUICK_DIR / name).is_file(), name)
        abl = pd.read_csv(QUICK_DIR / "ablation_study_results.csv")
        self.assertEqual(len(abl), 7)
        stress = pd.read_csv(QUICK_DIR / "adversarial_robustness_results.csv")
        self.assertEqual(sorted(stress["poison_rate"].unique().round(2).tolist()), [0.05, 0.1, 0.15, 0.2, 0.25])

    def test_latency_profile_recorded(self) -> None:
        lat = self.manifest["latency"]
        self.assertGreaterEqual(lat["n"], 100)
        self.assertLessEqual(lat["p50_ms"], lat["p95_ms"])
        self.assertLess(lat["p95_ms"], 5.0)

    def test_scored_upload_strips_leakage(self) -> None:
        raw = pd.read_csv(QUICK_DIR / "sample_transactions.csv")
        scored = pd.read_csv(QUICK_DIR / "sample_transactions_scored.csv")
        self.assertEqual(len(raw), len(scored))
        self.assertIn("oldbalanceOrg", raw.columns)
        self.assertEqual(set(self.manifest["batch_summary"]["dropped_leakage_columns"]) >= {"oldbalanceOrg",
                                                                                              "newbalanceOrig"},
                         True)
        self.assertTrue(set(scored["action_label"]) <= {"Approve", "Step-Up", "Hold", "Block"})


class TestPaperTables(unittest.TestCase):

    def _check_pair(self, folder: Path) -> None:
        tex = (folder / "paper_tables.tex").read_text(encoding="utf-8")
        md = (folder / "paper_tables.md").read_text(encoding="utf-8")
        for key in ("dataset_summary", "benchmark", "ablation", "stress_test"):
            self.assertIn(f"\\label{{tab:{key}}}", tex)
        self.assertEqual(tex.count("\\begin{tabular}"), 4)
        self.assertEqual(tex.count("\\begin{tabular}"), tex.count("\\end{tabular}"))
        self.assertEqual(tex.count("\\begin{table"), tex.count("\\end{table"))
        self.assertTrue(latex_balanced(tex), "unbalanced braces or math delimiters in paper_tables.tex")
        text = "\n".join(line for line in tex.splitlines() if not line.lstrip().startswith("%"))
        text = re.sub(r"\\label\{[^}]*\}", "", re.sub(r"\$[^$]*\$", "", text))
        self.assertIsNone(re.search(r"[^\\]_[A-Za-z]", text), "unescaped underscore in LaTeX text")
        self.assertEqual(len(re.findall(r"^### Table \d", md, re.M)), 4)
        for name in ("MG-ECFM", "BP1", "BP4", "B1 XGBoost only", "B5 Assist ON"):
            self.assertIn(name, md)

    def test_quick_demo_tables(self) -> None:
        if not (QUICK_DIR / "paper_tables.tex").is_file():
            self.skipTest("quick demo did not produce tables")
        self._check_pair(QUICK_DIR)

    def test_tables_from_published_reports(self) -> None:
        from reports.generate_paper_tables import REQUIRED_INPUTS, generate_tables
        src = ROOT / "reports"
        if not all((src / n).is_file() for n in REQUIRED_INPUTS):
            self.skipTest("Phase-7 published outputs not present")
        with tempfile.TemporaryDirectory() as tmp:
            summary = generate_tables(src, Path(tmp), verbose=False)
            self.assertEqual(set(summary["tables"]), {"dataset_summary", "benchmark", "ablation", "stress_test"})
            self._check_pair(Path(tmp))
            md = (Path(tmp) / "paper_tables.md").read_text(encoding="utf-8")
            res = pd.read_csv(src / "drift_evaluation_results.csv")
            pr = res[res["model"] == "mg_ecfm"]["pr_auc"].mean()
            self.assertIn(f"{pr:.3f}", md, "table 2 must report the MG-ECFM mean PR-AUC from the CSV")

    def test_missing_inputs_raise(self) -> None:
        from reports.generate_paper_tables import load_inputs
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_inputs(Path(tmp))


class TestPaperFigures(unittest.TestCase):
    STEMS = ("fig1_pr_auc_drift_trajectories", "fig2_martingale_wealth_trajectories",
             "fig3_cs_acrc_threshold_dynamics", "fig4_tdtpf_recovery_vs_detection_delay")

    def test_quick_demo_figures(self) -> None:
        folder = QUICK_DIR / "figures"
        if not folder.is_dir():
            self.skipTest("quick demo did not produce figures")
        for stem in self.STEMS:
            pdf, png = folder / f"{stem}.pdf", folder / f"{stem}.png"
            self.assertTrue(pdf.is_file() and png.is_file(), stem)
            self.assertEqual(pdf.read_bytes()[:5], b"%PDF-")
            self.assertGreater(pdf.stat().st_size, 5_000)
            self.assertAlmostEqual(png_dpi(png), 300.0, delta=1.0)
        traj = pd.read_csv(folder / "figure_data" / "fig2_martingale_trajectories.csv")
        thr = traj["threshold"].iloc[0]
        stops = traj.drop_duplicates(["test", "rule_id"])
        crossed = traj[traj["stopping_time"] > 0].groupby(["test", "rule_id"])["wealth"].last()
        self.assertTrue((crossed >= thr * (1 - 1e-9)).all(), "a stopping time must coincide with M_t >= 1/alpha")
        self.assertGreater(len(stops), 0)
        curves = pd.read_csv(folder / "figure_data" / "fig4_tdtpf_recovery_curves.csv")
        late = curves[curves["delay_hours"] > curves["window_hours"]]
        self.assertTrue((late["recovery_rate_pct"] == 0).all(), "no recovery after the settlement window")
        self.assertTrue(((curves["recovery_rate_pct"] >= 0) & (curves["recovery_rate_pct"] <= 100 + 1e-9)).all())


class TestPredictCLI(unittest.TestCase):

    def test_predict_writes_decisions(self) -> None:
        sample = QUICK_DIR / "sample_transactions.csv"
        if not sample.is_file() or not QUICK_SERVICE.is_file():
            self.skipTest("quick demo artifacts missing")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scored.csv"
            proc = run_cli("--predict", str(sample), "--output", str(out), "--model", str(QUICK_SERVICE),
                           timeout=300)
            self.assertEqual(proc.returncode, 0, msg=proc.stdout[-2000:] + proc.stderr[-2000:])
            scored = pd.read_csv(out)
            self.assertEqual(len(scored), len(pd.read_csv(sample)))
            for col in ("p_hat", "action", "action_label", "route_reason", "mg_ecfm_step"):
                self.assertIn(col, scored.columns)
            self.assertTrue(((scored["p_hat"] >= 0) & (scored["p_hat"] <= 1)).all())
            self.assertIn("stripped post-event leakage columns", proc.stdout)

    def test_cli_rejects_missing_file(self) -> None:
        proc = run_cli("--predict", str(ROOT / "does_not_exist.csv"), timeout=120)
        self.assertNotEqual(proc.returncode, 0)

    def test_cli_help(self) -> None:
        proc = run_cli("--help", timeout=60)
        self.assertEqual(proc.returncode, 0)
        for mode in ("quick-demo", "train", "drift-eval", "ablation", "stress-test", "--predict", "--ui"):
            self.assertIn(mode, proc.stdout)


class TestLiveService(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if not QUICK_SERVICE.is_file():
            raise unittest.SkipTest("quick demo service not found")
        from src.orchestrator.service import MGECFMService
        cls.service = MGECFMService.load(QUICK_SERVICE)

    def test_what_if_scoring_is_side_effect_free(self) -> None:
        svc = self.service
        name, ex = next(iter(svc.example_transactions().items()))
        lam_before = tuple(svc.system.router.controller.thresholds)
        counts_before = list(svc.system.router.action_counts)
        n_ent = svc.engineer.n_entities
        a = svc.score_transaction(**ex, explain=False)
        b = svc.score_transaction(**ex, explain=True)
        self.assertAlmostEqual(a["p_hat"], b["p_hat"], places=12)
        self.assertEqual(tuple(svc.system.router.controller.thresholds), lam_before)
        self.assertEqual(list(svc.system.router.action_counts), counts_before)
        self.assertEqual(svc.engineer.n_entities, n_ent)
        self.assertIn(b["action_label"], ("Approve", "Step-Up", "Hold", "Block"))
        attrib = b["attribution"]
        margin = b["base_margin"] + attrib["shap_margin"].sum()
        self.assertAlmostEqual(margin, b["margin"], places=3, msg="TreeSHAP must add up to the XGBoost margin")

    def test_presets_cover_dashboard(self) -> None:
        ex = self.service.example_transactions()
        self.assertGreaterEqual(len(ex), 2)
        for spec in ex.values():
            self.assertEqual(set(spec), {"src", "dst", "amount", "tx_type"})

    def test_graph_explanation(self) -> None:
        name, ex = next(iter(self.service.example_transactions().items()))
        g = self.service.explain_graph(**ex)
        nodes, edges = g["nodes"], g["edges"]
        self.assertIn("sender", set(nodes["role"]))
        self.assertIn("receiver", set(nodes["role"]))
        if len(edges):
            self.assertTrue(((edges["similarity"] >= 0) & (edges["similarity"] <= 1)).all())
            self.assertEqual(g["stats"]["n_pruned"], int((~edges[edges["kind"] != "transaction"]["kept"]).sum()))

    def test_kb_opinions_are_valid(self) -> None:
        kb = self.service.kb_table()
        self.assertGreater(len(kb), 0)
        s = kb["belief"] + kb["disbelief"] + kb["uncertainty"]
        self.assertTrue(np.allclose(s, 1.0, atol=1e-9))
        self.assertTrue(set(kb["status"]) <= {"quarantined", "active", "rejected", "retired"})
        self.assertTrue(self.service.kb_summary()["integrity"]["ok"])

    def test_recovery_trace(self) -> None:
        cands = self.service.recovery_candidates()
        tr = cands[cands["traceable"]] if len(cands) else cands
        if tr.empty:
            self.skipTest("no traceable fraud in the demo history")
        rec = self.service.recovery_trace(int(tr.iloc[0]["tx_id"]), 2.0)
        self.assertLess(abs(rec["trace_metrics"]["conservation_residual"]), 1e-6)
        late = self.service.recovery_trace(int(tr.iloc[0]["tx_id"]), 30.0)
        self.assertEqual(late["metrics"]["recovered_amount"], 0.0)


class TestDashboard(unittest.TestCase):
    APP = ROOT / "app" / "app.py"

    def test_app_compiles(self) -> None:
        py_compile.compile(str(self.APP), doraise=True)

    def test_app_renders_headless(self) -> None:
        if importlib.util.find_spec("streamlit") is None:
            self.skipTest("streamlit not installed")
        if not QUICK_SERVICE.is_file():
            self.skipTest("quick demo service not found")
        from streamlit.testing.v1 import AppTest
        os.environ["MG_ECFM_SERVICE"] = str(QUICK_SERVICE)
        at = AppTest.from_file(str(self.APP), default_timeout=300)
        at.run()
        self.assertEqual(len(at.exception), 0, msg=str([e.value for e in at.exception]))
        self.assertGreaterEqual(len(at.tabs), 5)
        text = " ".join(m.value for m in at.markdown)
        self.assertTrue(any(lbl in text for lbl in ("Approve", "Step-Up", "Hold", "Block")),
                        "the action badge must be rendered")
        self.assertTrue(any("latency" in m.label.lower() for m in at.metric))
        # Score the second preset through the UI controls.
        if len(at.selectbox) and len(at.selectbox[0].options) > 1:
            at.selectbox[0].select_index(1).run()
            self.assertEqual(len(at.exception), 0, msg=str([e.value for e in at.exception]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
