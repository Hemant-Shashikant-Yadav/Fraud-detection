"""
main.py - MG-ECFM unified command-line entry point (Phase 8 central orchestrator)
==================================================================================

Martingale-Gated Epistemic-Conformal Fraud Mitigation: one CLI for every stage
of the research pipeline.

Modes
-----
``--mode quick-demo``      end-to-end smoke run on a small synthetic stream (target < 60 s):
                           MG-ECFM + BP1-BP4 drift evaluation, significance tests, B1-B5
                           ablation, poisoning sweep, live service (real-time scoring, latency
                           profile, camouflage graph, batch CSV, TD-TPF trace, save/load),
                           camera-ready tables and figures -> ``reports/quick_demo/``
``--mode train``           fit the live MG-ECFM service on the training split and save it to
                           ``models/mg_ecfm_service.pkl`` (used by ``--predict`` and ``--ui``)
``--mode drift-eval``      50-window chronological drift benchmark (Phase 7)
``--mode ablation``        B1-B5 ablation study (Phase 7)
``--mode stress-test``     adversarial poisoning sweeps 5 %-25 % + Ville's-bound checks (Phase 7)
``--mode full-pipeline``   drift-eval -> significance -> ablation -> stress-test -> paper artifacts
``--mode paper-artifacts`` regenerate ``reports/paper_tables.{tex,md}`` and ``reports/figures/``
``--predict FILE.csv``     score a transaction CSV (PaySim or canonical schema) -> ``FILE_scored.csv``
``--ui``                   launch the Streamlit dashboard (``streamlit run app/app.py``)

Examples::

    python main.py --mode quick-demo
    python main.py --mode train --profile fast
    python main.py --predict data/samples/transactions.csv
    python main.py --ui
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 42
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
QUICK_DIR = REPORTS_DIR / "quick_demo"
DEFAULT_SERVICE_PATH = MODELS_DIR / "mg_ecfm_service.pkl"
QUICK_SERVICE_PATH = MODELS_DIR / "quick_demo_service.pkl"
APP_PATH = ROOT / "app" / "app.py"
QUICK_DEMO_BUDGET_S = 60.0

# Small stream and reduced baseline capacity for the quick demo (same architectures,
# fewer epochs) - the published numbers come from --mode drift-eval / full-pipeline.
QUICK_STREAM: Dict[str, Any] = {"n_steps": 160, "tx_per_step": (40, 70), "n_customers": 900,
                                "n_merchants": 90, "fraud_rate": 0.03, "drift_strength": 0.5, "seed": SEED}
QUICK_WINDOWS = 10
QUICK_BASELINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "bp1": {"earn_epochs": 2, "rxt_epochs": 2, "batch_size": 128, "ae_hidden": 16, "resnet_width": 16,
            "resnet_embedding": 8, "rxt_tokens": 4, "rxt_channels": 8, "cardinality": 2,
            "bottleneck_width": 4, "gru_hidden": 16, "smote_ratio": 0.5, "jaya_population": 3,
            "jaya_iterations": 1},
    "bp2": {"k_neighbors": 5, "gnn_hidden": 16, "gnn_epochs": 6, "gnn_batch_size": 128,
            "ae_epochs": 4, "ae_batch_size": 256, "patience": 4},
    "bp3": {"k_neighbors": 5, "embed_dim": 16, "epochs": 3, "batch_size": 256, "learning_rate": 0.01},
    "bp4": {"d_model": 16, "n_heads": 2, "n_blocks": 1, "ssm_state": 8, "epochs": 4,
            "batch_size": 64, "mc_samples": 4, "patience": 4},
}


def _jsonable(obj: Any) -> Any:
    from experiments.run_drift_eval import _jsonable as conv
    return conv(obj)


def _print_header(title: str) -> None:
    print("=" * 78)
    print(f" {title}")
    print("=" * 78, flush=True)


# --------------------------------------------------------------------------- #
# Quick demo                                                                  #
# --------------------------------------------------------------------------- #
class StageTimer:
    """Wall-clock bookkeeping of the demo stages."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.stages: Dict[str, float] = {}

    def run(self, name: str, fn: Callable[[], Any]) -> Any:
        t = time.perf_counter()
        print(f"[stage] {name} ...", flush=True)
        out = fn()
        self.stages[name] = time.perf_counter() - t
        print(f"[stage] {name} done in {self.stages[name]:.1f}s", flush=True)
        return out

    @property
    def total(self) -> float:
        return time.perf_counter() - self.t0


def make_sample_transactions(path: Path, n_steps: int = 6, seed: int = 7) -> Path:
    """A PaySim-schema sample upload (including the leakage columns, which the scorer strips)."""
    from src.data.dataset_loaders import make_synthetic_paysim_frame
    raw = make_synthetic_paysim_frame(n_steps=n_steps, tx_per_step=(25, 40), n_customers=900,
                                      n_merchants=90, fraud_rate=0.03, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path, index=False)
    return path


def run_quick_demo(out_dir: Path = QUICK_DIR, service_path: Path = QUICK_SERVICE_PATH,
                   seed: int = SEED) -> Dict[str, Any]:
    """End-to-end pipeline on a small synthetic stream; writes artifacts + a manifest of checks."""
    import numpy as np
    import pandas as pd

    from experiments.run_ablation_study import run_ablation_study
    from experiments.run_adversarial_stress_test import run_adversarial_stress_test
    from experiments.run_drift_eval import (DriftEvaluator, MODEL_ORDER, build_systems, load_configs,
                                            make_loss, prepare_splits, set_global_seed, summarise)
    from experiments.statistical_significance import run_statistical_tests
    from reports.generate_paper_figures import generate_figures
    from reports.generate_paper_tables import generate_tables
    from src.orchestrator.service import MGECFMService, ServiceConfig, profile_hyperparams

    _print_header("MG-ECFM quick demo (Phase 8)")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    timer = StageTimer()
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    hyperparams, project_config = load_configs()
    hp = profile_hyperparams("demo", hyperparams)
    stream_kwargs = dict(QUICK_STREAM, seed=seed)

    # 1. data -------------------------------------------------------------------
    splits = timer.run("causal stream + chronological splits",
                       lambda: prepare_splits("synthetic", QUICK_WINDOWS, project_config, None, stream_kwargs))
    s = splits.summary()
    print(f"        train={s['train']['transactions']} cal={s['calibration']['transactions']} "
          f"op={s['operating']['transactions']} test={s['test']['transactions']} "
          f"({s['test']['frauds']} frauds) in {s['n_windows']} windows", flush=True)

    # 2. all five models through the drift protocol -----------------------------
    systems = build_systems(splits, hp, project_config, QUICK_BASELINE_CONFIGS, MODEL_ORDER, 0.10, 1, seed)
    evaluator = DriftEvaluator(splits, systems, make_loss(project_config))
    fit_reports = timer.run("fit MG-ECFM + BP1-BP4", lambda: evaluator.fit_all(verbose=True))
    metrics, predictions = timer.run("prequential drift evaluation", lambda: evaluator.evaluate(verbose=True))
    manifest = {"protocol": "Menezes & Filho (2025) chronological drift evaluation (quick-demo profile)",
                "n_windows": QUICK_WINDOWS, "profile": "quick-demo", "seed": seed, "models": list(MODEL_ORDER),
                "label_delay_windows": 1, "candidate_poison_rate": 0.10,
                "splits": s, "fit_reports": _jsonable(fit_reports),
                "summary": _jsonable(summarise(metrics).to_dict(orient="records"))}
    metrics.to_csv(out_dir / "drift_evaluation_results.csv", index=False)
    predictions.to_csv(out_dir / "drift_evaluation_predictions.csv", index=False)
    (out_dir / "drift_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pooled = summarise(metrics).set_index("model")
    for m in MODEL_ORDER:
        sub = predictions[predictions["model"] == m]
        ok = len(sub) == s["test"]["transactions"] and bool(np.isfinite(sub["p"]).all()) \
            and bool(((sub["p"] >= 0) & (sub["p"] <= 1)).all())
        check(f"predictions[{m}]", ok, f"{len(sub)} scored test transactions, mean PR-AUC "
                                        f"{pooled.loc[m, 'pr_auc_mean']:.3f}")

    # 3. significance -----------------------------------------------------------
    sig = timer.run("statistical significance", lambda: run_statistical_tests(
        metrics, predictions, n_resamples=100, seed=seed, project_config=project_config))
    (out_dir / "statistical_significance_results.json").write_text(json.dumps(_jsonable(sig), indent=2),
                                                                   encoding="utf-8")
    check("significance tests", set(sig["mcnemar"]) == {"bp1", "bp2", "bp3", "bp4"},
          "McNemar + Wilcoxon + block bootstrap vs BP1-BP4")

    # 4. ablation (shares the splits) -------------------------------------------
    ablation, _ = timer.run("B1-B5 ablation", lambda: run_ablation_study(
        splits=splits, hyperparams_override=hp, results_path=out_dir / "ablation_study_results.csv",
        windows_path=out_dir / "ablation_study_windows.csv", seed=seed, verbose=False))
    check("ablation arms", len(ablation) == 7, ", ".join(ablation["arm"].tolist()))

    # 5. poisoning sweep ----------------------------------------------------------
    stress, _ = timer.run("adversarial stress test", lambda: run_adversarial_stress_test(
        n_candidates=100, n_cycles=20, ville_horizon=500, n_ville_reps=2000,
        results_path=out_dir / "adversarial_robustness_results.csv",
        ttr_path=out_dir / "adversarial_time_to_rollback.csv", seed=seed, verbose=False))
    gate_rows = stress[stress["policy"] == "mg_ecfm_gate"]
    check("Ville bound (all poisoning rates)", bool(gate_rows["ville_bound_satisfied"].astype(bool).all()),
          f"exact crossing prob {gate_rows['ville_exact_crossing_prob'].max():.4f} <= alpha")

    # 6. live service -------------------------------------------------------------
    service = timer.run("live service", lambda: MGECFMService.from_evaluated(
        systems["mg_ecfm"], splits, predictions, ServiceConfig.for_profile("demo"), hp))
    examples = service.example_transactions()
    first_name, first = next(iter(examples.items()))
    scored = service.score_transaction(**first, explain=True)
    check("single-transaction scoring", 0.0 <= scored["p_hat"] <= 1.0,
          f"'{first_name}': p={scored['p_hat']:.3f} -> {scored['action_label']} "
          f"({scored['route_reason']}), {len(scored['fired_rules'])} KB rules fired")
    lat = timer.run("latency profile (200 tx)", lambda: service.latency_benchmark(n=200))
    check("authorisation latency p95 < 5 ms", lat["p95_ms"] < 5.0,
          f"p50={lat['p50_ms']:.2f} ms p95={lat['p95_ms']:.2f} ms p99={lat['p99_ms']:.2f} ms "
          f"(causal featurisation p50={lat['featurize_p50_ms']:.2f} ms, timed separately)")
    graph = service.explain_graph(**first)
    check("camouflage graph", graph["stats"]["n_nodes"] >= 2,
          f"{graph['stats']['n_nodes']} nodes, {graph['stats']['n_neighbour_edges']} neighbour edges, "
          f"{graph['stats']['n_pruned']} pruned")
    sample_path = make_sample_transactions(out_dir / "sample_transactions.csv")
    batch, batch_summary = service.score_frame(pd.read_csv(sample_path))
    batch.to_csv(out_dir / "sample_transactions_scored.csv", index=False)
    check("batch CSV scoring", len(batch) > 0 and set(batch_summary["dropped_leakage_columns"]) >= {"oldbalanceOrg"},
          f"{len(batch)} rows in {batch_summary['ms_per_txn']:.3f} ms/tx; leakage columns stripped: "
          f"{batch_summary['dropped_leakage_columns']}")
    cands = service.recovery_candidates()
    traceable = cands[cands["traceable"]] if not cands.empty else cands
    if not traceable.empty:
        rec = service.recovery_trace(int(traceable.iloc[0]["tx_id"]), 1.0)
        check("TD-TPF recovery trace", abs(rec["trace_metrics"].get("conservation_residual", 0.0)) < 1e-6,
              f"tx {int(traceable.iloc[0]['tx_id'])}: recovered {rec['metrics']['recovery_rate_pct']:.1f}% "
              f"after 1 h, {int(rec['metrics']['n_accounts_frozen'])} accounts frozen")
    else:
        check("TD-TPF recovery trace", False, "no traceable fraud in the test period")
    service.kb_table().to_csv(out_dir / "knowledge_base_snapshot.csv", index=False)

    def roundtrip() -> float:
        service.save(service_path)
        loaded = MGECFMService.load(service_path)
        again = loaded.score_transaction(**first, explain=False)
        return abs(again["p_hat"] - scored["p_hat"])
    diff = timer.run("save / load service", roundtrip)
    check("service persistence", diff < 1e-9, f"{service_path.name}: |p_reload - p| = {diff:.2e}")

    # 7. camera-ready artifacts ---------------------------------------------------
    tables = timer.run("paper tables", lambda: generate_tables(out_dir, out_dir, verbose=False))
    check("tables", Path(tables["tex"]).is_file() and Path(tables["markdown"]).is_file(),
          f"{len(tables['tables'])} tables -> paper_tables.tex / .md")
    figs = timer.run("paper figures", lambda: generate_figures(out_dir, out_dir / "figures", horizon=120,
                                                               max_frauds=100, verbose=False))
    n_files = sum(Path(p).is_file() for f in figs.values() for p in f.values())
    check("figures", n_files == 8, f"{n_files} files (4 figures x PDF/PNG, 300 DPI)")

    total = timer.total
    check(f"runtime < {QUICK_DEMO_BUDGET_S:.0f}s", total < QUICK_DEMO_BUDGET_S, f"{total:.1f}s end-to-end")
    report = {
        "runtime_seconds": total, "stages_seconds": timer.stages, "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "pooled_metrics": _jsonable(pooled.reset_index()[["model", "pr_auc_mean", "f1_mean", "brier_mean",
                                                          "expected_monetary_loss_mean"]].to_dict("records")),
        "latency": lat, "example": {k: v for k, v in scored.items()
                                    if k not in ("attribution", "feature_deviation")},
        "batch_summary": batch_summary, "outputs": {"dir": str(out_dir), "service": str(service_path),
                                                    "tables": tables, "figures": figs},
    }
    (out_dir / "quick_demo_manifest.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    _print_header(f"quick demo finished in {total:.1f}s - "
                  f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed")
    return report


# --------------------------------------------------------------------------- #
# Other modes                                                                 #
# --------------------------------------------------------------------------- #
def run_train(profile: str, source: str, output: Path, seed: int, max_rows: Optional[int] = None) -> Dict[str, Any]:
    from src.orchestrator.service import MGECFMService, ServiceConfig
    _print_header(f"MG-ECFM train (profile={profile}, source={source}, max_rows={max_rows})")
    cfg = ServiceConfig.for_profile(profile, source=source, seed=seed, max_rows=max_rows)
    service = MGECFMService.build(cfg, verbose=True)
    lat = service.latency_benchmark(n=200)
    service.save(output)
    report = {"service": str(output), **service.describe(), "latency": lat}
    output.with_suffix(".json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print(f"[train] authorisation latency p50={lat['p50_ms']:.2f} ms p95={lat['p95_ms']:.2f} ms")
    print(f"[saved] {output}")
    return report


def load_or_build_service(model_path: Optional[Path], seed: int = SEED):
    """Trained service if present (models/), otherwise a demo service built on the fly and cached."""
    from src.orchestrator.service import MGECFMService, ServiceConfig
    for path in [p for p in (model_path, DEFAULT_SERVICE_PATH, QUICK_SERVICE_PATH) if p is not None]:
        if Path(path).is_file():
            print(f"[service] loading {path}", flush=True)
            return MGECFMService.load(path)
    print("[service] no trained service found - building the demo profile (python main.py --mode train "
          "for the full model)", flush=True)
    service = MGECFMService.build(ServiceConfig.for_profile("demo", seed=seed), verbose=True)
    service.save(QUICK_SERVICE_PATH)
    return service


def run_predict(csv_path: Path, output: Optional[Path], model_path: Optional[Path], seed: int) -> Path:
    import pandas as pd
    _print_header(f"MG-ECFM batch prediction: {csv_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    service = load_or_build_service(model_path, seed)
    scored, summary = service.score_frame(pd.read_csv(csv_path))
    out = output or csv_path.with_name(csv_path.stem + "_scored.csv")
    scored.to_csv(out, index=False)
    print(f"[predict] {summary['n']} transactions in {summary['seconds']:.2f}s "
          f"({summary['ms_per_txn']:.3f} ms/tx); actions: {summary['actions']}")
    if summary["dropped_leakage_columns"]:
        print(f"[predict] stripped post-event leakage columns: {summary['dropped_leakage_columns']}")
    if "metrics" in summary:
        m = summary["metrics"]
        print(f"[predict] labels found: PR-AUC={m['pr_auc']:.3f} F1={m['f1']:.3f} Brier={m['brier']:.4f}")
    print(f"[saved] {out}")
    return out


def run_paper_artifacts(reports_dir: Path) -> Dict[str, Any]:
    from reports.generate_paper_figures import generate_figures
    from reports.generate_paper_tables import generate_tables
    _print_header(f"camera-ready artifacts from {reports_dir}")
    tables = generate_tables(reports_dir)
    out_fig = REPORTS_DIR / "figures" if reports_dir.resolve() == REPORTS_DIR.resolve() else reports_dir / "figures"
    figs = generate_figures(reports_dir, out_fig)
    return {"tables": tables, "figures": figs}


def run_full_pipeline(args: argparse.Namespace) -> None:
    from experiments.run_ablation_study import run_ablation_study
    from experiments.run_adversarial_stress_test import run_adversarial_stress_test
    from experiments.run_drift_eval import run_drift_evaluation
    from experiments.statistical_significance import main as significance_main
    t0 = time.perf_counter()
    synth = {"drift_strength": args.drift_strength, "n_steps": args.n_steps, "seed": args.seed}
    _print_header("full pipeline 1/5: drift evaluation")
    run_drift_evaluation(source=args.source, profile=args.profile, n_windows=args.n_windows,
                         synthetic_kwargs=synth, max_rows=args.max_rows, seed=args.seed)
    _print_header("full pipeline 2/5: statistical significance")
    significance_main(["--seed", str(args.seed)])
    _print_header("full pipeline 3/5: ablation study")
    run_ablation_study(source=args.source, profile=args.profile, n_windows=args.n_windows,
                       synthetic_kwargs=synth, max_rows=args.max_rows, seed=args.seed)
    _print_header("full pipeline 4/5: adversarial stress test")
    run_adversarial_stress_test(seed=args.seed)
    _print_header("full pipeline 5/5: paper artifacts")
    run_paper_artifacts(REPORTS_DIR)
    print(f"[full-pipeline] finished in {time.perf_counter() - t0:.1f}s")


def launch_ui(port: int, model_path: Optional[Path]) -> int:
    if not APP_PATH.is_file():
        raise FileNotFoundError(APP_PATH)
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("streamlit is not installed: pip install -r requirements.txt", file=sys.stderr)
        return 2
    env = dict(os.environ)
    if model_path is not None:
        env["MG_ECFM_SERVICE"] = str(model_path)
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.port", str(port),
           "--server.headless", "false"]
    print("[ui] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
MODES = ("quick-demo", "train", "drift-eval", "ablation", "stress-test", "full-pipeline", "paper-artifacts")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python main.py",
                                 description="MG-ECFM unified CLI (Phase 8 orchestrator)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("Examples::")[1] if "Examples::" in __doc__ else None)
    ap.add_argument("--mode", choices=MODES, help="pipeline stage to run")
    ap.add_argument("--predict", type=Path, metavar="FILE.csv", help="score a transaction CSV")
    ap.add_argument("--ui", action="store_true", help="launch the Streamlit dashboard")
    ap.add_argument("--output", type=Path, default=None, help="output path (--predict / --mode train)")
    ap.add_argument("--model", type=Path, default=None, help="saved service (.pkl) for --predict / --ui")
    ap.add_argument("--profile", default="fast", choices=["demo", "fast", "full"],
                    help="model capacity for train / experiments (experiments accept fast|full)")
    ap.add_argument("--source", default="auto", choices=["auto", "synthetic", "paysim", "ieee_cis"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                    help="compute device for neural networks and XGBoost (auto, cuda, mps, cpu)")
    ap.add_argument("--n-windows", type=int, default=50)
    ap.add_argument("--n-steps", type=int, default=400, help="synthetic stream length")
    ap.add_argument("--drift-strength", type=float, default=0.5)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--reports-dir", type=Path, default=REPORTS_DIR, help="input of --mode paper-artifacts")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not (args.mode or args.predict or args.ui):
        ap.print_help()
        return 2
    if args.device != "auto":
        os.environ["MG_ECFM_DEVICE"] = args.device
    from src.utils.device import get_hardware_info
    hw = get_hardware_info()
    desc = hw.get("gpu_name") or f"CPU ({hw.get('cpu_threads', 1)} threads)"
    print(f"[device] Compute device: {hw['device']} ({desc})", flush=True)
    exp_profile = "full" if args.profile == "full" else "fast"
    synth = {"drift_strength": args.drift_strength, "n_steps": args.n_steps, "seed": args.seed}
    try:
        if args.ui:
            return launch_ui(args.port, args.model)
        if args.predict:
            run_predict(args.predict, args.output, args.model, args.seed)
            return 0
        if args.mode == "quick-demo":
            report = run_quick_demo(seed=args.seed)
            return 0 if report["all_passed"] else 1
        if args.mode == "train":
            source = "synthetic" if args.source == "auto" and args.profile == "demo" else args.source
            from experiments.run_drift_eval import load_configs, resolve_source
            source = resolve_source(source, load_configs()[1])
            run_train(args.profile, source, args.output or DEFAULT_SERVICE_PATH, args.seed, max_rows=args.max_rows)
            return 0
        if args.mode == "drift-eval":
            from experiments.run_drift_eval import run_drift_evaluation
            run_drift_evaluation(source=args.source, profile=exp_profile, n_windows=args.n_windows,
                                 synthetic_kwargs=synth, max_rows=args.max_rows, seed=args.seed)
            return 0
        if args.mode == "ablation":
            from experiments.run_ablation_study import run_ablation_study
            run_ablation_study(source=args.source, profile=exp_profile, n_windows=args.n_windows,
                               synthetic_kwargs=synth, max_rows=args.max_rows, seed=args.seed)
            return 0
        if args.mode == "stress-test":
            from experiments.run_adversarial_stress_test import run_adversarial_stress_test
            run_adversarial_stress_test(seed=args.seed)
            return 0
        if args.mode == "full-pipeline":
            args.profile = exp_profile
            run_full_pipeline(args)
            return 0
        if args.mode == "paper-artifacts":
            run_paper_artifacts(args.reports_dir)
            return 0
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        return 130
    except Exception:  # surface the full trace, non-zero exit for CI / batch runners
        traceback.print_exc()
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
