#!/usr/bin/env python
"""
scripts/setup_scaffolding.py
============================

Idempotent scaffolding generator for the MG-ECFM research repository
(Martingale-Gated Epistemic-Conformal Fraud Mitigation).

The script creates the canonical directory tree, Python package markers
(``__init__.py``), ``.gitkeep`` placeholders for data/report folders that must
exist but stay empty under version control, and a ``.gitignore`` that keeps raw
and processed datasets out of git. Existing files are never overwritten.

Usage
-----
    python scripts/setup_scaffolding.py                 # scaffold repo root
    python scripts/setup_scaffolding.py --root D:/x     # scaffold elsewhere
    python scripts/setup_scaffolding.py --dry-run       # show what would change
    python scripts/setup_scaffolding.py --verify        # exit 1 if incomplete
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

SEED = 42

DIRECTORIES = (
    "configs",
    "data",
    "data/raw",
    "data/raw/paysim",
    "data/raw/ieee_cis",
    "data/raw/amlsim",
    "data/processed",
    "data/synthetic",
    "data/synthetic/rule_feeds",
    "src",
    "src/data",
    "src/graph",
    "src/detection",
    "src/conformal",
    "src/epistemic_gate",
    "src/recovery",
    "src/closed_loop",
    "src/baselines",
    "experiments",
    "tests",
    "reports",
    "reports/figures",
    "reports/tables",
    "reports/manifests",
    "scripts",
)

PACKAGE_DOCSTRINGS: Dict[str, str] = {
    "src": "MG-ECFM: Martingale-Gated Epistemic-Conformal Fraud Mitigation.",
    "src/data": "Leakage-safe ingestion, causal feature engineering and "
                "adversarial rule-feed generation.",
    "src/graph": "Temporal node memory, camouflage similarity filtering and "
                 "inductive GraphSAGE.",
    "src/detection": "Tabular XGBoost, denoising autoencoder residuals and "
                     "stacked fusion scoring.",
    "src/conformal": "Cost-sensitive adaptive conformal risk control (CS-ACRC).",
    "src/epistemic_gate": "Subjective-logic opinions, SAVI martingale "
                          "e-processes and the versioned knowledge store.",
    "src/recovery": "Time-Decaying Tainted Proportional Flow (TD-TPF) and "
                    "recovery action simulation.",
    "src/closed_loop": "Delayed-label verification queue and feedback adapter.",
    "src/baselines": "Re-implemented state-of-the-art comparison baselines.",
    "experiments": "Experiment drivers for drift, ablation, stress and "
                   "significance studies.",
    "tests": "Phase-wise executable test suites.",
}

KEEP_DIRECTORIES = (
    "data/raw",
    "data/raw/paysim",
    "data/raw/ieee_cis",
    "data/raw/amlsim",
    "data/processed",
    "data/synthetic",
    "data/synthetic/rule_feeds",
    "reports",
    "reports/figures",
    "reports/tables",
    "reports/manifests",
)

GITIGNORE_CONTENT = """# --- MG-ECFM generated .gitignore ---
__pycache__/
*.py[cod]
.pytest_cache/
.ipynb_checkpoints/
.venv/
venv/
*.egg-info/
mlruns/

# Datasets and derived artefacts never enter version control
data/raw/**
data/processed/**
data/synthetic/**
!data/**/.gitkeep
!data/**/

reports/figures/**
reports/tables/**
reports/manifests/**
!reports/**/.gitkeep
"""

REQUIRED_CONFIGS = ("configs/project_config.json", "configs/hyperparams.json")


def build_scaffold(root: Path, dry_run: bool = False) -> Dict[str, List[str]]:
    """Create directories, package markers, keep-files and .gitignore.

    Args:
        root: Repository root directory.
        dry_run: If True, report planned actions without touching disk.

    Returns:
        Dictionary with keys ``created_dirs``, ``created_files`` and
        ``existing`` listing relative POSIX paths.
    """
    root = root.resolve()
    report: Dict[str, List[str]] = {
        "created_dirs": [], "created_files": [], "existing": []
    }

    for rel in DIRECTORIES:
        target = root / rel
        if target.is_dir():
            report["existing"].append(rel + "/")
            continue
        report["created_dirs"].append(rel + "/")
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)

    planned_files: Dict[str, str] = {}
    for pkg, doc in PACKAGE_DOCSTRINGS.items():
        planned_files[f"{pkg}/__init__.py"] = f'"""{doc}"""\n'
    for rel in KEEP_DIRECTORIES:
        planned_files[f"{rel}/.gitkeep"] = ""
    planned_files[".gitignore"] = GITIGNORE_CONTENT

    for rel, content in planned_files.items():
        target = root / rel
        if target.exists():
            report["existing"].append(rel)
            continue
        report["created_files"].append(rel)
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    return report


def verify_scaffold(root: Path) -> List[str]:
    """Return a list of missing directories / package markers / configs."""
    root = root.resolve()
    missing: List[str] = []
    for rel in DIRECTORIES:
        if not (root / rel).is_dir():
            missing.append(rel + "/")
    for pkg in PACKAGE_DOCSTRINGS:
        if not (root / pkg / "__init__.py").is_file():
            missing.append(f"{pkg}/__init__.py")
    for rel in REQUIRED_CONFIGS:
        if not (root / rel).is_file():
            missing.append(rel)
    return missing


def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="MG-ECFM repository scaffolding")
    parser.add_argument("--root", type=Path, default=_default_root(),
                        help="Repository root (default: parent of scripts/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned actions without writing")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify; exit 1 if anything is missing")
    args = parser.parse_args(argv)

    if args.verify:
        missing = verify_scaffold(args.root)
        if missing:
            print("[scaffold] INCOMPLETE - missing entries:")
            for item in missing:
                print(f"  - {item}")
            return 1
        print(f"[scaffold] OK - repository at {args.root.resolve()} is complete.")
        return 0

    report = build_scaffold(args.root, dry_run=args.dry_run)
    verb = "Would create" if args.dry_run else "Created"
    print(f"[scaffold] Root: {args.root.resolve()}")
    print(f"[scaffold] {verb} {len(report['created_dirs'])} directories, "
          f"{len(report['created_files'])} files; "
          f"{len(report['existing'])} entries already present.")
    for rel in report["created_dirs"] + report["created_files"]:
        print(f"  + {rel}")
    if not args.dry_run:
        missing_cfg = [c for c in REQUIRED_CONFIGS if not (args.root / c).is_file()]
        if missing_cfg:
            print("[scaffold] NOTE: config files not found yet: "
                  + ", ".join(missing_cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
