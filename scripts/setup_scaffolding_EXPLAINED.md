# Code Walkthrough: `scripts/setup_scaffolding.py`

**Purpose:** An **idempotent scaffolding generator** that created the repository's canonical layout: directory tree, Python package markers (`__init__.py`), `.gitkeep` placeholders for data/report folders that must exist but stay empty in git, and a `.gitignore` that keeps datasets out of version control. Existing files are **never overwritten**.

```bash
python scripts/setup_scaffolding.py              # scaffold the repo root
python scripts/setup_scaffolding.py --root D:/x  # scaffold elsewhere
python scripts/setup_scaffolding.py --dry-run    # show what would change
python scripts/setup_scaffolding.py --verify     # exit 1 if anything is missing
```

## Section-by-section
* **Constants (≈28–116):** `SEED`, `DIRECTORIES` (data/raw/{paysim,ieee_cis,amlsim}, data/processed, data/synthetic, src/*, experiments, reports/*, tests, configs, ...), the package list, the `.gitkeep` locations and the `.gitignore` text (datasets, derived artefacts, large result files).
* **`build_scaffold(root, dry_run)` (118–161):** creates missing directories, package markers and placeholders and writes `.gitignore` only if absent; returns a dict of what was created/skipped (and only reports in dry-run mode).
* **`verify_scaffold(root)` (163–177):** returns the list of missing required paths.
* **`_default_root` / `main` (179–end):** the project root (parent of `scripts/`) and the CLI; `--verify` exits with status 1 when the scaffold is incomplete.

## Notes
* Safe to re-run at any time. It is a one-time project-setup helper, not part of the experiments.
