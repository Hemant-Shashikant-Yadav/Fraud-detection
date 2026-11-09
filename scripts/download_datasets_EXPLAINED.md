# Code Walkthrough: `scripts/download_datasets.py`

**Purpose:** One command to **download and verify** the raw datasets into `data/raw/`:
1. PaySim (`ealaxi/paysim1` → `data/raw/paysim/PS_20174392719_1491204439457_log.csv`)
2. IEEE-CIS (`ieee-fraud-detection` → `data/raw/ieee_cis/train_transaction.csv`, `train_identity.csv`)
3. AMLSim (IBM sample files from GitHub → `data/raw/amlsim/tx.csv`, `accounts.csv`)

```bash
python scripts/download_datasets.py                 # all
python scripts/download_datasets.py --dataset paysim
python scripts/download_datasets.py --force         # re-download
```

## Section-by-section
* **Constants (≈30–50):** target directories, expected file lists per dataset (`EXPECTED_FILES`), and the raw-GitHub URLs of the AMLSim sample files.
* **`parse_env_file(path)` (52–67):** minimal `.env` reader (`KEY=value`, ignores comments/blank lines).
* **`configure_kaggle_credentials(env_path)` (69–122):** reads `KAGGLE_API_TOKEN` **or** `KAGGLE_USERNAME` + `KAGGLE_KEY` from `.env` and exports them as environment variables for the Kaggle CLI. **`.env` is git-ignored — never commit it.**
* **`extract_all_zips(target_dir, remove_zips)` (124–138):** unpacks downloaded archives.
* **`download_paysim` / `download_ieee_cis` / `download_amlsim` (140–233):** skip if the expected files already exist (unless `--force`); otherwise call the Kaggle CLI through `python -m kaggle …` (so `pip install kaggle` is required — the cause of a "kaggle is not recognized" error; AMLSim uses `urllib`), extract, and verify the files exist. IEEE-CIS requires accepting the competition rules on Kaggle first.
* **`print_status_table` (235–260):** prints a ✓/✗ table of expected files and sizes; returns whether everything is present.
* **`main` (262–287):** CLI with `--dataset`, `--env-file`, `--force`; returns exit code 0 only if all selected datasets are present.

## Notes
* AMLSim: the downloaded GitHub files are only a **small sample**; the paper's graph/recovery claims need a larger simulator run.
* Raw data is git-ignored (`data/raw/**`) — Kaggle terms do not allow redistribution.
