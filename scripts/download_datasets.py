"""
scripts/download_datasets.py - Unified Dataset Ingestion Script
==============================================================

Downloads and verifies all raw datasets required by MG-ECFM:
  1. PaySim    (ealaxi/paysim1 -> data/raw/paysim)
  2. IEEE-CIS  (ieee-fraud-detection -> data/raw/ieee_cis)
  3. AMLSim    (IBM AMLSim / local fallback -> data/raw/amlsim)

Reads Kaggle API credentials directly from `.env` in the project root:
  - KAGGLE_API_TOKEN=KGAT_...
  OR
  - KAGGLE_USERNAME=... and KAGGLE_KEY=...

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --dataset paysim
    python scripts/download_datasets.py --dataset ieee_cis
    python scripts/download_datasets.py --force
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PAYSIM_DIR = RAW_DIR / "paysim"
IEEE_DIR = RAW_DIR / "ieee_cis"
AMLSIM_DIR = RAW_DIR / "amlsim"

EXPECTED_FILES = {
    "paysim": [PAYSIM_DIR / "PS_20174392719_1491204439457_log.csv"],
    "ieee_cis": [IEEE_DIR / "train_transaction.csv", IEEE_DIR / "train_identity.csv"],
    "amlsim": [AMLSIM_DIR / "tx.csv", AMLSIM_DIR / "accounts.csv"],
}

AML_URL_TX = "https://raw.githubusercontent.com/IBM/AMLSim/master/sample/outputs/tx.csv"
AML_URL_ACC = "https://raw.githubusercontent.com/IBM/AMLSim/master/sample/outputs/accounts.csv"


# --------------------------------------------------------------------------- #
# Environment & Credential Setup                                              #
# --------------------------------------------------------------------------- #
def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse key=value pairs from a .env file."""
    env: Dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        env[k] = v
    return env


def configure_kaggle_credentials(env_path: Path) -> Dict[str, str]:
    """Load credentials from .env, inject into os.environ, and write to ~/.kaggle/."""
    # 1. Load .env
    file_env = parse_env_file(env_path)
    for k, v in file_env.items():
        if k not in os.environ:
            os.environ[k] = v

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("KAGGLE_API_TOKEN") or file_env.get("KAGGLE_API_TOKEN")
    username = os.environ.get("KAGGLE_USERNAME") or file_env.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or file_env.get("KAGGLE_KEY")

    if token:
        try:
            token_file = kaggle_dir / "access_token"
            if not token_file.is_file() or token_file.read_text(encoding="utf-8").strip() != token.strip():
                token_file.write_text(token.strip(), encoding="utf-8")
                print(f"[auth] Kaggle API token written -> {token_file}")
        except Exception:
            pass
        os.environ["KAGGLE_API_TOKEN"] = token.strip()
        print("[auth] Kaggle API token configured in environment")
        return {"type": "token", "token": token[:8] + "..."}

    if username and key:
        try:
            json_file = kaggle_dir / "kaggle.json"
            json_file.write_text(f'{{"username":"{username.strip()}","key":"{key.strip()}"}}', encoding="utf-8")
            print(f"[auth] Kaggle credentials written -> {json_file}")
        except Exception:
            pass
        os.environ["KAGGLE_USERNAME"] = username.strip()
        os.environ["KAGGLE_KEY"] = key.strip()
        print("[auth] Kaggle credentials configured in environment")
        return {"type": "key", "username": username}

    # Check existing files
    if (kaggle_dir / "access_token").is_file():
        print(f"[auth] Found existing access_token in {kaggle_dir}")
        return {"type": "token_existing"}
    if (kaggle_dir / "kaggle.json").is_file():
        print(f"[auth] Found existing kaggle.json in {kaggle_dir}")
        return {"type": "key_existing"}

    print("[warn] No Kaggle credentials found in .env or ~/.kaggle/.", file=sys.stderr)
    print("       Create a .env file with KAGGLE_API_TOKEN=KGAT_... or place kaggle.json in ~/.kaggle/", file=sys.stderr)
    return {}


# --------------------------------------------------------------------------- #
# Unzipping Helpers                                                           #
# --------------------------------------------------------------------------- #
def extract_all_zips(target_dir: Path, remove_zips: bool = False) -> None:
    """Extract any .zip archives in target_dir and flatten if needed."""
    for zpath in list(target_dir.glob("*.zip")):
        print(f"[unzip] Extracting {zpath.name} -> {target_dir} ...")
        try:
            with zipfile.ZipFile(zpath, "r") as z:
                z.extractall(target_dir)
            if remove_zips:
                zpath.unlink()
        except Exception as exc:
            print(f"[error] Failed to extract {zpath}: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Dataset Download Handlers                                                   #
# --------------------------------------------------------------------------- #
def download_paysim(force: bool = False) -> bool:
    """Download PaySim dataset from Kaggle."""
    PAYSIM_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPECTED_FILES["paysim"][0]
    if target.is_file() and not force:
        size_mb = target.stat().st_size / (1024 ** 2)
        print(f"[paysim] Already present: {target.name} ({size_mb:.1f} MB). Skipping download.")
        return True

    print("[paysim] Downloading ealaxi/paysim1 via Kaggle CLI ...")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", "ealaxi/paysim1",
           "-p", str(PAYSIM_DIR), "--unzip"]
    res = subprocess.run(cmd)
    extract_all_zips(PAYSIM_DIR)
    ok = target.is_file()
    if ok:
        print(f"[paysim] Successfully downloaded and extracted to {target}")
    else:
        print(f"[paysim] Error: {target.name} not found after download.", file=sys.stderr)
    return ok


def download_ieee_cis(force: bool = False) -> bool:
    """Download IEEE-CIS dataset from Kaggle."""
    IEEE_DIR.mkdir(parents=True, exist_ok=True)
    all_present = all(p.is_file() for p in EXPECTED_FILES["ieee_cis"])
    if all_present and not force:
        print(f"[ieee_cis] Already present in {IEEE_DIR}. Skipping download.")
        return True

    print("[ieee_cis] Downloading ieee-fraud-detection files via Kaggle CLI ...")
    files_to_download = ["train_transaction.csv", "train_identity.csv"]
    success = True
    for fname in files_to_download:
        dest_file = IEEE_DIR / fname
        if dest_file.is_file() and not force:
            continue
        print(f"[ieee_cis] Downloading {fname} ...")
        cmd = [sys.executable, "-m", "kaggle", "competitions", "download", "-c", "ieee-fraud-detection",
               "-f", fname, "-p", str(IEEE_DIR)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err = res.stderr or res.stdout
            if "403" in err or "rules" in err.lower():
                print("\n" + "=" * 78, file=sys.stderr)
                print(" [ERROR] IEEE-CIS Rules Not Accepted Yet!", file=sys.stderr)
                print(" Please visit https://www.kaggle.com/competitions/ieee-fraud-detection/rules", file=sys.stderr)
                print(" in your browser, click 'I Understand and Accept', then run this script again.", file=sys.stderr)
                print("=" * 78 + "\n", file=sys.stderr)
                return False
            print(f"[ieee_cis] Warning while downloading {fname}: {err.strip()}", file=sys.stderr)
            success = False

    extract_all_zips(IEEE_DIR)
    ok = all(p.is_file() for p in EXPECTED_FILES["ieee_cis"])
    if ok:
        print(f"[ieee_cis] Files verified in {IEEE_DIR}")
    return ok


def download_amlsim(force: bool = False) -> bool:
    """Download or copy AMLSim accounts.csv and tx.csv."""
    AMLSIM_DIR.mkdir(parents=True, exist_ok=True)
    all_present = all(p.is_file() for p in EXPECTED_FILES["amlsim"])
    if all_present and not force:
        print(f"[amlsim] Already present in {AMLSIM_DIR}. Skipping.")
        return True

    # Check local fallback directory first
    local_candidates = [
        ROOT.parent / "Fraud-detection" / "old-code",
        Path.home() / "Downloads" / "Telegram Desktop" / "old-code",
    ]
    for cdir in local_candidates:
        if (cdir / "tx.csv").is_file() and (cdir / "accounts.csv").is_file():
            print(f"[amlsim] Found local files in {cdir}. Copying ...")
            shutil.copy(cdir / "tx.csv", AMLSIM_DIR / "tx.csv")
            shutil.copy(cdir / "accounts.csv", AMLSIM_DIR / "accounts.csv")
            return True

    # Otherwise fetch from official IBM AMLSim GitHub
    print("[amlsim] Fetching sample from IBM AMLSim repository ...")
    try:
        urllib.request.urlretrieve(AML_URL_TX, AMLSIM_DIR / "tx.csv")
        urllib.request.urlretrieve(AML_URL_ACC, AMLSIM_DIR / "accounts.csv")
        print(f"[amlsim] Downloaded tx.csv and accounts.csv to {AMLSIM_DIR}")
        return True
    except Exception as exc:
        print(f"[amlsim] Failed to download from GitHub: {exc}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------- #
# Status Audit Report                                                         #
# --------------------------------------------------------------------------- #
def print_status_table() -> bool:
    """Print an audit table of expected dataset files."""
    print("\n" + "=" * 78)
    print(" MG-ECFM Raw Datasets Status Audit")
    print("=" * 78)
    print(f"{'Dataset':<12} {'Filename':<35} {'Status':<10} {'Size (MB)':<12}")
    print("-" * 78)
    all_ok = True
    for ds_name, files in EXPECTED_FILES.items():
        for fpath in files:
            exists = fpath.is_file()
            if exists:
                size_mb = fpath.stat().st_size / (1024 ** 2)
                status = "OK"
                size_str = f"{size_mb:,.2f}"
            else:
                status = "MISSING"
                size_str = "-"
                all_ok = False
            print(f"{ds_name:<12} {fpath.name:<35} {status:<10} {size_str:<12}")
    print("=" * 78 + "\n")
    return all_ok


# --------------------------------------------------------------------------- #
# Main Entry Point                                                            #
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify MG-ECFM raw datasets using .env token.")
    parser.add_argument("--dataset", choices=["all", "paysim", "ieee_cis", "amlsim"], default="all",
                        help="Which dataset to download (default: all)")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env",
                        help="Path to .env file (default: .env in project root)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if files already exist locally")
    args = parser.parse_args()

    print("[init] Starting MG-ECFM dataset download manager ...")
    configure_kaggle_credentials(args.env_file)

    selected = ["paysim", "ieee_cis", "amlsim"] if args.dataset == "all" else [args.dataset]
    results = {}

    if "paysim" in selected:
        results["paysim"] = download_paysim(force=args.force)
    if "amlsim" in selected:
        results["amlsim"] = download_amlsim(force=args.force)
    if "ieee_cis" in selected:
        results["ieee_cis"] = download_ieee_cis(force=args.force)

    all_ok = print_status_table()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
