"""
src/data/dataset_loaders.py
===========================

Chronological, memory-safe streaming loaders for PaySim, IEEE-CIS and AMLSim.

Every loader yields *canonical* chunks with the columns

    tx_id, source_tx_id, step, event_time, src, dst, amount, tx_type, label

plus optional ``raw_<col>`` passthrough columns. ``tx_id`` is the global
chronological ordinal of the row in the stream. Chunks are guaranteed to be
non-decreasing in ``step`` both within and across chunks; a violation raises
:class:`~src.data.causal_preprocessor.TemporalOrderError`.

Leakage control happens at ingestion: PaySim is read with ``usecols`` that
exclude the post-decision balance columns and ``isFlaggedFraud``, so those
values never enter memory.

Sorting: each loader first scans only its time column. If the file is already
chronological it is streamed in chunks (constant memory); otherwise only the
required columns are loaded, stably sorted by time, and re-chunked.

The module also provides schema-faithful synthetic fixture writers used by the
test-suite and for smoke runs without the real datasets.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.causal_preprocessor import (
    FORBIDDEN_FEATURE_COLUMNS,
    MISSING_ENTITY,
    PASSTHROUGH_PREFIX,
    PAYSIM_LEAKAGE_COLUMNS,
    SEED,
    TemporalOrderError,
    assert_no_leakage,
    strip_leakage_columns,
)

CANONICAL_ORDER: Tuple[str, ...] = (
    "tx_id", "source_tx_id", "step", "event_time", "src", "dst",
    "amount", "tx_type", "label",
)


class TransactionEvent(NamedTuple):
    """Single stream event ``e_t = (u_t, v_t, x_t, A_t, tau_t)``."""

    tx_id: int
    step: int
    event_time: float
    src: str
    dst: str
    amount: float
    tx_type: str
    label: int
    extras: Dict[str, object]


@dataclass
class _ScanResult:
    is_sorted: bool
    n_rows: int
    min_key: float
    max_key: float


def _entity_string(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Compose a deterministic entity key ``col=value|col=value`` with 'NA' for missing."""
    parts: List[pd.Series] = []
    for col in columns:
        s = frame[col]
        if pd.api.types.is_float_dtype(s):
            non_na = s.dropna()
            if non_na.empty or np.all(np.mod(non_na.to_numpy(), 1.0) == 0.0):
                s = s.astype("Int64")
        s = s.astype(object).where(s.notna(), "NA").astype(str)
        s = s.replace({"<NA>": "NA", "nan": "NA", "None": "NA"})
        parts.append(col + "=" + s)
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


class BaseStreamLoader(abc.ABC):
    """Abstract chronological chunk streamer producing canonical frames."""

    dataset_name = "base"
    time_column = "step"
    step_seconds = 3600

    def __init__(
        self,
        path: Path,
        chunksize: int = 250_000,
        max_rows: Optional[int] = None,
        ensure_sorted: bool = True,
        passthrough_columns: Sequence[str] = (),
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"[{self.dataset_name}] transaction file not found: {self.path}. "
                "Place the raw dataset under data/raw/ as configured in "
                "configs/project_config.json."
            )
        if chunksize <= 0:
            raise ValueError("chunksize must be positive")
        self.chunksize = int(chunksize)
        self.max_rows = int(max_rows) if max_rows is not None else None
        self.ensure_sorted = bool(ensure_sorted)
        self.passthrough_columns = tuple(passthrough_columns)
        leaked = [c for c in self.passthrough_columns if c in FORBIDDEN_FEATURE_COLUMNS]
        if leaked:
            raise ValueError(f"Passthrough of leakage columns is forbidden: {leaked}")
        self._header = list(pd.read_csv(self.path, nrows=0).columns)
        self._time_origin: Optional[float] = None
        self.last_scan: Optional[_ScanResult] = None

    # --------------------------------------------------------- abstract API
    @abc.abstractmethod
    def required_columns(self) -> List[str]:
        """Raw columns that must exist in the source file."""

    @abc.abstractmethod
    def _canonicalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map a raw chunk to canonical columns (without ``tx_id``)."""

    # ------------------------------------------------------------- helpers
    def optional_columns(self) -> List[str]:
        return []

    def _usecols(self) -> List[str]:
        required = self.required_columns()
        missing = [c for c in required if c not in self._header]
        if missing:
            raise KeyError(f"[{self.dataset_name}] missing required columns {missing} in {self.path}")
        missing_pt = [c for c in self.passthrough_columns if c not in self._header]
        if missing_pt:
            raise KeyError(f"[{self.dataset_name}] passthrough columns not in file: {missing_pt}")
        optional = [c for c in self.optional_columns() if c in self._header]
        cols = list(dict.fromkeys(required + optional + list(self.passthrough_columns)))
        return [c for c in cols if c not in FORBIDDEN_FEATURE_COLUMNS]

    def _time_key(self, series: pd.Series) -> np.ndarray:
        """Convert the raw time column to a sortable float array."""
        return pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)

    def scan_time_column(self) -> _ScanResult:
        """Single pass over the time column: sortedness, row count, min/max."""
        last = -np.inf
        is_sorted = True
        n_rows = 0
        lo, hi = np.inf, -np.inf
        reader = pd.read_csv(self.path, usecols=[self.time_column], chunksize=max(self.chunksize, 100_000))
        for chunk in reader:
            keys = self._time_key(chunk[self.time_column])
            if keys.size == 0:
                continue
            if np.isnan(keys).any():
                raise ValueError(f"[{self.dataset_name}] NaN in time column {self.time_column}")
            if is_sorted and (keys[0] < last or np.any(np.diff(keys) < 0)):
                is_sorted = False
            last = keys[-1]
            lo, hi = min(lo, keys.min()), max(hi, keys.max())
            n_rows += keys.size
        self.last_scan = _ScanResult(is_sorted, n_rows, float(lo), float(hi))
        self._time_origin = float(lo) if n_rows else 0.0
        return self.last_scan

    def _raw_chunks(self) -> Iterator[pd.DataFrame]:
        usecols = self._usecols()
        scan = self.scan_time_column()
        if scan.is_sorted or not self.ensure_sorted:
            reader = pd.read_csv(self.path, usecols=usecols, chunksize=self.chunksize,
                                 nrows=self.max_rows, low_memory=False)
            for chunk in reader:
                yield chunk
            return
        frame = pd.read_csv(self.path, usecols=usecols, low_memory=False)
        frame["__sort_key"] = self._time_key(frame[self.time_column])
        frame = frame.sort_values("__sort_key", kind="mergesort").drop(columns="__sort_key")
        if self.max_rows is not None:
            frame = frame.iloc[: self.max_rows]
        frame = frame.reset_index(drop=True)
        for start in range(0, len(frame), self.chunksize):
            yield frame.iloc[start:start + self.chunksize]

    def _attach_passthrough(self, raw: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
        for col in self.passthrough_columns:
            canon[PASSTHROUGH_PREFIX + col] = raw[col].to_numpy()
        return canon

    # ----------------------------------------------------------------- API
    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """Yield canonical, chronologically ordered, leakage-free chunks."""
        offset = 0
        last_step: Optional[int] = None
        for raw in self._raw_chunks():
            raw = strip_leakage_columns(raw.reset_index(drop=True))
            canon = self._canonicalize(raw)
            canon = self._attach_passthrough(raw, canon)
            canon.insert(0, "tx_id", np.arange(offset, offset + len(canon), dtype=np.int64))
            offset += len(canon)
            ordered = [c for c in CANONICAL_ORDER if c in canon.columns]
            extra = [c for c in canon.columns if c not in ordered]
            canon = canon[ordered + extra]

            steps = canon["step"].to_numpy(dtype=np.int64)
            if steps.size:
                if np.any(np.diff(steps) < 0) or (last_step is not None and steps[0] < last_step):
                    raise TemporalOrderError(
                        f"[{self.dataset_name}] chronological order violated near tx_id {offset}"
                    )
                last_step = int(steps[-1])
            if canon["amount"].isna().any():
                raise ValueError(f"[{self.dataset_name}] NaN transaction amount encountered")
            assert_no_leakage(canon)
            yield canon

    __iter__ = iter_chunks

    def iter_events(self) -> Iterator[TransactionEvent]:
        """Row-level event stream ``e_t`` (convenience; slower than chunks)."""
        base = set(CANONICAL_ORDER)
        for chunk in self.iter_chunks():
            extra_cols = [c for c in chunk.columns if c not in base]
            for row in chunk.itertuples(index=False):
                rec = row._asdict()
                yield TransactionEvent(
                    tx_id=int(rec["tx_id"]), step=int(rec["step"]),
                    event_time=float(rec["event_time"]), src=str(rec["src"]),
                    dst=str(rec["dst"]), amount=float(rec["amount"]),
                    tx_type=str(rec["tx_type"]), label=int(rec["label"]),
                    extras={c: rec[c] for c in extra_cols},
                )

    def read_all(self) -> pd.DataFrame:
        """Materialise the whole stream (use only for small data / tests)."""
        chunks = list(self.iter_chunks())
        if not chunks:
            return pd.DataFrame(columns=list(CANONICAL_ORDER))
        return pd.concat(chunks, ignore_index=True)


# --------------------------------------------------------------------------- #
# PaySim                                                                      #
# --------------------------------------------------------------------------- #
class PaySimLoader(BaseStreamLoader):
    """PaySim mobile-money simulator log (Lopez-Rojas et al., 2016).

    The balance columns are excluded through ``usecols``; ``isFlaggedFraud`` is
    excluded because it is the simulator's own rule output, not an observable.
    """

    dataset_name = "paysim"
    time_column = "step"
    step_seconds = 3600
    RAW_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]

    def required_columns(self) -> List[str]:
        return list(self.RAW_COLUMNS)

    def _canonicalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        steps = raw["step"].to_numpy(dtype=np.int64)
        return pd.DataFrame({
            "source_tx_id": np.arange(len(raw), dtype=np.int64),
            "step": steps,
            "event_time": steps.astype(np.float64) * self.step_seconds,
            "src": raw["nameOrig"].astype(str).to_numpy(),
            "dst": raw["nameDest"].fillna(MISSING_ENTITY).astype(str).to_numpy(),
            "amount": raw["amount"].to_numpy(dtype=np.float64),
            "tx_type": raw["type"].astype(str).to_numpy(),
            "label": raw["isFraud"].to_numpy(dtype=np.int8),
        })

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        for chunk in super().iter_chunks():
            # source_tx_id = global row ordinal in the (sorted) source file
            chunk["source_tx_id"] = chunk["tx_id"].to_numpy()
            yield chunk


# --------------------------------------------------------------------------- #
# IEEE-CIS                                                                    #
# --------------------------------------------------------------------------- #
class IEEECISLoader(BaseStreamLoader):
    """IEEE-CIS (Vesta) fraud detection, streamed in TransactionDT order.

    The source entity ``u_t`` is a composite card-holder key (default
    ``card1|addr1|P_emaildomain``); IEEE-CIS has no explicit beneficiary, so
    ``dst`` is empty unless ``destination_columns`` are given. ``step`` is
    ``TransactionDT // step_seconds``.
    """

    dataset_name = "ieee_cis"
    time_column = "TransactionDT"

    def __init__(
        self,
        path: Path,
        identity_path: Optional[Path] = None,
        entity_columns: Sequence[str] = ("card1", "addr1", "P_emaildomain"),
        destination_columns: Sequence[str] = (),
        identity_columns: Sequence[str] = ("DeviceType", "DeviceInfo"),
        step_seconds: int = 3600,
        **kwargs,
    ) -> None:
        self.entity_columns = tuple(entity_columns)
        self.destination_columns = tuple(destination_columns)
        self.step_seconds = int(step_seconds)
        super().__init__(path, **kwargs)
        self._identity: Optional[pd.DataFrame] = None
        self.identity_columns = tuple(identity_columns)
        if identity_path is not None and Path(identity_path).is_file():
            header = list(pd.read_csv(identity_path, nrows=0).columns)
            cols = ["TransactionID"] + [c for c in self.identity_columns if c in header]
            ident = pd.read_csv(identity_path, usecols=cols, low_memory=False)
            self._identity = ident.drop_duplicates("TransactionID").set_index("TransactionID")

    def required_columns(self) -> List[str]:
        return (["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"]
                + list(self.entity_columns) + list(self.destination_columns))

    def optional_columns(self) -> List[str]:
        return ["isFraud"]

    def _canonicalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        dt = raw["TransactionDT"].to_numpy(dtype=np.int64)
        label = (raw["isFraud"].to_numpy(dtype=np.int8) if "isFraud" in raw.columns
                 else np.full(len(raw), -1, dtype=np.int8))
        dst = (_entity_string(raw, self.destination_columns).to_numpy()
               if self.destination_columns else np.full(len(raw), MISSING_ENTITY, dtype=object))
        canon = pd.DataFrame({
            "source_tx_id": raw["TransactionID"].to_numpy(dtype=np.int64),
            "step": dt // self.step_seconds,
            "event_time": dt.astype(np.float64),
            "src": _entity_string(raw, self.entity_columns).to_numpy(),
            "dst": dst,
            "amount": raw["TransactionAmt"].to_numpy(dtype=np.float64),
            "tx_type": raw["ProductCD"].astype(str).to_numpy(),
            "label": label,
        })
        if self._identity is not None:
            joined = self._identity.reindex(raw["TransactionID"].to_numpy())
            for col in joined.columns:
                canon[PASSTHROUGH_PREFIX + col] = joined[col].to_numpy()
        return canon


# --------------------------------------------------------------------------- #
# AMLSim                                                                      #
# --------------------------------------------------------------------------- #
class AMLSimLoader(BaseStreamLoader):
    """IBM AMLSim transaction log (v1 ``tx.csv`` or v2 ``transactions.csv``).

    Column names are auto-detected from known aliases. Integer timestamps are
    treated as simulation steps (days); datetime strings are converted to days
    elapsed since the earliest timestamp in the file.
    """

    dataset_name = "amlsim"
    step_seconds = 86400
    ALIASES: Dict[str, Tuple[str, ...]] = {
        "id": ("tran_id", "TX_ID", "tx_id", "TXN_ID"),
        "src": ("orig_acct", "SENDER_ACCOUNT_ID", "nameOrig", "src", "ACCOUNT_ID"),
        "dst": ("bene_acct", "RECEIVER_ACCOUNT_ID", "nameDest", "dst", "COUNTER_PARTY_ACCOUNT_NUM"),
        "amount": ("base_amt", "TX_AMOUNT", "amount", "TXN_AMOUNT_ORIG"),
        "time": ("tran_timestamp", "TIMESTAMP", "step", "start"),
        "label": ("is_sar", "IS_FRAUD", "isFraud", "is_fraud", "IS_SAR"),
        "type": ("tx_type", "TX_TYPE", "type", "TXN_SOURCE_TYPE_CODE"),
    }

    def __init__(self, path: Path, accounts_path: Optional[Path] = None, **kwargs) -> None:
        header = list(pd.read_csv(path, nrows=0).columns) if Path(path).is_file() else []
        self.schema = self._resolve_schema(header) if header else {}
        if self.schema:
            self.time_column = self.schema["time"]
        self.accounts_path = Path(accounts_path) if accounts_path else None
        self._datetime_time: Optional[bool] = None
        super().__init__(path, **kwargs)

    @classmethod
    def _resolve_schema(cls, header: Sequence[str]) -> Dict[str, str]:
        schema: Dict[str, str] = {}
        for key, aliases in cls.ALIASES.items():
            found = next((a for a in aliases if a in header), None)
            if found is None and key in {"src", "dst", "amount", "time"}:
                raise KeyError(f"[amlsim] cannot find a column for '{key}' (aliases {aliases})")
            if found is not None:
                schema[key] = found
        return schema

    def required_columns(self) -> List[str]:
        return [self.schema[k] for k in ("src", "dst", "amount", "time")]

    def optional_columns(self) -> List[str]:
        return [self.schema[k] for k in ("id", "label", "type") if k in self.schema]

    def _time_key(self, series: pd.Series) -> np.ndarray:
        if self._datetime_time is None:
            numeric = pd.to_numeric(series, errors="coerce")
            self._datetime_time = bool(numeric.isna().any())
        if self._datetime_time:
            ts = pd.to_datetime(series, utc=True, errors="raise")
            epoch = pd.Timestamp("1970-01-01", tz="UTC")
            return (ts - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
        return pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)

    @staticmethod
    def _parse_label(series: pd.Series) -> np.ndarray:
        if series.dtype == bool:
            return series.to_numpy(dtype=np.int8)
        mapped = series.astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
        )
        if mapped.isna().any():
            raise ValueError("[amlsim] unrecognised label values")
        return mapped.to_numpy(dtype=np.int8)

    def _canonicalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        s = self.schema
        keys = self._time_key(raw[s["time"]])
        if self._datetime_time:
            origin = self._time_origin if self._time_origin is not None else float(keys.min())
            event_time = keys - origin
            steps = np.floor(event_time / self.step_seconds).astype(np.int64)
        else:
            steps = keys.astype(np.int64)
            event_time = steps.astype(np.float64) * self.step_seconds
        n = len(raw)
        return pd.DataFrame({
            "source_tx_id": (raw[s["id"]].to_numpy() if "id" in s else np.arange(n)),
            "step": steps,
            "event_time": event_time,
            "src": raw[s["src"]].astype(str).to_numpy(),
            "dst": raw[s["dst"]].fillna(MISSING_ENTITY).astype(str).to_numpy(),
            "amount": raw[s["amount"]].to_numpy(dtype=np.float64),
            "tx_type": (raw[s["type"]].astype(str).to_numpy() if "type" in s
                        else np.full(n, "TRANSFER", dtype=object)),
            "label": (self._parse_label(raw[s["label"]]) if "label" in s
                      else np.full(n, -1, dtype=np.int8)),
        })

    def load_accounts(self) -> pd.DataFrame:
        """Load the (small) AMLSim account table, if configured."""
        if self.accounts_path is None or not self.accounts_path.is_file():
            raise FileNotFoundError(f"[amlsim] accounts file not found: {self.accounts_path}")
        return pd.read_csv(self.accounts_path, low_memory=False)


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
def build_loader(
    dataset: str,
    project_config: Mapping,
    repo_root: Path = Path("."),
    chunksize: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> BaseStreamLoader:
    """Instantiate the configured loader for ``dataset``."""
    ds = project_config["datasets"][dataset]
    root = Path(repo_root)
    size = int(chunksize or ds.get("chunksize", 250_000))
    if dataset == "paysim":
        return PaySimLoader(root / ds["transaction_path"], chunksize=size, max_rows=max_rows)
    if dataset == "ieee_cis":
        ident = ds.get("identity_path")
        rows = max_rows if max_rows is not None else ds.get("subset_rows")
        return IEEECISLoader(
            root / ds["transaction_path"],
            identity_path=(root / ident) if ident else None,
            entity_columns=ds.get("entity_columns", ("card1", "addr1", "P_emaildomain")),
            destination_columns=ds.get("destination_columns", ()),
            identity_columns=ds.get("identity_columns", ("DeviceType", "DeviceInfo")),
            step_seconds=int(ds.get("step_seconds", 3600)),
            passthrough_columns=ds.get("passthrough_columns", ()),
            chunksize=size,
            max_rows=rows,
        )
    if dataset == "amlsim":
        acc = ds.get("accounts_path")
        return AMLSimLoader(root / ds["transaction_path"],
                            accounts_path=(root / acc) if acc else None,
                            chunksize=size, max_rows=max_rows)
    raise ValueError(f"Unknown dataset '{dataset}'")


# --------------------------------------------------------------------------- #
# Schema-faithful synthetic fixtures                                          #
# --------------------------------------------------------------------------- #
_PAYSIM_TYPE_PROBS = {"CASH_IN": 0.22, "CASH_OUT": 0.35, "DEBIT": 0.01,
                      "PAYMENT": 0.34, "TRANSFER": 0.08}


def make_synthetic_paysim_frame(
    n_steps: int = 48,
    tx_per_step: Tuple[int, int] = (15, 40),
    n_customers: int = 300,
    n_merchants: int = 60,
    fraud_rate: float = 0.03,
    seed: int = SEED,
) -> pd.DataFrame:
    """PaySim-schema frame *including* the leakage columns and their artefacts.

    Fraud rows reproduce the documented PaySim cancellation artefacts (origin
    drained to zero, destination balances left at zero), so tests can verify
    that these columns are removed before feature engineering.
    """
    rng = np.random.default_rng(seed)
    types = np.array(list(_PAYSIM_TYPE_PROBS))
    probs = np.array(list(_PAYSIM_TYPE_PROBS.values()))
    rows = []
    for step in range(1, n_steps + 1):
        n = int(rng.integers(tx_per_step[0], tx_per_step[1] + 1))
        for _ in range(n):
            t = str(rng.choice(types, p=probs))
            orig = f"C{int(rng.integers(0, n_customers)):09d}"
            dest = (f"M{int(rng.integers(0, n_merchants)):09d}" if t == "PAYMENT"
                    else f"C{int(rng.integers(0, n_customers)):09d}")
            amount = float(np.round(rng.lognormal(mean=9.0, sigma=1.2), 2))
            is_fraud = int(t in {"TRANSFER", "CASH_OUT"} and rng.random() < fraud_rate * 4)
            old_org = float(np.round(rng.uniform(0, 3e5), 2))
            old_dst = float(np.round(rng.uniform(0, 5e5), 2))
            if is_fraud:
                amount = max(old_org, 1.0)
                new_org, old_dst, new_dst = 0.0, 0.0, 0.0
            elif t == "CASH_IN":
                new_org, new_dst = old_org + amount, max(old_dst - amount, 0.0)
            else:
                new_org, new_dst = max(old_org - amount, 0.0), old_dst + amount
            rows.append({
                "step": step, "type": t, "amount": amount,
                "nameOrig": orig, "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
                "nameDest": dest, "oldbalanceDest": old_dst, "newbalanceDest": new_dst,
                "isFraud": is_fraud,
                "isFlaggedFraud": int(is_fraud and t == "TRANSFER" and amount > 200_000),
            })
    return pd.DataFrame(rows)


def write_synthetic_paysim_csv(path: Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_paysim_frame(**kwargs).to_csv(path, index=False)
    return path


def write_synthetic_ieee_cis_csv(
    path: Path,
    n_rows: int = 600,
    shuffle: bool = True,
    seed: int = SEED,
    identity_path: Optional[Path] = None,
) -> Path:
    """IEEE-CIS-schema transaction file, optionally row-shuffled (unsorted)."""
    rng = np.random.default_rng(seed)
    dt = np.sort(86400 + rng.integers(0, 30 * 86400, size=n_rows))
    addr = rng.choice([299.0, 315.0, 325.0, np.nan], size=n_rows)
    email = rng.choice(["gmail.com", "yahoo.com", "anonymous.com", None], size=n_rows)
    frame = pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + n_rows),
        "isFraud": (rng.random(n_rows) < 0.035).astype(int),
        "TransactionDT": dt,
        "TransactionAmt": np.round(rng.lognormal(4.0, 1.0, n_rows), 3),
        "ProductCD": rng.choice(["W", "H", "C", "S", "R"], size=n_rows),
        "card1": rng.integers(1000, 1040, size=n_rows),
        "addr1": addr,
        "P_emaildomain": email,
        "C1": rng.integers(0, 10, size=n_rows).astype(float),
        "D1": rng.integers(0, 500, size=n_rows).astype(float),
        "V1": rng.choice([1.0, np.nan], size=n_rows),
    })
    if shuffle:
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if identity_path is not None:
        ids = frame["TransactionID"].sample(frac=0.4, random_state=seed).to_numpy()
        pd.DataFrame({
            "TransactionID": ids,
            "DeviceType": rng.choice(["mobile", "desktop"], size=ids.size),
            "DeviceInfo": rng.choice(["Windows", "iOS Device", "MacOS"], size=ids.size),
        }).to_csv(identity_path, index=False)
    return path


def write_synthetic_amlsim_csv(
    path: Path,
    schema: str = "v1",
    n_accounts: int = 80,
    n_days: int = 30,
    tx_per_day: Tuple[int, int] = (5, 20),
    seed: int = SEED,
    accounts_path: Optional[Path] = None,
) -> Path:
    """AMLSim-schema transaction file in v1 (tx.csv) or v2 (transactions.csv) form."""
    rng = np.random.default_rng(seed)
    records = []
    tx_id = 0
    origin = pd.Timestamp("2017-01-01", tz="UTC")
    for day in range(n_days):
        for _ in range(int(rng.integers(tx_per_day[0], tx_per_day[1] + 1))):
            src, dst = rng.choice(n_accounts, size=2, replace=False)
            sar = bool(rng.random() < 0.05)
            amount = float(np.round(rng.uniform(100, 5000), 2))
            seconds = int(rng.integers(0, 86400))
            if schema == "v1":
                records.append({"TX_ID": tx_id, "SENDER_ACCOUNT_ID": int(src),
                                "RECEIVER_ACCOUNT_ID": int(dst), "TX_TYPE": "TRANSFER",
                                "TX_AMOUNT": amount, "TIMESTAMP": day,
                                "IS_FRAUD": sar, "ALERT_ID": (int(tx_id % 7) if sar else -1)})
            elif schema == "v2":
                ts = origin + pd.Timedelta(days=day, seconds=seconds)
                records.append({"tran_id": tx_id, "orig_acct": int(src), "bene_acct": int(dst),
                                "tx_type": "TRANSFER", "base_amt": amount,
                                "tran_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "is_sar": "True" if sar else "False",
                                "alert_id": (int(tx_id % 7) if sar else -1)})
            else:
                raise ValueError("schema must be 'v1' or 'v2'")
            tx_id += 1
    frame = pd.DataFrame(records)
    if schema == "v2":
        frame = frame.sort_values("tran_timestamp", kind="mergesort").reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if accounts_path is not None:
        pd.DataFrame({
            "ACCOUNT_ID": np.arange(n_accounts),
            "CUSTOMER_ID": [f"C_{i}" for i in range(n_accounts)],
            "INIT_BALANCE": np.round(rng.uniform(1000, 50000, n_accounts), 2),
            "COUNTRY": "US",
            "ACCOUNT_TYPE": "SAV",
            "IS_FRAUD": rng.random(n_accounts) < 0.05,
            "TX_BEHAVIOR_ID": rng.integers(1, 5, n_accounts),
        }).to_csv(accounts_path, index=False)
    return path


__all__ = [
    "AMLSimLoader", "BaseStreamLoader", "IEEECISLoader", "PaySimLoader",
    "TransactionEvent", "build_loader", "make_synthetic_paysim_frame",
    "write_synthetic_amlsim_csv", "write_synthetic_ieee_cis_csv",
    "write_synthetic_paysim_csv", "PAYSIM_LEAKAGE_COLUMNS",
]
