# Detailed Code Walkthrough: `src/data/dataset_loaders.py`

**File Location:** `src/data/dataset_loaders.py`  
**Purpose:** Memory-safe, chronological streaming ingestion for PaySim, IEEE-CIS, and AMLSim datasets, enforcing schema canonicalization and physical leakage stripping at the file read boundary.

---

## 1. High-Level Summary & Role in Architecture

When training or running fraud detection on real-world transaction logs:
1. Files are often gigabytes in size (e.g., PaySim has 6.36 million rows). Loading them all into RAM causes crashes.
2. Different datasets use completely different column names (PaySim uses `nameOrig`, IEEE-CIS uses `card1` + `addr1`, AMLSim uses `SENDER_ACCOUNT_ID`).
3. PaySim contains post-decision balance columns (`oldbalanceOrg`, `newbalanceOrig`, etc.) which leak future information.

`dataset_loaders.py` solves all three issues:
* It reads data **chunk by chunk** (default: 50,000 to 250,000 rows at a time) so RAM usage stays under 180 MB.
* It standardizes every dataset into a single **Canonical Event Format** (`TransactionEvent`).
* It **refuses to read balance columns from disk**, so future data never enters memory.

---

## 2. Line-by-Line / Section-by-Section Code Explanation

### 2.1 Imports & Global Constants (Lines 27–52)
```python
CANONICAL_ORDER: Tuple[str, ...] = (
    "tx_id", "source_tx_id", "step", "event_time", "src", "dst",
    "amount", "tx_type", "label",
)
```
* **Line 27–46:** Standard library imports (`abc`, `dataclass`, `Path`, `typing`) and pandas/numpy.
* **Line 37–46:** Imports security constants from `causal_preprocessor`:
  * `FORBIDDEN_FEATURE_COLUMNS`: List of illegal balance columns (`oldbalanceOrg`, `newbalanceOrig`, etc.).
  * `TemporalOrderError`: Exception raised if a transaction's timestamp goes backward.
  * `assert_no_leakage`: Guard function to stop execution if leakage is found.
* **Line 48–52 (`CANONICAL_ORDER`):** Defines the fixed 9-column standard that all datasets must match. Every downstream model only expects these exact columns.

---

### 2.2 Canonical Data Structure: `TransactionEvent` (Lines 54–74)
```python
class TransactionEvent(NamedTuple):
    tx_id: int          # Global unique transaction number (0, 1, 2, ...)
    step: int           # Discrete time bucket (e.g., hour 1, hour 2)
    event_time: float   # Exact continuous time in seconds
    src: str            # Sender account / entity ID
    dst: str            # Receiver account / entity ID
    amount: float       # Transaction monetary amount
    tx_type: str        # Payment type: TRANSFER, CASH_OUT, PAYMENT, etc.
    label: int          # Ground truth: 1 for Fraud, 0 for Legitimate
    extras: Dict[str, object]  # Optional passthrough metadata
```
* **Why NamedTuple?** NamedTuples are lightweight, immutable, and consume far less memory than standard Python dictionaries or DataFrame rows.
* **`_ScanResult` (Lines 68–74):** A dataclass that records if a file is already sorted (`is_sorted`), the total rows (`n_rows`), and the minimum/maximum timestamps.

---

### 2.3 Identity String Creator: `_entity_string` (Lines 76–92)
```python
def _entity_string(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
```
* **The Problem:** In card datasets like IEEE-CIS, there is no single "account number". A user is identified by a combination of card ID, billing address, and email domain.
* **How it works:**
  1. Loops through each specified column (`card1`, `addr1`, `P_emaildomain`).
  2. If numbers are whole numbers (integers stored as floats), it casts them to `Int64` to avoid `1234.0` string bugs.
  3. Replaces missing values (`NaN`, `None`) with `"NA"`.
  4. Joins them into a single string: `card1=1234|addr1=300|P_emaildomain=gmail.com`.
* **Result:** Creates a stable, unique sender ID for every cardholder.

---

### 2.4 The Core Base Class: `BaseStreamLoader` (Lines 94–188)

Every dataset loader inherits from this class. It provides the streaming engine.

#### A. Initializer: `__init__` (Lines 101–128)
* Checks if the dataset file exists on disk; if not, gives a clear error message telling the user where to place it.
* Verifies `chunksize > 0`.
* **Security Check (Lines 122–124):**
  ```python
  leaked = [c for c in self.passthrough_columns if c in FORBIDDEN_FEATURE_COLUMNS]
  if leaked:
      raise ValueError(f"Passthrough of leakage columns is forbidden: {leaked}")
  ```
  Prevents any developer from accidentally passing balance columns through optional passthrough arguments.
* Reads row 0 (`nrows=0`) just to grab column headers without loading any data into RAM.

#### B. Dynamic Column Pruning: `_usecols()` (Lines 142–153)
```python
def _usecols(self) -> List[str]:
    ...
    return [c for c in cols if c not in FORBIDDEN_FEATURE_COLUMNS]
```
* **How it saves memory and prevents cheating:** When reading a 5 GB CSV, pandas only parses the columns listed in `_usecols`. By excluding `newbalanceOrig` and `oldbalanceOrg` here, **those columns are physically never read from disk into computer memory.**

#### C. Fast Sorting Scan: `scan_time_column()` (Lines 158–188)
* Scans *only* the time column from start to finish.
* Checks if every timestamp is greater than or equal to the previous one (`t >= last`).
* If the file is already sorted chronologically, we can stream it directly chunk-by-chunk without sorting.
* If it is not sorted, it flags `is_sorted = False` so the loader can sort it safely.

#### D. Stream Generator: `__iter__()` (Lines 190–240)
* Iterates through the file and yields clean, standardized `pd.DataFrame` chunks.
* Tracks a running counter `global_tx_id` starting from 0.
* Assigns `tx_id = [global_tx_id, global_tx_id + 1, ...]` to guarantee global, uninterrupted chronological indexing across chunk boundaries.
* Yields each chunk to the caller. When the chunk is processed, Python releases it from memory.

---

### 2.5 PaySim Loader: `PaySimLoader` (Lines 242–305)
```python
class PaySimLoader(BaseStreamLoader):
    dataset_name = "paysim"
    time_column = "step"
    step_seconds = 3600
```
* **`required_columns()`:** Requires `["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]`.
* **Notice:** `oldbalanceOrg` and `newbalanceOrig` are intentionally **omitted**.
* **`_canonicalize(raw)`:**
  * Maps `nameOrig` $\to$ `src` (Sender).
  * Maps `nameDest` $\to$ `dst` (Receiver).
  * Maps `type` $\to$ `tx_type` (`TRANSFER`, `CASH_OUT`, etc.).
  * Maps `isFraud` $\to$ `label` (1 or 0).
  * Computes `event_time = raw["step"] * 3600.0` (converts hour step into seconds).

---

### 2.6 IEEE-CIS Loader: `IEEECISLoader` (Lines 307–395)
```python
class IEEECISLoader(BaseStreamLoader):
    dataset_name = "ieee_cis"
    time_column = "TransactionDT"
```
* Handles real-world credit card transactions.
* **Identity Merging:** If an identity file (`train_identity.csv`) is provided, it loads identity features (like `DeviceType` and `DeviceInfo`) and joins them with the main transaction table on `TransactionID`.
* **Sender Creation:** Uses `_entity_string` on `card1`, `addr1`, and `P_emaildomain` to create a virtual account identifier.
* **Hourly Steps:** Computes `step = TransactionDT // 3600` so hourly rolling velocities can be computed identical to PaySim.

---

### 2.7 AMLSim Loader: `AMLSimLoader` (Lines 397–465)
```python
class AMLSimLoader(BaseStreamLoader):
    dataset_name = "amlsim"
    time_column = "TIMESTAMP"
    step_seconds = 86400  # 1 day steps
```
* Handles IBM's multi-hop money laundering simulation.
* Maps `SENDER_ACCOUNT_ID` $\to$ `src` and `RECEIVER_ACCOUNT_ID` $\to$ `dst`.
* Converts integer day timestamps into continuous seconds (`TIMESTAMP * 86400.0`).

---

### 2.8 Loader Factory & Synthetic Fixture Writers (Lines 467–658)

* **`build_loader(name, project_config, ...)`:**
  * Factory function that inspects `project_config.json`, finds the file path, and instantiates the correct loader (`PaySimLoader`, `IEEECISLoader`, or `AMLSimLoader`).
* **`write_synthetic_paysim_csv()`, `write_synthetic_ieee_cis_csv()`, `write_synthetic_amlsim_csv()`:**
  * Generates miniature, synthetic CSV files with the exact same columns and data types as the real datasets.
  * Used by the test suite so all tests can run and pass in seconds without needing the 5 GB real dataset files on disk.

---

## 3. Summary of How This File Protects the Pipeline

1. **Memory Safety:** Uses Python generator `yield` so that only one chunk of data is in RAM at any given moment ($\mathcal{O}(1)$ memory).
2. **Cheating Prevention:** Explicitly excludes post-decision balance fields at the CSV parsing level.
3. **Multi-Dataset Uniformity:** Whether the input is mobile money, card payments, or bank wires, the output is always the exact same 9-column `TransactionEvent` table.

