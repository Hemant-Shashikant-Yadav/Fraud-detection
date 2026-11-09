# Code Walkthrough: `src/baselines/__init__.py`

**Purpose:** Registry and factory for the four comparison baselines (BP1–BP4), so experiments can build any baseline from a short name.

---

## 1. Role in the architecture
The paper compares MG-ECFM against four re-implemented state-of-the-art models. This file gives them one entry point so `experiments/run_drift_eval.py` can loop over `["bp1", "bp2", "bp3", "bp4"]` without knowing the class names.

| Key | Class | Reference |
|---|---|---|
| `bp1` | `EARNRXTJ` | Almazroi & Ayub (2023) |
| `bp2` | `LambdaGNNAutoencoder` | Alarfaj & Shahzadi (2025) |
| `bp3` | `CAREGNN` | Dou et al. (2020) |
| `bp4` | `ARGUSConformal` | Nayak & Bushara (2026) |

## 2. Section-by-section

### 2.1 Docstring and imports (lines 1–26)
The docstring is the table above. Imports re-export each model, its config dataclass and a few helpers (`jaya_minimize`, `smote_oversample`, `crc_fnr_threshold`) so callers can write `from src.baselines import CAREGNN`.

### 2.2 `BASELINE_REGISTRY` (lines 28–33)
A dict `key -> class`. This is the single source of truth for "which baselines exist".

### 2.3 `_ALIASES` (lines 35–44)
Maps friendly names (`"care-gnn"`, `"argus"`, `"dou"`, ...) and each class's own `.name` to the canonical key. Lets configs and CLI users type the name they remember.

### 2.4 `resolve_baseline_name(name)` (lines 47–54)
Lower-cases the input, applies `_ALIASES`, and raises a `KeyError` listing every valid name and alias if it is unknown. Failing loudly prevents a silently skipped baseline.

### 2.5 `get_baseline_model(name, config=None, seed=SEED)` (lines 57–71)
Factory returning an **unfitted** model. `config` can be:
* `None` – use the published defaults;
* a plain mapping or the config dataclass for this baseline;
* a mapping keyed by baseline (`{"bp3": {...}, "bp4": {...}}`) – only this baseline's entry is used. Line 69 detects this form (`any(k in BASELINE_REGISTRY for k in config)`), so one experiment config can tune all four.

### 2.6 `__all__` (lines 74–79)
Declares the public API.

## 3. Notes
* Adding a fifth baseline = subclass `BaselineModel`, add to `BASELINE_REGISTRY` and (optionally) `_ALIASES`.
* The same `seed` is passed to every model for a fair comparison.
