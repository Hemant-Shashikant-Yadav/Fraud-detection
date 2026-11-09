# Code Walkthrough: `src/utils/device.py`

**Purpose:** One place that decides *where* the models run (NVIDIA CUDA, Apple-Silicon MPS, or CPU) so no other file hard-codes a device.

**Used by:** every PyTorch model (`src/baselines/base.py`, GraphSAGE, autoencoder) and XGBoost (`tabular_engine.py`) through `resolve_device()` / `resolve_xgb_device()`.

---

## 1. Why this file exists

The project is developed on both Windows (NVIDIA) and macOS (Apple Silicon) machines, and must also run on plain CPUs (CI, the reviewer's laptop). Instead of scattering `torch.device("cuda")` everywhere, all code asks this module. The device is then an *input* to the experiment, not a code change.

## 2. Section-by-section

### 2.1 Module docstring and imports (lines 1–21)
Documents the **priority order** for choosing a device:
1. An explicit argument (`"cuda" | "mps" | "cpu" | "auto"`).
2. The environment variable `MG_ECFM_DEVICE`.
3. Automatic hardware detection.

Imports `os`/`platform` (environment and OS info), `torch` (the only heavy dependency).

### 2.2 `resolve_device(requested=None)` (lines 24–55)
```python
target = requested or os.environ.get("MG_ECFM_DEVICE") or "auto"
```
* **Line 32–33:** Applies the priority order and normalises to lowercase text.
* **Lines 35–39 (CUDA / "gpu"):** returns `cuda` only if `torch.cuda.is_available()`; otherwise prints a warning and **falls back to CPU** instead of crashing.
* **Lines 41–45 (MPS / "apple" / "metal"):** same pattern for Apple Silicon. `hasattr(torch.backends, "mps")` protects older PyTorch builds that do not have MPS.
* **Lines 47–48:** explicit `"cpu"` is honoured immediately.
* **Lines 50–55 (auto):** tries CUDA first, then MPS, then CPU.

**Design point:** a requested-but-missing GPU never raises; it degrades to CPU with a printed message, so a script written on one machine runs on another.

### 2.3 `resolve_xgb_device(device=None)` (lines 58–63)
XGBoost only understands `"cuda"` or `"cpu"` (it has no MPS support). The function maps the torch device to that vocabulary: CUDA stays `"cuda"`, everything else (including MPS) becomes `"cpu"`.

### 2.4 `get_hardware_info(device=None)` (lines 66–85)
Builds a dictionary of diagnostics (device, OS, Python and torch versions). Adds `gpu_name`, `gpu_count`, `cuda_version` for CUDA; a fixed label for MPS; `cpu_threads` for CPU. Intended to be written into run manifests so latency numbers in the paper can be reported against the exact hardware.

## 3. Things to remember
* No state is kept; each call re-inspects the machine.
* GraphSAGE/autoencoder can use CUDA/MPS; XGBoost uses the GPU only on CUDA.
* To force CPU for reproducible timings: set `MG_ECFM_DEVICE=cpu`.
