# Code Walkthrough: `scripts/run_phase1_demo.py`

**Purpose:** A **live demonstration and benchmark of Phase 1** (data ingestion and causal feature extraction), used for the first project presentation. Launched by `run_phase1.bat`.

It demonstrates:
1. **Ingestion and physical leakage stripping** — the post-decision balance columns are removed at read time.
2. **Chunked, memory-safe streaming** with boundary buffering (steps split across chunks are held back until complete).
3. **Causal rolling features** — every feature of a transaction uses only strictly earlier steps (`τ < t`).
4. **Intra-step permutation invariance** — shuffling transactions within one step must not change any feature.
5. **Per-transaction latency and throughput benchmark.**

## Section-by-section
* **Setup (1–40):** sets a temp `MPLCONFIGDIR`, puts the project root on `sys.path`, imports `CausalFeatureConfig`, `CausalFeatureEngineer` and the synthetic-data/loader helpers from `src/data/`.
* **`print_banner` (42–47):** title banner (project and candidate information).
* **`run_demo` (49–211):** runs five printed stages: (1) ingestion and leakage stripping — if the **real PaySim CSV is present** it streams its first 10 000 rows through `PaySimLoader`, otherwise it falls back to a small schema-faithful synthetic frame; (2) canonical harmonisation and chronological-order check; (3) causal rolling features and timing; then the permutation-invariance and latency/throughput checks. It prints the detected leakage columns, stripped columns, feature count, ms per batch, etc.

## Notes
* It works with or without the real dataset (it detects the file), and is a demonstration, not an experiment. The banner is hard-coded text from the first presentation (it prints a fixed PaySim row count and fraud rate, not values measured from the data).
