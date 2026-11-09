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
* **`run_demo` (49–211):** generates a small synthetic PaySim-style stream (including leakage columns), writes it to a temp CSV, then runs the five demonstrations above in order, printing results (columns stripped, rows streamed, feature count, invariance check pass/fail, ms per transaction and transactions per second). Returns a status used by the batch file for the `[SUCCESS]`/`[ERROR]` message.

## Notes
* It uses **synthetic** data by design (to be self-contained); it is a demonstration, not an experiment.
