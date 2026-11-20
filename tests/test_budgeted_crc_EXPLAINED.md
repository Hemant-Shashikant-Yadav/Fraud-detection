# Code Walkthrough: `tests/test_budgeted_crc.py`

**Purpose:** Unit tests of the budgeted Bayes router controller (`src/conformal/budgeted_crc.py`) and of its integration with `PreTransactionRouter`.

**Run:** `MG_ECFM_DEVICE=cpu python -m pytest -q tests/test_budgeted_crc.py`

---

## 1. Test data
`drift_stream(n, rates)` makes a stream whose fraud rate rises from `rates[0]` to `rates[1]`. The scores are informative but do not follow the drift, and the amounts are log-normal.

## 2. Test classes

| Class | What it checks |
|:---|:---|
| `TestDecisionRule` | The rule equals the Bayes argmin of the cost matrix at ν = 0. The action is monotone in the probability and in the price. The batch and scalar paths agree. The reported cut points are strictly increasing. |
| `TestGuarantees` | The intervention-budget bound and the telescoping identity hold, from a cold start and from a warm start. The calibration bound holds under drift. Static mode freezes the state. The certificates stay correct after a mode switch and after `state_dict` / `load_state_dict`. The warm start meets the budget. At a 5% budget, the controller loses less than the legacy λ-band router. |
| `TestRouterIntegration` | `build_controller` follows `decision_rule`. The router uses the amount. The rule floor and the feedback work. A router without a `_bayes` attribute (an old pickle) still works. The routing p99 latency is under 1 ms. |
