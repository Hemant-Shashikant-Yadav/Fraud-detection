# Detailed Code Walkthrough: `src/data/perturbation_generator.py`

**File Location:** `src/data/perturbation_generator.py`  
**Purpose:** Synthetic generator of contaminated threat-intelligence rule feeds. Simulates genuine rules, concept-drift decay (stale rules), intelligence disagreements (conflicting rules), malicious injections (poisoned rules), and coordinated attacker clusters (Sybil attacks) with controllable contamination rates ($\rho \in [5\%, 25\%]$).

---

## 1. High-Level Summary & Role in Architecture

In modern banking consortiums:
* Anti-fraud rules are constantly proposed by machine learning pipelines, external analyst feeds, and partner banks.
* **The Danger:** What happens if an adversary hacks an analyst account or spins up 100 fake partner feeds (a Sybil attack) to propose bad rules that block legitimate customers?
* If the bank accepts these rules blindly, it crashes customer service.

`perturbation_generator.py` creates realistic, executable candidate rules containing both **honest** and **adversarial** evidence. This is the stress-test environment used to verify that our Phase 4 **Subjective Logic Gate** and **Martingale SAVI Process** can detect and reject fake rules without human intervention.

---

## 2. Line-by-Line / Section-by-Section Code Breakdown

### 2.1 Operators and Rule Categories (Lines 48–56)
```python
RULE_CATEGORIES: Tuple[str, ...] = ("genuine", "stale", "conflicting", "poisoned", "sybil")
ADVERSARIAL_CATEGORIES: Tuple[str, ...] = ("poisoned", "sybil")
HONEST_SOURCE_TYPES: Tuple[str, ...] = ("tree_path", "graph_motif", "analyst_alert", "consortium_feed")
_OPS = {
    ">": np.greater, ">=": np.greater_equal, "<": np.less,
    "<=": np.less_equal, "==": np.equal, "!=": np.not_equal,
}
```
* **Line 49–51:** Defines the 5 test categories of candidate rules.
* **Line 52–55 (`_OPS`):** Maps string operators (`">"`, `"=="`, etc.) directly to NumPy vectorized functions. This allows a rule condition to be evaluated across hundreds of thousands of transactions in a single C-speed vector operation.

---

### 2.2 Atomic Rule Condition: `Condition` (Lines 61–86)
```python
@dataclass(frozen=True)
class Condition:
    feature: str            # e.g., "log1p_amount", "tx_type"
    op: str                 # e.g., ">", "=="
    value: Union[float, str]# e.g., 9.0, "TRANSFER"

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        column = frame[self.feature]
        if isinstance(self.value, str):
            values = column.astype(str).to_numpy()
            return np.asarray(_OPS[self.op](values, self.value), dtype=bool)
        return np.asarray(_OPS[self.op](column.to_numpy(dtype=np.float64), float(self.value)), dtype=bool)
```
* **What is a Condition?** A single mathematical check, such as `log1p_amount > 9.0` or `tx_type == 'TRANSFER'`.
* **`evaluate(frame)`:** Takes a DataFrame of transactions and returns a fast boolean array (`True` where the condition is met, `False` where it is not).
* **Immutability (`frozen=True`):** Prevents rules from being accidentally modified after creation.

---

### 2.3 Source Intelligence Evidence: `SourceEvidence` (Lines 88–115)
```python
@dataclass(frozen=True)
class SourceEvidence:
    source_id: str          # Name of the analyst or automated system (e.g., "feed_01")
    source_type: str        # e.g., "analyst_alert", "consortium_feed"
    reliability: float      # Historical trustworthiness score between 0.0 and 1.0
    r_plus: int             # Number of times this rule correctly caught fraud
    r_minus: int            # Number of times this rule caused a false alarm
    observed_at_step: int   # Timestamp when this evidence was gathered
    source_age_steps: int   # How long this source has existed in the system
    sybil_cluster_id: Optional[int] = None # Cluster ID if this source is an attacker
```
* **Why this matters for Subjective Logic:**
  * When an external intelligence feed proposes a rule, it doesn't just say "this is fraud". It reports evidence counts: $r_+$ (successes) and $r_-$ (false alarms).
* **Laplace Claimed Precision:**
  $$\text{claimed\_precision} = \frac{r_+ + 1}{r_+ + r_- + 2}$$
  Provides a smoothed estimate of what the source *claims* the rule's accuracy is.

---

### 2.4 Candidate Rule Class: `CandidateRule` (Lines 117–141)
```python
@dataclass
class CandidateRule:
    rule_id: str
    category: str
    conditions: Tuple[Condition, ...]
    evidence: Tuple[SourceEvidence, ...]
    emitted_at_step: int
    true_precision: float
    historical_precision: float
    is_harmful: bool
    base_rate: float = 0.5
```
* **`evaluate(frame)` (Lines 131–136):**
  Combines all atomic conditions using a logical AND ($\&$). A rule fires on a transaction only if **every** condition is met:
  $$\text{fires} = \text{condition}_1 \land \text{condition}_2 \land \dots \land \text{condition}_k$$
* **`simulate_outcomes(n_firings, rng)` (Lines 138–140):**
  Simulates live transaction firings using a Bernoulli process with parameter $p = \text{true\_precision}$. This allows our sequential martingale wealth process in Phase 4 to test the rule against actual incoming transactions.

---

### 2.5 The Five Rule Categories Explained

1. **`genuine` (Honest & Accurate):**
   * Proposed by trusted sources (`reliability >= 0.85`).
   * Both claimed precision and true precision are high ($p \ge 0.75 > p_0$).
   * *Expected Gate Decision:* Must be **admitted** into the active knowledge base.
2. **`stale` (Decayed over Time):**
   * Historically had high precision, but the fraudster changed tactics (concept drift).
   * Historical precision is high, but true precision today has decayed below $p_0 = 0.50$.
   * *Expected Gate Decision:* Must be **rejected** by freshness time-decay.
3. **`conflicting` (Sources Disagree):**
   * Two honest sources report contradicting evidence on the same rule.
   * *Expected Gate Decision:* Must be held in high uncertainty until more evidence arrives.
4. **`poisoned` (Malicious Injection):**
   * A low-reliability or rogue source fabricates high positive counts ($r_+ = 200, r_- = 5$) for a useless rule whose true precision is garbage ($p = 0.20$).
   * *Expected Gate Decision:* Trust discounting must inflate uncertainty $u \to 1$, blocking admission.
5. **`sybil` (Coordinated Attacker Swarm):**
   * A cluster of 10 to 50 brand-new, low-reputation feeds all propose the exact same harmful rule to simulate false consensus.
   * *Expected Gate Decision:* Cluster-aware consensus operator detects correlated sources and rejects the swarm.

---

### 2.6 The Feed Generator: `PerturbationGenerator` (Lines 317–426)

#### `validate_poison_rate(rate)` (Lines 335–346)
* Ensures the user-requested poison rate lies within the valid research testing range: $\rho \in [0.05, 0.25]$ (5% to 25% adversarial injection), or exactly `0.0` for a clean control feed.

#### `category_counts(n, poison_rate, stale_rate, conflict_rate)` (Lines 348–361)
* Calculates the exact integer breakdown of rules across the 5 categories based on the requested percentages.
* For example, for 100 candidate rules with a 20% poison rate:
  * 14 Poisoned rules
  * 6 Sybil rules
  * 10 Stale rules
  * 10 Conflicting rules
  * 60 Genuine rules

#### `generate_feed(...)` (Lines 363–421)
* The main factory method.
* **Deterministic Seeding (Lines 382–385):** Generates an exact NumPy random generator using:
  ```python
  rng = np.random.default_rng([self.seed, poison_rate, n, stale_rate, now])
  ```
* **Guarantee:** Running with the exact same seed and parameters produces the exact same rule objects every single time, enabling bit-identical experiment replays.
* Serializes candidate feeds to structured JSON files for audit trails and repeatable benchmarks.

---

## 3. How to Run This Generator Directly

You can generate test rule feeds from the command line:

```powershell
# Generate 50 candidate rules with 20% poisoned/Sybil feeds:
python -m src.data.perturbation_generator --n-candidates 50 --poison-rate 0.20
```

* Output is saved into `data/synthetic/rule_feeds/` as structured JSON, ready for ingestion by downstream testing suites.

