# Code Walkthrough: `experiments/run_adversarial_stress_test.py`

**Purpose:** The **adversarial knowledge-poisoning stress test** of the martingale gate. It asks: *if an attacker (or a bad data source) feeds the system poisoned, stale, conflicting or Sybil-backed rules, does the gate keep them out of the knowledge base, and does it throw them out quickly if they get in?* It also verifies **Ville's inequality** numerically.

**Outputs:** `reports/adversarial_robustness_results.csv` (one row per poisoning rate × policy) and `reports/adversarial_time_to_rollback.csv` (one row per injected harmful rule).

> Important: this experiment runs on **simulated candidate feeds** (from `src/data/perturbation_generator.py`), not on PaySim transactions — so its numbers are identical regardless of the dataset chosen.

---

## 1. Design (docstring)
For each poisoning rate `ρ ∈ {5, 10, 15, 20, 25 %}` a contaminated feed (genuine, stale, conflicting, poisoned, Sybil-backed rules) is processed by three admission policies:
* `mg_ecfm_gate` — Subjective-Logic screening + martingale admission (`H0: precision ≤ p0`) + rollback monitoring (`H0^good: FPR ≤ q0`);
* `sl_screen_only` — Subjective-Logic screening alone (no sequential test, no rollback);
* `auto_admit` — gate OFF, everything enters the KB.

Firing outcomes are Bernoulli draws from each rule's latent *current* precision, one batch of `firings_per_cycle` verified firings per rule per update cycle. Two experiments per rate:
1. **Admission:** submit all candidates; after `n_cycles`, record the **false-admission rate** (harmful rules ever admitted ÷ harmful rules; harmful = latent precision ≤ `p0`), the adversarial-only rate (poisoned + Sybil), the genuine admission rate and admission accuracy.
2. **Injection / rollback:** inject harmful rules **directly** into the active KB (an attacker or stale legacy import bypassing admission) and record the **time-to-rollback** (TTR, in update cycles), plus the false-rollback frequency of the good rules (`FPR ≤ q0`).

**Ville verification** — `P(∃t: M_t^roll ≥ 1/α) ≤ α` checked three ways per rate: (i) the *exact* crossing probability at the least-favourable null `FPR = q0` (dynamic programme); (ii) a Monte-Carlo estimate over many independent monitors with a Clopper-Pearson 95 % interval and a one-sided binomial test; (iii) the empirical false-rollback frequency of null-good rules inside the full gate. The bound is "satisfied" when the exact probability ≤ α and neither empirical frequency is significantly above α.

## 2. Section-by-section (by outline)

### 2.1 Binomial utilities (lines 94–140)
Dependency-free `binomial_sf`, `binomial_cdf` (log-space PMF for stability) and `clopper_pearson(k, n, confidence)` (exact interval by bisection).

### 2.2 Experiments (lines 145–292)
* `simulate_cycles(gate, precision_of, n_cycles, …)`: drives the gate with simulated verified firings.
* `admission_experiment(feed, hyperparams, policy, …)`: runs one admission policy and returns the false-admission / genuine-admission / accuracy statistics.
* `injection_experiment(...)`: injects harmful rules and measures TTR and false rollbacks.
* `ville_verification(gate_cfg, horizon, n_reps, seed, …)`: the three-way Ville check.

### 2.3 Runner (lines 298–388)
`run_adversarial_stress_test(rates, …)` loops over the poison rates, runs both experiments for each policy, assembles the result rows and TTR rows, and writes the two CSVs. `main` is the CLI.

## 3. How to read the results
In the PaySim-2M deliverable: gate FAR 0–1.6 %, SL-only up to 2.1 %, auto-admit 100 %; all harmful injected rules were rolled back; mean TTR 1.0–1.2 cycles but **max TTR up to 5** at ρ = 20–25 %; Ville exact bound 0.0365 ≤ 0.05. The "≤ 3 cycles" requirement is therefore an **empirical** property (and not met for the maximum at high poisoning), not a theorem.
