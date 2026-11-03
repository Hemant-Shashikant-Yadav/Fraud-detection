# MG-ECFM Phase 1 Presentation Outline & Word-for-Word Presenter Script

**Project Title:** Martingale-Gated Epistemic-Conformal Fraud Mitigation (MG-ECFM): A Statistically Certified Framework for Risk-Aware Prevention, Relational Detection, and Reversible Knowledge Evolution in Digital Payment Systems  
**Presentation Topic:** Phase 1 — Data Collection, Cleaning, and Fast Streaming Preprocessing  
**Researcher:** Shradha Vijay Kolekar (URN: 1025230008)  
**Guide:** Dr. Anisa B. Shikalgar, Professor, Dept. of Computer Science & Engineering  
**Institution:** Annasaheb Dange College of Engineering & Technology, Ashta, Sangli  

---

## Slide 1: Title & Project Overview

### [SLIDE CONTENT]
* **Slide Title:** Data Collection, Cleaning, and Real-Time Preprocessing
* **Subtitle:** Phase 1 Foundation for the MG-ECFM Fraud Mitigation Framework
* **Visual Idea:** Clean card layout with student and guide details on the left, and key project highlights on the right.
* **Key Slide Points:**
  * **Researcher:** Shradha Vijay Kolekar (URN: 1025230008)
  * **Guide:** Dr. Anisa B. Shikalgar, Professor, Dept. of Computer Science & Engineering
  * **College:** Annasaheb Dange College of Engineering & Technology, Ashta
  * **Today's Focus:** **Phase 1** — Building a clean, fast, and honest data pipeline.
  * **Key Achievement:** Creates fraud features in **under 1 millisecond (0.78 ms)** with zero data cheating and low memory usage.

### [PRESENTER SCRIPT / WHAT TO SAY TO THE PANEL]
> "Respected Chairperson, esteemed guide Dr. Anisa Shikalgar Ma'am, and distinguished members of the review panel. Good morning.
> 
> I am Shradha Vijay Kolekar, research candidate in the Department of Computer Science and Engineering at Annasaheb Dange College of Engineering and Technology, Ashta. 
> 
> Today, I am presenting **Phase 1** of our research framework, titled **'Martingale-Gated Epistemic-Conformal Fraud Mitigation' (MG-ECFM)**.
> 
> *[Pause for emphasis]* While our overall research introduces advanced graph detection and certified decision routing, today's presentation focuses specifically on the most critical foundation: **Phase 1 — Data Collection, Cleaning, and Real-Time Preprocessing**. 
> 
> In real-world banking, if the input data pipeline is flawed, slow, or peeks into future information, even the most sophisticated deep learning models will fail in production. Today, I will walk you through how we gathered multi-million transaction streams, eradicated the widespread data-leakage flaw found in academic literature, and engineered a streaming pipeline that creates rich predictive features in just **0.78 milliseconds**—well within banking real-time limits.
> 
> Let us begin by looking at why standard fraud detection models fail at the data level."

---

## Slide 2: The Problem: Fake Accuracy & Real-World Banking

### [SLIDE CONTENT]
* **Slide Title:** Why Standard Fraud Detection Fails in Real Life
* **Subtitle:** Rare fraud events, future data leaks, and strict bank time limits
* **Visual Idea:** 3 side-by-side cards comparing the three main data issues in banking.
* **Key Slide Points:**
  * **Fraud is Very Rare:**
    * In **PaySim Mobile Money**: Only **8,213 frauds out of 6.36 million payments** (0.13%).
    * In **IEEE-CIS Card Data**: **20,663 frauds out of 590,000 payments** (3.5%).
    * Because fraud is so rare, basic models just guess "Normal" and miss all stolen money.
  * **The "Future Data" Trap (Data Leakage):**
    * Many research papers use the account balance *after* the payment is made.
    * In fraud, thieves empty the account to zero. Models simply look for "balance = 0" and claim fake 99.9% accuracy.
  * **Real Banks Cannot Cheat:**
    * In a live payment system (like UPI or credit cards), the bank must decide in **less than 25 milliseconds**.
    * The new balance does not exist until *after* the payment is approved.

### [PRESENTER SCRIPT / WHAT TO SAY TO THE PANEL]
> "To understand the motivation behind Phase 1, we must look at why existing fraud detection research breaks down when deployed into real banking systems. There are three fundamental problems.
> 
> *[Point to Problem 1]* First: **Extreme Class Imbalance**. Fraud is exceptionally rare. In our primary dataset, PaySim, there are only 8,213 fraud cases out of more than 6.36 million transactions—that is barely 0.13%, or roughly one fraud in every one thousand payments. If a machine learning model simply predicts 'Normal' for every single transaction, it achieves 99.9% statistical accuracy, but it fails to stop a single rupee of theft.
> 
> *[Point to Problem 2 - Gesture with emphasis]* Second: **The Future Data Trap**, known academically as Target Leakage. When reviewing existing research papers that use benchmarks like PaySim, we discovered a serious recurring flaw: many models use the sender's account balance *after* the payment has executed. When fraudsters strike, they deliberately drain the victim's account to zero. Standard machine learning models notice this arithmetic pattern, check if the post-balance is zero, and claim near-perfect 99.9% accuracy. 
> 
> *[Anticipating committee question: Why is that a problem?]* But in a live banking payment switch—such as UPI or Visa—that post-transaction balance **does not exist yet** when the payment is being cleared! It is calculated only after the bank approves the transaction. Models trained on post-balances are essentially cheating on historical data and collapse completely in real-world deployment.
> 
> *[Point to Problem 3]* Third: **The 25-Millisecond SLA**. A live banking rail requires an authorization decision in under 25 milliseconds. You cannot pause the payment switch to reload large databases into memory.
> 
> Our Objective 1 was formulated specifically to solve these three real-world data bottlenecks."

---

## Slide 3: Research Objectives (The Big Picture)

### [SLIDE CONTENT]
* **Slide Title:** Project Goals: How Phase 1 Fits into the Big Picture
* **Subtitle:** Step-by-step plan to solve modern fraud problems
* **Visual Idea:** 4-step roadmap with Step 1 highlighted in bright blue.
* **Key Slide Points:**
  * **Objective 1 (Today's Focus): Honest & Fast Streaming Data Pipeline**
    * Remove balance leaks, clean data safely, and build speed features using past history only.
  * **Objective 2: Graph-Based Mule Detection**
    * Catch money mules who try to hide by sending money to normal shops and users.
  * **Objective 3: Smart Risk Action Router**
    * Instead of just "Block or Allow", use 4 safe choices: **Approve, Send OTP (2FA), Hold for Review, or Block** in 1.5 microseconds.
  * **Objective 4: Safe Rule Updates & Money Recovery**
    * Stop attackers from poisoning detection rules, and freeze stolen money across multiple hops within 24 hours.

### [PRESENTER SCRIPT / WHAT TO SAY TO THE PANEL]
> "This slide presents our four formal research objectives, showing where Phase 1 anchors the complete MG-ECFM framework.
> 
> *[Highlight Objective 1]* **Objective 1 is the subject of today's presentation**: building an honest, leak-free, and ultra-fast streaming data pipeline. Our goal here is to eradicate balance lookahead leaks, handle multi-source transaction feeds, and generate causal velocity features in under 1 millisecond.
> 
> *[Briefly summarize downstream objectives]* To see how this powers the rest of the project:
> * In **Objective 2**, the clean features produced by Phase 1 feed into our Camouflage-Resistant Graph Neural Network to detect money mules who attempt to blend in by transacting with normal merchants.
> * In **Objective 3**, we replace rigid 'Block or Allow' rules with a 4-tier Conformal Risk Router that chooses between approving, prompting for 2FA, holding for manual review, or hard blocking in just 1.5 microseconds.
> * And in **Objective 4**, we use anytime-valid martingale mathematics to safely update fraud rules without allowing malicious attacker feeds to poison the system, combined with a time-decaying flow algorithm to freeze stolen funds within 24 hours.
> 
> Everything downstream depends strictly on Objective 1. If Objective 1 leaks future information or slows down, the entire architecture fails. Let us look at where Phase 1 sits in the execution architecture."

---

## Slide 4: System Architecture: Where Phase 1 Works

### [SLIDE CONTENT]
* **Slide Title:** System Architecture: The Real-Time Fast Path
* **Subtitle:** Processing incoming payments before any deep learning runs
* **Visual Idea:** Simple 3-lane diagram showing the live payment flow from left to right.
* **Key Slide Points:**
  * **The Three System Paths:**
    * **1. Real-Time Fast Path:** Approves or flags payments in less than 25 milliseconds.
    * **2. Background Learning Path:** Checks and updates new fraud rules without slowing down payments.
    * **3. Money Recovery Path:** Tracks stolen money moving through accounts and freezes it.
  * **Phase 1 is the Entry Gate:**
    * Catches the raw payment as soon as it arrives.
    * Cleans missing values and checks payment order.
    * Creates all mathematical features in **0.78 ms** and hands them directly to the AI models.

### [PRESENTER SCRIPT / WHAT TO SAY TO THE PANEL]
> "This diagram shows the high-level system architecture of MG-ECFM. To handle high payment volumes safely, we decouple the architecture into three separate execution swimlanes.
> 
> *[Point to Swimlane 1 - Fast Path]* Swimlane 1 is the **Real-Time Fast Path**. This is where every live transaction must be evaluated and cleared within our strict 25-millisecond budget.
> 
> *[Highlight Phase 1's position]* **Phase 1 sits as the very first gateway of this Fast Path.** When a payment arrives from the banking rail:
> 1. Our ingestion loader strips out any illegal future balance columns.
> 2. It verifies that the payment timestamp is in proper chronological order.
> 3. It generates our rolling behavioral velocity tensor in just **0.78 milliseconds**.
> 4. It passes this clean feature vector directly to our downstream GraphSAGE and XGBoost models.
> 
> *[Point to Swimlanes 2 and 3]* Swimlanes 2 and 3 operate completely in the background: Swimlane 2 tests and admits new fraud rules out-of-band so the payment rail never experiences database lockups, and Swimlane 3 models the delayed dispute queue to freeze stolen money across downstream mule hops.
> 
> Because Phase 1 executes in less than 1 millisecond, it leaves over 24 milliseconds of free headroom for the rest of the fast-path pipeline.
> 
> Next, let us inspect the datasets we gathered to train and evaluate this pipeline."

---

## Slide 5: Data Sources & Datasets Used

### [SLIDE CONTENT]
* **Slide Title:** Data Collection: Three Real-World Payment Datasets
* **Subtitle:** Standardizing mobile money, card payments, and banking transfers
* **Visual Idea:** Clear table showing dataset name, payment type, size, and fraud count.
* **Key Slide Points:**
  * **1. PaySim Mobile Money (6.36 Million Payments):**
    * Created from real African mobile money logs by Lopez-Rojas et al.
    * 744 hours (31 days) of activity with 8,213 real fraud cases.
  * **2. IEEE-CIS Fraud Benchmark (590,000 Card Payments):**
    * Real-world e-commerce card transactions from Vesta Corporation.
    * Includes card type, device info, and payment amounts over 182 days.
  * **3. AMLSim Banking Simulator (100,000 Transfer Rows):**
    * Built by IBM Research to simulate money laundering rings.
    * Tracks money moving across multi-hop accounts (smurfing and layering).

### Dataset Summary Table

| Dataset Name | Real-World Source | Total Rows | Fraud Rows | Fraud Rate | Main Purpose in Project |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **PaySim** | Mobile Money Logs | 6,362,620 | 8,213 | 0.13% | High-volume streaming & testing leakage removal |
| **IEEE-CIS** | Vesta Card Switch | 590,540 | 20,663 | 3.50% | Real card payment fields & time sorting |
| **AMLSim** | IBM Bank Simulator | 100,000 | Topology | Variable | Testing multi-account transfer chains |

### [PRESENTER SCRIPT / WHAT TO SAY TO THE PANEL]
> "Slide 5 outlines the three diverse, real-world data sources gathered and standardized for Phase 1. Rather than relying on a single synthetic file, we evaluate across three distinct payment modalities: mobile money, commercial cards, and multi-hop banking transfers.
> 
> *[Point to Row 1 - PaySim]* Our primary high-volume stream is **PaySim**, created by Lopez-Rojas and colleagues. It contains **6,362,620 transactions** modeled on actual anonymized financial logs from an African mobile money provider across 31 continuous days. Out of over 6.36 million payments, exactly 8,213 are fraud events, occurring exclusively in `TRANSFER` and `CASH_OUT` operations. This gives us our realistic 0.13% base rate.
> 
> *[Point to Row 2 - IEEE-CIS]* Second, we ingest the **IEEE-CIS Fraud Detection Benchmark**, provided by Vesta Corporation, a major e-commerce payment prevention company. This consists of **590,540 real-world card clearing records** with 20,663 confirmed frauds over 182 days, allowing us to test card-specific fields, device attributes, and chronological sorting.
> 
> *[Point to Row 3 - AMLSim]* Third, we incorporate **AMLSim**, developed by IBM Research Financial Services. This simulates agent-based multi-hop banking networks, which we use to verify money laundering patterns like smurfing, fan-out, and circular fund movements.
> 
> *[Summarize canonical mapping]* In Phase 1, our loader engine harmonizes all three datasets into a single, unified 9-column format: Transaction ID, Time Step, Seconds, Sender, Receiver, Amount, Type, and Label. 
> 
> Now, let us look at the most critical data engineering fix we implemented: eliminating the balance leakage flaw."

---

## Slide 6: Fixing the "Balance Leak" Flaw

### [SLIDE CONTENT]
* **Slide Title:** Stopping Data Leaks: Removing Future Information
* **Subtitle:** How we fixed the artificial 99.9% accuracy flaw found in earlier papers
* **Visual Idea:** Left side shows "Old Mistake (Using Future Balance)" in red; Right side shows "Our Fix (Delete Future Columns)" in green.
* **Key Slide Points:**
  * **The Trap in Existing Papers:**
    * PaySim provides: $\text{New Balance} = \text{Old Balance} - \text{Amount}$.
    * Thieves empty the sender's account, so the new balance is zero.
    * Old models simply check if the new balance equals zero. This is cheating because that number does not exist yet!
  * **Our Strict Fix:**
    * We physically delete all balance columns before the data enters our system:
      * Removed: `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`.
    * If any file tries to send these columns, our system immediately throws an error and stops.
  * **The Result:**
    * Our model learns real fraud behavior (speed, amount spikes, strange hours) instead of memorizing a balance trick.

### [PRESENTER SCRIPT / WHAT TO SAY TO THE PANEL]
> "Slide 6 highlights what we consider the most important scientific correction in Phase 1: eradicating post-decision balance leakage.
> 
> *[Gesture to the left side - Old Mistake]* When you download PaySim, the dataset includes columns like `oldbalanceOrg` and `newbalanceOrig`—the sender's balance before and after the transaction. In fraudulent transfers, the criminal typically steals every single rupee in the account, leaving the new balance at exactly zero. 
> 
> Many published machine learning studies feed all raw columns into algorithms like Random Forest or standard XGBoost. The model quickly discovers a trivial mathematical rule: if `newbalanceOrig == 0` while the transaction amount is greater than zero, flag it as fraud. The author then reports an astonishing 99.9% accuracy. 
> 
> *[Defensive reasoning - Why this is critical]* But this is a complete illusion. In an active payment authorization gateway—when you swipe your card or click 'Pay' on UPI—the ledger has not settled yet! The transaction has not happened. Therefore, the post-transaction balance is fundamentally unknowable at decision time.
> 
> *[Point to the right side - Our Fix]* In MG-ECFM Phase 1, we implement strict, physical column stripping. Our ingestion loader rejects and permanently drops `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest` at the file-reading boundary. If any input frame contains these columns, our pipeline raises an immediate `LeakageError` and halts.
> 
> By forcing our models to be completely blind to post-transaction balances, our system is forced to learn genuine fraud signatures: rapid payment bursts, uncharacteristic amount spikes, and abnormal transaction hours. This guarantees that our results reflect true operational performance."

---

## Slide 7: Data Cleaning & Low-Memory Streaming

### [SLIDE CONTENT]
* **Slide Title:** Data Cleaning & Low-Memory Stream Engine
* **Subtitle:** Handling millions of payments on simple computer hardware
* **Visual Idea:** Step-by-step pipeline showing Disk File $\to$ Small Chunks $\to$ Clean Format $\to$ Model.
* **Key Slide Points:**
  * **Why We Cannot Load Everything at Once:**
    * 6.36 million payments will crash computer memory (RAM) if loaded all together.
  * **Chunked Streaming Solution:**
    * We read data in small batches of 50,000 to 200,000 rows.
    * Uses **less than 180 MB of RAM**, running easily on any basic server or laptop.
  * **Handling Boundaries Safely:**
    * If an hour step is split across two chunks, a temporary buffer holds it until the hour finishes.
    * Running the data in chunks gives the **exact same output** as running the entire file at once.
  * **Missing Data Cleaning:**
    * Replaces missing user or shop IDs with standard tags (`MISSING_ORIGIN`).
    * Cleans out negative numbers and invalid text characters.

### [PRESENTER SCRIPT / WHAT TO SAY TO THE PANEL]
> "Slide 7 explains our data cleaning pipeline and our memory-safe streaming design.
> 
> *[Address the practical engineering hurdle]* PaySim contains over 6.3 million records, and real banking logs reach hundreds of millions of events. Loading an entire multi-gigabyte dataset into memory all at once will crash server RAM or cause severe garbage-collection lag.
> 
> To solve this, Phase 1 implements a chunked, generator-based streaming architecture. Instead of reading the entire file, our engine processes data in configurable batches of 50,000 to 200,000 rows at a time. As a result, our resident RAM consumption remains constant at **under 180 megabytes**, meaning this system can run seamlessly on basic, low-cost commodity hardware.
> 
> *[Explain the pending buffer innovation]* Now, streaming in chunks introduces an engineering risk: what if a chunk cuts off in the middle of an hour step, say at 2:30 PM, and the remaining 2:00 PM payments arrive in the next chunk? If you compute features immediately, those early payments will be computed using an incomplete picture of that hour.
> 
> We solved this by implementing a causal `pending_buffer`. The preprocessor detects the final, incomplete step of a chunk, holds those rows in a safe buffer, and finalizes them only when the next chunk delivers the start of the next hour. We mathematically verified that processing a dataset in chunks produces results that are **100% bit-identical** to processing the entire file at once.
> 
> Finally, our cleaning routines resolve missing sender or receiver IDs using standardized identity tokens, while purging negative amounts and formatting errors.
> 
> Let us now examine how we generate predictive features from past history alone."

---

## Slide 8: Smart Feature Creation (History Only)

### [SLIDE CONTENT]
* **Slide Title:** Feature Engineering: Using Past History Only
* **Subtitle:** Computing payment velocity and size without seeing the future
* **Visual Idea:** Diagram showing 1-hour, 24-hour, and 7-day sliding history windows behind the current payment.
* **Key Slide Points:**
  * **Golden Rule of Causality:**
    * A payment at 2:00 PM can only use transactions from **before 2:00 PM**. It cannot peek ahead.
  * **Features We Calculate on the Fly:**
    * **Payment Velocity:** How many payments this user made in the past 1 hour, 24 hours, and 7 days.
    * **Average Amount:** The normal average amount spent by this user over time.
    * **Sudden Spike Score (Z-Score):** Compares today's amount to the user's past average:
      $$\text{Z-Score} = \frac{\text{Current Amount} - \text{Average Amount}}{\text{Standard Deviation}}$$
      * If a user who spends \$20 suddenly sends \$5,000, this score jumps immediately.
    * **Account Recency:** How many hours since this user last made a payment.
    * **New Account Flag:** Marks brand-new accounts that have zero payment history.

### [PRESENTER SCRIPT / WHAT TO SAY TO THE PANEL]
> "Slide 8 details our causal feature engineering engine. 
> 
> *[Emphasize the Golden Rule]* The golden rule of Phase 1 is **strict chronological causality**: a transaction arriving at step $t$ is allowed to compute features derived **exclusively** from events that occurred strictly prior to $t$ ($\tau < t$). It can never look at concurrent transactions or future transactions.
> 
> *[Explain the feature set]* From this strictly historical data, we compute several powerful behavioral indicators on the fly:
> 
> 1. **Rolling Velocity Windows**: We track the frequency of payments made by an entity across three sliding horizons: the past 1 hour, the past 24 hours, and the past 7 days. If an account that normally makes one payment a day suddenly executes 15 transfers in an hour, the velocity counter spikes instantly.
> 
> 2. **Causal Amount Moments (Mean and Variance)**: To compute the running average spend and standard deviation without numerical overflow, we implemented the **parallel Welford algorithm**. This guarantees mathematical precision even when tracking accounts moving millions of dollars.
> 
> 3. **The Amount Z-Score**: By calculating:
>    $$\text{Z-Score} = \frac{\text{Current Amount} - \text{Average Amount}}{\text{Standard Deviation}}$$
>    we quantify how uncharacteristic the current payment is. If a customer normally buys \$20 groceries and suddenly attempts a \$5,000 wire transfer, the Z-score jumps immediately.
> 
> 4. **Recency and New-Account Flags**: We track how many hours have elapsed since the user was last active, and flag brand-new accounts with zero history, which are prime targets for synthetic identity theft.
> 
> And to keep things fast, older history is automatically evicted from our memory queues in constant time."

---

## Slide 9: Account Balance Flow & Safety Checks

### [SLIDE CONTENT]
* **Slide Title:** Flow Balancing & Order Safety
* **Subtitle:** Tracking money direction and making sure random order does not change results
* **Visual Idea:** Visual showing Money In vs. Money Out, plus a simple order check badge.
* **Key Slide Points:**
  * **Two-Way Money Flow:**
    * Most fraud models only look at money leaving an account.
    * Our engine also tracks money coming in (`CASH_IN`), so legitimate deposits balance out outbound transfers.
  * **Same-Second Order Safety:**
    * In high-speed banking, many transactions arrive within the exact same millisecond.
    * Our math guarantees that even if transactions inside the same second shuffle their order, **the output features stay 100% identical**.
  * **Time Clock Guard:**
    * The streaming engine checks every timestamp.
    * If any data line has a timestamp older than the previous line, the system halts instantly with a time error.

### [PRESENTER SCRIPT / WHAT TO SAY TO THE PANEL]
> "Slide 9 presents two sophisticated safeguards built into Phase 1: our Double-Entry Flow Ledger and our Same-Second Order Safety guarantee.
> 
> *[Explain Two-Way Money Flow]* First: **Two-Way Flow Balancing**. Many traditional fraud models only monitor outbound payments. But legitimate business accounts frequently receive large deposits (`CASH_IN`) before making outbound transfers. MG-ECFM maintains a causal, double-entry observed ledger. When money comes in, it increments the entity's observed liquidity, correctly balancing out legitimate downstream disbursements and preventing false alarms on high-volume merchants.
> 
> *[Explain Same-Second Order Safety - Point to second card]* Second: **Same-Second Order Safety**, known technically as intra-step permutation invariance. In high-speed digital payments, dozens of transactions can arrive at the payment switch within the exact same millisecond. Due to network latency, their arrival order can be completely random. 
> 
> In computer science, floating-point addition is not strictly associative: $(a + b) + c$ can differ slightly from $(c + b) + a$ at the fifteenth decimal place. If you are not careful, two identical transactions could produce slightly different feature values simply based on which one arrived first!
> 
> To eliminate this, our preprocessor uses a **lexicographical sort (`lexsort`)** on transaction IDs and amounts before accumulation. We proved that even if transactions within the same time step are shuffled into completely random orders, our output features are **100% bit-identical**.
> 
> Finally, we enforce a strict **Time Clock Guard**: if a data stream ever presents a transaction with a timestamp earlier than the previous one, the engine throws a `TemporalOrderError` immediately to prevent retroactive data poisoning."

---

## Slide 10: Performance Results & Next Steps

### [SLIDE CONTENT]
* **Slide Title:** Performance Summary & Transition to Phase 2
* **Subtitle:** A fast, honest data engine ready for machine learning models
* **Visual Idea:** 4 speed & memory result boxes on the left; Arrow pointing to Phase 2 models on the right.
* **Key Slide Points:**
  * **Phase 1 Performance Numbers:**
    * **Feature Speed:** **0.78 milliseconds per payment** (Budget is 25 ms, leaving 24 ms free for AI models).
    * **Throughput:** Processes over **18,500 payments per second**.
    * **Memory Usage:** Stays under **180 MB RAM** even on 6.36 million payments.
    * **Zero Future Data:** 100% checked to ensure no future balance data leaks in.
  * **Ready for Phase 2 (Tomorrow's Next Step):**
    * The clean features created here are now fed directly into:
      1. **Graph Neural Networks (GraphSAGE)** to catch money mule networks.
      2. **XGBoost AI Classifier** to score payment risk.
  * **Takeaway:**
    * *"Phase 1 gives our system an honest, lightning-fast data foundation that works in real banks without cheating."*

### [PRESENTER SCRIPT / WHAT TO SAY TO THE PANEL]
> "To conclude our Phase 1 presentation, Slide 10 summarizes our measured engineering benchmarks and shows how this foundation connects directly to Phase 2.
> 
> *[Point to the 4 Performance Boxes]* When evaluated across our multi-million transaction streams, Phase 1 delivers outstanding production performance:
> * **Feature Processing Speed**: It extracts all causal velocity features in just **0.78 milliseconds at the 95th percentile**. Against our strict 25-millisecond banking SLA, this consumes only 3% of our time budget, leaving over 24 milliseconds of free headroom for our downstream AI models.
> * **Throughput**: Our memory-safe engine sustains over **18,500 transactions per second**.
> * **Memory Footprint**: Resident memory usage remains bounded at **under 180 megabytes of RAM**, even when processing the massive 6.36-million-row PaySim stream.
> * **Data Integrity**: We have verified 0.00% lookahead leakage and 100% chronological causality.
> 
> *[Point to Transition to Phase 2]* Because Phase 1 is complete and verified, we have an honest, high-speed data bedrock. Tomorrow, this clean causal tensor $x_{\text{causal}}$ feeds directly into:
> 1. Our **Camouflage-Resistant Graph Neural Network (GraphSAGE)** to detect money mule accounts, and
> 2. Our **600-tree XGBoost classifier** calibrated using Venn-Abers predictors.
> 
> *[Formal closing]* In summary: Phase 1 proves that we can process high-velocity financial streams cleanly, honestly, and in sub-millisecond time without cheating on future balance data.
> 
> I express my sincere gratitude to my supervisor, Dr. Anisa Shikalgar Ma'am, and to you, the respected members of the panel, for your valuable time and attention. 
> 
> I am now ready to answer any questions you may have regarding Phase 1. Thank you very much."
