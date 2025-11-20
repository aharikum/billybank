# Billy Bank Analysis

# Annual Insider Threat Probability Calculation

Calculate the annual probability that an employee in each role becomes a malicious insider, based on historical behavioral data.

`Annual Probability (per role) = (# users with ≥1 malicious day) / (total users in role)`


- User-Level Aggregation:
    For each user, determine if they had at least one malicious day across the 240-day period

    Binary outcome: had_incident = 1 if is_malicious.sum() > 0, else 0


- Role-Level Aggregation:
    Calculate the proportion of users in each role who had ≥1 incident

    This becomes the annual probability (Probability of Action)

For example:
```
    # If 7 out of 700 Analysts had at least one malicious day:
    Annual Probability (Analyst) = 7 / 700 = 0.01 (1.0%)

    # If 2 out of 100 Traders had at least one malicious day:
    Annual Probability (Trader) = 2 / 100 = 0.02 (2.0%)
```


# Machine Learning Behavioral Analysis

Build a machine learning model to identify which behavioral patterns best predict insider threats, and quantify the behavioral differences between normal users and malicious insiders.

Will write about this more...


# Monte Carlo Simulation

## Overview

 The simulation runs 10,000 iterations to generate a probabilistic distribution of potential annual losses, providing executives with P5, median, P95, and mean Expected Annual Loss (EAL) figures.

---

## Key Components

### 1. **Probability of Action (PoA)**

The probability that an employee becomes a malicious insider in a given year.

- Used the `billybank_activity_updated.csv` dataset containing 240 days of simulated employee behavior
- For each user, determined if they had at least one malicious day (`is_malicious = 1`)
- Calculated PoA per role as: **% of users who had at least one incident**

Example:
```python
# If 7 out of 700 Analysts had at least one malicious day
PoA_Analyst = 7 / 700 = 0.01 (1%)
```

### 1. **Threat Event Frequency (TEF)**
Expected number of insider events per year

TEF[role] = headcount[role] x PoA[role]

Example:
```python
TEF_Analyst = 700 employees × 0.01 = 7.0 events/year
```
---

### 2. **Vulnerability (V)**

The probability that a malicious insider's attack attempt succeeds.

- **Base vulnerability = 75%** (industry standard for insiders who already have legitimate access)
- **With mitigation:** `Effective Vulnerability = 0.75 × (1 - mitigation_weight)`
  - Example: 60% mitigation → `0.75 × (1 - 0.6) = 0.30 (30%)`

Insiders bypass many perimeter controls, so their success rate is naturally high. 

---

### 3. **Contact Frequency (CF)**

How many attack attempts a malicious insider makes per year.

- Used a **Poisson distribution** with mean = 3.5 attempts/year
- Rationale: Insiders don't continuously attack; they wait for opportune moments (e.g., before resignation, during stressful periods)

**Why Poisson?** It models rare, independent events over time.

---

### 4. **Loss Magnitude (LM)**

The financial damage caused by a single successful attack.

**How we calculated it:**
- Used `employee_loss_ranges.csv` which maps roles to real-world incident loss ranges
- **Mapped simulation roles to loss categories:**

| Simulation Role | Loss Category | 
|----------------|---------------|
| C_Level | C-Level Executives | 
| Trader | Team Leads | $100K | 
| IT_Admin | Team Leads | $100K |
| Analyst | Employees (full-time) |
| Exec_Assistant | Employees (full-time) | 
| Contractor | Contractors / Temp Staff | 

- Sampled losses using a **Lognormal distribution**:
  - `log_mean = (log(min_loss) + log(max_loss)) / 2`
  - `log_std = (log(max_loss) - log(min_loss)) / 4`
  - Clipped values to stay within [min, max]

**Why Lognormal?** Financial losses are right-skewed. Most incidents cause moderate damage, but catastrophic events create extreme outliers.

---

## Monte Carlo Simulation Process

### For Each of 10,000 Iterations:

#### **Step 1: Sample Malicious Insiders per Role**
```python
n_insiders = Binomial(headcount, PoA)
```
- Example: 700 Analysts, TEF = 1% → on average, 7 become malicious

#### **Step 2: Sample Attack Attempts per Insider**
```python
attempts_per_insider = Poisson(3.5)
total_attempts = sum(attempts_per_insider)
```
- Example: 7 insiders × avg 3.5 attempts = ~24-25 total attempts

#### **Step 3: Sample Successful Attacks**
```python
n_successful = Binomial(total_attempts, effective_vulnerability)
```
- Example: 25 attempts × 30% success rate = ~7-8 successful attacks

#### **Step 4: Sample Loss per Attack**
```python
losses = Lognormal(log_mean, log_std, n_successful)
losses = clip(losses, min_loss, max_loss)
```
- Example: For Analysts, sample 7-8 losses between $50K–$1M

#### **Step 5: Aggregate Losses**
```python
role_loss = sum(losses)
total_loss += role_loss
```

---

## FAIR Framework Mapping

| FAIR Component | Our Implementation |
|---------------|-------------------|
| **Probability of Action (PoA)** | % of users per role who had ≥1 malicious day |
| **Threat Event Frequency (TEF)** | Expected insiders per year = Headcount × PoA |
| **Vulnerability (V)** | 75% base success rate, adjusted by mitigation weight |
| **Contact Frequency (CF)** | Poisson(3.5) attempts per insider per year |
| **Loss Event Frequency (LEF)** | TEF × CF × V = `n_insiders × attempts × success_rate` |
| **Loss Magnitude (LM)** | Lognormal distribution based on role-specific loss ranges |
| **Expected Annual Loss (EAL)** | LEF × LM across all roles, repeated 10,000 times |

---


# Running the code

## Setting up the environment (do this first)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## To Run the dataset generator (while .venv is activated)
```bash
python3 generator.py
```

## To Run ipynb (while .venv is activated) 
```bash
python -m ipykernel install --user --name=billybank --display-name="Python (billybank)"
jupyter notebook

#Select Analysis.ipynb and select kernel - Python (billybank)
```

## To Run monte carlo analysis (while .venv is activated) 
```bash
python3 monte_carlo.py
```
