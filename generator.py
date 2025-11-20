import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(1337)
np.random.seed(1337)

# Organization: how many users per role, regions, days
NUM_USERS_BY_ROLE = {
    "C_Level":         9,
    "Analyst":       700,
    "Trader":        100,
    "IT_Admin":       50,
    "Exec_Assistant": 20,
    "Contractor":    130,
}

REGIONS = ["NA", "EU", "APAC"]
DAYS_TO_SIMULATE = 240  

# Role baselines behavior per day
# These are "typical" counts when the day is not malicious.
ROLE_BEHAVIOR_BASE = {
    "C_Level": {
        "after_hours_logons": 2.5,
        "sensitive_file_reads": 50,
        "usb_device_mounts": 0.02,
        "external_emails_sent": 20,
        "emails_with_attachments": 10,
        "cloud_upload_events": 0.1,
        "failed_logins": 0.3,
        "files_deleted": 0.5,
        "http_competitor_visits": 2.0,
    },
    "Trader": {
        "after_hours_logons": 1.5,
        "sensitive_file_reads": 30,
        "usb_device_mounts": 0.05,
        "external_emails_sent": 4,
        "emails_with_attachments": 2,
        "cloud_upload_events": 0.05,
        "failed_logins": 0.5,
        "files_deleted": 1,
        "http_competitor_visits": 1.5,
    },
    "IT_Admin": {
        "after_hours_logons": 2.0,
        "sensitive_file_reads": 15,
        "usb_device_mounts": 0.2,
        "external_emails_sent": 1,
        "emails_with_attachments": 0.5,
        "cloud_upload_events": 0.05,
        "failed_logins": 1.0,
        "files_deleted": 5,
        "http_competitor_visits": 0.5,
    },
    "Analyst": {
        "after_hours_logons": 0.7,
        "sensitive_file_reads": 20,
        "usb_device_mounts": 0.05,
        "external_emails_sent": 6,
        "emails_with_attachments": 3,
        "cloud_upload_events": 0.1,
        "failed_logins": 0.7,
        "files_deleted": 1,
        "http_competitor_visits": 1.0,
    },
    "Contractor": {
        "after_hours_logons": 0.4,
        "sensitive_file_reads": 10,
        "usb_device_mounts": 0.4,
        "external_emails_sent": 2,
        "emails_with_attachments": 1,
        "cloud_upload_events": 0.15,
        "failed_logins": 1.2,
        "files_deleted": 1,
        "http_competitor_visits": 0.5,
    },
    "Exec_Assistant": {
        "after_hours_logons": 0.5,
        "sensitive_file_reads": 5,
        "usb_device_mounts": 0.05,
        "external_emails_sent": 15,
        "emails_with_attachments": 5,
        "cloud_upload_events": 0.05,
        "failed_logins": 0.4,
        "files_deleted": 0.5,
        "http_competitor_visits": 0.2,
    },
}

# Role standard deviations per day
ROLE_BEHAVIOR_STD = {
    "C_Level": {
        "after_hours_logons": 1.0,
        "sensitive_file_reads": 8,
        "usb_device_mounts": 0.3,
        "external_emails_sent": 5,
        "emails_with_attachments": 3,
        "cloud_upload_events": 0.3,
        "failed_logins": 0.5,
        "files_deleted": 1,
        "http_competitor_visits": 1.0,
    },
    "Trader": {
        "after_hours_logons": 0.8,
        "sensitive_file_reads": 5,
        "usb_device_mounts": 0.5,
        "external_emails_sent": 2,
        "emails_with_attachments": 1.5,
        "cloud_upload_events": 0.5,
        "failed_logins": 0.7,
        "files_deleted": 2,
        "http_competitor_visits": 0.8,
    },
    "IT_Admin": {
        "after_hours_logons": 0.8,
        "sensitive_file_reads": 5,
        "usb_device_mounts": 0.5,
        "external_emails_sent": 2,
        "emails_with_attachments": 1.5,
        "cloud_upload_events": 0.5,
        "failed_logins": 0.7,
        "files_deleted": 2,
        "http_competitor_visits": 0.8,
    },
    "Analyst": {
        "after_hours_logons": 0.8,
        "sensitive_file_reads": 5,
        "usb_device_mounts": 0.5,
        "external_emails_sent": 2,
        "emails_with_attachments": 1.5,
        "cloud_upload_events": 0.5,
        "failed_logins": 0.7,
        "files_deleted": 2,
        "http_competitor_visits": 0.8,
    },
    "Contractor": {
        "after_hours_logons": 0.8,
        "sensitive_file_reads": 5,
        "usb_device_mounts": 0.5,
        "external_emails_sent": 2,
        "emails_with_attachments": 1.5,
        "cloud_upload_events": 0.5,
        "failed_logins": 0.7,
        "files_deleted": 2,
        "http_competitor_visits": 0.8,
    },
    "Exec_Assistant": {
        "after_hours_logons": 0.8,
        "sensitive_file_reads": 5,
        "usb_device_mounts": 0.5,
        "external_emails_sent": 2,
        "emails_with_attachments": 1.5,
        "cloud_upload_events": 0.5,
        "failed_logins": 0.7,
        "files_deleted": 2,
        "http_competitor_visits": 0.8,
    },
}

# Role "opportunity weights": which features matter most
ROLE_OPPORTUNITY_WEIGHTS = {
    "C_Level": {
        "sensitive_file_reads": 2.5,
        "external_emails_sent": 2.0,
        "cloud_upload_events": 1.5,
        "after_hours_logons": 1.0,
    },
    "Trader": {
        "cloud_upload_events": 2.0,
        "sensitive_file_reads": 1.5,
        "after_hours_logons": 1.0,
        "external_emails_sent": 1.0,
    },
    "IT_Admin": {
        "after_hours_logons": 2.0,
        "sensitive_file_reads": 1.5,
        "files_deleted": 1.5,
        "usb_device_mounts": 1.2,
    },
    "Analyst": {
        "emails_with_attachments": 2.0,
        "external_emails_sent": 1.5,
        "sensitive_file_reads": 1.0,
    },
    "Contractor": {
        "usb_device_mounts": 2.0,
        "sensitive_file_reads": 1.5,
        "cloud_upload_events": 1.2,
    },
    "Exec_Assistant": {
        "emails_with_attachments": 2.0,
        "external_emails_sent": 1.5,
        "sensitive_file_reads": 1.0,
    },
}


def nonnegative_int(x: float) -> int:
    return max(int(round(x)), 0)


def generate_psychometrics(role: str):
    """
    Assign per-user psychometric scores:
      - conscientiousness (rule-following)
      - neuroticism (stress reactivity)
    """
    if role == "C_Level":
        # Very high discipline, very calm - low base risk but catastrophic opportunity
        conscientiousness = np.random.normal(80, 5)
        neuroticism       = np.random.normal(40, 6)
    elif role == "Trader":
        # Traders are average in rule-following, but highly stress-sensitive
        conscientiousness = np.random.normal(60, 8)
        neuroticism       = np.random.normal(60, 10)
    elif role == "IT_Admin":
        # Admins are rule-driven and calm under pressure - low base predisposition, but high opportunity
        conscientiousness = np.random.normal(70, 6)
        neuroticism       = np.random.normal(45, 8)
    elif role == "Analyst":
        # Balanced: medium discipline, medium stress - moderate risk posture
        conscientiousness = np.random.normal(62, 7)
        neuroticism       = np.random.normal(55, 9)
    elif role == "Contractor":
        # High variability – some very careful, some careless
        conscientiousness = np.random.normal(50, 10)
        neuroticism       = np.random.normal(52, 10)
    elif role == "Exec_Assistant":
        # Very disciplined, emotionally steady; low predisposition, but high data sensitivity
        conscientiousness = np.random.normal(75, 5)
        neuroticism       = np.random.normal(50, 7)
    else:
        conscientiousness = np.random.normal(60, 10)
        neuroticism       = np.random.normal(55, 10)

    conscientiousness = min(max(conscientiousness, 0), 100)
    neuroticism       = min(max(neuroticism, 0), 100)
    return conscientiousness, neuroticism


def hr_flag_chance(role: str) -> float:
    return {
        "C_Level":        0.0001,  # 0.01% per day - very rare for C-level
        "Trader":         0.001,   # 0.1% per day
        "IT_Admin":       0.0015,
        "Analyst":        0.001,
        "Contractor":     0.002,
        "Exec_Assistant": 0.0005,
    }[role]


def opportunity_score(pre_row: dict) -> float:
    """
    Role-weighted 'opportunity' from *pre-injection* behavior.
    Only count rare deviations: max(0, z - 2.0) so > ~97.5th percentile.
    """
    role    = pre_row["role"]
    mu      = ROLE_BEHAVIOR_BASE[role]
    sigma   = ROLE_BEHAVIOR_STD[role]
    weights = ROLE_OPPORTUNITY_WEIGHTS[role]

    score = 0.0
    for feat, w in weights.items():
        x = pre_row[feat]
        s = sigma.get(feat, 0.0)
        if s <= 0:
            continue
        z = (x - mu[feat]) / s
        spike = max(0.0, z - 2.0)     # only count the tail beyond 2σ
        if spike > 0:
            score += w * spike
    return min(score, 5.0)            # cap for stability


def decide_and_inject_malicious(row: dict,
                                conscientiousness: float,
                                neuroticism: float,
                                is_hr_flagged: int):
    """
    Calibrated to produce realistic annual rates of 0.8-3% per employee.
    
    Daily base probabilities are set extremely low (0.005% to 0.02% per day)
    to achieve annual rates in the 0.8-3% range when compounded over 240 days.
    
    Formula: P(malicious) = base_role_prob + stress_factor × 0.0003 + opportunity_term
    where opportunity_term = 0.0001 × opportunity_score(pre)
    """
    # --- Human/HR stress (kept tiny to maintain realistic rates) ---
    stress_factor = 0
    if neuroticism > 65:
        stress_factor += 1
    if is_hr_flagged:
        stress_factor += 1
    if conscientiousness < 50:
        stress_factor += 1

    # Base daily probabilities calibrated for realistic annual rates
    # These produce approximately:
    # - C_Level: ~1.2% annual
    # - Trader: ~2.4% annual
    # - IT_Admin: ~3.6% annual (highest risk)
    # - Analyst: ~1.9% annual
    # - Contractor: ~4.8% annual (highest risk - temporary staff)
    # - Exec_Assistant: ~2.4% annual
    base_role_prob = {
        "C_Level":        0.00005,  # 0.005% per day → ~1.2% annual
        "Trader":         0.00010,  # 0.01% per day → ~2.4% annual
        "IT_Admin":       0.00015,  # 0.015% per day → ~3.6% annual
        "Analyst":        0.00008,  # 0.008% per day → ~1.9% annual
        "Contractor":     0.00020,  # 0.02% per day → ~4.8% annual
        "Exec_Assistant": 0.00010,  # 0.01% per day → ~2.4% annual
    }[row["role"]]

    opp = opportunity_score(row)      # usually 0, occasionally >0 on tail days
    opp_term = 0.00001 * opp          # 0.001% per 'opp' unit (very small contribution)

    prob = base_role_prob + 0.000003 * stress_factor + opp_term
    prob = max(0.0, min(prob, 0.0005))  # hard cap at 0.05% per day

    malicious_today = (random.random() < prob)
    if not malicious_today:
        return False, row

    # Inject role-typical spikes to mark an incident day
    role = row["role"]
    if role == "C_Level":
        row["sensitive_file_reads"]   += random.randint(100, 200)
        row["external_emails_sent"]   += random.randint(30, 50)
        row["cloud_upload_events"]    += random.randint(5, 10)
        row["after_hours_logons"]     += random.randint(5, 10)
    elif role == "Trader":
        row["cloud_upload_events"]    += random.randint(3, 6)
        row["sensitive_file_reads"]   += random.randint(20, 40)
        row["after_hours_logons"]     += random.randint(2, 4)
    elif role == "IT_Admin":
        row["after_hours_logons"]     += random.randint(3, 6)
        row["sensitive_file_reads"]   += random.randint(40, 80)
        row["usb_device_mounts"]      += random.randint(1, 3)
        row["files_deleted"]          += random.randint(20, 50)
    elif role == "Analyst":
        row["external_emails_sent"]   += random.randint(5, 10)
        row["emails_with_attachments"]+= random.randint(5, 10)
        row["sensitive_file_reads"]   += random.randint(15, 30)
    elif role == "Contractor":
        row["usb_device_mounts"]      += random.randint(2, 5)
        row["sensitive_file_reads"]   += random.randint(30, 60)
        row["cloud_upload_events"]    += random.randint(1, 3)
    elif role == "Exec_Assistant":
        row["external_emails_sent"]   += random.randint(8, 15)
        row["emails_with_attachments"]+= random.randint(8, 15)
        row["sensitive_file_reads"]   += random.randint(10, 20)

    return True, row


# Build the synthetic user directory (id, role, region, traits)
users = []
for role, count in NUM_USERS_BY_ROLE.items():
    for _ in range(count):
        uid = "BB-" + uuid.uuid4().hex[:8]
        region = random.choice(REGIONS)
        conscientiousness, neuroticism = generate_psychometrics(role)
        users.append({
            "user_id": uid,
            "role": role,
            "region": region,
            "conscientiousness": conscientiousness,
            "neuroticism": neuroticism
        })


# Simulate day-by-day activity and compute is_malicious
rows = []
start_date = datetime(2025, 9, 1)

for u in users:
    for day_offset in range(DAYS_TO_SIMULATE):
        day = start_date + timedelta(days=day_offset)
        base = ROLE_BEHAVIOR_BASE[u["role"]]
        std = ROLE_BEHAVIOR_STD[u["role"]]

        # Sample behavior around role means using the STDs
        after_hours_logons      = nonnegative_int(np.random.normal(base["after_hours_logons"],      std["after_hours_logons"]))
        sensitive_file_reads    = nonnegative_int(np.random.normal(base["sensitive_file_reads"],    std["sensitive_file_reads"]))
        usb_device_mounts       = nonnegative_int(np.random.normal(base["usb_device_mounts"],       std["usb_device_mounts"]))
        external_emails_sent    = nonnegative_int(np.random.normal(base["external_emails_sent"],    std["external_emails_sent"]))
        emails_with_attachments = nonnegative_int(np.random.normal(base["emails_with_attachments"], std["emails_with_attachments"]))
        cloud_upload_events     = nonnegative_int(np.random.normal(base["cloud_upload_events"],     std["cloud_upload_events"]))
        failed_logins           = nonnegative_int(np.random.normal(base["failed_logins"],           std["failed_logins"]))
        files_deleted           = nonnegative_int(np.random.normal(base["files_deleted"],           std["files_deleted"]))
        http_competitor_visits  = nonnegative_int(np.random.normal(base["http_competitor_visits"],  std["http_competitor_visits"]))

        # HR stressor
        is_hr_flagged = 1 if random.random() < hr_flag_chance(u["role"]) else 0

        # Assemble the "pre-injection" row (this is what opp score reads)
        row = {
            "user_id": u["user_id"],
            "role": u["role"],
            "region": u["region"],
            "day": day.strftime("%Y-%m-%d"),

            "after_hours_logons": after_hours_logons,
            "sensitive_file_reads": sensitive_file_reads,
            "usb_device_mounts": usb_device_mounts,
            "external_emails_sent": external_emails_sent,
            "emails_with_attachments": emails_with_attachments,
            "cloud_upload_events": cloud_upload_events,
            "failed_logins": failed_logins,
            "files_deleted": files_deleted,
            "http_competitor_visits": http_competitor_visits,

            "is_hr_flagged": is_hr_flagged,
            "conscientiousness": u["conscientiousness"],
            "neuroticism": u["neuroticism"],
        }

        # Decide maliciousness using base + stress + opportunity from pre-injection counts
        is_mal, row = decide_and_inject_malicious(
            row,
            conscientiousness=u["conscientiousness"],
            neuroticism=u["neuroticism"],
            is_hr_flagged=is_hr_flagged
        )

        row["is_malicious"] = int(is_mal)
        rows.append(row)

df = pd.DataFrame(rows)

# Calculate and display statistics
total_users = len(users)
total_days = DAYS_TO_SIMULATE
malicious_count = df["is_malicious"].sum()
total_observations = len(df)

# Calculate per-role statistics
role_stats = df.groupby("role").agg({
    "is_malicious": ["sum", "count"]
}).reset_index()
role_stats.columns = ["role", "malicious_days", "total_days"]
role_stats["annual_rate"] = (role_stats["malicious_days"] / role_stats["total_days"]) * 100

print("\n" + "="*70)
print("DATASET GENERATION COMPLETE")
print("="*70)
print(f"\nTotal rows: {len(df):,}")
print(f"Total users: {total_users:,}")
print(f"Days per user: {total_days}")
print(f"Total malicious days: {malicious_count:,}")
print(f"Overall malicious rate: {(malicious_count/total_observations)*100:.3f}%")
print(f"Approximate annual rate: {(malicious_count/total_observations)*100*240:.2f}%")

print("\n" + "-"*70)
print("PER-ROLE ANNUAL RATES (Expected: 0.8-3% for most roles)")
print("-"*70)
print(f"{'Role':<20} {'Malicious Days':>15} {'Total Days':>12} {'Annual %':>10}")
print("-"*70)
for _, row in role_stats.iterrows():
    print(f"{row['role']:<20} {int(row['malicious_days']):>15,} {int(row['total_days']):>12,} {row['annual_rate']:>9.2f}%")
print("="*70)

# Save to CSV
output_file = "billybank_activity_updated.csv"
df.to_csv(output_file, index=False)
print(f"\nDataset saved to: {output_file}")
print("\nNote: These are DAILY probabilities in the dataset.")
print("The linear regression model will aggregate these to annual probabilities.")
print("="*70)