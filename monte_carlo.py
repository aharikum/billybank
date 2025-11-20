import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

OUTPUT_DIR = Path("monte_carlo_results")
OUTPUT_DIR.mkdir(exist_ok=True)  

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

ROLES = ["C_Level", "Analyst", "Trader", "IT_Admin", "Exec_Assistant", "Contractor"]

ROLE_HEADCOUNT = {
    "C_Level": 9,
    "Analyst": 700,
    "Trader": 100,
    "IT_Admin": 50,
    "Exec_Assistant": 20,
    "Contractor": 130
}

BASE_VULNERABILITY = 0.75  # 75% baseline success rate
ATTEMPTS_MEAN = 3.5  # Average attempts per insider per year (Poisson distribution)

N_ITER = 10000
RANDOM_STATE = 42

# Mapping from dataset roles to loss data csv roles
ROLE_MAPPING = {
    'C_Level': 'C-Level Executives',
    'Trader': 'Team Leads',
    'IT_Admin': 'Team Leads',
    'Analyst': 'Employees (full-time)',
    'Exec_Assistant': 'Employees (full-time)',
    'Contractor': 'Contractors / Temporary Staff'
}

df = pd.read_csv("billybank_activity_updated.csv", keep_default_na=False)
df['region'] = df['region'].replace({np.nan: "NA"})

user_had_incident = df.groupby(['user_id', 'role'])['is_malicious'].agg([
    ('had_incident', lambda x: int(x.sum() > 0))
]).reset_index()

role_tef = (
    user_had_incident.groupby('role')['had_incident']
    .mean() 
    .reindex(ROLES)
)

loss_ranges = pd.read_csv('employee_loss_ranges.csv')
loss_dict = {}
for _, row in loss_ranges.iterrows():
    loss_dict[row['Level']] = {
        'min': row['Min Loss (USD)'],
        'max': row['Max Loss (USD)']
    }

# =====================================================
# MONTE CARLO SIMULATION WITH MITIGATION WEIGHT
# =====================================================

def run_monte_carlo_simulation(mitigation_weight=0.0, n_iterations=N_ITER):
    """
    Monte Carlo using FAIR Framework with mitigation weight
    
    Args:
        mitigation_weight (float): 0.0 to 1.0, represents % reduction in vulnerability
                                   0.0 = no mitigation (75% success rate)
                                   0.6 = 60% reduction (30% success rate)
    
    For each iteration:
      1. Sample number of malicious insiders per role ~ Binomial(headcount, TEF)
      2. Sample number of attempts per insider ~ Poisson(ATTEMPTS_MEAN)
      3. Sample number of successful attacks ~ Binomial(attempts, Vulnerability * (1 - weight))
      4. Sample loss per attack => Lognormal(loss_range)
      5. Aggregate losses by role and total
    """
    # Calculate effective vulnerability after mitigation. effective vulnerability = base_vulnerability (75%) with no mitigation
    effective_vulnerability = BASE_VULNERABILITY * (1 - mitigation_weight)
    
    results = {
        'total_loss': [],
        'by_role': {role: [] for role in ROLES},
        'incidents_by_role': {role: [] for role in ROLES},
        'mitigation_weight': mitigation_weight,
        'effective_vulnerability': effective_vulnerability
    }
    
    np.random.seed(RANDOM_STATE)
    
    for iteration in range(n_iterations):
        total_loss = 0
        role_losses = {role: 0 for role in ROLES}
        role_incidents = {role: 0 for role in ROLES}
        
        for role in ROLES:
            headcount = ROLE_HEADCOUNT[role]
            tef = role_tef[role]
            
            n_insiders = np.random.binomial(headcount, tef)
            
            if n_insiders == 0:
                continue
            
            attempts_per_insider = np.random.poisson(ATTEMPTS_MEAN, n_insiders)
            total_attempts = attempts_per_insider.sum()
            
            if total_attempts == 0:
                continue
            
            # Apply mitigation weight to vulnerability
            n_successful_attacks = np.random.binomial(total_attempts, effective_vulnerability)
            
            role_incidents[role] = n_successful_attacks
            
            if n_successful_attacks == 0:
                continue
            
            loss_category = ROLE_MAPPING.get(role, 'Employees (full-time)')
            min_loss = loss_dict[loss_category]['min']
            max_loss = loss_dict[loss_category]['max']
            
            log_mean = (np.log(min_loss) + np.log(max_loss)) / 2
            log_std = (np.log(max_loss) - np.log(min_loss)) / 4
            
            attack_losses = np.random.lognormal(log_mean, log_std, n_successful_attacks)
            attack_losses = np.clip(attack_losses, min_loss, max_loss)
            
            role_loss = attack_losses.sum()
            role_losses[role] = role_loss
            total_loss += role_loss
        
        results['total_loss'].append(total_loss)
        for role in ROLES:
            results['by_role'][role].append(role_losses[role])
            results['incidents_by_role'][role].append(role_incidents[role])
    
    return results


def generate_monte_carlo_results(mitigation_weight=0.0):
    # Run simulation with mitigation
    results_with_mitigation = run_monte_carlo_simulation(mitigation_weight)
    
    # Run baseline simulation (no mitigation) for comparison
    results_baseline = run_monte_carlo_simulation(0.0)
    
    total_losses = np.array(results_with_mitigation['total_loss'])
    mean_loss = total_losses.mean()
    p5, p50, p95 = np.percentile(total_losses, [5, 50, 95])
    
    baseline_losses = np.array(results_baseline['total_loss'])
    baseline_mean = baseline_losses.mean()
    
    savings = baseline_mean - mean_loss
    savings_pct = (savings / baseline_mean * 100) if baseline_mean > 0 else 0
    
    role_data = {}
    role_baseline_means = {}
    
    for role in ROLES:
        role_mean_loss = np.array(results_with_mitigation['by_role'][role]).mean()
        role_mean_incidents = np.array(results_with_mitigation['incidents_by_role'][role]).mean()
        role_baseline_mean = np.array(results_baseline['by_role'][role]).mean()
        
        role_baseline_means[role] = role_baseline_mean
        
        role_data[role] = {
            'mean_loss': float(role_mean_loss),
            'mean_incidents': float(role_mean_incidents),
            'p5': float(np.percentile(results_with_mitigation['by_role'][role], 5)),
            'median': float(np.percentile(results_with_mitigation['by_role'][role], 50)),
            'p95': float(np.percentile(results_with_mitigation['by_role'][role], 95)),
            'max': float(np.array(results_with_mitigation['by_role'][role]).max())
        }
    

    output_data = {
        'total_company_loss': {
            'mean_eal': float(mean_loss),
            'p5': float(p5),
            'median': float(p50),
            'p95': float(p95),
            'max': float(total_losses.max()),
            'min': float(total_losses.min())
        },
        'loss_by_role': role_data,
        'comparison': {
            'baseline_mean_eal': float(baseline_mean),
            'with_mitigation_mean_eal': float(mean_loss),
            'total_savings': float(savings),
            'savings_percentage': float(savings_pct)
        }
    }
    
    json_path = OUTPUT_DIR / 'monte_carlo_results.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Overlapping loss distribution by role
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    role_colors = {
        'C_Level': '#8B0000',
        'Trader': '#4682B4',
        'IT_Admin': '#228B22',
        'Analyst': '#FF8C00',
        'Exec_Assistant': '#9370DB',
        'Contractor': '#DC143C'
    }
    
    for role in ROLES:
        role_losses = np.array(results_with_mitigation['by_role'][role]) / 1e6
        
        if role_losses.max() > 0:
            role_losses_nonzero = role_losses[role_losses > 0]
            if len(role_losses_nonzero) > 0:
                weights = np.ones_like(role_losses_nonzero) / N_ITER * 100
                
                ax.hist(role_losses_nonzero, bins=50, alpha=0.6, label=role, 
                       edgecolor='black', color=role_colors[role], linewidth=0.8,
                       weights=weights)
    
    ax.set_xlabel('Annual Loss ($ Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability (%)', fontsize=14, fontweight='bold')
    ax.set_title('Insider Threat Loss Distribution by Role', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9, 
             title='Employee Role', title_fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    stats_text = f"Total Company EAL: ${mean_loss/1e6:.1f}M\nP5-P95: ${p5/1e6:.1f}M - ${p95/1e6:.1f}M"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / 'monte_carlo_loss_distribution.jpg'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()

    
    # Bar chart for comparison against no mitigation
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Calculate mean losses for each role
    role_means_baseline = [role_baseline_means[role] / 1e6 for role in ROLES]
    role_means_mitigation = [role_data[role]['mean_loss'] / 1e6 for role in ROLES]
    
    x = np.arange(len(ROLES))
    width = 0.35
    
    # Plot baseline (faded/transparent) behind
    bars_baseline = ax.bar(x, role_means_baseline, width, 
                          label=f'Without Solution (EAL: ${baseline_mean/1e6:.1f}M)',
                          alpha=0.4, color='gray', edgecolor='black', linewidth=1.5)
    
    # Plot with mitigation (solid) in front
    bars_mitigation = ax.bar(x, role_means_mitigation, width,
                             label=f'With Solution (EAL: ${mean_loss/1e6:.1f}M)',
                             alpha=0.85, color=[role_colors[role] for role in ROLES],
                             edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (baseline_val, mitig_val) in enumerate(zip(role_means_baseline, role_means_mitigation)):
        if baseline_val > 0:
            # Show savings amount
            savings_val = baseline_val - mitig_val
            if savings_val > 0:
                ax.text(i, max(baseline_val, mitig_val) + 1, f'-${savings_val:.1f}M',
                       ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
    
    ax.set_xlabel('Employee Role', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Annual Loss ($ Millions)', fontsize=14, fontweight='bold')
    ax.set_title(f'Impact of Mitigation Solution on Insider Threat Losses\n'
                 f'Mitigation Effectiveness: {mitigation_weight*100:.0f}% | '
                 f'Total Savings: ${savings/1e6:.1f}M ({savings_pct:.1f}%)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(ROLES, rotation=45, ha='right')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # Add stats box
    stats_text = (f"Baseline Vulnerability: {BASE_VULNERABILITY*100:.0f}%\n"
                 f"With Solution: {BASE_VULNERABILITY * (1 - mitigation_weight)*100:.0f}%\n"
                 f"Risk Reduction: {mitigation_weight*100:.0f}%")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            family='monospace')
    
    plt.tight_layout()
    comparison_path = OUTPUT_DIR / 'mitigation_comparison.jpg'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
        
    

if __name__ == "__main__":
    # Example: Run with no mitigation
    generate_monte_carlo_results(mitigation_weight=0.7)
    
    # Example: Run with 60% mitigation
    # generate_monte_carlo_results(mitigation_weight=0.6)