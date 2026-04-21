"""
07_rotemberg_weight.py

Replication script for Table 9: GPS (2020) Rotemberg Weights Decompositions.
This script calculates the just-identified IV estimates and Rotemberg weights 
for the component industries of the Standard (non-LOO) Bartik instrument.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS


# 1. LOAD DATA & PREPARE VARIABLES
df = pd.read_csv('data/processed/bartik_ready.csv')

# Get 2009 local employment shares for each sector
df['share_2009_5112'] = df['software_emp_2009'] / df['total_emp_2009']
df['share_2009_5182'] = df['dataproc_emp_2009'] / df['total_emp_2009']
df['share_2009_5415'] = df['it_consult_emp_2009'] / df['total_emp_2009']

# Fill any NaNs created by 0/0 division with 0 + drop na
df.fillna({'share_2009_5112': 0, 'share_2009_5182': 0, 'share_2009_5415': 0}, inplace=True)
df = df.dropna(subset=['delta_log_rest_emp_10_19', 'delta_tech_share_10_19', 'pre_retail_growth', 'total_emp_2010'])

# Define vars
weights = df['total_emp_2010']
endog = df['delta_tech_share_10_19']
exog_controls = sm.add_constant(df['pre_retail_growth'])

# Tech Sectors NAICS code
tech_sectors = {
    '5112': 'Software Publishers',
    '5182': 'Data Processing',
    '5415': 'IT Consulting'
}

# Hardcoded national growth rates (g_k)
g_k = {
    '5112': 0.5566, 
    '5182': 0.3734,
    '5415': 0.3798
}

shares = {
    '5112': df['share_2009_5112'],
    '5182': df['share_2009_5182'],
    '5415': df['share_2009_5415']
}


# 2. HELPER FUNCTIONS
def residualize(y, X, w):
    """Residualizes variable y against controls X using WLS."""
    model = sm.WLS(y, X, weights=w).fit()
    return model.resid


# 3. GET ROTEMBERG WEIGHTS
# A. Residualize the endogenous variable (Delta Tech Share)
endog_resid = residualize(endog, exog_controls, weights)

# B. Reconstruct the Standard Bartik & Residualize it
df['standard_bartik'] = sum(shares[k] * g_k[k] for k in tech_sectors.keys())
bartik_resid = residualize(df['standard_bartik'], exog_controls, weights)

# C. Calculate the Denominator of the Rotemberg Weight
alpha_denominator = np.sum(weights * bartik_resid * endog_resid)

results = []

print("==================================================================")
print("GPS (2020) Rotemberg Decomposition (Standard Bartik)")
print("==================================================================")

for k, name in tech_sectors.items():
    # 1. Calculate % CZs > 0
    pct_cz_gt_zero = (shares[k] > 0).mean() * 100
    
    # 2. Get Rotemberg Weight
    share_resid = residualize(shares[k], exog_controls, weights)
    alpha_numerator = g_k[k] * np.sum(weights * share_resid * endog_resid)
    alpha = alpha_numerator / alpha_denominator
    
    # 3. Get IV Estimate (beta_k)
    iv_model = IV2SLS(
        dependent=df['delta_log_rest_emp_10_19'],
        exog=exog_controls,
        endog=endog,
        instruments=shares[k],
        weights=weights
    ).fit(cov_type='robust')
    
    beta_k = iv_model.params['delta_tech_share_10_19']
    se_k = iv_model.std_errors['delta_tech_share_10_19']
    
    results.append({
        'Sector': name,
        'g_k': round(g_k[k], 4),
        'alpha_k': round(alpha, 4),
        'beta_k': round(beta_k, 4),
        'SE(beta_k)': round(se_k, 4),
        '%CZs>0': f"{pct_cz_gt_zero:.1f}%"
    })


# 4. PRINT FORMATTED TABLE
results_df = pd.DataFrame(results)
implied_bartik = (results_df['alpha_k'] * results_df['beta_k']).sum()

print("\nPanel A: Tech Sectors -> Restaurant Employment")
print(results_df.to_string(index=False))
print("-" * 65)
print(f"Sum of alpha_k:                 {results_df['alpha_k'].sum():.4f}")
print(f"Implied Bartik (Sum a_k * b_k): {implied_bartik:.4f}")