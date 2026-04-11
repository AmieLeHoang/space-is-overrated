"""
06b_retail_regression.py

This script runs the main retail regression, which is a placebo test to see if we find any significant effects 
of the tech shock on retail employment. If we do find significant effects here, it would suggest that our main 
results may be picking up some other confounding factor that is correlated with the tech shock but is actually 
driving retail employment changes, which would cast doubt on our main findings. On the other hand, if we find 
no significant effects here, it would strengthen our confidence that the main results are truly capturing the 
impact of the tech shock on retail employment.

Input: data/processed/bartik_ready.csv (created by 03_build_bartik.py)
Output: Regression summary printed to console for the retail placebo test  

Note: Ensure that the linearmodels package is installed in your Python environment to run this script, and that 
the input file from the previous step exists at the specified path. The regression will use the 
delta_log_retail_emp_10_19 variable, which captures retail employment changes, as the dependent variable.
"""


import pandas as pd
import os
from linearmodels.iv import IV2SLS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'bartik_ready.csv')

def run_retail_placebo():
    print("Loading retail placebo data")
    df = pd.read_csv(DATA_PATH)
    df['const'] = 1
    
    print("Running Model E: The Retail 'Apocalypse' Placebo Test")
    
    base_vars = ['delta_log_retail_emp_10_19', 'delta_tech_share_10_19', 'bartik_shock_pure', 'weight_2010', 'pre_retail_growth']
    df_base = df.dropna(subset=base_vars).copy()
    
    iv_base = IV2SLS(
        dependent=df_base['delta_log_retail_emp_10_19'],
        exog=df_base[['const', 'pre_retail_growth']], 
        endog=df_base['delta_tech_share_10_19'],
        instruments=df_base['bartik_shock_pure'],
        weights=df_base['weight_2010'] 
    ).fit(cov_type='robust')
    
    print(iv_base.summary)

if __name__ == "__main__":
    run_retail_placebo()