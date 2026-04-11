"""
05b_wage_regression.py

This script runs the wage regression analyses using the delta_log_rest_wage_10_19 variable created in 05a_extract_wages.py. 
It estimates both a baseline wage spillover model and a housing interaction model, printing out robust standard errors and
first stage results for each.    

Input: data/processed/bartik_ready.csv (created by 05a_extract_wages.py)
Output: Regression summaries printed to console

Note: Ensure that the linearmodels package is installed in your Python environment to run this script.
"""


import pandas as pd
import os
from linearmodels.iv import IV2SLS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'bartik_ready.csv')

def run_wage_analysis():
    print("Loading wage regression data")
    df = pd.read_csv(DATA_PATH)
    
    # Create the interaction using the correct share difference
    if 'log_zhvi_2009' in df.columns:
        df['tech_X_zhvi'] = df['delta_tech_share_10_19'] * df['log_zhvi_2009']
        df['bartik_X_zhvi'] = df['bartik_shock_pure'] * df['log_zhvi_2009']
    
    df['const'] = 1
    
    # ==========================================
    # MODEL A: BASELINE WAGE SPILLOVER
    # ==========================================
    print("Running Model A: Baseline Wage Spillover")
    
    base_vars = ['delta_log_rest_wage_10_19', 'delta_tech_share_10_19', 'bartik_shock_pure', 'weight_2010', 'pre_retail_growth']
    df_base = df.dropna(subset=base_vars).copy()
    
    iv_base = IV2SLS(
        dependent=df_base['delta_log_rest_wage_10_19'],
        exog=df_base[['const', 'pre_retail_growth']], 
        endog=df_base['delta_tech_share_10_19'],
        instruments=df_base['bartik_shock_pure'],
        weights=df_base['weight_2010'] 
    ).fit(cov_type='robust')
    
    print(iv_base.summary)
    
    # ==========================================
    # MODEL B: HOUSING COST INTERACTION (WAGES)
    # ==========================================
    if 'log_zhvi_2009' in df.columns:
        print("\n\nRunning Model B: Housing Cost Interaction on Wages")
        
        interact_vars = [
            'delta_log_rest_wage_10_19', 
            'delta_tech_share_10_19', 
            'bartik_shock_pure', 
            'weight_2010', 
            'pre_retail_growth',  
            'log_zhvi_2009', 
            'tech_X_zhvi', 
            'bartik_X_zhvi'
        ]
        
        df_int = df.dropna(subset=interact_vars).copy()
        
        iv_int = IV2SLS(
            dependent=df_int['delta_log_rest_wage_10_19'],
            exog=df_int[['const', 'pre_retail_growth', 'log_zhvi_2009']],
            endog=df_int[['delta_tech_share_10_19', 'tech_X_zhvi']],
            instruments=df_int[['bartik_shock_pure', 'bartik_X_zhvi']],
            weights=df_int['weight_2010']
        ).fit(cov_type='robust')
        
        print(iv_int.summary)

if __name__ == "__main__":
    run_wage_analysis()