"""
05_build_quatile_regression.py

This script builds on the previous one by creating quartiles of housing prices and interacting them with 
the treatment and instrument. This allows us to directly estimate the treatment effect for each housing 
quartile without needing to do any manual calculations after the regression.

Input: data/processed/bartik_ready.csv (created by 03_build_bartik.py)
Output: Regression summary printed to console with quartile-specific treatment effects

Note: Ensure that the linearmodels package is installed in your Python environment to run this script.
"""

import pandas as pd
import os
from linearmodels.iv import IV2SLS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'bartik_ready.csv')

def run_quartile_analysis():
    print("Loading regression data")
    df = pd.read_csv(DATA_PATH)
    
    # We only want rows that have housing data
    req_cols = ['log_zhvi_2009', 'delta_log_rest_emp_10_19', 'delta_tech_share_10_19', 
                'bartik_shock_pure', 'pre_retail_growth', 'weight_2010']
    df = df.dropna(subset=req_cols).copy()
    
    df['const'] = 1
    
    # 1. CREATE HOUSING QUARTILES
    # q=4 creates 4 equal-sized bins (25% of CZs each) based on 2009 housing prices
    df['zhvi_quartile'] = pd.qcut(df['log_zhvi_2009'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # Create dummy variables for the quartiles (converts True/False to 1/0)
    quartile_dummies = pd.get_dummies(df['zhvi_quartile']).astype(int)
    df = pd.concat([df, quartile_dummies], axis=1)
    
   
    # 2. CREATE ENDOGENOUS INTERACTIONS
    # By interacting the treatment with ALL 4 quartiles, the model will output the exact multiplier for each quartile directly.
    df['tech_Q1'] = df['delta_tech_share_10_19'] * df['Q1'] # Cheapest 25%
    df['tech_Q2'] = df['delta_tech_share_10_19'] * df['Q2']
    df['tech_Q3'] = df['delta_tech_share_10_19'] * df['Q3']
    df['tech_Q4'] = df['delta_tech_share_10_19'] * df['Q4'] # Most Expensive 25%
    
    
    # 3. CREATE INSTRUMENT INTERACTIONS
    df['bartik_Q1'] = df['bartik_shock_pure'] * df['Q1']
    df['bartik_Q2'] = df['bartik_shock_pure'] * df['Q2']
    df['bartik_Q3'] = df['bartik_shock_pure'] * df['Q3']
    df['bartik_Q4'] = df['bartik_shock_pure'] * df['Q4']
    
    
    # 4. RUN 2SLS REGRESSION
    print("\n\nRunning Model D: Non-Linear Housing Quartile Effects")
    
    iv_quartile = IV2SLS(
        dependent=df['delta_log_rest_emp_10_19'],
        # Exogenous controls. We omit Q1 to act as the baseline intercept to avoid the dummy variable trap.
        exog=df[['const', 'pre_retail_growth', 'Q2', 'Q3', 'Q4']], 
        endog=df[['tech_Q1', 'tech_Q2', 'tech_Q3', 'tech_Q4']],
        instruments=df[['bartik_Q1', 'bartik_Q2', 'bartik_Q3', 'bartik_Q4']],
        weights=df['weight_2010']
    ).fit(cov_type='robust')
    
    print(iv_quartile.summary)

if __name__ == "__main__":
    run_quartile_analysis()