"""
06_retail_housing_interaction.py

This script runs an additional regression that interacts the tech shock with housing costs, but with retail employment 
as the dependent variable instead of wages or total employment. This serves as a robustness check to see if the retail 
"apocalypse" effects of the tech shock are also stronger in high housing cost areas, which would be consistent with the 
idea that housing constraints amplify the negative impacts of the tech shock on retail employment. If we find similar 
interaction effects here as we do in the main wage and employment regressions, it would further strengthen our argument 
that housing constraints are a key mechanism driving the spatial variation in the tech shock's impacts.

The setup of this regression is very similar to the main housing interaction regression, except that the dependent variable 
is now the change in retail employment (delta_log_retail_emp_10_19) instead of wages or total employment. We will still 
interact the tech shock with the log of 2009 housing prices to see if the amplification effect holds for retail employment 
as well. By including the interaction directly in the regression, we can see the differential impact of the tech shock on 
retail employment across housing quartiles without needing to do any manual calculations after the regression.

Input: data/processed/bartik_ready.csv (created by 02_build_bartik.py)
Output: Regression summary printed to console for the retail-housing interaction model

Note: Ensure that the linearmodels package is installed in your Python environment to run this script, and that the input 
file from the previous step exists at the specified path. The regression will use the delta_log
"""


import pandas as pd
import os
from linearmodels.iv import IV2SLS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'bartik_ready.csv')

def run_retail_interaction():
    print("Loading retail regression data")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure interaction variables exist
    if 'tech_X_zhvi' not in df.columns:
        df['tech_X_zhvi'] = df['delta_tech_share_10_19'] * df['log_zhvi_2009']
    if 'bartik_X_zhvi' not in df.columns:
        df['bartik_X_zhvi'] = df['bartik_shock_pure'] * df['log_zhvi_2009']
        
    df['const'] = 1
    
    print("\nRunning Model F: Retail Spillovers under Housing Constraints")
    
    interact_vars = [
        'delta_log_retail_emp_10_19', 
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
        dependent=df_int['delta_log_retail_emp_10_19'],
        exog=df_int[['const', 'pre_retail_growth', 'log_zhvi_2009']],
        endog=df_int[['delta_tech_share_10_19', 'tech_X_zhvi']],
        instruments=df_int[['bartik_shock_pure', 'bartik_X_zhvi']],
        weights=df_int['weight_2010']
    ).fit(cov_type='robust')
    
    print(iv_int.summary)

if __name__ == "__main__":
    run_retail_interaction()