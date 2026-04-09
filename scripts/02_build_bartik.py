"""
02_build_bartik.py

This script constructs the Bartik instrument for the pure tech shock, following Moretti's functional form. It also prepares 
the dataset for regression analysis by pivoting the data and calculating necessary variables. The output is saved as 
'bartik_ready.csv' in the processed data directory."

Input: data/processed/czone_master_panel.csv (created by 01_build_housing_cz.py)
Output: data/processed/bartik_ready.csv (ready for regression analysis in 03_build_regression.py)

Note: Ensure that the input file from the previous step exists before running this script.
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'czone_master_panel.csv')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'bartik_ready.csv')

def construct_bartik():
    print("Loading CZ panel data")
    df_cz = pd.read_csv(DATA_PATH)
    
    # Keep necessary years
    df_years = df_cz[df_cz['year'].isin([2005, 2009, 2010, 2019])].copy()
    
    # Set indices to czone
    df_2009 = df_years[df_years['year'] == 2009].set_index('czone')
    df_2019 = df_years[df_years['year'] == 2019].set_index('czone')
    
    # =========================================================
    # 2. CONSTRUCT THE LEAVE-ONE-OUT (LOO) PURE TECH BARTIK
    # =========================================================
    tech_sectors = ['software_emp', 'dataproc_emp', 'it_consult_emp']
    
    print("Calculating 2009 Local Shares and Leave-One-Out (LOO) Bartik")
    
    # Initialize an empty series for the shock
    bartik_shock_pure = pd.Series(0.0, index=df_2009.index)

    for k in tech_sectors:
        # 1. Get total national employment for sector k
        nat_2009_total = df_2009[k].sum()
        nat_2019_total = df_2019[k].sum()
        
        # 2. Subtract the local CZ's employment to create the LOO national totals
        # This creates a unique pandas Series of national totals for every single CZ
        loo_nat_2009 = nat_2009_total - df_2009[k]
        loo_nat_2019 = nat_2019_total - df_2019[k]
        
        # 3. Calculate the LOO growth rate (Log difference)
        # We use np.where to handle any weird edge cases where a single CZ contained 
        # 100% of the national employment (which would make loo_nat_2009 = 0)
        loo_g_rate = np.where(
            loo_nat_2009 > 0, 
            np.log(loo_nat_2019.replace(0, np.nan)) - np.log(loo_nat_2009), 
            0
        )
        
        # 4. Multiply the local 2009 share by the LOO growth rate
        share_2009 = df_2009[k] / df_2009['total_emp']
        bartik_shock_pure += (share_2009.fillna(0) * loo_g_rate)

    # =========================================================
    # 3. PIVOT AND CALCULATE MORETTI FUNCTIONAL FORM
    # =========================================================

    # Save Zillow data if it exists
    housing_data = df_years[['czone', 'log_zhvi_2009']].drop_duplicates() if 'log_zhvi_2009' in df_years.columns else None

    df_pivot = df_years.pivot(
        index='czone', columns='year', 
        values=['restaurant_accommodation_emp', 'retail_emp', 'total_emp'] + tech_sectors
    ).reset_index()
    
    df_pivot.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df_pivot.columns]
    
    if housing_data is not None:
        df_pivot = pd.merge(df_pivot, housing_data, on='czone', how='left')

    # Calculate pure tech aggregates
    df_pivot['pure_tech_2009'] = df_pivot['software_emp_2009'] + df_pivot['dataproc_emp_2009'] + df_pivot['it_consult_emp_2009']
    df_pivot['pure_tech_2010'] = df_pivot['software_emp_2010'] + df_pivot['dataproc_emp_2010'] + df_pivot['it_consult_emp_2010']
    df_pivot['pure_tech_2019'] = df_pivot['software_emp_2019'] + df_pivot['dataproc_emp_2019'] + df_pivot['it_consult_emp_2019']
    
    # # --- Moretti Multipliers ---
    # # New Jobs per Initial Worker
    df_pivot['actual_tech_growth_10_19'] = (df_pivot['pure_tech_2019'] - df_pivot['pure_tech_2010']) / df_pivot['total_emp_2010']
    df_pivot['rest_growth_10_19'] = (df_pivot['restaurant_accommodation_emp_2019'] - df_pivot['restaurant_accommodation_emp_2010']) / df_pivot['total_emp_2010']

    # 1. Calculate shares
    df_pivot['tech_share_2019'] = df_pivot['pure_tech_2019'] / df_pivot['total_emp_2019']
    df_pivot['tech_share_2009'] = df_pivot['pure_tech_2009'] / df_pivot['total_emp_2009']

    # 2. Calculating raw diff
    df_pivot['delta_tech_share_pp'] = (df_pivot['tech_share_2019'] - df_pivot['tech_share_2009'])

    # 3. Calculate Outcome: Delta Log Restaurant Employment (2009 to 2019)
    # Replacing 0 with np.nan to avoid -inf issues in logs
    # OUTCOME: Log Difference (2010 to 2019)
    rest_19 = df_pivot['restaurant_accommodation_emp_2019'].replace(0, np.nan)
    rest_10 = df_pivot['restaurant_accommodation_emp_2010'].replace(0, np.nan)
    df_pivot['delta_log_rest_emp_10_19'] = np.log(rest_19) - np.log(rest_10)

    # TREATMENT: Decimal Share Difference (2010 to 2019)
    share_19 = df_pivot['pure_tech_2019'] / df_pivot['total_emp_2019']
    share_10 = df_pivot['pure_tech_2010'] / df_pivot['total_emp_2010']
    df_pivot['delta_tech_share_10_19'] = share_19 - share_10
    
    # Pre-trends and weights
    df_pivot['pre_retail_growth'] = np.log(df_pivot['retail_emp_2009'].replace(0, np.nan)) - np.log(df_pivot['retail_emp_2005'].replace(0, np.nan))
    df_pivot['weight_2010'] = df_pivot['total_emp_2010']

    df_pivot['bartik_shock_pure'] = df_pivot['czone'].map(bartik_shock_pure)

    df_pivot['tech_2009_total'] = df_pivot['software_emp_2009'] + df_pivot['dataproc_emp_2009'] + df_pivot['it_consult_emp_2009']

    # Filter to match the presentation's N=480
    df_pivot = df_pivot[df_pivot['tech_2009_total'] > 0]
    
    df_pivot = df_pivot.replace([np.inf, -np.inf], np.nan)
    df_pivot.to_csv(OUT_PATH, index=False)
    print(f"Phase 02 Complete: {OUT_PATH}")

if __name__ == "__main__":
    construct_bartik()