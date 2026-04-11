import pandas as pd
import numpy as np
import os

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMP_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'employment_cleaned.csv')
ZHVI_DATA = os.path.join(BASE_DIR, 'data', 'raw', 'County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
CZ_CWALK = os.path.join(BASE_DIR, 'data', 'raw', 'cw_cty_czone.dta')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'czone_master_panel.csv')

def build_spatial_data():
    print("Loading datasets...")
    df = pd.read_csv(EMP_DATA, dtype={'area_fips': str})
    cw = pd.read_stata(CZ_CWALK)

    # 1. Clean FIPS (to ensure match)
    df['area_fips'] = df['area_fips'].astype(str).str.strip().str.split('.').str[0].str.zfill(5)
    cw['area_fips'] = cw['cty_fips'].astype(str).str.strip().str.split('.').str[0].str.zfill(5)
    if 'cz' in cw.columns:
        cw = cw.rename(columns={'cz': 'czone'})

    # 2. Merge Employment to CZ
    df = pd.merge(df, cw[['area_fips', 'czone']], on='area_fips', how='inner')
    df['tech_fin_emp'] = df['software_emp'] + df['dataproc_emp'] + df['it_consult_emp'] + df['finance_emp']

    # 3. Aggregate Employment to CZ Level
    agg_dict = {
        'total_emp': 'sum', 
        'tech_fin_emp': 'sum', 
        'restaurant_accommodation_emp': 'sum', 
        'retail_emp': 'sum',
        'software_emp': 'sum',
        'dataproc_emp': 'sum',
        'it_consult_emp': 'sum',
        'finance_emp': 'sum'
    }
    df_cz = df.groupby(['czone', 'year']).agg(agg_dict).reset_index()

    # ==========================================
    # 4. PROCESS ZILLOW HOUSING DATA
    # ==========================================
    print("Processing Zillow Housing Data...")
    if os.path.exists(ZHVI_DATA):
        df_zhvi = pd.read_csv(ZHVI_DATA, dtype={'StateCodeFIPS': str, 'MunicipalCodeFIPS': str})
        
        # Create 5-digit FIPS for housing
        df_zhvi['area_fips'] = df_zhvi['StateCodeFIPS'].str.zfill(2) + df_zhvi['MunicipalCodeFIPS'].str.zfill(3)
        
        # Calculate 2009 baseline mean home value for each county
        cols_2009 = [c for c in df_zhvi.columns if c.startswith('2009')]
        df_zhvi['mean_zhvi_2009'] = df_zhvi[cols_2009].mean(axis=1)
        
        # Merge housing with crosswalk to get CZ mapping
        df_housing = pd.merge(df_zhvi[['area_fips', 'mean_zhvi_2009']], cw[['area_fips', 'czone']], on='area_fips', how='inner')
        
        # Grab 2010 employment to use as weights (to avoid small towns skewing the city average)
        weights_2010 = df[df['year'] == 2010][['area_fips', 'total_emp']]
        df_housing = pd.merge(df_housing, weights_2010, on='area_fips', how='left').dropna(subset=['mean_zhvi_2009'])
        
        # Calculate Weighted Average at the CZ level
        df_housing['weighted_2009'] = df_housing['mean_zhvi_2009'] * df_housing['total_emp']
        cz_housing = df_housing.groupby('czone').sum(numeric_only=True, min_count=1).reset_index()
        cz_housing['cz_zhvi_2009'] = cz_housing['weighted_2009'] / cz_housing['total_emp']
        
        # Calculate final LOG value
        cz_housing['log_zhvi_2009'] = np.log(cz_housing['cz_zhvi_2009'])
        
        # Merge this baseline constraint back into the master employment panel
        df_cz = pd.merge(df_cz, cz_housing[['czone', 'log_zhvi_2009']], on='czone', how='left')
        print("Successfully attached 'log_zhvi_2009' to master panel")
    else:
        print(f"WARNING: Zillow data not found at {ZHVI_DATA}. Skipping housing merge.")

    # 5. Save result
    df_cz.to_csv(OUT_PATH, index=False)
    print(f"Phase 01 Complete: {OUT_PATH}")

if __name__ == "__main__":
    build_spatial_data()