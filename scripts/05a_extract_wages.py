"""
05a_extract_wages.py

This script extracts average wage data from the raw QCEW files, aggregates it to the commuting zone level, and merges it 
into our master panel dataset.

Steps:
1. Load the raw QCEW zip files for 2010 and 2019.
2. Identify the files containing wage data (NAICS 72 - Accommodation and Food Services).
3. Extract total annual wages and average employment, then calculate average wage per worker for each county.
4. Aggregate the data to the commuting zone level using the county-to-CZ crosswalk.
5. Calculate the log difference in average wages between 2010 and 2019 for each CZ.
6. Merge the resulting wage change variable into the bartik_ready.csv dataset.      

Input:
- data/raw/QCEW_Zips/*.zip (Raw QCEW data for 2010 and 2019)
- data/raw/cw_cty_czone.dta (County to Commuting Zone crosswalk)   

Output:
- data/processed/bartik_ready.csv (Updated with wage change variable)  

Note: This script assumes that the raw QCEW zip files are organized in a specific way and that the necessary columns are present.
"""




import pandas as pd
import zipfile, glob, os, re
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'QCEW_Zips')
CZ_CWALK = os.path.join(BASE_DIR, 'data', 'raw', 'cw_cty_czone.dta')
BARTIK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'bartik_ready.csv')

def extract_and_append_wages():
    print("Extracting wage data from raw QCEW zips")
    all_zips = glob.glob(os.path.join(RAW_DIR, "*.zip"))
    dfs = []
    
    for zip_path in all_zips:
        year = int(os.path.basename(zip_path)[:4])
        if year not in [2010, 2019]:
            continue
            
        with zipfile.ZipFile(zip_path, 'r') as z:
            for f in z.namelist():
                clean_name = re.sub(r'[^a-zA-Z0-9\-]', ' ', f)
                if '72' in clean_name.split() and 'annual' in clean_name.split():
                    with z.open(f) as file_data:
                        df = pd.read_csv(file_data, low_memory=False)
                        if 'own_code' in df.columns:
                            df = df[df['own_code'] == 5] # Private sector only
                        
                        # Keep employment AND total annual wages
                        if 'total_annual_wages' in df.columns:
                            df = df[['area_fips', 'year', 'annual_avg_emplvl', 'total_annual_wages']]
                            dfs.append(df)
                            
    master = pd.concat(dfs, ignore_index=True)
    
    # Clean FIPS codes to match crosswalk
    master['area_fips'] = master['area_fips'].astype(str).str.zfill(5)
    master = master[~master['area_fips'].str.endswith('999') & master['area_fips'].str.isnumeric()]
    
    print("Aggregating to Commuting Zones...")
    cw = pd.read_stata(CZ_CWALK)
    cw['area_fips'] = cw['cty_fips'].astype(str).str.strip().str.split('.').str[0].str.zfill(5)
    if 'cz' in cw.columns:
        cw = cw.rename(columns={'cz': 'czone'})
        
    df_merged = pd.merge(master, cw[['area_fips', 'czone']], on='area_fips', how='inner')
    
    # Sum total employment and total wages by CZ and Year
    cz_wages = df_merged.groupby(['czone', 'year'])[['annual_avg_emplvl', 'total_annual_wages']].sum().reset_index()
    
    # Calculate average annual wage: (Total Wages / Total Workers)
    cz_wages['avg_wage'] = cz_wages['total_annual_wages'] / cz_wages['annual_avg_emplvl'].replace(0, np.nan)
    
    # Pivot to get 2010 and 2019 columns
    cz_pivot = cz_wages.pivot(index='czone', columns='year', values='avg_wage').reset_index()
    cz_pivot.columns = ['czone', 'wage_2010', 'wage_2019']
    
    # Calculate the log difference
    cz_pivot['delta_log_rest_wage_10_19'] = np.log(cz_pivot['wage_2019']) - np.log(cz_pivot['wage_2010'])
    
    print("Merging into bartik_ready.csv...")
    df_bartik = pd.read_csv(BARTIK_PATH)
    
    # Safely drop the column if it already exists from a previous run
    if 'delta_log_rest_wage_10_19' in df_bartik.columns:
        df_bartik = df_bartik.drop(columns=['delta_log_rest_wage_10_19'])
        
    df_bartik = pd.merge(df_bartik, cz_pivot[['czone', 'delta_log_rest_wage_10_19']], on='czone', how='left')
    df_bartik.to_csv(BARTIK_PATH, index=False)
    print(f"Complete. Wage data appended to {BARTIK_PATH}")

if __name__ == "__main__":
    extract_and_append_wages()