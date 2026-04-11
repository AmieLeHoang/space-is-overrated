"""
06a_extract_retail.py
This script extracts retail employment data from the raw QCEW files, aggregates it to the commuting zone level, and merges it 
into our master panel dataset. It follows a similar structure to 05a_extract_wages.py, but focuses on NAICS 44-45 (Retail Trade) 
instead of Accommodation and Food Services.

Steps:
1. Load the raw QCEW zip files for 2010 and 2019.
2. Identify the files containing retail employment data (NAICS 44-45).
3. Extract average employment for each county, then aggregate to the commuting zone level using the county-to-CZ crosswalk.
4. Calculate the log difference in retail employment between 2010 and 2019 for each CZ.
5. Merge the resulting retail employment change variable into the bartik_ready.csv dataset.      

Input:
- data/raw/QCEW_Zips/*.zip (Raw QCEW data for 2010 and 2019)
- data/raw/cw_cty_czone.dta (County to Commuting Zone crosswalk)   

Output:
- data/processed/bartik_ready.csv (Updated with retail employment change variable)

Note: This script assumes that the raw QCEW zip files are organized in a specific way and that the necessary columns are present. 
Ensure that the input files exist and are correctly formatted before running this script.
"""



import pandas as pd
import zipfile, glob, os, re
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'QCEW_Zips')
CZ_CWALK = os.path.join(BASE_DIR, 'data', 'raw', 'cw_cty_czone.dta')
BARTIK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'bartik_ready.csv')

def extract_retail_employment():
    print("Extracting Retail (NAICS 44-45) data from raw QCEW zips...")
    all_zips = glob.glob(os.path.join(RAW_DIR, "*.zip"))
    dfs = []
    
    for zip_path in all_zips:
        year = int(os.path.basename(zip_path)[:4])
        if year not in [2010, 2019]:
            continue
            
        with zipfile.ZipFile(zip_path, 'r') as z:
            for f in z.namelist():
                clean_name = re.sub(r'[^a-zA-Z0-9\-]', ' ', f)
                # QCEW often groups Retail as '44-45'
                if '44-45' in clean_name and 'annual' in clean_name.split():
                    with z.open(f) as file_data:
                        df = pd.read_csv(file_data, low_memory=False)
                        if 'own_code' in df.columns:
                            df = df[df['own_code'] == 5] # Private sector only
                        
                        if 'annual_avg_emplvl' in df.columns:
                            df = df[['area_fips', 'year', 'annual_avg_emplvl']]
                            dfs.append(df)
                            
    master = pd.concat(dfs, ignore_index=True)
    master['area_fips'] = master['area_fips'].astype(str).str.zfill(5)
    master = master[~master['area_fips'].str.endswith('999') & master['area_fips'].str.isnumeric()]
    
    print("Aggregating to Commuting Zones...")
    cw = pd.read_stata(CZ_CWALK)
    cw['area_fips'] = cw['cty_fips'].astype(str).str.strip().str.split('.').str[0].str.zfill(5)
    if 'cz' in cw.columns:
        cw = cw.rename(columns={'cz': 'czone'})
        
    df_merged = pd.merge(master, cw[['area_fips', 'czone']], on='area_fips', how='inner')
    cz_retail = df_merged.groupby(['czone', 'year'])['annual_avg_emplvl'].sum().reset_index()
    
    cz_pivot = cz_retail.pivot(index='czone', columns='year', values='annual_avg_emplvl').reset_index()
    cz_pivot.columns = ['czone', 'retail_2010', 'retail_2019']
    
    # Calculate the log difference for the outcome
    cz_pivot['delta_log_retail_emp_10_19'] = np.log(cz_pivot['retail_2019'].replace(0, np.nan)) - np.log(cz_pivot['retail_2010'].replace(0, np.nan))
    
    print("Merging into bartik_ready.csv...")
    df_bartik = pd.read_csv(BARTIK_PATH)
    
    if 'delta_log_retail_emp_10_19' in df_bartik.columns:
        df_bartik = df_bartik.drop(columns=['delta_log_retail_emp_10_19'])
        
    df_bartik = pd.merge(df_bartik, cz_pivot[['czone', 'delta_log_retail_emp_10_19']], on='czone', how='left')
    df_bartik.to_csv(BARTIK_PATH, index=False)
    print(f"Success! Retail data appended to {BARTIK_PATH}")

if __name__ == "__main__":
    extract_retail_employment()