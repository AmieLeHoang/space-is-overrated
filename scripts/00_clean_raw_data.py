import pandas as pd
import zipfile, glob, os, re

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'QCEW_Zips')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'employment_cleaned.csv')

code_map = {'10': 'total_emp', '52': 'finance_emp', '5112': 'software_emp', 
            '5182': 'dataproc_emp', '5415': 'it_consult_emp', 
            '44-45':'retail_emp', '72': 'restaurant_accommodation_emp'}

def clean_employment():
    all_zips = glob.glob(os.path.join(RAW_DIR, "*.zip"))
    dfs = []
    
    for zip_path in all_zips:
        year = os.path.basename(zip_path)[:4]
        with zipfile.ZipFile(zip_path, 'r') as z:
            for code, var_name in code_map.items():
                for f in z.namelist():
                    clean_name = re.sub(r'[^a-zA-Z0-9\-]', ' ', f)
                    if code in clean_name.split() and 'annual' in clean_name.split():
                        with z.open(f) as file_data:
                            df = pd.read_csv(file_data, low_memory=False)
                            if 'own_code' in df.columns:
                                df = df[df['own_code'] == 5]
                            df = df[['area_fips', 'year', 'annual_avg_emplvl']]
                            df['industry_type'] = var_name
                            dfs.append(df)

    master = pd.concat(dfs, ignore_index=True)
    final_df = master.pivot_table(index=['area_fips', 'year'], columns='industry_type', values='annual_avg_emplvl').reset_index().fillna(0)
    
    # Tech Definition & FIPS fix 
    final_df['area_fips'] = final_df['area_fips'].astype(str).str.zfill(5)
    final_df = final_df[~final_df['area_fips'].str.endswith('999') & final_df['area_fips'].str.isnumeric()]
    
    final_df.to_csv(OUT_PATH, index=False)
    print(f"Phase 00 Complete: {OUT_PATH}")

if __name__ == "__main__":
    clean_employment()