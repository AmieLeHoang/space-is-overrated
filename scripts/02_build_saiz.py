"""
02_build_saiz.py

This script creates a bridge between the Saiz elasticity dataset and our commuting zone panel, allowing us to incorporate local 
housing supply elasticity estimates into our regression analyses.
It performs the following steps:
1. Loads the necessary datasets, including the Saiz elasticity data and geographic crosswalks.
2. Applies an enhanced string matching logic to link Saiz's MSA-level elasticities to our county-level crosswalk.
3. Aggregates the matched elasticities up to the commuting zone level.
4. Merges the resulting CZ-level elasticity estimates into our master panel dataset.

Input:
- data/raw/Saiz_elasticity.csv (Saiz's MSA-level elasticity estimates)
- data/raw/crosswalk_msa.csv (County to MSA crosswalk)
- data/raw/cw_cty_czone.dta (County to Commuting Zone crosswalk)
- data/processed/czone_master_panel.csv (Master panel dataset from previous steps)

Output:
- data/processed/czone_master_panel_with_saiz.csv (Master panel with Saiz elasticity merged in) 

Note: The matching logic for linking Saiz's MSA names to our crosswalk is intentionally flexible to account for naming discrepancies.
Ensure that the input files exist and are correctly formatted before running this script.
"""


import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- FIXED PATHS ---
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'czone_master_panel.csv')
CROSSWALK_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'crosswalk_msa.csv')
CZ_CWALK = os.path.join(BASE_DIR, 'data', 'raw', 'cw_cty_czone.dta')
SAIZ_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Saiz_elasticity.csv') 

OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'czone_master_panel_with_saiz.csv')

def find_saiz_match(census_title, saiz_df):
    """Refined matcher for MSA titles"""
    if pd.isna(census_title) or census_title == 'nan': 
        return None
        
    for _, row in saiz_df.iterrows():
        saiz_full_name = row['saiz_clean']
        parts = saiz_full_name.split(',')
        if len(parts) < 2: continue
        
        first_city = parts[0].split('-')[0].strip()
        state_abbr = parts[1].strip()[:2] 
        
        if first_city in census_title and state_abbr in census_title:
            return row['Supply Elasticity']
    return None

def build_saiz_bridge():
    print("Loading geographic crosswalks...")
    df_crosswalk = pd.read_csv(CROSSWALK_PATH, dtype={'County Code': str})
    df_saiz = pd.read_csv(SAIZ_RAW_PATH, dtype={'MSA/NECMA Name': str})
    cw_cz = pd.read_stata(CZ_CWALK)

    # 1. Clean FIPS
    df_crosswalk['area_fips'] = df_crosswalk['County Code'].astype(str).str.zfill(5)
    cw_cz['area_fips'] = cw_cz['cty_fips'].astype(str).str.strip().str.split('.').str[0].str.zfill(5)
    if 'cz' in cw_cz.columns:
        cw_cz = cw_cz.rename(columns={'cz': 'czone'})

    # 2. String Match Saiz to MSA Crosswalk
    print("Applying enhanced matching logic for Saiz Elasticities...")
    df_crosswalk['msa_clean'] = df_crosswalk['MSA Title'].astype(str).str.lower().str.strip()
    df_saiz['saiz_clean'] = df_saiz['MSA/NECMA Name'].astype(str).str.lower().str.strip()
    
    df_crosswalk['saiz_elasticity'] = df_crosswalk['msa_clean'].apply(
        lambda x: find_saiz_match(x, df_saiz)
    )

    # 3. Bridge County MSA data to Commuting Zones
    print("Aggregating MSAs to Commuting Zones...")
    bridge_df = pd.merge(cw_cz[['area_fips', 'czone']], 
                         df_crosswalk[['area_fips', 'saiz_elasticity']], 
                         on='area_fips', how='inner')
    
    # Drop counties that didn't match an MSA/Saiz value
    bridge_df = bridge_df.dropna(subset=['saiz_elasticity'])
    
    # Calculate the mean elasticity for each Commuting Zone
    cz_saiz = bridge_df.groupby('czone')['saiz_elasticity'].mean().reset_index()
    
    # 4. Merge into the Master Panel
    print("Merging into Master Panel...")
    df_main = pd.read_csv(DATA_PATH)
    df_final = pd.merge(df_main, cz_saiz, on='czone', how='left')
    
    df_final.to_csv(OUT_PATH, index=False)
    
    matched_czs = cz_saiz['czone'].nunique()
    print(f"Phase 02 Complete. Successfully mapped Saiz elasticity to {matched_czs} Commuting Zones.")
    print(f"Saved to: {OUT_PATH}")

if __name__ == "__main__":
    build_saiz_bridge()