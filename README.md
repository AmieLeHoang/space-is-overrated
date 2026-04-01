# Trading Tech for Space: Service Exports, Housing Constraints, and Local Employment Spillovers

**Author:** Amie Le Hoang

**Date:** March 31, 2026

## Abstract
This research investigates the distributional consequences of the 21st-century **"App Economy"** boom on local labor markets in the United State. Specifically, it examines whether growth driven by export booms in tradable high-skilled technology sectors (Software, Data Processing, and IT Consulting) generates a positive local employment multiplier for the non-tradable service sector, such as restaurants and accommodations. Using a balanced long-difference panel of **480 US Commuting Zones** from 2010 to 2019, the study explores how housing supply constraints and rising rents may "crowd out" these potential economic gains.

---

## Key Methodology
The study utilizes a **Reduced-Form Shift-Share (Bartik) regression** to establish causality by isolating exogenous labor demand shocks originating from national export trends.

### Econometric Specification
The secondary specification tests the displacement hypothesis in supply-constrained markets using the following interaction model:

$$\Delta ln(Emp_{c,rest}) = \alpha + \beta_{1}Shock_{c} + \beta_{2}(Shock_{c} \times Elasticity_{c}) + \delta Elasticity_{c} + \gamma \Delta ln M_{c} + X_{c}' \theta + \epsilon_{c}$$

* **$\Delta ln(Emp_{c,rest})$**: Log change in restaurant and accommodation employment (NAICS 722).
* **$Shock_{c}$**: Log growth of tech employment.
* **$Elasticity_{c}$**: Housing supply elasticity index.
* **$\Delta ln M_{c}$**: Log change in manufacturing employment (control for deindustrialization).

---

## Data Sources
* **Labor Market Data**: Bureau of Labor Statistics (BLS) Quarterly Census of Employment and Wages (QCEW).
* **Housing Prices**: Zillow Home Value Index (ZHVI).
* **Housing Elasticity**: Saiz (2010) Housing Supply Elasticity index.
* **Geographic Unit**: County-level data aggregated to **David Dorn’s Commuting Zones (CZs)**.

---

## Principal Findings
* **The Multiplier**: A one standard deviation (1 SD) increase in tech employment share causes a **10.6% increase** in local restaurant employment over a decade.
* **Job Creation**: For the median Commuting Zone, this translates to approximately **779 new local service jobs**.
* **Housing Constraint**: The multiplier is highly dependent on baseline housing costs. In inelastic markets, rising rents capture a larger share of the income shock, significantly dampening job growth in the service sector.

---

## Repository Structure
* `/data`: Contains instructions for accessing BLS and Zillow datasets (raw data excluded for licensing).
* `/scripts`:
    * `cleaning.py`: Python script for FIPS code string transformation (zfill logic) and CZ aggregation.
    * `estimation.do`: Stata code for the 2SLS Bartik IV models.
* `/results`: Tables and figures, including the Tech Concentration maps.
* `Paper_Draft.pdf`: The full preliminary estimation text.

---

## Citation
If you use this research or code, please cite:
> Hoang Gia Nguyen, Le. (2026). *Trading Tech for Space: Service Exports, Housing Constraints, and Local Employment Spillovers*. Working Paper.
