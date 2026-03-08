# v5 Implementation Instructions for Agents

## Overview

This document provides all context needed to implement the GeoTERYT cross-table estimation pipeline (v5.0–v5.6). Each section below gives precise technical details that agents need to avoid pitfalls and produce correct code on the first attempt.

The master task list is in `Code/todo.md`. The project context is in `Code/context.md`. The core database module is `Code/tools/geoTERYT_db.py`.

---

## 1. Repository Structure & Key Files

| File | Role |
|------|------|
| `Code/tools/geoTERYT_db.py` | Core database module (~5000 lines, v4.3). Classes: `DataSeries`, `CrossTable`, `TERYTRecord`, `GeoTERYTDatabase`. Do NOT modify unless explicitly required (e.g., extending `YEAR_RANGE_FULL`). |
| `Code/tools/demographic_estimator.py` | **TO CREATE** — estimation logic. Keep all numerical methods here. |
| `Code/analysis/GUS04A_gTdb_prerequisites.ipynb` | **TO CREATE** — Work Chunk A notebook. |
| `Code/analysis/GUS04B_estimator_setup.ipynb` | **TO CREATE** — Work Chunk B notebook. |
| `Code/analysis/GUS04C_core_algorithm.ipynb` | **TO CREATE** — Work Chunk C notebook. |
| `Code/analysis/GUS04D_variable_estimation.ipynb` | **TO CREATE** — Work Chunk D notebook. |
| `Code/analysis/GUS04E_validation.ipynb` | **TO CREATE** — Work Chunk E notebook. |
| `Code/analysis/GUS04F_full_pipeline.ipynb` | **TO CREATE** — Work Chunk F notebook. |
| `geoteryt_O.pkl` | Serialized database (1.5GB, v4.3, 4612 records, 27 subjects). Load with `pickle.load()`. |
| `Code/todo.md` | Master task list with all items and specifications. |
| `Code/context.md` | Project background (survey re-weighting for regional income inequality). |

---

## 2. Database Architecture

### Loading the database
```python
import pickle, sys
sys.path.insert(0, 'Code/tools')
from geoTERYT_db import GeoTERYTDatabase, TERYTRecord, DataSeries, CrossTable

with open('geoteryt_O.pkl', 'rb') as f:
    db = pickle.load(f)
```

### Key constants in `geoTERYT_db.py`
```python
YEAR_RANGE_FULL = list(range(1988, 2026))      # TO BE CHANGED TO range(1986, 2026)
DATETIME_INDEX_FULL = pd.DatetimeIndex(...)     # Corresponding DatetimeIndex
_YEAR_BASE = 1988                               # TO BE CHANGED TO 1986
_N_YEARS_FULL = 38                              # TO BE CHANGED TO 40
LEVEL_VOIVODESHIP = 2  # teryt like '02xxxxx'
LEVEL_POWIAT = 5       # teryt like '0201xxx'
LEVEL_GMINA = 6        # teryt like '0201011'
RODZ_SUB_DIVISIONS = ['4', '5']                 # town-part, village-part of urban-rural gminas
RODZ_SUB_DIVISIONS_AND_DISTRICTS = ['4', '5', '8', '9']  # includes Warsaw districts + delegatury
```

### TERYTRecord structure
Each record (`db.records[teryt_id]`) has:
- `.teryt_id: str` — 7-digit code (e.g., '0201011')
- `.name: str` — name of the unit
- `.level: int` — 0 (country), 2 (voivodeship), 5 (powiat), 6 (gmina)
- `.rodz: str` — type: '1'=urban, '2'=rural, '3'=urban-rural, '4'=town-part, '5'=village-part, '8'=district, '9'=delegatura
- `.data: Dict[str, DataSeries]` — subject_id → DataSeries
- `.cross_tables: Dict[str, CrossTable]` — subject_id → CrossTable
- `.pop: pd.Series` — total population (DatetimeIndex, 1988–2025); .pop[year] gives int or NaN
- `.pop_class: pd.Series` — population size class
- `.parent: TERYTRecord` — parent record (powiat for gmina, voivodeship for powiat, etc.)
- `.children: List[TERYTRecord]` — child records
- `.years_valid: range` — years when this TERYT code was active (typically range(1999, 2025)), BUT pop data may exist before 1999
- `.historical_codes: List[Tuple[str, range]]` — list of (old_teryt_id, year_range) for pre-reform affiliations
- `.old_woj_id: str` — 2-digit old voivodeship code (49+1 system), available for 4104 of 4162 level-6 records

### DataSeries structure
```python
ds = record.data['P2137']
ds.values       # pd.Series with DatetimeIndex, float values or NaN
ds.source_type  # 'BDL', 'Census', 'Estimated', etc.
ds.subject_id   # 'P2137'
```

### CrossTable structure
```python
ct = record.cross_tables['P2137']
ct.tables       # Dict[int, np.ndarray] — year → M-dimensional array; value is None for years without data
ct.dim_labels   # List[List[str]] — label names for each dimension, ALPHABETICALLY SORTED
ct.dim_names    # List[str] — dimension names (e.g., ['wiek', 'płeć'])
```

**Example:** P2137 age×sex table for a gmina in 2020:
- Shape: (21, 3) — 21 age labels × 3 sex labels
- `dim_labels = [['0-4', '0-14', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70 i więcej', '70-74', '75-79', '80-84', '85 lat i więcej', '5-9', 'ogółem'], ['kobiety', 'mężczyźni', 'ogółem']]`
- `dim_labels[0]` is ALPHABETICALLY sorted. "ogółem" is at index 20 (last), "0-14" at index 1, "70 i więcej" at index 14.
- M_pop__age_sex has 19 labels (removes "0-14" and "70 i więcej" aggregate rows from P2137's 21).

**CRITICAL:** Label indices are NOT in logical order. Always index by label name:
```python
labels = ct.dim_labels[0]
ogoltem_idx = labels.index('ogółem')
```

---

## 3. Data Inventory

### Subjects by type and level

| Subject ID | Description | Level | Years | Dims | Shape |
|------------|-------------|-------|-------|------|-------|
| **P2137** | Age × sex (BDL) | 6 (gmina) | 1995–2024 | 2D | (21, 3) |
| **P2884** | Age (1988 census) | 6 | 1988 | 1D | (8,) |
| **P2883** | Sex (1988 census) | 6 | 1988 | 1D | (3,) |
| **P2885** | Education (1988 census) | 6 | 1988 | 1D | (4,) |
| **P2114** | Sex × age (2002 census) | 6 | 2002 | 2D | (3, 19) |
| **P2402** | Sex × education (2002 census) | 6 | 2002 | 2D | (3, 8) |
| **P2871** | Household size (2002 census) | 6 | 2002 | 1D | (7,) |
| **P2887** | Household size (1988 census) | 6 | 1988 | 1D | (4,) |
| **P3304** | Sex × age (2011 census) | 6 | 2011 | 2D | (3, 19) |
| **P3309** | Sex × education (2011 census) | 5 (powiat) | 2011 | 2D | (3, 8) |
| **P3420** | Household size (2011 census) | 5 (powiat) | 2011 | 1D | (7,) |
| **P4253** | Sex × age (2021 census) | 6 | 2021 | 2D | (3, 19) |
| **P4315** | Sex × education (2021 census) | 6 | 2021 | 2D | (3, 8) |
| **P4287** | Household size (2021 census) | 6 | 2021 | 1D | (7,) |
| **P2350** | Education by voi (BDL) | 2 (voivodeship) | 1995–2020 | 1D | varies |
| **P4092** | Education by voi (BDL) | 2 (voivodeship) | 2010–2024 | 1D | varies |
| **H_age_sex** | Age × sex (historical) | old voi (49+PL) | 1986–1994 | 2D | varies |
| **H_sex_educ** | Sex × education (historical) | 0 (country) | 1986–88, 91–94 | 2D | varies |
| **H_educ_age** | Sex × educ × age (census) | 0 (country) | 1988 | 3D | (3, 5, 9) |
| **P2403** | Age × education (2002 census) | 5 (powiat) | 2002 | 2D | varies |
| **P3311** | Age × education (2011 census) | 5 (powiat) | 2011 | 2D | varies |
| **P4320** | Age × education (2021 census) | 5 (powiat) | 2021 | 2D | varies |

### Already-created merged subjects (M_ prefix)

| Subject ID | Description | Source subjects |
|------------|-------------|-----------------|
| **M_pop__age_sex** | Unified age × sex | P2137, P2114, P3304, P4253 (19 age bins × 3 sex) |
| **M_pop__educ** | Unified education | P2885 only so far (MISSING P2402!) |
| **M_pop__age_educ** | Age × education | P2403, P3311, P4320 (powiat-level only) |

### M_ subjects TO BE CREATED in v5.0

See `todo.md` items 1–5 for detailed label mappings.

| Subject ID | Variables united | Prediction section |
|------------|-----------------|-------------------|
| M_hh_size_1990 | P2887 + P2871 | Prediction1990 (1986–2002) |
| M_hh_size_2000 | P2871 + P3420 + P4287 | Prediction2000 (1999–2024) |
| M_age_sex (already exists) | P2137 + censuses | Both |
| M_age_1990 | P2884 (1988, 1D age only) | Prediction1990 |
| M_educ_1990 | P2885 + P2402 (ogółem sex dim) | Prediction1990 |
| M_educ_2000 | P2402 + P3309 + P4315 (ogółem sex dim) | Prediction2000 |
| M_educ_sex_1990 | P2885 × H_sex_educ + P2402 | Prediction1990 |
| M_educ_sex_2000 | P2402 + P3309 + P4315 | Prediction2000 |

---

## 4. Critical Pitfalls — MUST READ

### Pitfall 1: Double-counting in hierarchical aggregation
When summing gminas to powiats/voivodeships, sum ONLY `rodz ∈ {'1', '2', '3'}`.
- Rodz '4' (town of urban-rural) and '5' (village of urban-rural) are SUB-PARTS of rodz '3' gminas. Their population is ALREADY included in the rodz-3 parent.
- Rodz '8' (Warsaw district) and '9' (delegatura) are sub-parts of rodz '1'.
- Including them causes 2×–5× overcounting. Test evidence: sum of all gminas in voivodeship 02 = 3,635,418, but voivodeship = 673,592 (5.4× overcount from including all rodz types).

**Rule:** Always filter with:
```python
children = [r for r in parent.children if r.rodz in ('1', '2', '3')]
```

### Pitfall 2: 1988 census data is 1D, not 2D
P2884 (age), P2883 (sex), P2885 (education) are all **separate 1D marginals** at gmina level for 1988. There is NO joint gmina-level age×sex or sex×educ table for 1988. You must construct the joint table via IPF:
- Use H_age_sex[old_voi, 1988] as the seed structure.
- Fit to P2884 (age marginal) and P2883 (sex marginal) using 2D IPF.
- This produces a plausible gmina-level age×sex joint table.

### Pitfall 3: Labels are alphabetically sorted
Cross table `dim_labels` are sorted alphabetically, NOT logically (not by age order). "0-4" comes before "0-14" before "10-14". "ogółem" is at the end. **Never hardcode integer indices.** Always look up by label string.

### Pitfall 4: Ogółem cells are redundant sums
In 2D cross tables: ogółem row = sum of non-ogółem rows; ogółem column = sum of non-ogółem columns; ogółem×ogółem = grand total = record.pop. When interpolating, interpolate ONLY non-ogółem cells and recompute ogółem as sums. Otherwise, interpolated ogółem may drift from sum of parts.

### Pitfall 5: Overlapping age groups in P2137
P2137 has 21 age labels including:
- "0-14" = aggregate of 0-4 + 5-9 + 10-14
- "70 i więcej" = aggregate of 70-74 + 75-79 + 80-84 + 85+

M_pop__age_sex correctly removes these two, leaving 19 non-overlapping bins. When using P2137 directly, always exclude aggregate labels.

### Pitfall 6: P2350/P4092 overlap for 2010–2020
Both provide voivodeship-level education data. Prefer P4092 in the overlap (more recent, extends to 2024). P2350 year 2000 has been set to NaN (data quality issue). Together, P2350+P4092 cover 1995–2024 with a shared range of 2010–2020.

### Pitfall 7: 2011 census education and hh_size are powiat-level only
P3309 (sex×educ) and P3420 (hh_size) have 379 records, ALL at powiat level (level=5). There is NO gmina-level data for 2011 for these variables. Must disaggregate to gmina level using 2002 and 2021 gmina-level data as structural proportions.

### Pitfall 8: 58 gminas lack old_woj_id assignment
These are defunct city-district subdivisions (e.g., Wrocław-Fabryczna). They existed only ~1999–2001. For Prediction1990 purposes, group them with their parent city's old voivodeship (infer from parent record or TERYT code prefix).

### Pitfall 9: H_educ_age is a unique 3D bridge tensor
Shape (3×5×9) = sex × education × age, available at country level for 1988 ONLY. This is the ONLY data source linking education and age distributions at any level. Use it as an IPF seed when estimating education distributions — it constrains the education-by-age relationship.

### Pitfall 10: years_valid starts at 1999 but pop exists from 1988
`record.years_valid` covers 1999–2024 (when the TERYT reform took effect). But `record.pop` has 1988 data for ~3,624 gminas from census. Do NOT use `years_valid` to gate pre-1999 processing — check data availability directly:
```python
has_pop_1988 = not pd.isna(record.pop.get(pd.Timestamp('1988-01-01'), np.nan))
```

### Pitfall 11: Temporal ordering of estimation matters
Run Prediction2000 BEFORE Prediction1990 for each variable. Prediction1990 uses 1995–2002 BDL data from the overlap. Running Prediction2000 first ensures validated estimates exist for 1999–2002.

### Pitfall 12: Cross-variable consistency as information multiplier
Estimate in this order: age×sex → education → education×sex → household_size → age×education. Each earlier variable provides marginals that constrain later ones (e.g., E_age_sex age marginals constrain E_educ via the H_educ_age bridge).

### Pitfall 13: Historical TERYT resolution must happen BEFORE creating M_ subjects
If you create M_ subjects before resolving historical TERYTs, the merged subjects will have NaN for many gminas/years where data actually exists under old TERYT codes. Resolve historical TERYTs first (fills NaN cells in raw P- subjects with real data), then merge into M_ subjects.

---

## 5. Work Chunks

The implementation is divided into 6 work chunks (A–F), each corresponding to a subsection in `todo.md`. They MUST be implemented in order because each chunk depends on the outputs of previous ones.

### Work Chunk A — Prerequisites (`v5.0`, todo items 1–7)
**What:** Data preparation. Extend year range, resolve historical TERYTs, create all M_ subjects, build cross tables, validate.
**Notebook:** `GUS04A_gTdb_prerequisites.ipynb`
**Modifies:** `geoTERYT_db.py` (year range constants), database pickle (adds M_ subjects)
**Output:** Updated `geoteryt_O.pkl` with year range 1986–2025, historical TERYTs resolved, all M_ subjects created with cross tables built.
**Entry criteria:** Current `geoteryt_O.pkl` exists, v4.3.
**Exit criteria:** All M_ subjects exist with correct shapes, cross tables built, grand totals match record.pop, no double-counting in aggregation.
**Dependencies:** None (first chunk).

### Work Chunk B — Architecture (`v5.1`, todo items 8–13)
**What:** Create `demographic_estimator.py` skeleton with class structure, constants, E_ subject storage, helper methods.
**Notebook:** `GUS04B_estimator_setup.ipynb`
**Creates:** `Code/tools/demographic_estimator.py`
**Output:** Working `DemographicEstimator` class that can be instantiated with a `GeoTERYTDatabase`, defines all constants/config, has storage methods for E_ subjects, and the `_get_aggregation_children()` helper passes unit tests.
**Entry criteria:** Chunk A completed, database has all M_ subjects.
**Exit criteria:** `DemographicEstimator(db)` instantiates without error, E_ prefix storage works, aggregation children helper returns correct results for sample gminas/powiats/voivodeships.
**Dependencies:** Chunk A.

### Work Chunk C — Core Algorithm (`v5.2`, todo items 14–19)
**What:** Implement the three-layer numerical pipeline: (1) log-linear seed generation, (2) multi-dimensional IPF, (3) Gurobi QP hierarchical consistency.
**Notebook:** `GUS04C_core_algorithm.ipynb`
**Modifies:** `Code/tools/demographic_estimator.py` (adds core methods)
**Output:** Working `_generate_seeds()`, `_cohort_adjust_seeds()`, `_fit_marginals_ipf()`, `_enforce_hierarchy_gurobi()`, `_enforce_hierarchy_ipf()` methods. Demonstrated on a small subset (1 voivodeship, 3 years, age×sex only).
**Entry criteria:** Chunk B completed, `DemographicEstimator` skeleton exists.
**Exit criteria:** For sample voivodeship: seeds generated match census at anchor years, IPF fits voivodeship marginals within tolerance, Gurobi QP produces hierarchically consistent results, fallback IPF method produces similar results.
**Dependencies:** Chunks A, B.

### Work Chunk D — Variable Pipelines (`v5.3`, todo items 20–28)
**What:** Variable-specific estimation pipelines for age×sex, education, education×sex, household size, age×education. Each uses the core algorithm with variable-specific anchors and marginal sources.
**Notebook:** `GUS04D_variable_estimation.ipynb`
**Modifies:** `Code/tools/demographic_estimator.py` (adds `estimate_*()` methods)
**Output:** All E_ subjects estimated and stored in the database for all variables and both prediction sections.
**Entry criteria:** Chunk C completed, core algorithm works.
**Exit criteria:** E_ subjects exist for all variable types, all census years reproduce actual data (provenance = observed), sample visualizations show smooth temporal evolution.
**Dependencies:** Chunks A, B, C.

### Work Chunk E — Validation (`v5.4`, todo items 29–31)
**What:** Leave-one-out cross-validation, consistency diagnostics, estimation quality metrics.
**Notebook:** `GUS04E_validation.ipynb`
**Modifies:** `Code/tools/demographic_estimator.py` (adds validation methods)
**Output:** Cross-validation RMSE tables, diagnostic report, confidence scores per territory.
**Entry criteria:** Chunk D completed, all E_ subjects exist.
**Exit criteria:** RMSE tables generated, no consistency violations found, confidence scores stored.
**Dependencies:** Chunks A, B, C, D.

### Work Chunk F — Orchestration & Documentation (`v5.5–v5.6`, todo items 32–35)
**What:** Performance optimization, full pipeline notebook, mathematical documentation, academic references.
**Notebook:** `GUS04F_full_pipeline.ipynb`
**Creates:** `Code/numerical_methods_documentation.md`
**Output:** Complete end-to-end pipeline that runs in <1 hour, comprehensive documentation.
**Entry criteria:** Chunk E completed, validation passes.
**Exit criteria:** Full database estimate saved, total runtime <1h, documentation complete.
**Dependencies:** Chunks A, B, C, D, E.

---

## 6. Coding Conventions

### Import patterns
```python
import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from geoTERYT_db import GeoTERYTDatabase, TERYTRecord, DataSeries, CrossTable
```

### Existing code style (match these patterns)
- Type hints used throughout
- Docstrings use triple quotes with parameter descriptions
- Constants are UPPER_SNAKE_CASE
- Private methods prefixed with `_`
- Database operations use `db.records[teryt_id]` for direct access
- Subjects accessed via `record.data[subject_id]` and `record.cross_tables[subject_id]`
- Population accessed via `record.pop[pd.Timestamp(f'{year}-01-01')]`
- Year-to-index: `year - _YEAR_BASE` (after extending, `year - 1986`)

### Cross table access patterns
```python
# Get cross table for a specific year
ct = record.cross_tables['M_pop__age_sex']
table_2020 = ct.tables[2020]  # np.ndarray or None

# Find label index
age_labels = ct.dim_labels[0]
idx_0_4 = age_labels.index('0-4')

# Get non-ogółem cells only
age_labels_no_og = [l for l in ct.dim_labels[0] if l != 'ogółem']
sex_labels_no_og = [l for l in ct.dim_labels[1] if l != 'ogółem']
```

### Aggregation pattern (SAFE)
```python
def aggregate_children(parent_record, subject_id, year, db):
    """Sum children's cross tables, excluding sub-divisions."""
    children = [c for c in parent_record.children if c.rodz in ('1', '2', '3')]
    tables = []
    for child in children:
        ct = child.cross_tables.get(subject_id)
        if ct and ct.tables.get(year) is not None:
            tables.append(ct.tables[year])
    if tables:
        return np.sum(tables, axis=0)
    return None
```

### Year iteration pattern
```python
for year in range(1986, 2026):
    ts = pd.Timestamp(f'{year}-01-01')
    pop_val = record.pop.get(ts, np.nan)
    if pd.notna(pop_val):
        # process
```

---

## 7. Solver-Specific Notes

### Gurobi QP
```python
import gurobipy as gp
from gurobipy import GRB

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)  # suppress output
env.start()

model = gp.Model('hierarchy_qp', env=env)
model.setParam('TimeLimit', 60)  # seconds

# Create variables (non-negative)
x = model.addMVar(n_vars, lb=0.0, name='x')

# Objective: min (x - x_hat)^T @ W @ (x - x_hat) where W = diag(1/(x_hat + eps))
# This is chi-squared / weighted least squares
Q = np.diag(1.0 / (x_hat + EPSILON))
model.setObjective(x @ Q @ x - 2 * (x_hat / (x_hat + EPSILON)) @ x, GRB.MINIMIZE)

# Linear constraints (aggregation)
model.addConstr(A @ x == b, name='aggregation')

model.optimize()
if model.status == GRB.OPTIMAL:
    result = x.X  # numpy array
```

### IPF via ipfn
```python
from ipfn import ipfn

# Fit a 2D seed to known row and column marginals
seed = np.array([[...]])  # 2D
row_marginals = np.array([...])  # 1D, target row sums
col_marginals = np.array([...])  # 1D, target column sums

IPF = ipfn.ipfn(seed, [row_marginals, col_marginals], [[0], [1]],
                convergence_rate=1e-6, max_iteration=1000)
result = IPF.iteration()
```

### scipy spline interpolation
```python
from scipy.interpolate import CubicSpline

years = [1988, 2002, 2011, 2021]
log_values = np.log(cell_values + EPSILON)  # per-cell time series
cs = CubicSpline(years, log_values, bc_type='natural')
interpolated = np.exp(cs(target_year))
```

---

## 8. Testing & Validation Checklist

For every implementation chunk, verify:

1. **Census reproduction:** At census anchor years, estimated values must equal observed values exactly (within floating-point tolerance). Provenance for these cells = True (observed).

2. **Grand total consistency:** For every (teryt_id, year): `E_age_sex[year].sum(non-ogółem cells) == record.pop[year]` (within 0.1%).

3. **Marginal consistency:** For 2D tables: ogółem row = sum of data rows; ogółem column = sum of data columns.

4. **Hierarchical consistency:** For every parent-child pair where both have data:
   ```python
   parent_table == sum(child.tables for child in parent.children if child.rodz in ('1','2','3'))
   ```

5. **Non-negativity:** All cells ≥ 0.

6. **Temporal smoothness:** No year-over-year changes > 20% for any non-ogółem cell (flag only, not a hard constraint).

7. **No double-counting:** Verify aggregation sums match expected voivodeship/national totals.

---

## 9. Quick Reference: Which Data Exists Where

### BDL annual data (near-complete, 1995–2024)
- Age×sex at gmina: P2137 → M_pop__age_sex (87–95% coverage per year)
- Education at voivodeship: P2350 (1995–2020) + P4092 (2010–2024)
- No annual household size data at any level

### Census data (100% coverage at their level)

| Census | Age×sex (gmina) | Education (gmina) | Education (powiat) | HH size (gmina) | HH size (powiat) |
|--------|-----------------|--------------------|--------------------|------------------|-------------------|
| 1988 | P2884 (1D age) + P2883 (1D sex) | P2885 (1D educ) | — | P2887 | — |
| 2002 | P2114 (2D) | P2402 (2D sex×educ) | P2403 (age×educ) | P2871 | — |
| 2011 | P3304 (2D) | — | P3309 (sex×educ), P3311 (age×educ) | — | P3420 |
| 2021 | P4253 (2D) | P4315 (2D sex×educ) | P4320 (age×educ) | P4287 | — |

### Historical data (old voivodeship / national level)
- H_age_sex: 49 old voivodeships + Poland, 1986–1994, 2D age×sex
- H_sex_educ: Poland only, 1986–88 + 1991–94, 2D sex×education
- H_educ_age: Poland only, 1988 only, 3D sex×education×age

### Data gaps requiring estimation
- **Gmina-level age×sex** for 1986–1994: have old-voi marginals (H_age_sex) + 1988 census 1D + 1995 BDL start
- **Gmina-level education** for all non-census years: have voi marginals (P2350/P4092) annually + national (H_sex_educ) 1986–94
- **Gmina-level household size** for all non-census years: NO annual marginals at any level — pure interpolation between censuses
- **Gmina-level 2011 educ and hh_size:** only powiat aggregates exist — disaggregate using 2002+2021

---

## 10. Implementation Priority & Information Flow

```
┌─────────────────────────────────────────────┐
│  Step 0: Historical TERYT resolution        │  ← Fills NaN in raw P- subjects
│  Step 1: Create M_ subjects                 │  ← Merges P- into unified M-
│  Step 2: Build cross tables                 │  ← Creates numpy arrays
├─────────────────────────────────────────────┤
│  Step 3: E_age_sex_2000                     │  ← Uses P2137/M_pop__age_sex (richest data)
│  Step 4: E_age_sex_1990                     │  ← Uses H_age_sex + 1988 census marginals
│  Step 5: E_educ_2000                        │  ← Uses P2350/P4092 voi marginals + E_age_sex
│  Step 6: E_educ_1990                        │  ← Uses H_sex_educ + H_educ_age bridge
│  Step 7: E_educ_sex_2000                    │  ← Constrained by E_educ marginals
│  Step 8: E_educ_sex_1990                    │  ← Constrained by E_educ marginals
│  Step 9: E_hh_size_2000                     │  ← Pure interpolation (no marginals)
│  Step 10: E_hh_size_1990                    │  ← Pure interpolation
│  Step 11: E_age_educ (powiat only)          │  ← Uses E_age_sex + E_educ as marginals
├─────────────────────────────────────────────┤
│  Step 12: Validation & diagnostics          │  ← Cross-validation, consistency checks
│  Step 13: Documentation & optimization      │  ← Final polish
└─────────────────────────────────────────────┘
```

Each step can use results from previous steps as marginal constraints, creating an increasingly constrained and consistent estimate.

---

## 11. Glossary

| Term | Meaning |
|------|---------|
| **BDL** | Bank Danych Lokalnych — Polish statistical office's local data bank |
| **TERYT** | Territorial register of administrative units (7-digit codes) |
| **rodz** | Rodzaj — type of gmina (1=urban, 2=rural, 3=urban-rural, 4=town, 5=village, 8=district, 9=delegatura) |
| **IPF** | Iterative Proportional Fitting — method for adjusting a table to match known marginals |
| **QP** | Quadratic Programming — optimization with quadratic objective and linear constraints |
| **ogółem** | "total" in Polish — aggregate row/column in cross tables |
| **powiat** | County (second-level administrative unit) |
| **gmina** | Municipality (third-level, smallest unit) |
| **voivodeship** | Province (first-level, largest unit below country) |
| **H_** prefix | Historical subjects (old administrative divisions, pre-1999) |
| **P_** prefix | Raw BDL/census subjects |
| **M_** prefix | Merged subjects (unified labels across sources) |
| **E_** prefix | Estimated subjects (numerical estimation output) |
| **Prediction1990** | Estimation for 1986–2002 (anchored by 1988 + 2002 censuses) |
| **Prediction2000** | Estimation for 1999–2024 (anchored by 2002 + 2011 + 2021 censuses + BDL) |
