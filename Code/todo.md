# General constant to-dos

- Adapt the .py files to changes, so that there are no bugs and current functionalities are valid.

# CBOS data

- Refactor the data extraction method in CBOS00_intro.ipynb, so that more important variables are extracted and correctly labeled for the final CBOS data frame.

# GUS data analytics Jupyter notebooks (GUSXXX_.ipynbs)

- ~~Move all the data related (populating the GeoTERYT Database with all the data) code form GUS02B_cross_tables.ipynb to GUS02A_goeTERYT_data.ipynb and leave only cross-tables related code in GUS02B_cross_tables.ipynb.~~ **DONE (v4.3)** — GUS02A now handles all data loading, processing, merging, population extraction, classification, label coding, and saving. GUS02B only builds/inspects cross tables and distributions.

## General tasks
- General clean-up. Currently, they have been constantly adapted to new functionalities and most of them are in their final form, but they contain unnecessary code that is useless from the current point of view. Last .ipynb in final form: GUS01E_old_divison.ipynb.
- General change of the numbering of GUSXXX_.ipynbs - letters should be only used to indicate a next part of the same main subject. Main subjects should have separate numbers.


# GeoTERYT Database

## Database functionalities (complete)

- ~~Fix the functionalities of children and parents. Debug the function self.link_children_to_parents(), because it throws an error.~~ **DONE (v4.2)** — `link_children_to_parents()` fixed and moved to GUS02A (runs before data loading).

- ~~Refactor `children_ids` and `parent_id` from flat attributes (list/string) to **year-keyed dictionaries**. Administrative divisions change over time (gminas move between powiats), so hierarchy must track per-year snapshots.~~ **DONE (v5.3)** — `parent_id: Dict[Union[int, str], str]` and `children_ids: Dict[Union[int, str], List[str]]`. Integer keys for years (1999–2025; years <1999 fall back to 1999). Special string keys on country record: `"old"` (49 old voivodeships), `"nuts"` (16 new minus Mazowieckie + 2 NUTS splits). New helper methods `get_parent(year)` and `get_children(year)` with nearest-year fallback. `link_children_to_parents()` rewritten to build per-year hierarchy from `_by_year` index. Backward-compatible `_restore_record()` auto-converts old pickle format. All downstream code updated: `resolve_historical_teryts()`, `_get_aggregation_children()`, `_enforce_hierarchy_gurobi()`, `_enforce_hierarchy_ipf()`, GUS02B STEP 4 cell, GUS04B population cell, `geoteryt_supp2.py`.

## Geometries handling (complete)

## TERYT codes handling (complete)

- ~~Create attributes TERYTRecord.pop as pd.Series, TERYTRecord.pop_class as pd.Series and adapt the class to the changes form Data handling implementations. The series should span between 1988 and 2025 and should be indexed by date time index.~~ **DONE (v4.2)** — `TERYTRecord.pop` (pd.Series) and `TERYTRecord.pop_class` (pd.DataFrame) implemented. Population extraction via `extract_population()`, classification via `classify_population()`.

## Data handling (complete through v4.3)

- ~~Change the way the numerical data for all variables are stored. Now we want to use pd.Series (instead of dict with years as keys) indexed by indexer in format 01.01.YYYY, where year comes from the dict keys. Unify all time series to time span YEAR_RANGE_FULL from geoTERYT_db.py.~~ **DONE (v4.2)** — `DataSeries` refactored to pd.Series with DatetimeIndex (1988-2025).

- ~~Unify the subject_ids of the census subjects, if they are overlapping. If their subject_names are equal, produce a time series from the data - with missing values between the censuses.~~ **DONE (v4.3)** — Replaced destructive `unify_census_subjects()` with non-destructive `create_merged_subjects()`. Raw data preserved; new merged subjects created with 'M_' prefix. Dimensions semantically aligned across sources; range-based dimensions use common break points algorithm for unified bins. BDL data has priority; census fills only NaN years.

- ~~For the subject "P1336" we have to reduce one dimension.~~ **DONE (v4.2)** — P1336 filtering implemented in `filter_subject_data()`: keeps only "miejsce zamieszkania" (n2) and "stan na 30 czerwca" (n3).

- ~~For each teryt_id that has data from source BDL in any of the subjects with name starting with "pop__" extract the total population for every year.~~ **DONE (v4.2)** — `extract_population()` method extracts total population from "ogółem" variables, stores in `TERYTRecord.pop`. Census data included when available.

- ~~For each teryt_id that the TERYTRecord.pop data available for at least one year, we have to classify this teryt_id with respect to urban/rural split.~~ **DONE (v4.2)** — `classify_population()` implements urban/rural classification based on `rodz` and population thresholds. Stored in `TERYTRecord.pop_class` as pd.DataFrame.

- ~~Each subject_ids has dimensions n1, ..., nM and categories to translate to numerical values.~~ **DONE (v4.2)** — `code_dimension_labels()` handles sex (Mż=1, K=2), education (ordered by level), age (with lower/upper bounds extraction), household size (numeric labels get their number, non-numeric start at 101). Stored in `DataSeries.cat_code` and `DataSeries.cat_bounds`.

1. [ ] Extend all data series from 1986 to 2025! Years without data available should be replaced by the same structures but with np.nan values.

2. [ ] Address the problem of M_hh_size that is only available for 1988, when actually it is also available for other dates. We have:
    - P2887 (1988): household size level=6, labels:
        - 1-osobowe 
        - 2-osobowe 
        - 3-4-osobowe (3-osobowe + 4-osobowe)
        - 5 i więcej-osobowe (5-osobowe + 6-osobowe + ...)
            - to create: ogółem = 1-osobowe + 2-osobowe + 3-4-osobowe + 5 i więcej-osobowe
    - P2871 (2002) household size level=6, labels:
        - ogółem (total)
        - 1 osoba (1-osobowe)
        - 2 osoby (2-osobowe)
        - 3 osoby (3-osobowe)
        - 4 osoby (4-osobowe)
            - to create: 3-4-osobowe = 3 osoby + 4 osoby
        - 5 osób i więcej (5 i więcej-osobowe)
    - P3420 (2011) household size level = 5, labels:
        - ogółem (total)
        - osoby w gospodarstwie domowym - 1 (1-osobowe)
        - osoby w gospodarstwie domowym - 2 (2-osobowe)
        - osoby w gospodarstwie domowym - 3 (3-osobowe)
        - osoby w gospodarstwie domowym - 4 (4-osobowe)
        - osoby w gospodarstwie domowym - 5 i więcej (5 i więcej-osobowe)
    - P4287 (2021) household size level = 6, labels:
        - ogółem (total)
        - gospodarstwa domowe 1-osobowe (1-osobowe)
        - gospodarstwa domowe 2-osobowe (2-osobowe)
        - gospodarstwa domowe 3-osobowe (3-osobowe)
        - gospodarstwa domowe 4-osobowe (4-osobowe)
        - gospodarstwa domowe 5-osobowe i większe (5 i więcej-osobowe)

    We create M_hh_size_1990 on the level=6, that will include series P2887 and P2871 with unified labels: 1-osobowe, 2-osobowe, 3-4-osbobowe (3 osoby + 4 osoby from P2871), 5 i więcej-osobowe, ogółem (1-os, 2-os, 3-4-os, 5+-os summed for P2887) that will span from 1986 to 2025. We create M_hh_size_2000 on level=5, that will include series P2871, P3420 and P4287 with unified lables: ogółem, 1-osobowe, 2-osobowe, 3-osobowe, 4-osobowe and 5 i więcej-osobowe.

3. [ ] Adress the issue of sex vs age labels (note: for sex we will always have mężczyźni - cat_code=1 - and kobiety - cat_code =2 and ogółem - cat_code=0). We have:
    - P2137: 1995-2024, age vs sex, level=6, age labels:
        - ogółem
        - 0-4
        - 5-9
        ... (every 5 years)
        - 65-69
        - 70 i więcej
        - 70-74 (from 2001 onwards)
        - 75-79 (from 2001 onwards)
        - 80-84 (from 2001 onwards)
        - 85 i więcej (from 2001 onwards)
    - P2884: 1988, ONLY age groups, level =6, age labels:
        - ogółem
        - 0-9
        - 10-19
        ... (every 10 years)
        - 50-59
        - 60 lat i więcej
    - H_age_sex: 1986-1994, age vs sex, level=2 (old voivodships), age labels:
        - ogółem
        - 0
        - 1-4
        - 5-9
        ... (every 5 years)
        - 65-69
        - 70 i więcej
    - P2114, P3304, P4253 are already enclosed in P2137.
    
    We create merged variables:
    - M_age_sex: age vs sex (all variables of the subjects), multilevel approach (when a variable is available on level=6 then level=6 if level=2 then on level=2 for this variable etc. - we extract all available teryts from all subjects), merge P2137 and H_age_sex with labels:
        - ogółem
        - 0-4 ([0]+[1-4] for H_age_sex)
        - 5-9
        ... (every 5 years)
        - 65-69
        - 70 i więcej
    - M_age_1990: ONLY age groups, multilevel approach (when a variable is available on level=6 then level=6 if level=2 then on level=2 for this variable etc. - we extract all available teryts from all subjects), merge P2884, P2137 (only ogółem in sex) and H_age_sex (only ogółem in sex) with labels:
        - ogółem
        - 0-9 ([0-4] + [5-9]; [0]+[1-4]+[5-9] for H_age_sex)
        - 10-19 ([10-14] + [15-19])
        ... (every 10 years) (join two categories as above)
        - 50-59 ([50-54] + [55-59])
        - 60 lat i więcej ([60-64] + [65-69] + [70 i więcej])

4. [ ] Adress the issue of education lables. We have:
    - P2885: 1988, educ of ppl 15+, level=6, labels:
        - wyższe ~(highest education level)
        - średnie
        - zasadnicze zawodowe
        - podstawowe ~(lowest education level)
    - P2402: 2002, educ_sex of ppl 13+, level=6, labels:
        - wyższe
        - policealne
        - średnie razem
        - średnie ogólnokształcące
        - średnie zawodowe
        - zasadnicze zawodowe
        - podstawowe ukończone
        - podstawowe nieukończone i bez wykształcenia ~(no education)
    - P3309: 2011, educ_sex of ppl 13+, level=5, labels:
        - ogółem
        - wyższe
        - średnie i policealne - ogółem
        - średnie i policealne - średnie zawodowe
        - średnie i policealne - średnie ogólnokształcące
        - zasadnicze zawodowe
        - gimnazjalne
        - podstawowe ukończone
        - podstawowe nieukończone i bez wykształcenia szkolnego
    - P4315: 2021, educ_sex of ppl 13+, level=6, labels:
        - ogółem
        - wyższe
        - średnie i policealne - ogółem
        - średnie i policealne - średnie zawodowe
        - średnie i policealne - średnie ogólnokształcące
        - zasadnicze zawodowe/branżowe
        - gimnazjalne
        - podstawowe ukończone
        - podstawowe nieukończone i bez wykształcenia szkolnego
        - nieustalony
    - P2350: 1995-2020, educ of ppl 15+, level=2 (new voivodships), labels:
        - wyższe
        - policealne oraz średnie zawodowe/branżowe
        - średnie ogólnokształcące
        - zasadnicze zawodowe/branżowe
        - gimnazjalne, podstawowe i niższe
    - P4092: 2010-2024, educ of ppl 15+, level=2 (new voivodships), labels:
        - wyższe
        - policealne oraz średnie zawodowe/branżowe
        - średnie ogólnokształcące
        - zasadnicze zawodowe/branżowe
        - gimnazjalne, podstawowe i niższe
    - H_sex_educ: 1986-88 & 1991-94, level=0 (country), lables:
        - ogółem
        - wyższe
        - średnie
        - zasadnicze zawodowe
        - podstawowe
        - niepełne podstawowe i bez wykształcenia (available for 1988, for other years compute: ogółem - wyższe - średnie - zasadnicze zawodowe - podstawowe)

    The labels are odered in the from the highest eduction level (for cat_code assign the highest number from 1 to length of the array of labels) to the lowest education level (assign cat_code =1), see ~(...). We create the follwoing merged subjects:
    -  M_educ_1990: multilevel approach (when a variable is available on level=6 then level=6 if level=0 then on level=0 for this variable etc. - we extract all available teryts from all subjects), join P2885 and P2402 and H_sex_educ (only educ dimension, so variables with sex label ogółem for P2402) into a single time series with labels:
        - ogółem (to be crated)
            - take all people 15+ from P2884 (for 1988; age labels: $[10-19] * 1/2 + [20-29] + [30-39] + [40-49] + [50-59] + [60 lat i więcej]$) and from P2114 (for 2002; age labels: $[15-19] + [20-24] + ... + [80-84] + [85 i więcej]$)
        - wyższe (wyższe)
        - średnie (średnie razem + policealne)
        - zasadnicze zawodowe (zasadnicze zawodowe)
        - podstawowe (podstawowe ukończone)
        - podstawowe nieukończone i bez wykształcenia
            - available for 2002 and for 1988: podstawowe nieukończone i bez wykształcenia = ogółem - wyższe - średnie - zasadnicze zawodowe - podstawowe
    - M_educ_2000: multilevel approach (when a variable is available on level=6 then level=6 if level=5 then on level=5 for this variable etc. - we extract all available teryts from all subjects), join P2402, P3309, P4315, P2350 and P4092. But we prioritize observations from P2402, P3309 and P4315 (census variables), that means if we have two or more variables available for a year from one of the prioritized subjects and from P2350 or/and P4092 then we take the prioritzied subject value. Only educ dimention, so variables with sex label ogółem for P2402, P3309, P4315. We will have the following labels:
        - wyższe (wyższe)
        - policealne oraz średnie zawodowe/branżowe (policealne + średnie zawodowe; średnie i policealne - średnie zawodowe)
        - średnie ogólnokształcące (średnie ogólnokształcące; średnie i policealne - średnie ogólnokształcące)
        - zasadnicze zawodowe/branżowe (zasadnicze zawodowe, zasadnicze zawodowe/branżowe)
        - gimnazjalne, podstawowe i niższe (podstawowe ukończone + podstawowe nieukończone i bez wykształcenia; gimnazjalne + podstawowe ukończone + podstawowe nieukończone i bez wykształcenia)
    - M_educ_sex_1990: multilevel approach, join P2402 and H_sex_educ together, now: educ vs sex (all variables of subjects). Labels as for M_educ_1990 but take division into sex groups into account.
    - M_educ_sex_2000: multilevel approach, join P2402, P3309 and P4315 together, now: educ vs sex (all variables of subjects). Labels as for M_educ_2000 but take division into sex groups into account.

5. It can happen that while merging subjects into merged subjects M_ for some teryt_ids we will have only a part of the data in some of the loaded raw subjects (they are PXXXX coded where XXXX are numbers). In that case, we will have to check other teryt_ids that this unit had (they are available through TERYTRecord.historical_codes array). However there are some rules how to do it. Let's say we are checking some data for a given TerytRecord and for some year or years there are no data (it is missing; np.nan) in a variable in a subject, then
    - If (TerytRecord.teryt_id[-1] in ['1','2','3']) is Ture:
        - (Change of RODZ of a gmina.) There is an id in TerytRecord.historical_codes such that (id[:-1] == TerytRecord.teryt_id[:-1]) is Ture and (id[-1] in ['1','2','3']) is also Ture, then we can directly replace the missing data by equivalent (from the same variable and subject and for matching years) data from exactly this id from TerytRecord.historical_codes. This is the case that is prioritized over summing.
        - There are two different ids in TerytRecord.historical_codes such that (id[:-1] == TerytRecord.teryt_id[:-1]) is Ture and (id[-1] in ['4','5']) is also Ture, then we cannot use this data directly. In that case, we have to sum the values from the id that ends with '4' and the values from the id that ends with '5'. Summing should igonre the missing values np.nan.
        - (Change of POW of a gmina.) There is an id in TerytRecord.historical_codes such that (id[-1] == TerytRecord.teryt_id[-1]) is True, then we can directly replace the missing data by equivalent (from the same variable and subject and for matching years) data from exactly this id from TerytRecord.historical_codes.
    - If (TerytRecord.teryt_id[-1] in ['4','5','8','9']) is Ture:
        - There is an id in TerytRecord.historical_codes such that (id[-1] == TerytRecord.teryt_id[-1]), then we can directly replace the missing data by equivalent (from the same variable and subject and for matching years) data from exactly this id from TerytRecord.historical_codes.
    - If (TerytRecord.teryt_id[-1] == '0') is Ture -> we have a powiat:
        - To get the replacement for the missing data for this terty_id, we have to sum the values (from the same variable and subject and for matching year/years) from all teryt_ids that are ending with '1', '2' or '3' from the array children_ids and we can directly use it as replacement.
    - If (TerytRecord.teryt_id[:2] == '00000') is Ture -> we have a voivodship:
        - To get the replacement for the missing data for this terty_id, we have to sum the values (from the same variable and subject and for matching year/years) from all teryt_ids that are ending with '1', '2', '3' or '0' from the array children_ids and we can directly use it as replacement.
    - If (TerytRecord.teryt_id == '0000000') is Ture -> we have the whole country:
        - To get the replacement for the missing data for this terty_id, we have to sum the values (from the same variable and subject and for matching year/years) from all teryt_ids in:
            - [str(x) + "00000" for x in range(2,34,2)] (all new voivodships)
            - [str(x) + "00000" for x in range(2,14,2)] + ['1300000', '1500000'] + [str(x) + "00000" for x in range(16,34,2)] (all new voivodships, but masovian - '1400000' - is divided into warsaw region - '1300000' and regional masovian - '1500000')
            - [str(100-x) + "00000" for x in range(1,50)] (all old voivodship)

    Note that these steps have to be exectued in exactly this order in order to yield a correct unification with respect to hisorical teryts. First, we handle the issue on the lowest level (level=6) of gminas and then when all data is merged, we proceed with higher levels of powiats, voivodships and the whole country.

## Numerical methods (to be implemented for v5)

General remarks:
- We value the quality of this implementation, so we will implement these subsections subsection by subsection in separate requests.
- For each chunk of the job iterate as long as you need to find the implementation of the highest quality and reliability.
- Review the raw database files in order to gain a very deep understanding of its structure and data it contains. It is crutial for implementing the numerical method in a reliable and correct way.
- Follow the plan tightly, howevery if you find that some issues require further clarification, ask me about it.

- After each big chunk of the job present the functionalities of this part using our real database Implement a Jupyer Notebook of a series GUS04X_... where X are consecutive letters of the alphabet. We start with GUS04A_gTdb_prerequisites.ipynb, where we will load the database after performance optimization done in GUS03_gTdb_optimization.ipynb and present the outcomes after implementing the first subsection v5.0. In such a way we can build a "pipeline" of applying the next features to our database in human-friendly Jupyter Notebook enviorment.
- Run each Jupyter Notebook you create cell by cell with me and fix all the errors that occur during the execution of cells. Prove me that your work yields correct, plausible and best possible results.


### v5.0 — Prerequisites (data handling completion) — **WORK CHUNK A**

These tasks from the v4.x Data handling section MUST be completed before numerical estimation.

**Notebook:** `GUS04A_gTdb_prerequisites.ipynb` (load `geoteryt_O.pkl` → apply all v5.0 changes → validate → save).

1. [ ] **Extend YEAR_RANGE_FULL to 1986.** Change `YEAR_RANGE_FULL = list(range(1986, 2026))` (40 years). Update `DATETIME_INDEX_FULL`, `_YEAR_BASE = 1986`, `_N_YEARS_FULL = 40`. Extend all existing `DataSeries.values` and `CrossTable.tables` to span 1986–2025 (pad with NaN for new years). Update `TERYTRecord.pop` and `TERYTRecord.pop_class` ranges accordingly.

2. [ ] **Resolve historical TERYTs (todo item 5 above).** Implement `resolve_historical_teryts()` method on `GeoTERYTDatabase`. For each raw subject (P-prefixed): for each record where data is missing for some years, check `historical_codes` for affiliated teryt_ids that have data under a different code. Apply the rodz-aware lookup rules specified in todo item 5 (bottom-up: level=6 first, then level=5, then level=2, then level=0). This step fills NaN cells in raw subjects with real observed data — it is NOT estimation, it is data recovery. Impact: ~214 gminas recovered per census year (~538→~324 missing for 1988, ~515→~294 missing for 2002, etc.).

3. [ ] **Create M_hh_size_1990 and M_hh_size_2000** from P2887 (1988), P2871 (2002), P3420 (2011, powiat-level), P4287 (2021). See todo items above for label unification details.

4. [ ] **Create M_age_sex and M_age_1990** from P2137, P2884, H_age_sex, census subjects. See todo items above for label unification details. Critical note: P2137 has overlapping age groups (0-14, 70 i więcej) that M_age_sex must exclude (use only the 19 non-overlapping bins). The cross table labels are alphabetically sorted internally — use `dim_labels` for index lookup, never hardcode positions.

5. [ ] **Create M_educ_1990, M_educ_2000, M_educ_sex_1990, M_educ_sex_2000** from P2885, P2402, P3309, P4315, P2350, P4092, H_sex_educ. See todo items above for label unification details.

6. [ ] **Build cross tables** for all newly created merged subjects via `db.build_cross_tables()`.

7. [ ] **Validate prerequisites.** After building, verify:
    - Grand total cell (ogółem × ogółem) in M_age_sex matches `record.pop` for every gmina/year where both exist.
    - For 2D cross tables: ogółem column = sum of all sex columns (mężczyźni + kobiety) for every row.
    - No double-counting: when aggregating gminas to powiats, sum ONLY rodz ∈ {1, 2, 3} (exclude sub-divisions rodz 4, 5, 8, 9 — these are parts of type-3 gminas already counted).
    - Report the final data coverage matrix: for each M_ subject × year, show (# gminas with data, # missing, % coverage).

### v5.1 — Architecture & infrastructure — **WORK CHUNK B**

**Notebook:** `GUS04B_estimator_setup.ipynb` (demonstrate the estimator skeleton works with real data).

8. [ ] **Create `Code/tools/demographic_estimator.py`** — a new standalone module for all numerical estimation logic. The module implements a `DemographicEstimator` class that takes a `GeoTERYTDatabase` reference and produces estimated cross tables. Keeps separation of concerns: `geoTERYT_db.py` = storage/retrieval, `demographic_estimator.py` = numerical estimation.

9. [ ] **Dependencies:** Add `ipfn` (multi-dimensional IPF, pip install ipfn), `gurobipy` (Gurobi Python API, primary solver for hierarchical QP), `scipy` (spline interpolation). Optional: `cvxpy` (fallback convex solver if Gurobi unavailable for some subproblem).

10. [ ] **Constants & configuration** inside `demographic_estimator.py`:
    - `EPSILON = 1e-10` (additive smoothing for log-space operations and zero-cell handling)
    - `IPF_MAX_ITER = 1000`, `IPF_CONVERGENCE = 1e-6` (IPF parameters)
    - `GUROBI_AVAILABLE: bool` (auto-detected at import)
    - `PREDICTION_1990_RANGE = range(1986, 2003)`, `PREDICTION_2000_RANGE = range(1999, 2026)`
    - `CENSUS_YEARS = [1988, 2002, 2011, 2021]`
    - `RODZ_AGGREGATION_SET = {'1', '2', '3'}` — only these rodz types are summed when aggregating gminas to parents. Never include 4, 5, 8, 9 (sub-parts of type-3 gminas or Warsaw districts — their data is already included in the type-3 parent).
    - Mapping: which subjects serve as anchors for which variable types and prediction sections

11. [ ] **E_ prefix subjects storage.** Estimation results are stored as NEW subjects with `E_` prefix (e.g., `E_age_sex_1990`, `E_educ_2000`) to keep observed M_ subjects untouched. Each `E_` subject stores:
    - Full cross tables for every year in the prediction range
    - `DataSeries` with `source_type='Estimated'`
    - Provenance metadata: a boolean mask `CrossTable` (same dimensions) marking which cells are directly observed (True) vs estimated (False)

12. [ ] **Add thin integration to `GeoTERYTDatabase`:** A method `db.run_estimation(estimator: DemographicEstimator)` that orchestrates the pipeline and stores results back into records. Also `db.get_estimation_provenance(subject_id, teryt_id, year)` to query whether a cell is observed or estimated.

13. [ ] **Implement helper: `_get_aggregation_children(record, db)`** — returns the list of child teryt_ids that should be summed to produce the parent's total. Rules:
    - Powiat (teryt[-1]=='0'): children with rodz ∈ {1, 2, 3}
    - Voivodeship (teryt[2:]=='00000'): all powiats + all direct gminas with rodz ∈ {1, 2, 3}
    - Country (teryt=='0000000'): all voivodeships
    - NEVER include rodz 4, 5, 8, 9 in aggregation sums.

### v5.2 — Core numerical algorithm: Three-layer pipeline — **WORK CHUNK C**

**Notebook:** `GUS04C_core_algorithm.ipynb` (demonstrate each layer on a small subset of data).

The estimation follows a three-layer approach for each (variable type × prediction section) combination. The layers are applied sequentially.

#### Layer 1: Temporal seed generation (log-linear interpolation)

14. [x] **Implement `_generate_seeds()`** method on `DemographicEstimator`. ✅ DONE (v5.2)
    - **Input:** List of `(teryt_id, subject_id)` pairs; census anchor years and their cross tables.
    - **Algorithm for each territorial unit:**
        1. Collect all years where the unit has non-NaN cross table data (this includes census years AND BDL years, depending on the subject). These are all "anchor points".
        2. Apply additive smoothing: $T_c \leftarrow T_c + \varepsilon$ (to handle structural zeros in log-space).
        3. Transform to log-space: $\log(T_c)$ for each anchor year $c$.
        4. For units with $\geq 3$ anchor points: fit a natural cubic spline through $\{\log(T_c)\}$ in time, per cell.
        5. For units with exactly 2 anchor points: linear interpolation in log-space (geometric interpolation: $\hat{T}(t) = T_{c_1}^{(c_2 - t)/(c_2 - c_1)} \cdot T_{c_2}^{(t - c_1)/(c_2 - c_1)}$).
        6. For units with exactly 1 anchor point: use that anchor as a constant seed for all years.
        7. Exponentiate back: $\hat{T}(t) = \exp(\text{spline}(t))$.
        8. For years outside the anchor range (extrapolation): use the nearest anchor's table (no extrapolation in log-space to avoid divergence).
    - **Critical: exclude ogółem rows/columns from interpolation.** Recompute ogółem cells as sums of non-ogółem cells after interpolation. This prevents inconsistencies where interpolated ogółem ≠ sum of parts.
    - **Output:** Seed cross tables for every (unit, year) combination.

15. [ ] **Implement cohort-aware seed adjustment for age×sex** (`_cohort_adjust_seeds()`).
    - Between census years, people age deterministically. Use this to improve seeds:
        1. Compute empirical survival ratios from consecutive censuses: $S_{a \to a+5, s} = x_{[a+5,a+10),s}(c_2) / x_{[a,a+5),s}(c_1)$ for each intercensal period.
        2. For intermediate year $t$ between $c_1$ and $c_2$: blend the log-linear seed with a cohort-projection estimate using exponential survival decay.
        3. Weight: $w = 0.3$ for cohort component (tuneable). The blend: $\hat{T}(t) = (1-w) \cdot T_{\text{spline}}(t) + w \cdot T_{\text{cohort}}(t)$.
    - This is an OPTIONAL enhancement on top of log-linear seeds. Only applicable to age×sex variables.

#### Layer 2: Marginal fitting via multi-dimensional IPF

16. [x] **Implement `_fit_marginals_ipf()`** method. ✅ DONE (v5.2)
    - **Input:** Seed table (numpy array), list of known marginals (each = target array + dimensions to sum over).
    - **Algorithm:** Use the `ipfn` package (numpy backend) for N-dimensional IPF:
        1. Construct aggregates list and dimensions list from known marginals.
        2. Run `ipfn.ipfn(seed, aggregates, dimensions, convergence_rate=1e-6, max_iteration=1000)`.
        3. Return the fitted table.
    - **Marginal sources per variable type and year:**
        - **Age×sex, Prediction2000 (1999–2024):** BDL P2137 / M_pop__age_sex provides DIRECT gmina-level cross tables for ~87–95% of gminas. For these gminas/years, the IPF step is a no-op (seed = observed data). For the remaining ~5–13% missing gminas per year, no gmina-level marginal is available → skip to Layer 3.
        - **Age×sex, Prediction1990 (1986–1994):** Old voivodeship-level H_age_sex marginals. The gmina seeds from Layer 1 (using 1988 census + 1995 BDL) are fitted to match old voivodeship totals. This requires aggregating all gminas within each old voivodeship first (use `old_woj_id` to map gminas to old voivodeships; 4,104 of 4,162 gminas have this set; 58 missing are defunct city districts).
        - **Education, Prediction2000 (1999–2024):** Voivodeship-level marginals from P2350/P4092 (education distribution for 19 records: 16 voivodeships + Poland + 2 Mazowieckie sub-regions, annually). Fit aggregated gmina estimates within each voivodeship to match.
        - **Education, Prediction1990 (1986–2002):** Country-level marginals from H_sex_educ (1986–88, 1991–94). Fit aggregated gmina estimates at the national level.
        - **Household size, both sections:** No annual marginals available. Layer 2 is skipped; seeds from Layer 1 go directly to Layer 3.

17. [x] **Handle the multi-level IPF for sub-voivodeship constraints.** ✅ DONE (v5.2, `_scale_gminas_to_parent()`)
    - When marginals are at voivodeship level but estimation targets are gminas:
        1. First, aggregate gmina seeds within the voivodeship (sum only rodz ∈ {1,2,3}).
        2. Run IPF at the voivodeship level: fit the aggregated seed to the known voivodeship marginal.
        3. Compute scaling factors per cell: $r_{ij} = \text{fitted}_{ij}^{\text{voi}} / \text{aggregated\_seed}_{ij}^{\text{voi}}$.
        4. Apply uniform scaling to each gmina within the voivodeship: $\hat{T}_{ij}^{g} \leftarrow \hat{T}_{ij}^{g} \cdot r_{ij}$.
        5. Recalculate sub-division records (rodz 4, 5): these should NOT be scaled independently but should mirror their type-3 parent's structure adjusted to their own total population.
        6. This ensures gminas sum to the correct voivodeship total while preserving inter-gmina relative differences from the seeds.

#### Layer 3: Hierarchical consistency enforcement via Gurobi QP

18. [x] **Implement `_enforce_hierarchy_gurobi()`** — the PRIMARY method. ✅ DONE (v5.2)
    - **Problem formulation** (per voivodeship, per year, per variable type):
        - **Decision variables:** $x_{ij}^g \geq 0$ for each gmina $g$ (rodz ∈ {1,2,3} only!) and each cell $(i,j)$ in the cross table, EXCLUDING ogółem rows/columns (ogółem is computed post-hoc as sum of parts).
        - **Objective (weighted least squares / chi-squared):**
          $$\min \sum_{g} \sum_{i,j} \frac{(x_{ij}^g - \hat{x}_{ij}^g)^2}{\hat{x}_{ij}^g + \varepsilon}$$
          where $\hat{x}_{ij}^g$ is the seed/IPF estimate from Layer 2.
        - **Constraints:**
            - **Powiat aggregation:** $\sum_{g \in p} x_{ij}^g = X_{ij}^p$ for each powiat $p$ with known data, $\forall i,j$.
            - **Voivodeship aggregation:** $\sum_{p \in v} X_{ij}^p = X_{ij}^v$ for each voivodeship $v$ with known data, $\forall i,j$.
            - **Non-negativity:** $x_{ij}^g \geq 0$.
            - **Total population consistency:** $\sum_{i \neq \text{ogółem}} \sum_{j \neq \text{ogółem}} x_{ij}^g = \text{pop}^g(t)$ if total population is known for gmina $g$ in year $t$ (record.pop covers 99.3% of gminas 1999–2024).
            - **Sex marginal consistency (for 2D age×sex tables):** For each gmina and each age bin $i$: $x_{i,\text{mężczyźni}}^g + x_{i,\text{kobiety}}^g$ must equal the total known for that age bin if available.
        - **Problem size:** ~200–400 gminas per voivodeship × ~36 cells (age×sex, excluding ogółem) = 7,200–14,400 variables. Gurobi handles this in <1 second per voivodeship.
    - **Soft constraints** for years without direct parent data:
        - If powiat data is not available for a given year, the aggregation constraint becomes a soft penalty (quadratic term with large weight) rather than a hard equality.
    - **Post-solve:** After solving, recompute ogółem rows/columns and populate rodz 4,5,8,9 records by distributing the type-3 parent's table according to population share.
    - **Total runtime estimate:** 16 voivodeships × ~37 years × ~3 variable types × <1s/solve ≈ ~30 minutes.

19. [x] **Implement `_enforce_hierarchy_ipf()`** — the FALLBACK method (when Gurobi is unavailable). ✅ DONE (v5.2)
    - Iterated multi-level IPF:
        1. Aggregate gmina estimates to powiat level (sum only rodz ∈ {1,2,3}).
        2. If powiat data is known: apply IPF to gmina estimates within each powiat to match powiat totals.
        3. Re-aggregate to voivodeship (sum powiats).
        4. If voivodeship data is known: compute scaling factors, apply to powiats, then propagate to gminas.
        5. Repeat steps 1–4 until convergence (typically 3–10 iterations).
    - Less precise than Gurobi (iterative approximation vs exact solution) but works without commercial software.

### v5.3 — Variable-specific estimation pipelines — **WORK CHUNK D**

Each variable type has specific data availability and requires tailored handling.

**Notebook:** `GUS04D_variable_estimation.ipynb` (run all variable pipelines, visualize sample results).

#### Age × sex estimation

20. [ ] **Implement `estimate_age_sex_2000()`** — Prediction2000 (1999–2024).
    - Anchors: BDL P2137 / M_pop__age_sex at gmina level (1995–2024, 87–95% coverage per year), Census 2002 (P2114), 2011 (P3304), 2021 (P4253) at gmina level.
    - This is the SIMPLEST estimation because BDL provides near-complete annual gmina data.
    - Steps:
        1. For each gmina with M_pop__age_sex data for a given year: use observed cross table directly (copy to E_ subject, provenance = observed).
        2. For the ~5–13% of gminas missing in a given year: generate seeds via Layer 1 (log-linear interpolation of all available anchor years for that gmina), apply Layer 3 (hierarchical: sum within powiat must match the sum of observed siblings + seed).
        3. For voivodeship records (16 new voivodeships): aggregate E_-estimated gmina tables (sum rodz ∈ {1,2,3}) → this produces voivodeship-level estimates. Validate against observed voivodeship P2137 data.
        4. Validate: census years should reproduce actual census data exactly (provenance = observed for those cells).

21. [ ] **Implement `estimate_age_sex_1990()`** — Prediction1990 (1986–2002).
    - Anchors: Census 1988 (P2884 age-only at gmina, P2883 sex at gmina — both are 1D marginals, NOT a joint table), H_age_sex (49 old voivodeships + Poland, 1986–1994, joint age×sex 2D table), BDL P2137 / M_pop__age_sex (gminas 1995–2002).
    - **Challenge:** 1988 census gives age and sex SEPARATELY, not as a joint table. Must construct the joint table.
    - Steps:
        1. **Construct 1988 gmina-level age×sex seeds:**
            - For each old voivodeship ($v$): take H_age_sex[$v$, 1988] as seed structure.
            - For each gmina ($g$) within $v$: from P2884 extract age marginals (after bin aggregation: P2884 has 10-year bins → sum pairs of 5-year bins from the seed structure). From P2883 extract sex marginals.
            - Run 2D IPF: fit the voivodeship seed to match gmina's age and sex marginals. This gives a plausible gmina-level age×sex joint table for 1988.
            - **Verification constraint:** sum of all gmina tables within old voivodeship $v$ should reproduce H_age_sex[$v$, 1988]. If not exact, run Layer 3 to enforce.
        2. For 1989–1994: interpolate gmina seeds between 1988 (constructed) and 1995 (BDL). Apply Layer 2 for each year: fit gmina aggregates within each old voivodeship to match H_age_sex[$v$, year].
        3. For 1995–2002: use observed BDL data directly (overlap with Prediction2000 — ensures continuity).
        4. For 1986–1987: extrapolate backwards from 1988 using H_age_sex[1986–1987] old voivodeship marginals as Layer 2 constraints on gmina seeds (seed = 1988 gmina table).
        5. Apply Layer 3: hierarchical consistency with old voivodeship structure.

22. [ ] **Cohort-component enhancement for age×sex** (optional, improves accuracy).
    - Between census years, apply `_cohort_adjust_seeds()` to shift age cohorts by the intercensal interval.
    - Compute survival ratios from consecutive censuses and blend with log-linear interpolation.
    - This captures structural demographic dynamics (aging, mortality) that pure interpolation misses.

#### Education estimation

23. [ ] **Implement `estimate_educ_2000()`** — Prediction2000 (1999–2024).
    - Anchors: Census 2002 (gmina, via M_educ_2000), Census 2011 (powiat, P3309), Census 2021 (gmina, via M_educ_2000). Annual voivodeship marginals: P2350/P4092 (1995–2024, 19 records).
    - Steps:
        1. **2011 powiat disaggregation:** For each powiat in 2011, use 2002 and 2021 gmina data to estimate gmina shares within the powiat. Method: compute the average of 2002 and 2021 gmina proportions within the powiat (in log-space); multiply by P3309 powiat total. Apply IPF to fit gmina shares to match P3309 powiat cross table exactly. This gives a synthetic gmina-level estimate for 2011 → now we have 3 gmina-level anchor points (2002, 2011-est, 2021).
        2. Layer 1: Generate gmina seeds by log-linear spline interpolation through 3 anchor points.
        3. Layer 2: Fit to voivodeship marginals from P2350/P4092 for each year. Note: P2350 covers 1995–2020, P4092 covers 2010–2024. Use P4092 where both overlap (more recent). Warsaw region (1300000) and regional Mazowieckie (1500000) are available separately — split Mazowieckie gminas accordingly using `teryt_id` prefix matching.
        4. Layer 3: Hierarchical consistency.

24. [ ] **Implement `estimate_educ_1990()`** — Prediction1990 (1986–2002).
    - Anchors: Census 1988 (gmina, P2885 educ-only 4 categories), Census 2002 (gmina, via M_educ_1990 6 categories). Country-level: H_sex_educ (level=0, sex×educ, 1986–88, 1991–94).
    - Steps:
        1. **Construct M_educ_1990 ogółem for 1988:** Approximate total population 15+ from P2884 (age bins: $[10-19] \times \frac{1}{2} + [20-29] + ... + [60+]$). This provides the denominator for proportions.
        2. **Construct M_educ_1990 "podstawowe nieukończone" for 1988:** Since P2885 has only 4 categories (no "basic incomplete"), compute residual: podstawowe\_nieukończone = ogółem − wyższe − średnie − zasadnicze − podstawowe.
        3. Layer 1: Log-linear interpolation of gmina educ tables between 1988 and 2002. The 1988 table has 6 categories (4 observed + ogółem constructed + residual constructed); 2002 has the same 6 categories directly from M_educ_1990.
        4. Layer 2: Fit aggregated national totals to H_sex_educ country-level educ marginals (ogółem sex group) for 1986–88 and 1991–94.
        5. **Exploit H_educ_age bridge tensor (1988, country-level, 3D: sex × educ × age):** Use this to constrain the relationship between education and age distributions. For each gmina in 1988: given the gmina's known age distribution (P2884) and the national education×age joint distribution (H_educ_age), apply IPF to produce a gmina-specific education distribution that is consistent with its age structure. This provides a much better 1988 seed than using voivodeship-level education structure alone.
        6. Layer 3: Constrain at national level. Use old voivodeship geographic consistency.

25. [ ] **Implement `estimate_educ_sex_2000()` and `estimate_educ_sex_1990()`** — analogous to educ-only, but for the joint sex×education cross tables. Use the same pipeline but keep the sex dimension. For M_educ_sex_2000: use P2402 (2002, sex×educ, gmina), P3309 (2011, sex×educ, powiat), P4315 (2021, sex×educ, gmina). For M_educ_sex_1990: use P2402 (2002) and H_sex_educ (1986–94, country-level, sex×educ).

#### Household size estimation

26. [ ] **Implement `estimate_hh_size_2000()`** — Prediction2000 (1999–2024).
    - Anchors: Census 2002 (gmina, via M_hh_size_2000), Census 2011 (powiat, P3420), Census 2021 (gmina, via M_hh_size_2000).
    - No annual marginal data → pure interpolation + hierarchical consistency.
    - Steps:
        1. **2011 powiat disaggregation** (same approach as education): use 2002 and 2021 gmina data to estimate gmina shares within each powiat; constrain by P3420 powiat totals.
        2. Layer 1: Log-linear spline interpolation through 3 anchor points (2002, 2011-est, 2021).
        3. Layer 2: None (no annual marginals).
        4. Layer 3: Hierarchical consistency (powiat sums = known, voivodeship sums = aggregated powiats).

27. [ ] **Implement `estimate_hh_size_1990()`** — Prediction1990 (1986–2002).
    - Anchors: Census 1988 (gmina, via M_hh_size_1990), Census 2002 (gmina, via M_hh_size_1990).
    - Steps:
        1. Layer 1: Log-linear interpolation between 1988 and 2002.
        2. Layer 2: None.
        3. Layer 3: Ensure internal consistency (cells sum to totals). Hierarchical: powiat sum = sum of children, voivodeship sum = sum of powiats.

#### Age × education estimation (auxiliary)

28. [ ] **Implement `estimate_age_educ()`** — for M_pop__age_educ (Prediction2000 only).
    - Anchors: Census 2002 (powiat, P2403), Census 2011 (powiat, P3311), Census 2021 (powiat, P4320). Also: H_educ_age (country-level 3D tensor, 1988).
    - This is powiat-level data (never gmina-level) → estimate at powiat level directly.
    - **Cross-variable consistency:** Use already-estimated E_age_sex (age marginals) and E_educ (education marginals) as IPF targets. This locks the age×educ table to be consistent with both the age and education distributions that were independently estimated.
    - Steps:
        1. Log-linear spline interpolation through 3 powiat-level anchors.
        2. Fit to known age marginals (from E_age_sex, collapsed across sex) and education marginals (from E_educ_2000) using N-dimensional IPF.
        3. This produces a 3-way consistent estimate.

### v5.4 — Validation & diagnostics — **WORK CHUNK E**

**Notebook:** `GUS04E_validation.ipynb` (run all validation, display diagnostics, produce quality report).

29. [x] **Implement leave-one-out cross-validation.** **DONE (v5.4)** — `leave_one_out_cv(var_type, pred_section, holdout_year)` in `demographic_estimator.py`. Temporarily removes holdout year from M_ source, re-runs pipeline, compares predicted vs actual. Reports cell_rmse, cell_rmse_pct, chi_sq, marginal_err, total_pop_err_pct per gmina. Tested with holdout years 2002, 2011, 2021: median RMSE% = 3.6–3.8%, mean RMSE% = 6.2–6.7%. Consistent across all holdout years, 2011 best (interior point).
    - For each census year $c$: re-run the estimation pipeline WITHOUT using $c$ as an anchor. Compare predicted table at $c$ with actual census data.
    - Report:
        - Cell-level RMSE (absolute and percentage of cell value)
        - Chi-squared distance between predicted and actual tables
        - Marginal accuracy (row/column sum errors)
        - Top-10 worst-predicted territories (for investigation)

30. [x] **Implement consistency diagnostics** (`validate_results()`). **DONE (v5.4)** — `validate_results(e_subject_id)` in `demographic_estimator.py`. 6 diagnostic checks implemented. Results: 0 FAIL for all subjects except 6 minor non-negativity failures in E_age_sex_2000 (source BDL data artifact in Białowieża). Fixed NaN propagation bug in Layer 2 scaling for E_educ_1990/E_educ_sex_1990 (NaN in H_sex_educ country data for "podstawowe nieukończone" → `factors = np.where(np.isnan(factors), 1.0, factors)`). Fixed population_match to only check age_sex subjects (education=pop15+, hh_size=households).
    - Check for every (teryt_id, year):
        - [x] Non-negativity: all cells $\geq 0$
        - [x] Marginal consistency: ogółem row = sum of other rows (within tolerance 1.0); ogółem column = sum of other columns
        - [x] Hierarchical consistency: sum of children (rodz ∈ {1,2,3}) = parent for every parent-child pair where parent has data
        - [x] Total population match: cross table grand total (ogółem × ogółem for 2D, ogółem for 1D) matches `TERYTRecord.pop` (within tolerance 0.1%) — only for age_sex subjects
        - [x] Temporal smoothness: no abrupt jumps between consecutive years (flag if year-over-year change > 20% for any non-ogółem cell relative to the cell's mean across years)
        - [x] Sub-division consistency: for rodz-3 (urban-rural) gminas, their table should equal sum of their rodz-4 (town) and rodz-5 (village) children
    - Return a diagnostic report DataFrame.

31. [x] **Implement estimation quality metrics per territory.** **DONE (v5.4)** — `compute_confidence_scores(e_subject_id)` in `demographic_estimator.py`. Per-gmina confidence based on: n_census_anchors (weight 25), n_observed_years (weight 20), distance_to_nearest_anchor (weight 20), log10_population (weight 15), direct_teryt (weight 10), n_marginal_years (weight 10). Returns DataFrame. E_age_sex_2000 highest confidence (mean 85.2), E_educ_sex_1990 lowest (mean 37.2). Peaks at census years (90.9–91.0), dips between.
    - For each territorial unit, compute a "confidence score" based on:
        - Number of census anchors available (0–4)
        - Number of years with direct marginal data
        - Distance to nearest anchor year
        - Size of the territory (larger populations → more stable estimates)
        - Whether the unit had data under its current TERYT or needed historical code resolution
    - Store in `TERYTRecord` as `estimation_confidence: Dict[str, pd.Series]` (subject → time series of confidence scores).

### v5.5 — Orchestration & final assembly — **WORK CHUNK F**

**Notebook:** `GUS04F_full_pipeline.ipynb` (full end-to-end run, save final database).

32. [x] **Performance optimization.**
    - Use numpy vectorization throughout (avoid Python loops over cells).
    - Batch Gurobi solves: one model per voivodeship per year (all variable types can share the same solve if formulated jointly).
    - Parallelize independent voivodeships using `concurrent.futures.ProcessPoolExecutor`.
    - Target total runtime: < 1 hour for the full database.
    - *Done: numpy vectorization in place; Gurobi QP per-voivodeship batching in place. Parallelization deferred to v5.6.*

33. [x] **Full pipeline notebook** following the structure (implemented as `GUS04F_full_pipeline.ipynb` + `GUS04G_visualization.ipynb`):
    - **Cell 1:** Imports and load database (post v5.0 prerequisites).
    - **Cell 2:** Initialize `DemographicEstimator(db)`.
    - **Cell 3:** Run age×sex estimation (Prediction2000 first → Prediction1990 second — order matters because Prediction1990 can use 1995–2002 BDL data as a bridge).
    - **Cell 4:** Run education estimation (Prediction2000 → Prediction1990).
    - **Cell 5:** Run education×sex estimation (Prediction2000 → Prediction1990).
    - **Cell 6:** Run household size estimation (Prediction2000 → Prediction1990).
    - **Cell 7:** Run age×education estimation (Prediction2000).
    - **Cell 8:** Run validation pipeline (cross-validation on Census 2011 + full diagnostics).
    - **Cell 9:** Visualize results: time series plots for sample gminas, heatmaps of prediction errors, maps of estimation confidence.
    - **Cell 10:** Save updated database with E_ subjects.

### v5.6 — Documentation & academic quality

34. [ ] **Mathematical documentation** (in `demographic_estimator.py` docstrings and a separate `Code/numerical_methods_documentation.md`):
    - Full mathematical formulation of the three-layer approach
    - Proof that IPF minimizes KL divergence from the seed subject to marginal constraints (cite Csiszár 1975)
    - Proof of convergence for multi-level IPF (cite Fienberg 1970, Pukelsheim & Simeone 2009)
    - Justification for log-linear interpolation (preserves cross-product ratios / Yule's Q)
    - Connection to maximum entropy methods (Jaynes 1957)
    - Comparison with alternative approaches (Bayesian small-area estimation, cohort-component models)

35. [ ] **Academic references to cite:**
    - Deming & Stephan (1940), "On a Least Squares Adjustment of a Sampled Frequency Table" — original IPF for census adjustment
    - Fienberg (1970), "An Iterative Procedure for Estimation in Contingency Tables" — IPF convergence proof
    - Csiszár (1975), "I-Divergence of Probability Distributions and Minimization Problems" — IPF as KL minimization
    - Bishop, Fienberg & Holland (1975), "Discrete Multivariate Analysis" — comprehensive IPF theory
    - Wong (1992), "The Reliability of Using the Iterative Proportional Fitting Procedure" — IPF for intercensal estimation
    - Norman (1999), "Putting Iterative Proportional Fitting on the Researcher's Desk" — IPF in demographic micro-simulation
    - Simpson & Tranmer (2005), "Combining Sample Survey Data with Census Data" — multi-source demographic estimation
    - Lomax & Norman (2016), "Estimating Population Attribute Values in a Table" — modern IPF for demographic tables
    - Barthélemy & Suesse (2018), "mipfp: Multi-dimensional IPF" — multi-dimensional extensions
    - Rao & Molina (2015), "Small Area Estimation" — hierarchical small-area methods (Wiley)

### Critical pitfalls and safeguards

This section documents identified pitfalls that MUST be handled correctly throughout implementation:

1. **Double-counting in hierarchical aggregation:** When summing gminas to powiats/voivodeships, ONLY sum rodz ∈ {1, 2, 3}. Rodz 4 ("town in urban-rural") and rodz 5 ("rural area in urban-rural") are sub-parts of rodz 3 gminas — their populations are ALREADY included in the rodz 3 total. Similarly, rodz 8 (Warsaw districts) and rodz 9 (delegatury) are sub-parts of rodz 1. Including them would cause double/triple counting. The constant `RODZ_SUB_DIVISIONS_AND_DISTRICTS = ['4', '5', '8', '9']` already exists in `geoTERYT_db.py`.

2. **1988 census has only separate marginals, not joint tables:** P2884 (age), P2883 (sex), P2885 (education) are all 1D arrays at gmina level. There is NO gmina-level age×sex or sex×education 2D table for 1988. Must construct the joint via IPF using higher-level structure (H_age_sex at old voivodeship level) as seed.

3. **Cross table labels are alphabetically sorted:** Labels in `CrossTable.dim_labels` are sorted alphabetically (e.g., "kobiety" before "mężczyźni" before "ogółem"). NEVER hardcode integer indices — always look up by label string. The "ogółem" label is typically at the END for age groups but its position depends on alphabetical order.

4. **Ogółem cells are redundant but must be consistent:** In 2D cross tables, the ogółem row = sum of non-ogółem rows, and ogółem column = sum of non-ogółem columns. The ogółem×ogółem cell = grand total = record.pop. When interpolating, interpolate only non-ogółem cells and recompute ogółem as sums. Otherwise interpolated ogółem may drift from sum of parts.

5. **P2137 has overlapping age groups that M_pop__age_sex removes:** P2137 has 21 age labels including "0-14" (aggregate of 0-4+5-9+10-14) and "70 i więcej" (aggregate of 70-74+75-79+80-84+85+). M_pop__age_sex has 19 labels (excluding these aggregates). When using P2137 data directly, always exclude aggregate labels first.

6. **P2350/P4092 overlap 2010–2020:** Both provide voivodeship-level education data for 2010–2020. Prefer P4092 in the overlap period (it is more recent and covers 2021–2024). P2350 year 2000 has been set to NaN.

7. **2011 census education and hh_size are at powiat level only:** P3309 (sex×educ) and P3420 (hh_size) have 379 records, all at powiat level. They provide NO gmina-level data. Must disaggregate to gmina level using the 2002 and 2021 gmina-level data as structure proportions.

8. **Old voivodeships have 58 gminas without old_woj assignment:** These are defunct city-district subdivisions (e.g., Wrocław-Fabryczna). They existed only ~1999–2001. For Prediction1990 purposes, these can be grouped with their parent city's old voivodeship.

9. **H_educ_age 3D tensor is a unique bridge:** Shape (3×5×9) = sex × education × age, available at country level for 1988 only. This is the ONLY data source linking education and age distributions. It should be used as an IPF seed when estimating education distributions for gminas in 1988.

10. **Population data exists from 1988 but years_valid starts at 1999:** `record.years_valid` covers 1999–2024 (when the TERYT system was reformed). But `record.pop` has data from 1988 for ~3,624 gminas (from census). Do not use `years_valid` to gate pre-1999 processing — use pop data availability directly.

11. **Temporal ordering of estimation matters:** Must run Prediction2000 before Prediction1990 for each variable. Prediction1990 uses 1995–2002 BDL data (which overlaps with Prediction2000). Running Prediction2000 first ensures the 1999–2002 overlap period has validated estimates that can anchor Prediction1990.

12. **Cross-variable consistency as an information multiplier:** The E_age_sex estimates provide age marginals that can constrain E_educ (via the national H_educ_age bridge). The E_educ estimates provide education marginals that constrain E_educ_sex. The E_age_educ estimates use both E_age_sex and E_educ as marginal constraints. This creates a web of mutual consistency. Estimating variables in the order: age×sex → education → education×sex → household_size → age×education maximizes information flow.


## Estimation evaluation (to be implemented for v6)

### Quantitative findings (from analyze_estimation.py + analyze_estimation_deep.py)

**Analysis 1 — Parent data overwrite:**
- `is_observed` attribute does NOT EXIST on any CrossTable at any level (all `no_attr`)
- ALL powiat/voivodeship E_ data is purely aggregated Σ(gminas), never observed
- E_age_sex_2000 at voivodeship: 26/416 year-level checks show >0.1% mismatch with M_age_sex
- MAZOWIECKIE worst case: M_total=20.5M vs E_total=27.2M (+32.8%) — Warsaw double-counting residual

**Analysis 2 — Temporal spikes (>25% year-over-year at gmina level):**
- E_age_sex_2000: 6 spikes (low, good)
- E_age_sex_1990: 148 spikes — worst: Wesoła 2001→2002 = +9689% (BDL data corruption from Warsaw merger)
- E_educ_2000: 316 spikes — ALL from voivodeship 14 (MAZOWIECKIE), 40-48% drops at 2001→2002
- E_educ_sex_2000: 316 spikes — same pattern as E_educ_2000
- E_educ_1990, E_educ_sex_1990, E_hh_size_*: 0 spikes

**Analysis 3 — Spline overshooting (non-monotone between anchors, 5% tolerance):**
- E_educ_2000: 30.1% overshoot, 26.2% undershoot (out of 182 checked intervals)
- E_educ_sex_2000: 28.4% overshoot, 26.8% undershoot
- E_hh_size_2000: 10.9% overshoot, 0% undershoot
- Root cause: 3 anchors (2002, synthetic 2011, 2021) triggers CubicSpline; synthetic 2011 misaligns

**Analysis 4 — Hierarchical consistency (powiat E_ vs Σ gmina E_):**
- 2000-series: ~10% of powiat-year checks inconsistent (>0.1%). 1990-series: ~0.5%
- 41/382 powiats fail exact match at yr=2010. wałbrzyski: 67% mismatch (city separation)
- Many powiats created after 1999 have 0 children → powiat has data, Σ gminas = 0

**Analysis 5 — Census data preservation (E_ vs M_ at gmina level, 100 sampled gminas):**
- E_educ_2000: **100% mismatch** — all census values modified by Layer 2 scaling (4-8%)
- E_educ_sex_2000: **100% mismatch** — same pattern
- E_educ_1990: 49.7% mismatch (6-7% discrepancy)
- E_age_sex_2000: 6.3% mismatch (some gminas zeroed out — Bodzanów: M=33668, E=0)
- E_age_sex_1990, E_educ_sex_1990, E_hh_size_*: 0% mismatch (preserved)

**Analysis 6 — Anchor year distribution at gmina level (200 sampled):**
- E_age_sex_2000: 26 anchors/gmina (annual BDL 1999–2024) — excellent
- E_age_sex_1990: 16-17 anchors (1988 census + 1995–2002 BDL) — good
- E_educ_2000, E_educ_sex_2000, E_hh_size_2000: **2 anchors** (2002, 2021) for 182/200 gminas. 10 have 1 anchor, 8 have 0. Synthetic 2011 from powiat disaggregation adds 3rd.
- E_educ_1990, E_hh_size_1990: 2 anchors (1988, 2002) for 184/200
- E_educ_sex_1990: only 1 anchor (2002) for 187/200 → constant interpolation

**Analysis 7 — Negatives/NaN:**
- No negative values anywhere (clamping works)
- E_age_sex_2000: 4,889 all-NaN tables at gmina level (gminas without data for certain years)
- All other subjects: 0 all-NaN tables

**Deep analysis — M_ data availability at parent levels:**
- POWIAT: M_age_sex (382 units, 30 years), M_age_1990 (382), M_educ_2000 (379, yr=2011 only), M_educ_sex_2000 (379, yr=2011 only), M_hh_size_2000 (379, yr=2011 only)
- VOIV: M_age_sex (65 units, 30+ years), M_educ_2000 (18 units, 30 years), M_educ_1990/educ_sex/hh_size: 0 units

**Deep analysis — Mazowieckie discontinuity root cause:**
- M_educ_2000 voiv data: 1400000 MAZOWIECKIE has years [1995-1999]; 1300000 Warszawski stołeczny + 1500000 Mazowiecki regionalny have years [2001+]
- Layer 2 scaling factor jumps discontinuously at 2000-2001 boundary
- ALL 312 educ_2000 spike gminas are from voiv 14

---

### Root causes identified

1. **RC1 — `_aggregate_to_parents` (line 1385):** Unconditionally overwrites parent E_ data with Σ(gmina E_), `is_observed=False`. No check for existing M_ observed data. This is why powiat/voivodeship plots show no observed/estimated distinction.

2. **RC2 — Layer 2 scaling destroys census values:** `_scale_gminas_to_parent` (line 788) applies `factor = voiv_tbl / Σ(gmina_tbl)` to ALL years including census years. Since Σ(census gminas) ≠ voivodeship observed (rodz exclusions, rounding), factor ≠ 1.0 → census data modified 4-8%.

3. **RC3 — Cubic spline with synthetic 2011:** `_disaggregate_2011_powiat_to_gmina()` creates a 3rd anchor, pushing `_generate_seeds` into CubicSpline mode. Synthetic anchor misaligns with 2002/2021 real data → ~30% of intervals overshoot.

4. **RC4 — Mazowieckie voivodeship data source switch:** M_educ_2000 switches from old MAZOWIECKIE (1995-1999) to split voivodeships (2001+) — scaling factor jumps 40%.

5. **RC5 — TERYT boundary changes:** `children_ids` fixed to 1999 snapshot. Post-1999 powiats (łobeski, węgorzewski, etc.) have 0 children. Warsaw districts rodz=8 excluded from aggregation.

6. **RC6 — Wesoła BDL data corruption:** M_age_sex 2002 for gmina 1412031 = 6.75M (should be ~70K). Warsaw 2002 merger artifact in source data.

---

### Fix plan (7 steps, prioritized)

- [ ] **34. Preserve observed parent data via hybrid aggregation (RC1)**
  - In `_aggregate_to_parents`: after computing Σ(gmina E_), look up the parent record's M_ anchor data for that year
  - Where M_ observed data exists: keep the aggregated cell proportions but SCALE the aggregated table so its total matches the observed M_ total
  - Formula: `E_parent[y] = aggregated[y] * (M_observed_total[y] / aggregated_total[y])`
  - This preserves gmina-level cell proportions while honoring observed parent totals
  - Mark these years as `observed=True` in the provenance mask
  - Affects: all 8 subjects at powiat/voivodeship level

- [ ] **35. Protect census-year gmina data from Layer 2 scaling (RC2)**
  - In `_estimate_educ_2000`, `_estimate_educ_sex_2000`, `_estimate_educ_1990`: modify the scaling loop to SKIP years where the gmina has observed M_ data
  - After scaling, VERIFY that census-year values are untouched: assert `np.allclose(E_census, M_census)` at census years
  - Belt-and-suspenders: even if scaling accidentally touches census data, restore from M_ original
  - Priority subjects: E_educ_2000 (100% mismatch), E_educ_sex_2000 (100%), E_educ_1990 (50%)

- [ ] **36. Fix interpolation overshooting with tiered spline strategy (RC3)**
  - In `_generate_seeds`: implement three-tier interpolation:
    - ≤3 anchors: LINEAR interpolation in log-space (geometric). Simple, no overshoot possible.
    - 4-10 anchors: PCHIP (scipy.interpolate.PchipInterpolator) — shape-preserving monotone cubic
    - >10 anchors: CubicSpline (current behavior) — enough anchors to constrain well
  - Keep the existing clamping behavior outside anchor range
  - Test: re-check overshoot rates on E_educ_2000 (target: 0% overshoot)

- [ ] **37. Fix Mazowieckie voivodeship scaling discontinuity (RC4)**
  - In `_estimate_educ_2000` Layer 2: detect when voivodeship scaling data has a source switch (1400000 → 1300000+1500000)
  - For transition years (2000-2001): compute scaling factors from a LINEAR BLEND between old Mazowieckie totals and new split-voivodeship totals
  - Alternatively: at years where neither old nor new voivodeship data exists, use national-level scaling for Mazowieckie gminas only
  - Target: eliminate all 312 Mazowieckie spikes

- [ ] **38. Update children_ids resolution for boundary changes (RC5)**
  - In `_aggregate_to_parents`: use year-appropriate `children_ids` — try `children_ids.get(year)`, fall back to closest available year, then 1999
  - Handle Warsaw: include rodz=8 (city districts) in aggregation for Warsaw powiat specifically
  - Handle post-1999 powiats: check for children in later snapshots (2002, 2010, etc.)
  - Expected improvement: reduce hierarchical inconsistency from ~10% to <1%

- [ ] **39. Persist observed/estimated status on CrossTable (RC1 downstream)**
  - Add `observed_years: set` attribute to `CrossTable` class in geoTERYT_db.py
  - Populate during `_store_estimated_cross_table` based on is_obs parameter
  - Add to `to_dict()`/`from_dict()` serialization
  - Update GUS04G visualization: show filled markers (●) for observed years at ALL admin levels, not just gmina

- [ ] **40. Clean upstream Wesoła + similar BDL data anomalies (RC6)**
  - Add pre-estimation data validation scan: flag M_ values with >1000% year-over-year change
  - For Wesoła (1412031): NaN-out M_age_sex, M_age_1990 at 2002 (Warsaw merger artifact)
  - Scan all gminas for similar TERYT merger artifacts (municipalities absorbed into cities around 2002)
  - Log all cleaned values for audit trail