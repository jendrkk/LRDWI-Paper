# Deep Analysis: Root Causes of Estimation Irregularities

## Executive Summary

After exhaustive analysis of the codebase (geoTERYT_db.py, demographic_estimator.py, pipeline notebooks GUS02B through GUS05, and the estimated database `geoteryt_E.pkl`), I identified **8 confirmed bugs** and **4 secondary issues**. The most impactful is a fundamental data propagation failure: **`load_subject_data()` does not use `code_by_year` to route census data to records that changed TERYT codes**, causing Warsaw and other reformed units to lack critical census anchors.

---

## BUG 1 (CRITICAL): Data Propagation Failure — `load_subject_data()` Ignores `code_by_year`

### Root Cause
`load_subject_data()` (geoTERYT_db.py:3466-3568) places census data on whichever record matches the TERYT code in the CSV. It does NOT consult `code_by_year` to propagate data to the "canonical" record for units that changed codes.

### Evidence
- Record `1431001` (Warsaw urban gmina) has `code_by_year = {1999: '1431001', 2000: '1431001', 2001: '1431001', 2002: '1465011', ...}` and `historical_codes = ['1465011', '1431001']`.
- The 2002 census education data (P2402) uses TERYT code `1465011`. `load_subject_data()` places it on record `1465011` — a **separate record** from `1431001`.
- Record `1431001` has **no P2402 data at all** — meaning it has no 2002 education anchor.
- `resolve_historical_teryts()` (STEP 11) recovers 0 points for P2402 because it skips records with no existing data for a subject (`if not subj_data: continue`).

### Impact
- `M_educ_1990` on record `1431001` has only **one anchor: 1988**.
- `_generate_seeds()` with 1 anchor → **constant seed** (line 613: "1 anchor → constant seed").
- Result: **E_educ_1990 for Warsaw (and its powiat 1431000) is completely flat from 1988 to 2002**. Verified in database:
  ```
  E_educ_1990 on 1431001: years 1989-2002 all have sum=2,690,175.0 (identical to 1988)
  E_educ_sex_1990: flat from 1989-2002
  E_hh_size_1990: flat for ALL years 1986-2001
  ```
- This affects **every gmina that changed TERYT codes between census years**, not just Warsaw.

### Affected Records
- Warsaw: 1431001 (missing P2402/2002, P4315/2021)
- Warsaw powiat: 1431000 (aggregated from flat child)
- Any gmina whose TERYT code changed between 1988 and 2002, or between 2002 and 2021

### Fix
**Option A (Recommended):** Modify `load_subject_data()` to check `code_by_year` and `historical_codes` on ALL records when exact match fails. If the incoming TERYT code matches a value in any record's `code_by_year` or `historical_codes`, place the data on that record.

**Option B:** Create a new post-load method `propagate_data_by_historical_codes()` that iterates over all records, and for each record with `historical_codes`, copies subject data from sibling records that share historical codes.

**Location:** `geoTERYT_db.py`, `load_subject_data()` method (lines 3466-3568)

---

## BUG 2 (CRITICAL): `resolve_historical_teryts()` Cannot Bridge Gaps for Missing Subjects

### Root Cause
`resolve_historical_teryts()` (geoTERYT_db.py:5204-5403) has a gate at line 5270:
```python
subj_data = record.get_data_by_subject(sid)
if not subj_data:
    continue
```
This means if a record has **zero DataSeries** for a subject, the method skips it entirely. It only fills NaN values within existing DataSeries.

### Impact
Because record 1431001 has NO P2402 DataSeries at all, `resolve_historical_teryts` cannot help. The method was designed for filling gaps in multi-year time series (e.g., BDL annual data with missing years), not for bridging across administrative reforms where the entire subject is absent.

### Fix
Modify `resolve_historical_teryts()` to also handle the case where a record has no data but shares `historical_codes` with another record that does have data. Alternatively, fix BUG 1 (which eliminates the need for this method to bridge the gap).

**Location:** `geoTERYT_db.py`, `resolve_historical_teryts()` (lines 5204-5403)

---

## BUG 3 (HIGH): Missing M_educ_2000 on Warsaw Gmina 1431001

### Root Cause
Same root cause as BUG 1. Record `1431001` has:
- No P2402 (2002 census education) — needed for M_educ_2000
- No P4315 (2021 census education) — needed for M_educ_2000
- No P3309 (2011 census education at powiat level) — this is on the powiat record

When `create_merged_subjects()` builds M_educ_2000, it reads from P2402/P4315 on each record. Since 1431001 has neither, M_educ_2000 is not created on 1431001.

### Evidence
Investigation confirms: `M_educ_2000: NOT FOUND on 1431001!`

### Impact
- Record 1431001 never gets E_educ_2000 (no source data)
- The powiat 1465000 gets E_educ_2000 from its child 1465011 (which DOES have data), but only for years 2002+
- Years 1999-2001 at the powiat level (under old code 1431000) have E_educ_2000 with only 4 years of data

### Fix
Resolved by fixing BUG 1.

---

## BUG 4 (HIGH): Mazowieckie Voivodeship 1400000 Children Include Both Powiats AND Gmina 1431001

### Root Cause
In `link_children_to_parents()`, Mazowieckie (1400000) has 43 children for years 1986-2001, including BOTH:
- `1431000` (Warsaw powiat) — level 5
- `1431001` (Warsaw urban gmina) — level 6

This means Warsaw gmina `1431001` appears as a **direct child** of the voivodeship AND as a child of powiat `1431000`.

### Evidence from database:
```
1400000 Children 1999: ['1401000', ..., '1431000', '1431001', '1432000', ...]
```
Warsaw powiat 1431000 has child `['1431001']` for all years.

### Impact
When `_collect_voivodeship_gminas()` traverses the hierarchy for Mazowieckie, it:
1. Gets direct level-6 children (including 1431001)
2. Gets powiats (including 1431000), then gets their children (1431001 again)

The `_get_aggregation_children()` method includes both powiats (level 5) and direct gminas (level 6 with rodz 1/2/3). Since 1431001 appears at both levels, it could be **double-counted** in voivodeship-level aggregation.

The `filter_aggregation_children()` function is designed for the "encompassing child" pattern (where a powiat-city's population equals the sum of its children), but this is a different pattern: the voivodeship directly contains both the powiat and its child gmina.

### Fix
In `link_children_to_parents()`, when building voivodeship children, exclude gminas whose parent powiat is also a child of the same voivodeship. A gmina should appear in the children list of its powiat, not directly as a child of the voivodeship.

**Location:** `geoTERYT_db.py`, `link_children_to_parents()` (lines 2075-2307), specifically the `_build_for_snapshot()` inner function.

---

## BUG 5 (HIGH): NUTS Split Records (1300000/1500000) Have Inconsistent Children

### Root Cause
Record `1500000` (Regional Mazowieckie NUTS-2) lists `1431001` as a direct child among powiats:
```
1500000 Children 2002: [...powiats..., '1431001', ...powiats...]
```
This is a cross-level inclusion: a gmina-level record among powiat-level records.

Meanwhile, record `1300000` (Warsaw agglomeration NUTS-2) lists powiats only (10 powiats including `1431000`).

### Impact
If the estimator ever operates on NUTS-level voivodeships (1300000/1500000), the hierarchy traversal would:
- For 1300000: traverse powiat 1431000 → gmina 1431001 (correct)
- For 1500000: include 1431001 directly AND through powiats → potential double-counting

Also, gmina `1431001` appears under BOTH 1300000 (through powiat 1431000) and 1500000 (directly), leading to double-counting across the NUTS split.

### Fix
Ensure NUTS split children are only powiats. Gminas should be reached through their parent powiats, not included directly. Remove 1431001 from 1500000's children and ensure 1431000 (or 1465000 post-reform) is in the appropriate NUTS split.

**Location:** `geoTERYT_db.py`, `link_children_to_parents()` or wherever NUTS children are set up (possibly in GUS02B notebook).

---

## BUG 6 (MEDIUM): Census Data Restoration Creates Temporal Discontinuities (Spikes)

### Root Cause
In `demographic_estimator.py`, the pipeline:
1. Generates seeds (Layer 1: log-linear interpolation between census anchors)
2. Applies Layer 2 scaling (growth-rate-deviation factors from voivodeship or national marginals)
3. **Restores original census values** (`_restore_census_data()`, lines 1572-1590)

Step 3 overwrites the Layer 2-adjusted values at census years with the original M_ data. If Layer 2 shifted values at years adjacent to a census year (e.g., 2001 and 2003), but the census year itself (2002) is restored to the original, this creates a discontinuity.

### Evidence
Spikes around 2002 in E_educ_2000 and E_educ_sex_2000 are observed, particularly at the voivodeship level.

### How it manifests
- Year 2001: Layer 2 factor might be e.g. 0.95 (scaled down)
- Year 2002: Restored to original census value (factor = 1.0 effectively)
- Year 2003: Layer 2 factor might be 1.03 (scaled up)
- Result: visible V-shape or spike at 2002

### Fix
**Option A:** Do not restore census data after Layer 2 scaling. Instead, exclude census years from Layer 2 scaling entirely (don't apply factors to census years).

**Option B:** Apply temporal smoothing to the restoration — blend the scaled and original values at census years instead of hard-overwriting.

**Option C (simplest):** In Layer 2, compute factors relative to the census year itself, and ensure the census year factor is always 1.0 by construction.

**Location:** `demographic_estimator.py`, `_restore_census_data()` (lines 1572-1590), and the Layer 2 scaling methods.

---

## BUG 7 (MEDIUM): Layer 2 for educ_1990 Has Very Sparse National Data

### Root Cause
`_estimate_educ_1990()` (line 3405) calls `_layer2_national_scaling_smoothed()` using country-level `M_educ_1990`. This data comes from `H_sex_educ` which only covers years **1986-1988 and 1991-1994**.

For years 1989-1990 and 1995-2002, there is **no national constraint**. The growth-rate-deviation approach falls back to the closest census year, but since `country_data` dict only has entries for years where data exists, years without national data get NO scaling at all.

### Impact
Years 1989-1990 and 1995-2002 are purely driven by Layer 1 interpolation. With only 1 anchor (1988 for Warsaw), these years are flat. Even for gminas with both 1988 and 2002 anchors, these years are pure log-linear interpolation with no correction from higher-level data.

### Fix
Consider using voivodeship-level M_educ_1990 as the Layer 2 constraint instead of national level, if voivodeship data is available. Alternatively, interpolate the national data to fill the gap years.

**Location:** `demographic_estimator.py`, `_estimate_educ_1990()` (lines 3332-3453)

---

## BUG 8 (MEDIUM): Wałbrzych Has Fragmented Records Missing Education Data

### Root Cause
Wałbrzych has multiple records due to administrative status changes:
- `0265011` (gmina miejska, current): E_educ_2000 observed=[2021], 27 years
- `0263011` (gmina miejska, former powiat-city period): E_educ_2000 observed=[2002], 27 years
- `0221091` (gmina miejska): **NO education cross tables at all**
- `0263000` (powiat, old period): E_educ_2000 only 4 years
- `0265000` (powiat, new period): E_educ_2000 only 13 years

### Impact
The fragmentation means different "incarnations" of the same city have partial data. The full time series should combine 2002 data from `0263011` and 2021 data from `0265011` into a single continuous estimate, but they exist on separate records.

### Fix
Same root cause as BUG 1 — `code_by_year` should route census data to the canonical record. After fixing BUG 1, ensure that the merged subjects (M_educ_2000) on the canonical Wałbrzych record contain data from all historical codes.

---

## SECONDARY ISSUES

### Issue S1: `_get_all_gminas()` vs `_collect_voivodeship_gminas()` Mismatch for Warsaw Districts

`_get_all_gminas()` (demographic_estimator.py:1940-1946) filters for `rodz in {1, 2, 3}`. Warsaw's post-reform structure has only rodz=8 districts under powiat 1465000. The `_get_aggregation_children()` fallback (lines 217-225) correctly picks up rodz=8 children when no rodz 1/2/3 exist, but `_get_all_gminas()` would miss them.

However, Warsaw's gmina record `1431001` has `rodz=1`, so it IS included in `_get_all_gminas()`. The issue is that `1431001` is parented under `1431000` (old powiat) for all years, meaning when the estimator traverses Mazowieckie → powiats → gminas, it finds 1431001 through pow 1431000 (which exists for years 1999-2001 only) but NOT through pow 1465000 (whose child is `1465011`, a different record).

**Fix:** Ensure that when records share `historical_codes`, seeds generated for one are available to the other for aggregation purposes.

### Issue S2: E_educ_2000 Observed Years Include Non-Census Years for Mazowieckie

For Mazowieckie (1400000), `E_educ_2000` has `observed=[2008, 2009, 2012, 2013, 2024]`. These are NOT census years — they appear to be years where the aggregated gmina total happened to closely match the voivodeship M_educ_2000 data (within the 0.5% hybrid scaling threshold).

This pollutes the "observed" flag with non-census data. The `is_observed` flag should only be True for actual census years.

**Fix:** In `_estimate_educ_2000()` storage loop, set `is_obs = True` only when the year is a known census year AND the source data exists.

### Issue S3: Hybrid Scaling 0.5% Threshold May Be Too Tight for Education

The `_hybrid_scale_to_observed()` method (demographic_estimator.py:2094-2145) uses `HIER_TOL_PCT = 0.5%` threshold. For education data where census definitions differ from BDL surveys, the aggregated gmina total may exceed this threshold.

**Fix:** Consider a larger threshold for education subjects (e.g., 2-5%).

### Issue S4: Household Size Estimation for 1431001 Is Completely Flat

Similar to the education issue, E_hh_size_1990 on record 1431001 is completely flat for all years 1986-2001, with every year having identical values (sum=1,280,844). This is because M_hh_size_1990 only has 1988 data (from P2887), and the 2002 data (P2871) was loaded onto the separate record 1465011.

**Fix:** Same as BUG 1.

---

## IMPLEMENTATION PLAN FOR FIXING AGENT

### Phase 1: Fix Data Propagation (BUG 1 — the root cause)

**File:** `geoTERYT_db.py`

**Step 1.1:** Create a new method `propagate_subject_data_by_historical_codes()` that:
1. Iterates over all records
2. For each record with `historical_codes` containing codes other than its own `teryt_id`:
   a. Finds the "sibling" records (records whose teryt_id appears in this record's historical_codes)
   b. For each subject on the sibling record:
      - Get all DataSeries for that subject
      - Copy the DataSeries to the current record, but only for years that are NOT already present
3. This should be called AFTER `load_subject_data()` and BEFORE `create_merged_subjects()`

**Step 1.2:** Alternatively (and more surgically), modify `load_subject_data()` to build a reverse lookup:
```python
# Build code_to_record mapping from historical_codes
code_to_records = {}
for tid, rec in self._records.items():
    for hc in rec.historical_codes:
        code_to_records.setdefault(hc, []).append(rec)
```
Then when loading data for a TERYT code, place it on ALL records that list that code in their `historical_codes`.

**Step 1.3:** Also modify `resolve_historical_teryts()` to handle records with zero DataSeries for a subject. Remove the `if not subj_data: continue` gate, and instead handle the case where the record has no data but a sibling does.

### Phase 2: Fix Children Hierarchy (BUGS 4 & 5)

**File:** `geoTERYT_db.py`, method `link_children_to_parents()`

**Step 2.1:** In `_build_for_snapshot()`, when building voivodeship children:
- After computing children for a voivodeship, remove any gmina-level (level 6) records whose parent powiat is also a child of the same voivodeship
- Specifically: if record `1431001` (level 6) has parent `1431000`, and `1431000` is already in the children list of `1400000`, then `1431001` should NOT appear directly in `1400000`'s children

**Step 2.2:** For NUTS split records (1300000/1500000), ensure children are only powiats. Remove any direct gmina references. This likely needs to be fixed in GUS02B where NUTS children are set up.

### Phase 3: Fix Census Restoration Spikes (BUG 6)

**File:** `demographic_estimator.py`

**Step 3.1:** Modify `_restore_census_data()` to use a blending approach instead of hard overwrite:
- For census years, compute a weighted average of the scaled value and the original value
- Or better: modify the Layer 2 scaling methods to exclude census years from factor application (set factor = 1.0 for census years)

**Step 3.2:** In `_layer2_voiv_scaling_smoothed()` and `_layer2_national_scaling_smoothed()`:
- When computing growth-rate-deviation factors, ensure that at census years the factor is 1.0
- This can be done by excluding census years from the factor interpolation, or by construction

### Phase 4: Fix Layer 2 Sparsity for educ_1990 (BUG 7)

**File:** `demographic_estimator.py`

**Step 4.1:** In `_estimate_educ_1990()`, consider using voivodeship-level data as Layer 2 constraint instead of national. Check if old voivodeships have H_sex_educ data (they do: for 1986-94).

**Step 4.2:** Alternatively, interpolate the national H_sex_educ data across the gap years (1989-90, 1995-2002) before applying Layer 2.

### Phase 5: Fix `is_observed` Flag (Issue S2)

**File:** `demographic_estimator.py`

Modify the storage loops in all estimation methods to only set `is_observed=True` when:
1. The year is a known census year (from `CENSUS_YEARS` constant), AND
2. The source cross table has non-NaN data for that year

### Phase 6: Re-run Pipeline

After implementing the fixes, the pipeline must be re-run in order:
1. Re-run GUS02B (with the new `propagate_subject_data_by_historical_codes()` step)
2. Re-run GUS03 (geometry optimization)
3. Re-run GUS04F (full estimation pipeline) — verify Warsaw education is no longer flat
4. Re-run GUS04H (validation) — check for new warnings
5. Re-run GUS04G (visualization) — visually verify estimates
6. Re-run GUS05 (population classification export)

---

## KEY CONTEXT FOR IMPLEMENTING AGENT

### File Structure
- `Code/tools/geoTERYT_db.py` (~7000 lines): Database management class
- `Code/tools/demographic_estimator.py` (~5000 lines): Estimation algorithms
- `Code/analysis/GUS02B_database_pipeline.ipynb`: Database construction pipeline
- `Code/analysis/GUS04F_full_pipeline.ipynb`: Estimation pipeline

### Critical TERYT Codes
- `1431001` → `1465011`: Warsaw urban gmina (code changed in 2002)
- `1431000` → `1465000`: Warsaw powiat
- `1400000`: Mazowieckie voivodeship
- `1300000`: Warsaw agglomeration NUTS-2
- `1500000`: Regional Mazowieckie NUTS-2
- `0263011` → `0265011`: Wałbrzych gmina (code changed ~2013)
- `0263000` → `0265000`: Wałbrzych powiat

### Critical Data Subjects
- P2885: 1988 census education (gmina, 1D)
- P2402: 2002 census education by sex (gmina, 2D, filter to sex='ogółem' for 1D)
- P4315: 2021 census education by sex (gmina, 2D)
- P3309: 2011 census education by sex (powiat level only)
- P2884: 1988 census age bins (gmina)
- P2883: 1988 census sex (gmina)
- P2871: 2002 census household size (gmina)
- P2887: 1988 census household size (gmina)
- P4287: 2021 census household size (gmina)
- P2137: BDL age x sex (gmina, annual 1995-2024)

### Critical Methods in geoTERYT_db.py
- `load_subject_data()`: lines 3466-3568 — THE method to fix
- `resolve_historical_teryts()`: lines 5204-5403 — secondary fix
- `create_merged_subjects()`: lines 4700-5103 — creates M_ subjects (reads from records)
- `link_children_to_parents()`: lines 2075-2307 — hierarchy construction
- `_build_for_snapshot()`: inner function, lines 2141-2224 — hierarchy per year

### Critical Methods in demographic_estimator.py
- `_generate_seeds()`: lines 592-800 — Layer 1 interpolation (1 anchor = flat)
- `_estimate_educ_1990()`: lines 3332-3453 — education 1990 estimation
- `_estimate_educ_2000()`: lines 3169-3330 — education 2000 estimation
- `_restore_census_data()`: lines 1572-1590 — census restoration (spike source)
- `_layer2_national_scaling_smoothed()`: lines 1087-1253 — Layer 2 scaling
- `_aggregate_to_parents()`: lines 2147-2257 — bottom-up aggregation
- `_collect_voivodeship_gminas()`: lines 1957-2004 — hierarchy traversal
- `_get_aggregation_children()`: lines 148-228 — child filtering with Warsaw fix

### Test Commands
After implementing fixes, run the investigation script:
```bash
cd Code/analysis && /Users/jedrek/miniforge3/envs/py313/bin/python _investigate_db.py
```
Check that:
1. Record 1431001 has P2402 data for 2002
2. M_educ_1990 on 1431001 has BOTH 1988 AND 2002 data
3. E_educ_1990 on 1431001 is NOT flat (different values for 1990 vs 1995 vs 2000)
4. M_educ_2000 exists on 1431001 (with 2002 and 2021 data)
5. 1400000 children do NOT contain both 1431000 and 1431001
