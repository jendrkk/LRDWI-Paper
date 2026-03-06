# Comprehensive Fix Plan: Label Unification & Estimation Bugs

## Date: 2026-03-05
## Priority: CRITICAL — these fixes change merged subject schemas, requiring full pipeline re-run

---

## Overview of Changes

This plan addresses 3 major bugs and their cascading effects:

| # | Bug | Severity | Files affected |
|---|-----|----------|----------------|
| A | E_age_sex_1990 uses 5yr bins instead of 10yr | HIGH | geoTERYT_db.py, demographic_estimator.py |
| B | M_educ_1990 has wrong 6-label scheme (should be 5) | HIGH | geoTERYT_db.py, demographic_estimator.py |
| C | P2350 not used for M_educ_1990 at voivodship level | HIGH | geoTERYT_db.py, demographic_estimator.py |

All three are interconnected: fixing B enables fixing C, and adding P2350 data enables switching Layer 2 from sparse national to rich voivodship-level constraints.

---

## PHASE 1: Create M_age_sex_1990 in geoTERYT_db.py

### File: `Code/tools/geoTERYT_db.py`
### Location: `create_merged_subjects()`, between sections 4 (M_age_1990) and 5 (M_educ_1990)

### Step 1.1: Add new section "4b. M_age_sex_1990"

Insert a new block after the M_age_1990 section (after line ~4958) to create M_age_sex_1990 as a 2D cross table (age_10yr × sex).

**Labels:**
```python
SID = 'M_age_sex_1990'
AGE_10YR_LABELS = ['ogółem', '0-9', '10-19', '20-29', '30-39', '40-49', '50-59',
                   '60 lat i więcej']
SEX_LABELS = ['ogółem', 'mężczyźni', 'kobiety']
```

**Sources and mappings:**

1. **P2137 (gmina, 5yr×sex, 1995+)**: aggregate 5yr→10yr for each sex group:
   ```python
   P2137_10YR_AGE_SUM = {
       '0-9': ['0-4', '5-9'],
       '10-19': ['10-14', '15-19'],
       '20-29': ['20-24', '25-29'],
       '30-39': ['30-34', '35-39'],
       '40-49': ['40-44', '45-49'],
       '50-59': ['50-54', '55-59'],
       '60 lat i więcej': ['60-64', '65-69', '70 i więcej'],
   }
   ```
   For each sex group (ogółem, mężczyźni, kobiety): sum the 5yr bins into 10yr bins.

2. **H_age_sex (old voivodships, 5yr×sex, 1986-1994)**: same aggregation but with extra bin:
   ```python
   HAGE_10YR_AGE_SUM = {
       '0-9': ['0', '1-4', '5-9'],
       '10-19': ['10-14', '15-19'],
       '20-29': ['20-24', '25-29'],
       '30-39': ['30-34', '35-39'],
       '40-49': ['40-44', '45-49'],
       '50-59': ['50-54', '55-59'],
       '60 lat i więcej': ['60-64', '65-69', '70 i więcej'],
   }
   ```

### Step 1.2: Implementation approach

For P2137: Use `_extract_2d_all_sex()` to get all (age_label, sex_label) → Series pairs, then apply the 10yr summation. This requires a new helper or modification of the extraction to handle age-group summing across the sex dimension.

**Recommended approach**: Create a wrapper that:
1. Calls `_extract_2d_all_sex(record, 'P2137', 'age', P2137_10YR_MAP, P2137_10YR_SUM)` — but this method currently only supports 1:1 mapping + summation on the FIRST (non-sex) dimension.
2. The existing `_extract_2d_all_sex()` already supports `agg_map` and `agg_sum` parameters for the non-sex dimension. Verify this works for 5yr→10yr summing.

For H_age_sex: Same approach with `HAGE_10YR_AGE_SUM`.

Store with: `self._store_2d_merged(record, SID, AGE_10YR_LABELS, SEX_LABELS, pairs, 'age_sex_1990')`

After storing, call `self._recompute_ogółem_2d(SID)` to ensure ogółem rows/columns are consistent sums.

### Step 1.3: Add to build_cross_tables()

Ensure M_age_sex_1990 is included in the cross table building step. Check `build_cross_tables()` to see if it auto-discovers new M_ subjects or needs explicit listing.

---

## PHASE 2: Fix M_educ_1990 Labels in geoTERYT_db.py

### File: `Code/tools/geoTERYT_db.py`
### Location: `create_merged_subjects()`, section 5 (lines 4960-5073)

### Step 2.1: Change label set

**Old (WRONG):**
```python
LABELS = ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
          'podstawowe', 'podstawowe nieukończone i bez wykształcenia']
```

**New (CORRECT):**
```python
LABELS = ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
          'gimnazjalne, podstawowe i niższe']
```

### Step 2.2: Fix P2885 mapping (1988 census, gmina)

**Old (WRONG):**
```python
P2885_MAP = {'wyższe': 'wyższe', 'średnie': 'średnie',
             'zasadnicze zawodowe': 'zasadnicze zawodowe',
             'podstawowe': 'podstawowe'}
```

**New (CORRECT):**
```python
P2885_MAP = {'wyższe': 'wyższe', 'średnie': 'średnie',
             'zasadnicze zawodowe': 'zasadnicze zawodowe',
             'podstawowe': 'gimnazjalne, podstawowe i niższe'}
```

And the residual computation changes. Currently the code computes `podstawowe nieukończone i bez wykształcenia = ogółem - wyższe - średnie - zasadnicze zawodowe - podstawowe`. With the new labels:

**New residual logic:**
```python
# 'gimnazjalne, podstawowe i niższe' = ogółem - wyższe - średnie - zasadnicze zawodowe
# (This includes BOTH 'podstawowe' and 'niepełne podstawowe')
self._compute_residual_label(u, 'ogółem',
    ['wyższe', 'średnie', 'zasadnicze zawodowe'],
    'gimnazjalne, podstawowe i niższe')
```

**IMPORTANT**: Remove the separate `P2885_MAP` entry for `'podstawowe'`. Instead, the entire `'gimnazjalne, podstawowe i niższe'` is computed as the residual from ogółem. The `'podstawowe'` value from P2885 should NOT be directly mapped — it becomes part of the residual.

Wait, there's a subtlety: P2885 gives us `podstawowe` as a measured value. The residual approach (`ogółem - wyższe - średnie - zasadnicze zawodowe`) will give us `podstawowe + niepełne podstawowe`, which is exactly `'gimnazjalne, podstawowe i niższe'`. So:

**Final approach for P2885:**
1. Extract wyższe, średnie, zasadnicze zawodowe from P2885 (3 direct mappings)
2. Compute ogółem from P2884 (pop 15+) — same as current
3. Compute `'gimnazjalne, podstawowe i niższe' = ogółem - wyższe - średnie - zasadnicze zawodowe`
4. Do NOT map `'podstawowe'` from P2885 to any label — it's implicitly included in the residual

### Step 2.3: Fix P2402 mapping (2002 census, gmina, sex=ogółem)

**Old (WRONG):**
```python
P2402_1990_MAP = {
    'wyższe': 'wyższe', 'policealne': 'średnie', 'średnie razem': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe ukończone': 'podstawowe',
    'podstawowe nieukończone i bez wykształcenia': 'podstawowe nieukończone i bez wykształcenia',
}
```

**New (CORRECT):**
```python
P2402_1990_MAP = {
    'wyższe': 'wyższe',
    'policealne': 'średnie',
    'średnie razem': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe ukończone': 'gimnazjalne, podstawowe i niższe',
    'podstawowe nieukończone i bez wykształcenia': 'gimnazjalne, podstawowe i niższe',
}
```

Both `podstawowe ukończone` and `podstawowe nieukończone i bez wykształcenia` map to `'gimnazjalne, podstawowe i niższe'` and are **summed** (the `_extract_2d_filter_sex` / `_extract_1d_labels` helper already handles many-to-one mappings by summing).

### Step 2.4: Fix H_sex_educ mapping (country level, sex=ogółem)

**Old (WRONG):**
```python
H_EDUC_1990_MAP = {
    'ogółem': 'ogółem', 'wyższe': 'wyższe', 'średnie': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe', 'podstawowe': 'podstawowe',
    'niepełne podstawowe i bez wykształcenia': 'podstawowe nieukończone i bez wykształcenia',
}
```

**New (CORRECT):**
```python
H_EDUC_1990_MAP = {
    'ogółem': 'ogółem', 'wyższe': 'wyższe', 'średnie': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe': 'gimnazjalne, podstawowe i niższe',
    'niepełne podstawowe i bez wykształcenia': 'gimnazjalne, podstawowe i niższe',
}
```

Both `podstawowe` and `niepełne podstawowe i bez wykształcenia` map to `'gimnazjalne, podstawowe i niższe'` and are summed.

**Update residual logic**: The current code at lines 5064-5067 checks if `'podstawowe nieukończone i bez wykształcenia'` is missing and computes it as residual. With the new labels, this residual computation is no longer needed for H_sex_educ because:
- For 1988: both `podstawowe` and `niepełne podstawowe` are directly available → sum them
- For other years (1986-87, 1991-94): `niepełne podstawowe` is NaN; compute `'gimnazjalne, podstawowe i niższe' = ogółem - wyższe - średnie - zasadnicze zawodowe`

The mapping-based summing will handle 1988 correctly. For other years, we need a residual fallback:
```python
# After extracting with H_EDUC_1990_MAP:
if 'ogółem' in u:
    # For years where 'podstawowe' or 'niepełne podstawowe' are NaN,
    # 'gimnazjalne, podstawowe i niższe' may be incomplete.
    # Recompute as residual where needed.
    gpi_key = 'gimnazjalne, podstawowe i niższe'
    og_series = u['ogółem']
    if gpi_key in u:
        gpi_series = u[gpi_key]
        for ts in og_series.index:
            og_val = og_series.get(ts, np.nan)
            gpi_val = gpi_series.get(ts, np.nan)
            if not pd.isna(og_val) and pd.isna(gpi_val):
                parts = sum(u.get(k, pd.Series(dtype=float)).get(ts, 0) or 0
                           for k in ['wyższe', 'średnie', 'zasadnicze zawodowe'])
                gpi_series[ts] = og_val - parts
```

### Step 2.5: Add P2350 as source for M_educ_1990 (voivodship level)

**New mapping:**
```python
P2350_1990_MAP = {
    'wyższe': 'wyższe',
    'zasadnicze zawodowe/branżowe': 'zasadnicze zawodowe',
    'gimnazjalne, podstawowe i niższe': 'gimnazjalne, podstawowe i niższe',
}
P2350_1990_SUM = {
    'średnie': ['policealne oraz średnie zawodowe/branżowe', 'średnie ogólnokształcące'],
}
```

**New code block** (add after the H_sex_educ block, before `result[SID] = ...`):
```python
# P2350 (voivodship level, 1D, 1995-2020)
for record in self._records.values():
    if record.level == LEVEL_VOIVODESHIP or record.teryt_id == '0000000':
        u = self._extract_1d_labels(record, 'P2350', 'educ',
                                     P2350_1990_MAP, P2350_1990_SUM)
        if u:
            n_created += self._store_1d_merged(record, SID, LABELS, u, 'educ_1990')
```

Update the source list:
```python
result[SID] = ['P2885', 'P2402', 'H_sex_educ', 'P2350']
```

### Step 2.6: Verify _extract_1d_labels supports combined map + sum

Check if `_extract_1d_labels()` supports both direct mapping (`agg_map`) AND summation (`agg_sum`) parameters. The method signature and behavior need to handle:
- Direct 1:1 mapping: `'wyższe' → 'wyższe'`
- Many-to-one summation: `'policealne...' + 'średnie ogólnokształcące' → 'średnie'`
- Direct mapping: `'gimnazjalne...' → 'gimnazjalne...'`

Look at how M_age_1990 handles this — it uses both `P2137_10YR_MAP` and `P2137_10YR_SUM` successfully (lines 4922-4928). The same pattern should work for P2350.

---

## PHASE 3: Fix M_educ_sex_1990 Labels in geoTERYT_db.py

### File: `Code/tools/geoTERYT_db.py`
### Location: `create_merged_subjects()`, section 7 (lines 5144-5206)

### Step 3.1: Change label set

**Old:**
```python
EDUC_1990_LABELS = ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
                    'podstawowe', 'podstawowe nieukończone i bez wykształcenia']
```

**New:**
```python
EDUC_1990_LABELS = ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
                    'gimnazjalne, podstawowe i niższe']
```

### Step 3.2: Fix P2402 sex mapping

**Old:**
```python
P2402_SEX1990_MAP = {
    'wyższe': 'wyższe', 'policealne': 'średnie', 'średnie razem': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe ukończone': 'podstawowe',
    'podstawowe nieukończone i bez wykształcenia': 'podstawowe nieukończone i bez wykształcenia',
}
```

**New:**
```python
P2402_SEX1990_MAP = {
    'wyższe': 'wyższe',
    'policealne': 'średnie',
    'średnie razem': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe ukończone': 'gimnazjalne, podstawowe i niższe',
    'podstawowe nieukończone i bez wykształcenia': 'gimnazjalne, podstawowe i niższe',
}
```

### Step 3.3: Fix H_sex_educ mapping (all sex groups)

**Old:**
```python
H_EDUC_SEX1990_MAP = {
    'ogółem': 'ogółem', 'wyższe': 'wyższe', 'średnie': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe', 'podstawowe': 'podstawowe',
    'niepełne podstawowe i bez wykształcenia': 'podstawowe nieukończone i bez wykształcenia',
}
```

**New:**
```python
H_EDUC_SEX1990_MAP = {
    'ogółem': 'ogółem', 'wyższe': 'wyższe', 'średnie': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe': 'gimnazjalne, podstawowe i niższe',
    'niepełne podstawowe i bez wykształcenia': 'gimnazjalne, podstawowe i niższe',
}
```

### Step 3.4: Fix ogółem computation and residual logic

The current ogółem computation block (lines 5166-5176) sums all non-ogółem education labels per sex group. Update the label list it iterates over:

**Old iteration:**
```python
for elbl in EDUC_1990_LABELS:
    if elbl == 'ogółem':
        continue
    ...
```
This will automatically work with the new 5-label set since it iterates over `EDUC_1990_LABELS`.

The residual computation block for H_sex_educ (lines 5186-5203) currently computes `'podstawowe nieukończone i bez wykształcenia'` as residual. Update to compute `'gimnazjalne, podstawowe i niższe'` as residual for years where the summed value is incomplete:

**New residual key:**
```python
pn_key = ('gimnazjalne, podstawowe i niższe', sex_lbl)
# Residual = ogółem - wyższe - średnie - zasadnicze zawodowe
for elbl in ['wyższe', 'średnie', 'zasadnicze zawodowe']:
    ...
```

---

## PHASE 4: Update demographic_estimator.py for E_age_sex_1990

### File: `Code/tools/demographic_estimator.py`
### Location: `_estimate_age_sex_1990()` (lines 2879-3170)

### Step 4.1: Change source_sid

**Old:** `source_sid = 'M_age_sex'` (line 2901)
**New:** `source_sid = 'M_age_sex_1990'`

This changes the output dimensions from (16, 3) to (8, 3).

### Step 4.2: Remove AGE_5_TO_10 mapping

The `AGE_5_TO_10` mapping (lines 2916-2924) and `group_idx_map` (lines 2926-2933) are used to map 5yr bins to 10yr bins for IPF marginal constraints. With M_age_sex_1990 already in 10yr bins, this mapping is no longer needed for the source data.

However, the IPF in Phase A still needs it for constructing 1988 tables from P2884 + P2883. Re-evaluate:

**Phase A (construct 1988 gmina tables via IPF)**: Currently:
- Gets old voivodeship M_age_sex seed (16×3, 5yr)
- Gets P2884 age marginals (10yr) → maps to 5yr rows via `group_idx_map`
- Runs grouped IPF to fit 16×3 table to 10yr age marginals + sex marginals

With the new source being M_age_sex_1990 (8×3, 10yr):
- Old voivodeship M_age_sex_1990 seed (8×3, 10yr) — this will exist after Phase 1 because H_age_sex is aggregated to 10yr
- P2884 age marginals (10yr) → direct 1:1 match with seed rows
- P2883 sex marginals → same as before
- IPF becomes simpler: no grouped IPF needed, standard 2D IPF

**Simplification**: Replace `_grouped_ipf_age_sex()` call with a standard IPF:
```python
# Seed: old voivodeship M_age_sex_1990 table for 1988 (core: 7×2)
# Fit to:
#   - Row marginals (age): from P2884 (7 values)
#   - Column marginals (sex): from P2883 (2 values)
result = self._standard_2d_ipf(seed_core, age_marginals, sex_marginals)
```

### Step 4.3: Update _get_1988_age_marginals()

Currently this method maps P2884 10yr bins to 5yr row groups. With 10yr source, it becomes a direct extraction:
- For each M_age_sex_1990 non-ogółem age label, get the matching P2884 value
- No group_idx_map needed

### Step 4.4: Update Phase B (seeds)

Seeds are built from anchor tables in M_age_sex_1990. The shapes will be (8, 3) instead of (16, 3). The core shape will be (7, 2) instead of (15, 2). No logic changes needed — the interpolation code is shape-agnostic.

### Step 4.5: Update Phase C (Layer 2)

Currently scales gmina aggregates to match old voivodship M_age_sex totals. With M_age_sex_1990 as source, it will use the 10yr×sex tables from old voivodships. The `_layer2_voiv_scaling_smoothed()` code is dimension-agnostic — it works on whatever shape the seeds have.

---

## PHASE 5: Update demographic_estimator.py for E_educ_1990 and E_educ_sex_1990

### File: `Code/tools/demographic_estimator.py`

### Step 5.1: _estimate_educ_1990() — update for 5-label M_educ_1990

**Location:** lines 3391-3512

The method uses `source_sid = 'M_educ_1990'` and reads dimensions from it. With the new 5 labels (instead of 6), the shape changes from (6,) to (5,). The code is largely shape-agnostic, but verify:

1. `_get_subject_dimensions()` will return the new labels automatically
2. `_identify_ogolem()` will find 'ogółem' among 5 labels → core has 4 non-ogółem labels
3. `_generate_seeds()` works on core tables → shape changes from (5,) to (4,)
4. `_layer2_national_scaling_smoothed()` uses country-level M_educ_1990 → this will now also have the new 5 labels

**No explicit code changes needed** for the method body — it's driven by the data dimensions. But verify all helper methods handle size 4 cores correctly.

### Step 5.2: Switch Layer 2 from national to voivodship scaling

**Current:** `_layer2_national_scaling_smoothed()` — uses country-level data (7 sparse years)
**New:** `_layer2_voiv_scaling_smoothed()` — uses voivodship-level M_educ_1990 from P2350 (1995-2020)

Replace:
```python
# OLD:
n_scaled = self._layer2_national_scaling_smoothed(
    seeds, source_sid, year_range,
    dim_names, dim_labels,
    observed_years_per_gmina,
)
```

With:
```python
# NEW:
n_scaled = self._layer2_voiv_scaling_smoothed(
    seeds, source_sid, year_range,
    dim_names, dim_labels,
    observed_years_per_gmina,
)
```

**IMPORTANT**: Keep the national scaling as a FALLBACK for years/voivodships without P2350 data (1986-1994 only has H_sex_educ). The combined approach:
1. For years 1995-2002 (within PREDICTION_1990_RANGE and covered by P2350): use voivodship-level scaling
2. For years 1986-1994 (only H_sex_educ available): use national-level scaling

This may require a hybrid approach — check if `_layer2_voiv_scaling_smoothed()` already falls back to national when voivodship data is unavailable, or if a new hybrid method is needed.

### Step 5.3: _estimate_educ_sex_1990() — update for 5-label M_educ_sex_1990

**Location:** lines 3696-3920

Same dimension changes: (6, 3) → (5, 3). The code reads dimensions from `M_educ_sex_1990` and is largely shape-agnostic.

Key updates:
1. `educ_1d_sid = 'M_educ_1990'` — the 1D education marginals used in Phase A IPF now have 4 non-ogółem labels instead of 5. The IPF will work correctly with the new sizes.
2. Layer 2: Same switch from national to voivodship scaling where possible.

### Step 5.4: Verify label_2d_core_to_1d_core mapping

At lines 3762-3771, the code builds a mapping from 2D educ row indices to 1D educ indices by matching label strings. With both M_educ_sex_1990 and M_educ_1990 using the same new 5-label set, the mapping will be a direct 1:1 correspondence. No code changes needed.

---

## PHASE 6: Update Downstream Notebooks

### Step 6.1: GUS04F_full_pipeline.ipynb

No changes needed — the pipeline calls `estimator.run()` which auto-discovers dimensions.

### Step 6.2: GUS04G_visualization.ipynb

Check for any hardcoded label references or shape assumptions for E_age_sex_1990 and E_educ_1990. Update any plot titles or label arrays that reference the old labels.

### Step 6.3: GUS04H_validation.ipynb

Check validation assertions that may reference specific label counts or names. Update any that assume 16 age bins for E_age_sex_1990 or 6 education labels for E_educ_1990.

### Step 6.4: GUS05_pop_class_export.ipynb

Check if the export references specific E_ label names. Update as needed.

---

## PHASE 7: Cleanup and Verification

### Step 7.1: Remove investigation scripts

Delete:
- `Code/analysis/_investigate_labels.py`
- `Code/analysis/_investigate_labels_output.txt`
- `Code/analysis/_investigate_bugs_deep.py`
- `Code/analysis/_investigate_bugs_deep_output.txt`

### Step 7.2: Full pipeline re-run

After all code changes, re-run the full pipeline in order:

1. **GUS02A** or **GUS04A**: Rebuild merged subjects with new label schemes
   - M_age_sex_1990 created (NEW)
   - M_educ_1990 with 5 labels (FIXED)
   - M_educ_sex_1990 with 5 labels (FIXED)
   - M_educ_1990 now includes P2350 at voivodship level (NEW)
2. **GUS03**: Geometry optimization (unchanged, just re-save)
3. **GUS04F**: Full estimation pipeline
   - E_age_sex_1990: now (8, 3) with 10yr bins
   - E_educ_1990: now (5,) with unified labels
   - E_educ_sex_1990: now (5, 3) with unified labels
4. **GUS04G**: Visualization — verify plots look correct
5. **GUS04H**: Validation — check for new warnings
6. **GUS05**: Export

### Step 7.3: Verification checks

After re-running, verify:
1. E_age_sex_1990 has shape (8, 3) and labels ['0-9', '10-19', ..., '60 lat i więcej', 'ogółem'] × ['kobiety', 'mężczyźni', 'ogółem']
2. E_educ_1990 has shape (5,) and labels ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe', 'gimnazjalne, podstawowe i niższe']
3. E_educ_sex_1990 has shape (5, 3) with same educ labels
4. M_educ_1990 exists on voivodships (from P2350) with years 1995-2020
5. Layer 2 for educ_1990 uses voivodship-level constraints for 1995-2002
6. Census data for 1988 and 2002 is preserved in E_ subjects
7. No spikes in E_educ_1990 around voivodship data transitions

---

## Execution Order Summary

```
1. geoTERYT_db.py: Add M_age_sex_1990 creation (Phase 1)
2. geoTERYT_db.py: Fix M_educ_1990 labels + add P2350 (Phase 2)
3. geoTERYT_db.py: Fix M_educ_sex_1990 labels (Phase 3)
4. demographic_estimator.py: Update _estimate_age_sex_1990 (Phase 4)
5. demographic_estimator.py: Update _estimate_educ_1990 (Phase 5)
6. demographic_estimator.py: Update _estimate_educ_sex_1990 (Phase 5)
7. Notebooks: Update references (Phase 6)
8. Full pipeline re-run (Phase 7)
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Shape mismatch in helper methods | Medium | All helpers are dimension-agnostic; verify with unit tests |
| P2350 'średnie' sum (2 categories) not handled by _extract_1d_labels | Low | Already works for M_age_1990 (uses same pattern) |
| Layer 2 voiv scaling missing for 1986-1994 | Medium | Keep national scaling as fallback for pre-P2350 years |
| Downstream notebooks break | Medium | Check all cells that reference E_ dimensions |
| Cross-variable consistency (E_age_sex_1990 age marginals feed into E_educ_1990) | Low | Only E_educ_2000 uses E_age_sex cross-constraints, not E_educ_1990 |
