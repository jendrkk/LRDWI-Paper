# Investigation Findings: Database Label and Estimation Bugs

## Date: 2026-03-05
## Database analyzed: geoteryt_E.pkl (post-GUS05, full pipeline)

---

## FINDING F1: E_age_sex_1990 Uses Wrong Age Bin Granularity

### What is happening
`E_age_sex_1990` has shape `(16, 3)` — 15 five-year age bins + ogółem, by 3 sex groups. The age labels are:
```
['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
 '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70 i więcej', 'ogółem']
```

### What should happen
Per the user's specification in todo.md (and confirmed in the first message), E_age_sex_1990 should use the same 10-year age thresholds as the 1988 census P2884:
```
['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60 lat i więcej', 'ogółem']
```

### Root cause
`_estimate_age_sex_1990()` at `demographic_estimator.py:2901` sets `source_sid = 'M_age_sex'`. M_age_sex has 5-year bins (from P2137 and H_age_sex). The estimator produces E_age_sex_1990 inheriting this 5-year structure.

There is **no M_age_sex_1990** merged subject in the database. M_age_1990 exists but is 1D (age only, no sex dimension). The 2D (age×sex) merged subject with 10-year bins was never created.

### Evidence
- Database label survey: M_age_sex_1990 found on 0 records
- M_age_1990: shape (8,), labels ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60 lat i więcej', 'ogółem'] — correct 10yr bins
- E_age_sex_1990: shape (16, 3) — 5yr bins inherited from M_age_sex

### Impact
The E_age_sex_1990 data is internally consistent (summing 5yr pairs matches M_age_1990 10yr values), but it uses the wrong granularity. This means E_age_sex_1990 is at a finer resolution than the source census data can support for 1988, introducing artificial precision. The 1988 census P2884 only has 10-year bins — disaggregating to 5-year bins relies entirely on voivodeship-level H_age_sex proportions via IPF, which may not accurately reflect within-gmina age distributions.

---

## FINDING F2: M_educ_1990 Has Wrong Label Aggregation

### What is happening
M_educ_1990 currently has **6 labels**:
```
['ogółem', 'podstawowe', 'podstawowe nieukończone i bez wykształcenia',
 'wyższe', 'zasadnicze zawodowe', 'średnie']
```

### What should happen
Per todo.md item 4, M_educ_1990 should have **5 labels**:
```
['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
 'gimnazjalne, podstawowe i niższe']
```

Key mapping changes:
- **P2885 (1988)**: `podstawowe` should NOT be a separate category. Instead: `'gimnazjalne, podstawowe i niższe' = ogółem_15+ - wyższe - średnie - zasadnicze zawodowe`
- **P2402 (2002)**: `'podstawowe ukończone' + 'podstawowe nieukończone i bez wykształcenia'` → `'gimnazjalne, podstawowe i niższe'`
- **P2350 (voivodship)**: direct mapping `'gimnazjalne, podstawowe i niższe'` → `'gimnazjalne, podstawowe i niższe'`
- **H_sex_educ (country)**: `'podstawowe' + 'niepełne podstawowe i bez wykształcenia'` → `'gimnazjalne, podstawowe i niższe'`

### Root cause
`create_merged_subjects()` at `geoTERYT_db.py:4962-4964` defines the wrong label set. The code keeps `'podstawowe'` and `'podstawowe nieukończone i bez wykształcenia'` as separate categories instead of merging them into `'gimnazjalne, podstawowe i niższe'`.

### Evidence
- M_educ_1990 on Warsaw (1431001): 6 labels, includes separate 'podstawowe' and 'podstawowe nieukończone i bez wykształcenia'
- M_educ_2000 (correctly implemented): 5 labels including 'gimnazjalne, podstawowe i niższe'
- P2350 labels: ['gimnazjalne, podstawowe i niższe', 'policealne oraz średnie zawodowe/branżowe', 'wyższe', 'zasadnicze zawodowe/branżowe', 'średnie ogólnokształcące']

---

## FINDING F3: P2350 Not Used as Source for M_educ_1990

### What is happening
`create_merged_subjects()` only uses P2885, P2402, and H_sex_educ as sources for M_educ_1990 (line 5069). P2350 is completely excluded.

### What should happen
Per todo.md item 4 line 186: M_educ_1990 should include P2350 as a source at voivodship level.
- `'wyższe'` → `'wyższe'`
- `'policealne oraz średnie zawodowe/branżowe'` + `'średnie ogólnokształcące'` → `'średnie'` (SUM)
- `'zasadnicze zawodowe/branżowe'` → `'zasadnicze zawodowe'`
- `'gimnazjalne, podstawowe i niższe'` → `'gimnazjalne, podstawowe i niższe'`

### Impact
Without P2350, voivodships have **no M_educ_1990 data at all** (confirmed: 0/16 voivodships have it). This means:
1. Layer 2 scaling for educ_1990 can only use country-level H_sex_educ (7 sparse years: 1986-88, 1991-94)
2. No voivodship-level constraint is available for estimation
3. P2350 has annual data 1995-2020 on all 19 voivodship-level records — this is a huge missed resource

### Evidence
- All 5 tested voivodships: M_educ_1990 NOT FOUND
- 18 records have P2350 with 5 labels (covering 1995-2020)
- Country record has M_educ_1990 for only 7 years from H_sex_educ

---

## FINDING F4: M_educ_sex_1990 Has Same Wrong Labels

### What is happening
M_educ_sex_1990 has the same wrong 6-label scheme as M_educ_1990:
```
educ labels: ['ogółem', 'podstawowe', 'podstawowe nieukończone i bez wykształcenia',
              'wyższe', 'zasadnicze zawodowe', 'średnie']
sex labels:  ['kobiety', 'mężczyźni', 'ogółem']
```

### What should happen
Same 5 educ labels as M_educ_1990:
```
educ labels: ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
              'gimnazjalne, podstawowe i niższe']
sex labels:  ['kobiety', 'mężczyźni', 'ogółem']
```

### Root cause
`create_merged_subjects()` at `geoTERYT_db.py:5146-5148` uses the same wrong label set.

---

## FINDING F5: H_sex_educ "niepełne podstawowe" Available Only for 1988

### What is happening
H_sex_educ at country level has data for 7 years (1986, 1987, 1988, 1991, 1992, 1993, 1994). The category `'niepełne podstawowe i bez wykształcenia'` has data **only for 1988**. For other years, it must be computed as residual.

### Current code behavior
The current code in `_compute_residual_label()` for H_sex_educ already handles this by computing `ogółem - wyższe - średnie - zasadnicze zawodowe - podstawowe` for years where the explicit value is NaN.

### Impact on new labels
With the new M_educ_1990 labels, the H_sex_educ mapping changes:
- Current: `'podstawowe'` → separate category, `'niepełne podstawowe...'` → separate category
- New: `'podstawowe' + 'niepełne podstawowe...'` → `'gimnazjalne, podstawowe i niższe'`
- For 1988: directly sum both
- For other years: `'gimnazjalne, podstawowe i niższe' = ogółem - wyższe - średnie - zasadnicze zawodowe`

---

## FINDING F6: P4287 Missing "3-osobowe" Category

### What is happening
P4287 (2021 census household size) has only 5 labels:
```
['gospodarstwa domowe 1-osobowe', 'gospodarstwa domowe 2-osobowe',
 'gospodarstwa domowe 4-osobowe', 'gospodarstwa domowe 5-osobowe i większe', 'ogółem']
```
The `'gospodarstwa domowe 3-osobowe'` category is **missing**.

### Expected labels (per todo.md)
```
['ogółem', '1-osobowe', '2-osobowe', '3-osobowe', '4-osobowe', '5 i więcej-osobowe']
```

### Impact
M_hh_size_2000 already has 6 labels including '3-osobowe' (correctly populated from P2871 and P3420 which do have it). But P4287 only contributes 2021 data for 5 of 6 categories. The missing '3-osobowe' for 2021 means:
- E_hh_size_2000 at 2021 anchor year has a gap in the 3-person household category
- This may be a source data issue (BDL genuinely not providing this category for 2021)

### Recommendation
Check if this is a data loading issue or a source data gap. If source data gap, compute 3-osobowe as `ogółem - 1-osobowe - 2-osobowe - 4-osobowe - 5+-osobowe` for 2021.

---

## FINDING F7: Layer 2 for educ_1990 Only Uses Sparse National Data

### What is happening
`_estimate_educ_1990()` uses `_layer2_national_scaling_smoothed()` for Layer 2 scaling. This only uses country-level M_educ_1990 data from H_sex_educ, which covers only 7 years (1986-88, 1991-94).

### What should happen
With P2350 added to M_educ_1990 at voivodship level, Layer 2 should use **voivodship-level scaling** (like `_layer2_voiv_scaling_smoothed()` already used for educ_2000). P2350 provides 1995-2020 data on 19 records, giving much better constraints.

### Impact
Years 1995-2002 in E_educ_1990 are currently unconstrained by Layer 2. Adding P2350 to M_educ_1990 and switching to voivodship-level Layer 2 would provide annual constraints for these years.

---

## FINDING F8: Downstream Impacts of Label Changes

### What will break
Changing E_age_sex_1990 from (16,3) to (8,3) and E_educ_1990 from (6,) to (5,) will affect:

1. **demographic_estimator.py**:
   - `_estimate_age_sex_1990()`: Must use M_age_sex_1990 as source
   - `_estimate_educ_1990()`: Must handle new 5-label M_educ_1990
   - `_estimate_educ_sex_1990()`: Must handle new 5-label M_educ_sex_1990
   - `_layer2_national_scaling_smoothed()`: Label count changes
   - `_aggregate_to_parents()`: Dimension changes

2. **GUS04G_visualization.ipynb**: Plots reference E_ dimensions
3. **GUS04H_validation.ipynb**: Validation checks reference dimensions
4. **GUS05_pop_class_export.ipynb**: Export may reference label names

### What will NOT break
- E_age_sex_2000 (unchanged, stays at 16×3)
- E_educ_2000 (unchanged, stays at 5 labels)
- E_educ_sex_2000 (unchanged)
- E_hh_size_1990 and E_hh_size_2000 (unchanged)
