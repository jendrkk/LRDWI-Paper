# Code-Level Change Reference

## This file contains the exact locations and code changes needed for each fix.

---

## FILE 1: geoTERYT_db.py

### Change 1A: Insert M_age_sex_1990 creation block

**Location:** After line ~4958 (after M_age_1990 section, before M_educ_1990 section)

**Insert the following new section (approximately 50 lines):**

```python
# ── 4b. M_age_sex_1990 ──
# P2137 (sex×age 5yr → 10yr) + H_age_sex (sex×age 5yr → 10yr)
SID = 'M_age_sex_1990'
AGE_10YR_LABELS = ['ogółem', '0-9', '10-19', '20-29', '30-39', '40-49',
                   '50-59', '60 lat i więcej']
# P2137: aggregate 5yr → 10yr for each sex group
P2137_10YR_MAP_SEX = {'ogółem': 'ogółem'}  # pass-through for ogółem age label
P2137_10YR_SUM_SEX = {
    '0-9': ['0-4', '5-9'], '10-19': ['10-14', '15-19'],
    '20-29': ['20-24', '25-29'], '30-39': ['30-34', '35-39'],
    '40-49': ['40-44', '45-49'], '50-59': ['50-54', '55-59'],
    '60 lat i więcej': ['60-64', '65-69', '70 i więcej'],
}
# H_age_sex: same 10yr aggregation with extra '0' bin
HAGE_10YR_MAP_SEX = {'ogółem': 'ogółem'}
HAGE_10YR_SUM_SEX = {
    '0-9': ['0', '1-4', '5-9'], '10-19': ['10-14', '15-19'],
    '20-29': ['20-24', '25-29'], '30-39': ['30-34', '35-39'],
    '40-49': ['40-44', '45-49'], '50-59': ['50-54', '55-59'],
    '60 lat i więcej': ['60-64', '65-69', '70 i więcej'],
}

n_created = 0
for record in self._records.values():
    # P2137 → aggregate 5yr age bins to 10yr, keeping all sex groups
    p2137_data = record.get_data_by_subject('P2137')
    if p2137_data:
        pairs = self._extract_2d_all_sex(record, 'P2137', 'age',
                                          P2137_10YR_MAP_SEX, P2137_10YR_SUM_SEX)
        if pairs:
            n_created += self._store_2d_merged(record, SID, AGE_10YR_LABELS,
                                               SEX_LABELS, pairs, 'age_sex_1990')
    # H_age_sex → same aggregation
    h_data = record.get_data_by_subject('H_age_sex')
    if h_data:
        pairs = self._extract_2d_all_sex(record, 'H_age_sex', 'age',
                                          HAGE_10YR_MAP_SEX, HAGE_10YR_SUM_SEX)
        if pairs:
            n_created += self._store_2d_merged(record, SID, AGE_10YR_LABELS,
                                               SEX_LABELS, pairs, 'age_sex_1990')
result[SID] = ['P2137', 'H_age_sex']
# NOTE: _recompute_ogółem_2d does NOT exist. M_age_sex (section 3) does not
# call any ogółem recomputation for 2D tables either — ogółem is stored
# directly from the source data via the label_map ('ogółem' → 'ogółem').
# For M_age_sex_1990, ogółem rows/columns come from the source data.
# If consistency is needed, build cross tables and verify via assertions.
if verbose:
    print(f"  {SID}: {n_created} entries stored")
```

**NOTE:** Verify that `_extract_2d_all_sex()` accepts `agg_map` and `agg_sum` parameters. Check the method signature:
- If it does NOT accept these, you need to modify it to support them (same as `_extract_2d_filter_sex` or `_extract_1d_labels`)
- If it does, the above code will work directly

**NOTE:** `_recompute_ogółem_2d()` does NOT exist. The existing M_age_sex (section 3) does not call any ogółem recompute either — ogółem rows/columns come from the source data directly via the label_map (`'ogółem': 'ogółem'`). Follow the same pattern: include `'ogółem': 'ogółem'` in both map and sum_groups. For the SUM versions, include `'ogółem'` in the AGE map. The source data already has correct ogółem values. If needed, verify after building cross tables.

---

### Change 1B: Fix M_educ_1990 labels and mappings

**Location:** Lines 4960-5073

**Replace lines 4962-4964 (LABELS):**
```python
# OLD:
LABELS = ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
          'podstawowe', 'podstawowe nieukończone i bez wykształcenia']
# NEW:
LABELS = ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
          'gimnazjalne, podstawowe i niższe']
```

**Replace lines 4965-4967 (P2885_MAP):**
```python
# OLD:
P2885_MAP = {'wyższe': 'wyższe', 'średnie': 'średnie',
             'zasadnicze zawodowe': 'zasadnicze zawodowe',
             'podstawowe': 'podstawowe'}
# NEW — do NOT map 'podstawowe' directly; it will be part of the residual:
P2885_MAP = {'wyższe': 'wyższe', 'średnie': 'średnie',
             'zasadnicze zawodowe': 'zasadnicze zawodowe'}
```

**Replace lines 4968-4973 (P2402_1990_MAP):**
```python
# OLD:
P2402_1990_MAP = {
    'wyższe': 'wyższe', 'policealne': 'średnie', 'średnie razem': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe ukończone': 'podstawowe',
    'podstawowe nieukończone i bez wykształcenia': 'podstawowe nieukończone i bez wykształcenia',
}
# NEW:
P2402_1990_MAP = {
    'wyższe': 'wyższe', 'policealne': 'średnie', 'średnie razem': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe ukończone': 'gimnazjalne, podstawowe i niższe',
    'podstawowe nieukończone i bez wykształcenia': 'gimnazjalne, podstawowe i niższe',
}
```

**Replace lines 4974-4978 (H_EDUC_1990_MAP):**
```python
# OLD:
H_EDUC_1990_MAP = {
    'ogółem': 'ogółem', 'wyższe': 'wyższe', 'średnie': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe', 'podstawowe': 'podstawowe',
    'niepełne podstawowe i bez wykształcenia': 'podstawowe nieukończone i bez wykształcenia',
}
# NEW:
H_EDUC_1990_MAP = {
    'ogółem': 'ogółem', 'wyższe': 'wyższe', 'średnie': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe': 'gimnazjalne, podstawowe i niższe',
    'niepełne podstawowe i bez wykształcenia': 'gimnazjalne, podstawowe i niższe',
}
```

**Replace lines 5011-5014 (residual computation for P2885):**
```python
# OLD:
if 'ogółem' in u:
    self._compute_residual_label(u, 'ogółem',
        ['wyższe', 'średnie', 'zasadnicze zawodowe', 'podstawowe'],
        'podstawowe nieukończone i bez wykształcenia')
# NEW:
if 'ogółem' in u:
    self._compute_residual_label(u, 'ogółem',
        ['wyższe', 'średnie', 'zasadnicze zawodowe'],
        'gimnazjalne, podstawowe i niższe')
```

**Replace lines 5063-5067 (residual for H_sex_educ):**
```python
# OLD:
if 'ogółem' in u and 'podstawowe nieukończone i bez wykształcenia' not in u:
    self._compute_residual_label(u, 'ogółem',
        ['wyższe', 'średnie', 'zasadnicze zawodowe', 'podstawowe'],
        'podstawowe nieukończone i bez wykształcenia')
# NEW — compute residual for years where the sum is incomplete:
# The summing via H_EDUC_1990_MAP handles 1988 correctly (both categories exist).
# For other years, 'niepełne podstawowe' is NaN → 'gimnazjalne...' only has 'podstawowe'.
# Use residual to fill: 'gimnazjalne...' = ogółem - wyższe - średnie - zasadnicze zawodowe
gpi_key = 'gimnazjalne, podstawowe i niższe'
if 'ogółem' in u and gpi_key in u:
    og = u['ogółem']
    gpi = u[gpi_key]
    for ts in og.index:
        og_val = og.get(ts, np.nan)
        gpi_val = gpi.get(ts, np.nan)
        if not pd.isna(og_val) and (pd.isna(gpi_val) or gpi_val <= 0):
            parts = 0.0
            for k in ['wyższe', 'średnie', 'zasadnicze zawodowe']:
                v = u.get(k, pd.Series(dtype=float)).get(ts, 0)
                if pd.isna(v):
                    v = 0
                parts += v
            res = og_val - parts
            if res >= 0:
                gpi[ts] = res
```

**Insert BEFORE `result[SID] = ...` line (add P2350 source):**
```python
# P2350 (voivodship level, 1D education, 1995-2020)
P2350_1990_MAP = {
    'wyższe': 'wyższe',
    'zasadnicze zawodowe/branżowe': 'zasadnicze zawodowe',
    'gimnazjalne, podstawowe i niższe': 'gimnazjalne, podstawowe i niższe',
}
P2350_1990_SUM = {
    'średnie': ['policealne oraz średnie zawodowe/branżowe', 'średnie ogólnokształcące'],
}
for record in self._records.values():
    if record.level == LEVEL_VOIVODESHIP or record.teryt_id == '0000000':
        u = self._extract_1d_labels(record, 'P2350', 'educ',
                                     P2350_1990_MAP, P2350_1990_SUM)
        if u:
            n_created += self._store_1d_merged(record, SID, LABELS, u, 'educ_1990')
```

**Update the result line:**
```python
# OLD:
result[SID] = ['P2885', 'P2402', 'H_sex_educ']
# NEW:
result[SID] = ['P2885', 'P2402', 'H_sex_educ', 'P2350']
```

**IMPORTANT about ogółem for P2350:** P2350 does NOT have an 'ogółem' label.
The `_store_1d_merged` will create the DataSeries with label 'ogółem' but its
values will be NaN for P2350 years. The `_recompute_ogółem_1d(SID)` call
(already present at line ~5071) will compute `ogółem = sum(non-ogółem)` for
all years, correctly filling the ogółem for P2350 voivodship data. This ensures
`_layer2_voiv_scaling_smoothed()` in `demographic_estimator.py` sees the
correct full_shape cross tables at voivodship level.

---

### Change 1C: Fix M_educ_sex_1990 labels and mappings

**Location:** Lines 5144-5206

**Replace lines 5147-5148 (EDUC_1990_LABELS):**
```python
# OLD:
EDUC_1990_LABELS = ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
                    'podstawowe', 'podstawowe nieukończone i bez wykształcenia']
# NEW:
EDUC_1990_LABELS = ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
                    'gimnazjalne, podstawowe i niższe']
```

**Replace lines 5149-5153 (P2402_SEX1990_MAP):**
```python
# OLD:
P2402_SEX1990_MAP = {
    'wyższe': 'wyższe', 'policealne': 'średnie', 'średnie razem': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe ukończone': 'podstawowe',
    'podstawowe nieukończone i bez wykształcenia': 'podstawowe nieukończone i bez wykształcenia',
}
# NEW:
P2402_SEX1990_MAP = {
    'wyższe': 'wyższe', 'policealne': 'średnie', 'średnie razem': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe ukończone': 'gimnazjalne, podstawowe i niższe',
    'podstawowe nieukończone i bez wykształcenia': 'gimnazjalne, podstawowe i niższe',
}
```

**Replace lines 5155-5158 (H_EDUC_SEX1990_MAP):**
```python
# OLD:
H_EDUC_SEX1990_MAP = {
    'ogółem': 'ogółem', 'wyższe': 'wyższe', 'średnie': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe', 'podstawowe': 'podstawowe',
    'niepełne podstawowe i bez wykształcenia': 'podstawowe nieukończone i bez wykształcenia',
}
# NEW:
H_EDUC_SEX1990_MAP = {
    'ogółem': 'ogółem', 'wyższe': 'wyższe', 'średnie': 'średnie',
    'zasadnicze zawodowe': 'zasadnicze zawodowe',
    'podstawowe': 'gimnazjalne, podstawowe i niższe',
    'niepełne podstawowe i bez wykształcenia': 'gimnazjalne, podstawowe i niższe',
}
```

**Replace lines 5186-5203 (residual logic for H_sex_educ):**
```python
# OLD:
pn_key = ('podstawowe nieukończone i bez wykształcenia', sex_lbl)
if og is not None and pn_key not in pairs:
    # ... compute residual = ogółem - wyższe - średnie - zasadnicze - podstawowe
# NEW:
gpi_key = ('gimnazjalne, podstawowe i niższe', sex_lbl)
if og is not None:
    gpi = pairs.get(gpi_key)
    if gpi is not None:
        # Fill NaN years with residual
        for ts in og.index:
            og_val = og.get(ts, np.nan)
            gpi_val = gpi.get(ts, np.nan) if gpi is not None else np.nan
            if not pd.isna(og_val) and (pd.isna(gpi_val) or gpi_val <= 0):
                parts = 0.0
                for elbl in ['wyższe', 'średnie', 'zasadnicze zawodowe']:
                    v = pairs.get((elbl, sex_lbl), pd.Series(dtype=float)).get(ts, 0)
                    if pd.isna(v):
                        v = 0
                    parts += v
                res = og_val - parts
                if res >= 0:
                    gpi[ts] = res
    else:
        # No 'gimnazjalne...' extracted at all → full residual
        residual = pd.Series(data=np.nan, index=DATETIME_INDEX_FULL, dtype=float)
        for ts in og.index:
            og_val = og.get(ts, np.nan)
            if pd.isna(og_val):
                continue
            parts = 0.0
            for elbl in ['wyższe', 'średnie', 'zasadnicze zawodowe']:
                v = pairs.get((elbl, sex_lbl), pd.Series(dtype=float)).get(ts, 0)
                if pd.isna(v):
                    v = 0
                parts += v
            res = og_val - parts
            if res >= 0:
                residual[ts] = res
        if residual.notna().any():
            pairs[gpi_key] = residual
```

---

## FILE 2: demographic_estimator.py

### Change 2A: _estimate_age_sex_1990 — change source

**Location:** Line 2901

```python
# OLD:
source_sid = 'M_age_sex'
# NEW:
source_sid = 'M_age_sex_1990'
```

### Change 2B: _estimate_age_sex_1990 — simplify Phase A IPF

**Location:** Lines 2915-2933 and 2994-2999

The `AGE_5_TO_10` mapping and `group_idx_map` can be simplified or removed since M_age_sex_1990 already uses 10yr bins. The IPF in Phase A becomes:

Instead of `_grouped_ipf_age_sex()`, use direct 2D IPF:
- Seed: old voivodship M_age_sex_1990 (already 10yr×sex)
- Row marginals: P2884 age values (10yr bins) — direct 1:1 match
- Column marginals: P2883 sex values

**Update `_get_1988_age_marginals()`**:
Currently this method must map P2884 to `group_idx_map` (5yr groups). With 10yr source, simplify to direct extraction of P2884 values in the same order as M_age_sex_1990 non-ogółem rows.

### Change 2C: _estimate_educ_1990 — switch Layer 2 to voivodship

**Location:** Lines 3459-3468

```python
# OLD:
self._log("  Layer 2: national marginal scaling (smoothed)…")
n_scaled = self._layer2_national_scaling_smoothed(
    seeds, source_sid, year_range,
    dim_names, dim_labels,
    observed_years_per_gmina,
)

# NEW — hybrid approach:
self._log("  Layer 2: voivodship+national marginal scaling (smoothed)…")
# First apply voivodship-level scaling (P2350 data, 1995-2020)
n_v = self._layer2_voiv_scaling_smoothed(
    seeds, source_sid, year_range,
    dim_names, dim_labels,
    observed_years_per_gmina,
)
# Then apply national-level scaling for remaining years (1986-94 from H_sex_educ)
n_n = self._layer2_national_scaling_smoothed(
    seeds, source_sid, year_range,
    dim_names, dim_labels,
    observed_years_per_gmina,
)
n_scaled = n_v + n_n
```

**IMPORTANT NOTE**: `_layer2_voiv_scaling_smoothed` must be checked to ensure it:
1. Finds voivodship M_educ_1990 data (now available from P2350)
2. Correctly maps gminas to voivodships for PREDICTION_1990_RANGE
3. Only scales years where voivodship data is available
4. Does not re-scale years already scaled by national method (or vice versa)

The order should be: voivodship first (more precise), then national for gaps.

### Change 2D: _estimate_educ_sex_1990 — same Layer 2 switch

**Location:** Inside `_estimate_educ_sex_1990()`, at the Layer 2 call

Apply the same hybrid voivodship + national scaling pattern.

**However**: M_educ_sex_1990 is 2D (educ×sex). P2350 is 1D (educ only). The voivodship scaling for 2D data would use the 1D education marginals from M_educ_1990 as row-sum constraints (same as educ_sex_2000 does). Check how `_estimate_educ_sex_2000()` handles this — it uses M_educ_2000 (1D) as voivodship constraint for M_educ_sex_2000 (2D). Apply the same approach.

---

## KEY VERIFICATION AFTER ALL CHANGES:

```python
# After rebuilding database:
for tid in ['1431001', '0265011', '0201011']:
    rec = db.get_by_teryt_id(tid)

    # 1. M_age_sex_1990 exists with 10yr bins
    ct = rec.get_cross_table('M_age_sex_1990')
    assert ct is not None, f"M_age_sex_1990 missing on {tid}"
    assert ct.shape == (8, 3), f"Wrong shape: {ct.shape}"
    assert ct.dim_labels['n1'] == ['0-9', '10-19', '20-29', '30-39', '40-49',
                                    '50-59', '60 lat i więcej', 'ogółem']

    # 2. M_educ_1990 has 5 labels
    ct = rec.get_cross_table('M_educ_1990')
    assert ct is not None, f"M_educ_1990 missing on {tid}"
    assert ct.shape == (5,), f"Wrong shape: {ct.shape}"
    assert 'gimnazjalne, podstawowe i niższe' in ct.dim_labels['n1']
    assert 'podstawowe' not in ct.dim_labels['n1']

    # 3. M_educ_sex_1990 has 5 educ labels
    ct = rec.get_cross_table('M_educ_sex_1990')
    assert ct.shape == (5, 3), f"Wrong shape: {ct.shape}"

# 4. M_educ_1990 on voivodships (from P2350)
for vid in ['0200000', '0400000', '1400000']:
    rec = db.get_by_teryt_id(vid)
    ct = rec.get_cross_table('M_educ_1990')
    assert ct is not None, f"M_educ_1990 missing on voivodship {vid}"
    # Should have data for years ~1995-2020
    assert any(not np.all(np.isnan(ct.get_table(yr)))
               for yr in range(1995, 2021)), f"No P2350 data on {vid}"
```
