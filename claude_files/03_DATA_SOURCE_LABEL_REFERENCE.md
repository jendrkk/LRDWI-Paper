# Data Source Label Reference

## Complete label inventory for all data sources relevant to Prediction1990

This file captures the exact labels as they exist in the database,
to serve as ground truth for the implementing agent.

---

## Raw Census Subjects (gmina level)

### P2884 — Census 1988 Age (level=6, 1D)
```
n1 labels (8): ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59',
                '60 lat i więcej', 'ogółem']
shape: (8,)
records: 4220
years: 1988 only
```

### P2883 — Census 1988 Sex (level=6, 1D)
```
n1 labels (3): ['kobiety', 'mężczyźni', 'ogółem']
shape: (3,)
records: 4220
years: 1988 only
```

### P2885 — Census 1988 Education (level=6, 1D)
```
n1 labels (4): ['podstawowe', 'wyższe', 'zasadnicze zawodowe', 'średnie']
shape: (4,)
records: 4220
years: 1988 only
NOTE: No 'ogółem', no 'podstawowe nieukończone'
```

### P2402 — Census 2002 Education × Sex (level=6, 2D)
```
n1 labels (3): ['kobiety', 'mężczyźni', 'ogółem']  ← sex dimension
n2 labels (8): ['podstawowe nieukończone i bez wykształcenia',
                'podstawowe ukończone', 'policealne', 'wyższe',
                'zasadnicze zawodowe', 'średnie ogólnokształcące',
                'średnie razem', 'średnie zawodowe']  ← educ dimension
shape: (3, 8)
records: 4268
years: 2002 only
```

### P2137 — BDL Age × Sex (level=6, 2D)
```
n1 labels: VARIES (3 variants)
  variant 1 (16): 5yr bins 0-4...65-69 + "70 i więcej" + ogółem
  variant 2 (20): adds 70-74, 75-79, 80-84, "85 i więcej"
  variant 3 (21): adds "0-14" aggregate
n2 labels (3): ['kobiety', 'mężczyźni', 'ogółem']
records: 4541
years: 1995-2024
```

---

## Historical Subjects

### H_age_sex — Historical Age × Sex (old voivodships, level=2, 2D)
```
n1 labels (17): ['0', '1-4', '10-14', '15-19', '20-24', '25-29',
                 '30-34', '35-39', '40-44', '45-49', '5-9',
                 '50-54', '55-59', '60-64', '65-69', '70 i więcej', 'ogółem']
n2 labels (3): ['kobiety', 'mężczyźni', 'ogółem']
shape: (17, 3)
records: 50 (old voivodships + country)
years: 1986-1994 (varies; typically 1986-1988 and/or 1991-1994)
IMPORTANT: has '0' as separate bin (not '0-4')
```

### H_sex_educ — Historical Sex × Education (country level=0, 2D)
```
n1 labels (3): ['kobiety', 'mężczyźni', 'ogółem']  ← sex dimension
n2 labels (6): ['niepełne podstawowe i bez wykształcenia', 'ogółem',
                'podstawowe', 'wyższe', 'zasadnicze zawodowe', 'średnie']  ← educ dimension
shape: (3, 6)
records: 1 (country only: '0000000')
years with data: 1986, 1987, 1988, 1991, 1992, 1993, 1994
IMPORTANT: 'niepełne podstawowe i bez wykształcenia' only has data for 1988
```

### P2350 — BDL Education (new voivodships, level=2, 1D)
```
n1 labels (5): ['gimnazjalne, podstawowe i niższe',
                'policealne oraz średnie zawodowe/branżowe',
                'wyższe', 'zasadnicze zawodowe/branżowe',
                'średnie ogólnokształcące']
shape: (5,)
records: 19 (16 voivodships + 2 NUTS splits + country)
years: 1995-2020
```

---

## Current Merged Subjects (WRONG — to be fixed)

### M_educ_1990 — CURRENT (wrong, 6 labels)
```
n1 labels (6): ['ogółem', 'podstawowe',
                'podstawowe nieukończone i bez wykształcenia',
                'wyższe', 'zasadnicze zawodowe', 'średnie']
shape: (6,)
records: 3872
sources: P2885, P2402, H_sex_educ (P2350 MISSING!)
```

### M_educ_sex_1990 — CURRENT (wrong, 6 educ labels)
```
n1 labels (6): same as M_educ_1990
n2 labels (3): ['kobiety', 'mężczyźni', 'ogółem']
shape: (6, 3)
records: 3871
sources: P2402, H_sex_educ
```

### M_age_sex_1990 — DOES NOT EXIST (to be created)

---

## Target Merged Subjects (CORRECT — after fixes)

### M_age_sex_1990 — NEW (10yr × sex)
```
n1 labels (8): ['ogółem', '0-9', '10-19', '20-29', '30-39', '40-49',
                '50-59', '60 lat i więcej']
n2 labels (3): ['ogółem', 'mężczyźni', 'kobiety']
shape: (8, 3)
sources: P2137 (5yr→10yr), H_age_sex (5yr→10yr)
```

### M_educ_1990 — FIXED (5 labels)
```
n1 labels (5): ['ogółem', 'wyższe', 'średnie', 'zasadnicze zawodowe',
                'gimnazjalne, podstawowe i niższe']
shape: (5,)
sources: P2885, P2402, H_sex_educ, P2350 (NEW!)
```

### M_educ_sex_1990 — FIXED (5 educ labels)
```
n1 labels (5): same as M_educ_1990
n2 labels (3): ['ogółem', 'mężczyźni', 'kobiety']
shape: (5, 3)
sources: P2402, H_sex_educ
```

---

## Label Mapping Tables

### P2885 → M_educ_1990 (FIXED)
| P2885 label | M_educ_1990 target | Method |
|---|---|---|
| wyższe | wyższe | direct |
| średnie | średnie | direct |
| zasadnicze zawodowe | zasadnicze zawodowe | direct |
| podstawowe | *(not mapped)* | included in residual |
| *(residual)* | gimnazjalne, podstawowe i niższe | ogółem - wyższe - średnie - zasadnicze zawodowe |

### P2402 → M_educ_1990 (FIXED, sex=ogółem filtered)
| P2402 label (n2) | M_educ_1990 target | Method |
|---|---|---|
| wyższe | wyższe | direct |
| policealne | średnie | sum |
| średnie razem | średnie | sum |
| zasadnicze zawodowe | zasadnicze zawodowe | direct |
| podstawowe ukończone | gimnazjalne, podstawowe i niższe | sum |
| podstawowe nieukończone i bez wykształcenia | gimnazjalne, podstawowe i niższe | sum |
| średnie ogólnokształcące | *(not mapped)* | included in średnie razem |
| średnie zawodowe | *(not mapped)* | included in średnie razem |

### H_sex_educ → M_educ_1990 (FIXED, sex=ogółem filtered)
| H_sex_educ label (n2) | M_educ_1990 target | Method |
|---|---|---|
| ogółem | ogółem | direct |
| wyższe | wyższe | direct |
| średnie | średnie | direct |
| zasadnicze zawodowe | zasadnicze zawodowe | direct |
| podstawowe | gimnazjalne, podstawowe i niższe | sum (+ next) |
| niepełne podstawowe i bez wykształcenia | gimnazjalne, podstawowe i niższe | sum (NaN for years ≠ 1988) |

For years where 'niepełne podstawowe' is NaN (1986-87, 1991-94):
`gimnazjalne, podstawowe i niższe = ogółem - wyższe - średnie - zasadnicze zawodowe`

### P2350 → M_educ_1990 (NEW)
| P2350 label | M_educ_1990 target | Method |
|---|---|---|
| wyższe | wyższe | direct |
| policealne oraz średnie zawodowe/branżowe | średnie | sum (+ next) |
| średnie ogólnokształcące | średnie | sum |
| zasadnicze zawodowe/branżowe | zasadnicze zawodowe | direct |
| gimnazjalne, podstawowe i niższe | gimnazjalne, podstawowe i niższe | direct |

### P2137 → M_age_sex_1990 (NEW, all sex groups)
| P2137 5yr labels | M_age_sex_1990 10yr target | Method |
|---|---|---|
| 0-4 + 5-9 | 0-9 | sum |
| 10-14 + 15-19 | 10-19 | sum |
| 20-24 + 25-29 | 20-29 | sum |
| 30-34 + 35-39 | 30-39 | sum |
| 40-44 + 45-49 | 40-49 | sum |
| 50-54 + 55-59 | 50-59 | sum |
| 60-64 + 65-69 + 70 i więcej | 60 lat i więcej | sum |
| ogółem | ogółem | direct |

### H_age_sex → M_age_sex_1990 (NEW, all sex groups)
| H_age_sex 5yr labels | M_age_sex_1990 10yr target | Method |
|---|---|---|
| 0 + 1-4 + 5-9 | 0-9 | sum |
| 10-14 + 15-19 | 10-19 | sum |
| 20-24 + 25-29 | 20-29 | sum |
| 30-34 + 35-39 | 30-39 | sum |
| 40-44 + 45-49 | 40-49 | sum |
| 50-54 + 55-59 | 50-59 | sum |
| 60-64 + 65-69 + 70 i więcej | 60 lat i więcej | sum |
| ogółem | ogółem | direct |

---

## Key Line Numbers Reference

### geoTERYT_db.py
- `create_merged_subjects()`: starts at line 4241
- M_age_sex (section 3): ~line 4840-4912
- M_age_1990 (section 4): ~line 4914-4958
- M_educ_1990 (section 5): ~line 4960-5073
- M_educ_2000 (section 6): ~line 5075-5142
- M_educ_sex_1990 (section 7): ~line 5144-5206
- M_educ_sex_2000 (section 8): ~line 5210-5237
- `_extract_1d_labels()`: search for def, supports (agg_map, agg_sum)
- `_extract_2d_filter_sex()`: search for def
- `_extract_2d_all_sex()`: search for def
- `_store_1d_merged()`: search for def
- `_store_2d_merged()`: search for def
- `_compute_residual_label()`: search for def
- `_recompute_ogółem_1d()`: search for def
- `_recompute_ogółem_2d()`: VERIFY EXISTS, if not create it

### demographic_estimator.py
- `_estimate_age_sex_1990()`: line 2879
- `_estimate_educ_1990()`: line 3391
- `_estimate_educ_sex_1990()`: line 3696
- `_layer2_national_scaling_smoothed()`: search for def
- `_layer2_voiv_scaling_smoothed()`: search for def
- `_generate_seeds()`: search for def
- `_get_1988_age_marginals()`: search for def
- `_grouped_ipf_age_sex()`: search for def
