# Implementation Summary

## Changes completed across all 7 phases

### Phase 1: Created M_age_sex_1990 (geoTERYT_db.py)
- **What:** Inserted new section "4b. M_age_sex_1990" in `create_merged_subjects()` after the existing M_age_1990 block.
- **Sources:** P2137 (BDL 5yr age bins aggregated to 10yr, all sex groups) + H_age_sex (historical 5yr aggregated to 10yr).
- **Result:** New 2D merged subject with shape (8, 3) — 7 age bins + ogółem × 3 sex groups.
- **Key detail:** 10yr age bins match P2884 census directly, eliminating the need for grouped IPF in the estimator.

### Phase 2: Fixed M_educ_1990 labels + added P2350 (geoTERYT_db.py)
- **What:** Changed education labels from 6 categories to 5, merging 'podstawowe' and 'podstawowe nieukończone i bez wykształcenia' into 'gimnazjalne, podstawowe i niższe'.
- **Changes:**
  - LABELS: 6 → 5 categories
  - P2885_MAP: Removed 'podstawowe' mapping (now part of residual)
  - P2402_1990_MAP: Both 'podstawowe ukończone' and 'podstawowe nieukończone...' → 'gimnazjalne, podstawowe i niższe'
  - H_EDUC_1990_MAP: Both 'podstawowe' and 'niepełne podstawowe...' → 'gimnazjalne, podstawowe i niższe'
  - Residual: Now computes 'gimnazjalne...' = ogółem - wyższe - średnie - zasadnicze zawodowe
  - **NEW:** Added P2350 as voivodship-level source with sum_groups for 'średnie'
  - Updated result[SID] source list to include P2350

### Phase 3: Fixed M_educ_sex_1990 labels (geoTERYT_db.py)
- **What:** Same 6→5 label change for the 2D education × sex subject.
- **Changes:**
  - EDUC_1990_LABELS: 6 → 5
  - P2402_SEX1990_MAP: Same merging as Phase 2
  - H_EDUC_SEX1990_MAP: Same merging as Phase 2
  - Residual logic: Complete rewrite to handle NaN years with two branches (partial gpi fill vs. full residual creation)

### Phase 4: Updated _estimate_age_sex_1990 (demographic_estimator.py)
- **What:** Switched source from M_age_sex (16×3, 5yr bins) to M_age_sex_1990 (8×3, 10yr bins).
- **Changes:**
  - `source_sid`: 'M_age_sex' → 'M_age_sex_1990'
  - Removed AGE_5_TO_10 mapping and group_idx_map (no longer needed)
  - Replaced `_grouped_ipf_age_sex()` with standard `_fit_marginals_ipf()` using direct 10yr marginals
  - Updated `_get_1988_age_marginals()` signature: now accepts `target_labels` and `full_age_labels` instead of `group_idx_map`
  - Phase A IPF now works on core tables (non-ogółem), matching `_fit_marginals_ipf` conventions

### Phase 5: Updated _estimate_educ_1990 and _estimate_educ_sex_1990 (demographic_estimator.py)
- **What:** Switched Layer 2 scaling from national-only to hybrid voivodship + national.
- **_estimate_educ_1990:**
  - Layer 2a: `_layer2_voiv_scaling_smoothed()` — uses P2350 data (1995–2020) at voivodship level
  - Layer 2b: `_layer2_national_scaling_smoothed()` — uses H_sex_educ (1986–94) at country level
- **_estimate_educ_sex_1990:**
  - Layer 2a: `_layer2_educ_sex_marginal_smoothed()` — uses M_educ_1990 (1D) at voivodship level to constrain education row sums (same pattern as educ_sex_2000)
  - Layer 2b: `_layer2_national_scaling_smoothed()` — national fallback
- **Docstrings:** Updated to reflect new shapes and data landscape

### Phase 6: Checked downstream notebooks
- **GUS04G_visualization.ipynb:** Updated `E_TO_ANCHOR` mapping: `E_age_sex_1990` now points to `['M_age_sex_1990', 'M_age_1990']`
- **GUS04H_validation.ipynb:** Updated `E_TO_M` mapping: `E_age_sex_1990` now points to `'M_age_sex_1990'`
- **GUS05_pop_class_export.ipynb:** No changes needed (uses E_ subjects dynamically)
- Old labels in notebook outputs will auto-correct on re-run

### Phase 7: Cleanup
- Deleted `Code/analysis/_investigate_labels.py`
- Deleted `Code/analysis/_investigate_labels_output.txt`
- Deleted `Code/analysis/_investigate_bugs_deep.py`
- Deleted `Code/analysis/_investigate_bugs_deep_output.txt`

---

## Files modified
| File | Type of change |
|------|---------------|
| `Code/tools/geoTERYT_db.py` | New M_age_sex_1990 block; fixed M_educ_1990 and M_educ_sex_1990 labels/mappings/residuals; added P2350 source |
| `Code/tools/demographic_estimator.py` | Changed source_sid for age_sex_1990; simplified IPF; updated _get_1988_age_marginals; added voivodship Layer 2 for educ_1990 and educ_sex_1990 |
| `Code/analysis/GUS04G_visualization.ipynb` | Fixed E_TO_ANCHOR mapping |
| `Code/analysis/GUS04H_validation.ipynb` | Fixed E_TO_M mapping |

## Files deleted
- `Code/analysis/_investigate_labels.py`
- `Code/analysis/_investigate_labels_output.txt`
- `Code/analysis/_investigate_bugs_deep.py`
- `Code/analysis/_investigate_bugs_deep_output.txt`

## Expected shapes after rebuild
| Subject | Old shape | New shape |
|---------|-----------|-----------|
| M_age_sex_1990 | did not exist | (8, 3) |
| E_age_sex_1990 | (16, 3) | (8, 3) |
| M_educ_1990 | (6,) | (5,) |
| E_educ_1990 | (6,) | (5,) |
| M_educ_sex_1990 | (6, 3) | (5, 3) |
| E_educ_sex_1990 | (6, 3) | (5, 3) |

## Required next steps
1. **Run full pipeline:** GUS02A → GUS04F → GUS04G → GUS04H → GUS05
2. **Verify assertions** from `02_CODE_LEVEL_CHANGES.md` (key verification section)
3. **Check that P2350 voivodship data appears** in M_educ_1990 for years 1995–2020
4. **Confirm Layer 2 voivodship scaling** actually fires for educ_1990 (check log output for "Voivodship: N voiv-year combinations" with N > 0)
