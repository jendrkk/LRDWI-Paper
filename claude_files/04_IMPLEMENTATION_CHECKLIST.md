# Implementation Checklist

## For the implementing Claude agent — use this to track progress

### Pre-implementation
- [ ] Read all files in `claude_files/` folder
- [ ] Read current `geoTERYT_db.py` (especially `create_merged_subjects()`)
- [ ] Read current `demographic_estimator.py` (especially age_sex_1990, educ_1990, educ_sex_1990)
- [ ] Read `todo.md` for user specifications

### Phase 1: Create M_age_sex_1990
- [ ] Add section 4b in `create_merged_subjects()` after M_age_1990
- [ ] Define AGE_10YR_LABELS and sex labels
- [ ] Add P2137 → 10yr aggregation with sum_groups
- [ ] Add H_age_sex → 10yr aggregation with sum_groups
- [ ] Include ogółem mapping in both P2137 and H_age_sex maps
- [ ] Test: verify M_age_sex_1990 appears on gmina records with expected shape (8,3)

### Phase 2: Fix M_educ_1990 labels
- [ ] Change LABELS from 6 to 5 categories
- [ ] Fix P2885_MAP: remove 'podstawowe' mapping
- [ ] Fix residual: change to `ogółem - wyższe - średnie - zasadnicze zawodowe = 'gimnazjalne, podstawowe i niższe'`
- [ ] Fix P2402_1990_MAP: merge basic educ categories
- [ ] Fix H_EDUC_1990_MAP: merge basic educ categories
- [ ] Fix H_sex_educ residual logic for years where niepełne is NaN
- [ ] Add P2350 source with sum_groups for 'średnie'
- [ ] Update result[SID] source list
- [ ] Test: verify M_educ_1990 has 5 labels on gminas
- [ ] Test: verify M_educ_1990 exists on voivodships with P2350 data

### Phase 3: Fix M_educ_sex_1990 labels
- [ ] Change EDUC_1990_LABELS from 6 to 5 categories
- [ ] Fix P2402_SEX1990_MAP
- [ ] Fix H_EDUC_SEX1990_MAP
- [ ] Fix residual logic (change key and known categories list)
- [ ] Test: verify M_educ_sex_1990 has (5,3) shape

### Phase 4: Update _estimate_age_sex_1990
- [ ] Change source_sid to 'M_age_sex_1990'
- [ ] Review/simplify Phase A IPF (10yr bins, no grouped IPF needed)
- [ ] Update _get_1988_age_marginals for direct 10yr match
- [ ] Verify Phase B seeds work with new shape
- [ ] Verify Phase C Layer 2 works with new shape
- [ ] Test: verify E_age_sex_1990 has (8,3) shape

### Phase 5: Update _estimate_educ_1990 and _estimate_educ_sex_1990
- [ ] Verify _estimate_educ_1990 is shape-agnostic (should auto-adapt)
- [ ] Switch Layer 2 to voivodship scaling (hybrid with national fallback)
- [ ] Verify _estimate_educ_sex_1990 is shape-agnostic
- [ ] Apply same Layer 2 switch for educ_sex_1990
- [ ] Test: verify E_educ_1990 has (5,) shape
- [ ] Test: verify E_educ_sex_1990 has (5,3) shape

### Phase 6: Update notebooks
- [ ] Check GUS04G for hardcoded label references
- [ ] Check GUS04H for hardcoded validation checks
- [ ] Check GUS05 for hardcoded export labels

### Phase 7: Cleanup
- [ ] Delete `Code/analysis/_investigate_labels.py`
- [ ] Delete `Code/analysis/_investigate_labels_output.txt`
- [ ] Delete `Code/analysis/_investigate_bugs_deep.py`
- [ ] Delete `Code/analysis/_investigate_bugs_deep_output.txt`
- [ ] Run full pipeline: GUS02A → GUS04F → GUS04G → GUS04H → GUS05
- [ ] Verify all assertions pass

### Critical Gotchas
1. `_extract_1d_labels` does `.lower()` on source labels before matching — make sure P2350_1990_MAP keys are lowercase
2. `_extract_2d_all_sex` sum_groups works on raw (not-yet-mapped) labels — use original source label strings in sum_groups
3. P2350 has no ogółem → `_recompute_ogółem_1d(SID)` must be called AFTER P2350 data is stored
4. H_sex_educ 'niepełne podstawowe' is NaN for years ≠ 1988 → residual fallback needed
5. `_layer2_voiv_scaling_smoothed` checks `voiv_tbl.shape != full_shape` → P2350 voivodship data must produce correct shape (5,) after ogółem recomputation
6. `_recompute_ogółem_2d` does NOT exist — for 2D subjects, ogółem comes from source data mappings
7. `_grouped_ipf_age_sex` may need to be replaced with standard IPF when source is 10yr bins
8. The order of sources matters in `create_merged_subjects`: first source's values take priority (NaN-filling)
