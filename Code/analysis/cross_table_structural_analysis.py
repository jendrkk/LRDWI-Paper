#!/usr/bin/env python3
"""
Structural analysis of cross tables in the GeoTERYT database.
Answers 7 specific questions about cross table structure, coverage, and consistency.
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import warnings
import random
warnings.filterwarnings('ignore')

# ── Load database ──────────────────────────────────────────────────────
DB_PATH = Path(os.path.expanduser(
    "~/Documents/Studium Volkswirschaftslehre/3. Semester/"
    "Long-run dynamics of wealth inequalities/Paper/Data/Geospatial/geoteryt_O.pkl"
))
print(f"Loading database from {DB_PATH} ...")

sys.path.insert(0, str(DB_PATH.parents[2] / 'local_repo' / 'LRDWI-Paper' / 'Code' / 'tools'))
from geoTERYT_db import load_complete_database

db = load_complete_database(DB_PATH, verbose=False)
records = db._records
print(f"Total records: {len(records)}")

gminas = {tid: r for tid, r in records.items() if r.level == 6}
print(f"Level-6 (gmina) records: {len(gminas)}")

SEP = "\n" + "=" * 100

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: f'{x:.0f}' if not np.isnan(x) else 'NaN')

# ════════════════════════════════════════════════════════════════════════
# Q1. P2137 CROSS TABLE STRUCTURE FOR A GMINA
# ════════════════════════════════════════════════════════════════════════
print(SEP)
print("Q1. P2137 CROSS TABLE STRUCTURE FOR A GMINA")
print("=" * 100)

# Find a gmina with P2137 that has 2020 data
target_gmina = None
for tid, r in gminas.items():
    if 'P2137' in r.cross_tables:
        ct = r.cross_tables['P2137']
        tbl = ct.tables.get(2020)
        if tbl is not None and not np.all(np.isnan(tbl)):
            target_gmina = (tid, r)
            break

if target_gmina:
    tid, r = target_gmina
    ct = r.cross_tables['P2137']
    print(f"\nSelected gmina: {tid} — {r.name} (kind={r.kind})")
    print(f"\nCrossTable object: {ct}")
    print(f"  subject_id:   {ct.subject_id}")
    print(f"  subject_name: {ct.subject_name}")
    print(f"  ndim:         {ct.ndim}")
    print(f"  shape:        {ct.shape}")
    print(f"  dim_names:    {ct.dim_names}")
    
    print(f"\n  dim_labels:")
    for dim_name in ct.dim_names:
        labels = ct.dim_labels[dim_name]
        print(f"    '{dim_name}' ({len(labels)} labels):")
        for i, lab in enumerate(labels):
            print(f"      [{i}] {lab}")
    
    print(f"\n  years_with_data: {ct.years_with_data}")
    print(f"  years_missing:   {ct.years_missing}")
    
    # Print FULL cross table for 2020
    print(f"\n  === FULL CROSS TABLE FOR YEAR 2020 ===")
    df_2020 = ct.get_as_dataframe(2020)
    print(df_2020.to_string())
    
    # Show grand total — find ogółem indices properly
    tbl_2020 = ct.tables[2020]
    d0_labels = ct.dim_labels[ct.dim_names[0]]
    d1_labels = ct.dim_labels[ct.dim_names[1]]
    ogolem_r = d0_labels.index('ogółem') if 'ogółem' in d0_labels else 0
    ogolem_c = d1_labels.index('ogółem') if 'ogółem' in d1_labels else 0
    grand_total = tbl_2020[ogolem_r, ogolem_c] if tbl_2020.ndim == 2 else tbl_2020.flat[0]
    print(f"\n  Grand total (ogółem × ogółem) = cell [{ogolem_r},{ogolem_c}] = {grand_total:.0f}")
    print(f"  (ogółem row label at index {ogolem_r}, ogółem col label at index {ogolem_c})")
    
    # Compare with record.pop
    ts_2020 = pd.Timestamp(year=2020, month=1, day=1)
    pop_2020 = r.pop[ts_2020] if ts_2020 in r.pop.index else np.nan
    print(f"  record.pop[2020]                       = {pop_2020:.0f}")
    if not np.isnan(pop_2020) and not np.isnan(grand_total):
        print(f"  Match: {'YES' if abs(grand_total - pop_2020) < 1 else 'NO (diff=' + str(grand_total - pop_2020) + ')'}")
else:
    print("  No gmina with P2137 and 2020 data found!")

# ════════════════════════════════════════════════════════════════════════
# Q2. M_pop__age_sex vs P2137 COMPARISON
# ════════════════════════════════════════════════════════════════════════
print(SEP)
print("Q2. M_pop__age_sex CROSS TABLE vs P2137 — COMPARISON AT YEAR 2002")
print("=" * 100)

if target_gmina:
    tid, r = target_gmina
    
    has_m = 'M_pop__age_sex' in r.cross_tables
    has_p = 'P2137' in r.cross_tables
    print(f"\n  Gmina: {tid} — {r.name}")
    print(f"  Has M_pop__age_sex cross table: {has_m}")
    print(f"  Has P2137 cross table:          {has_p}")
    
    if has_m and has_p:
        ct_m = r.cross_tables['M_pop__age_sex']
        ct_p = r.cross_tables['P2137']
        
        print(f"\n  ── M_pop__age_sex ──")
        print(f"  dim_names: {ct_m.dim_names}")
        print(f"  shape:     {ct_m.shape}")
        for dn in ct_m.dim_names:
            print(f"  dim_labels['{dn}']: {ct_m.dim_labels[dn]}")
        print(f"  years_with_data: {ct_m.years_with_data}")
        
        print(f"\n  ── P2137 ──")
        print(f"  dim_names: {ct_p.dim_names}")
        print(f"  shape:     {ct_p.shape}")
        for dn in ct_p.dim_names:
            print(f"  dim_labels['{dn}']: {ct_p.dim_labels[dn]}")
        print(f"  years_with_data: {ct_p.years_with_data}")
        
        print(f"\n  KEY DIFFERENCES:")
        # Compare label sets
        for dn in sorted(set(ct_m.dim_names) | set(ct_p.dim_names)):
            lm = set(ct_m.dim_labels.get(dn, []))
            lp = set(ct_p.dim_labels.get(dn, []))
            only_m = lm - lp
            only_p = lp - lm
            if only_m or only_p:
                print(f"  Dim '{dn}':")
                if only_m:
                    print(f"    Only in M_pop__age_sex: {sorted(only_m)}")
                if only_p:
                    print(f"    Only in P2137:          {sorted(only_p)}")
        
        # Compare at year 2002
        if 2002 in ct_m.years_with_data and 2002 in ct_p.years_with_data:
            tbl_m = ct_m.tables[2002]
            tbl_p = ct_p.tables[2002]
            print(f"\n  ── Comparison at year 2002 ──")
            print(f"  M_pop__age_sex shape: {tbl_m.shape}")
            print(f"  P2137 shape:          {tbl_p.shape}")
            print(f"  Shapes identical: {tbl_m.shape == tbl_p.shape}")
            
            # Find ogółem totals properly
            m_d0 = ct_m.dim_labels[ct_m.dim_names[0]]
            m_d1 = ct_m.dim_labels[ct_m.dim_names[1]]
            p_d0 = ct_p.dim_labels[ct_p.dim_names[0]]
            p_d1 = ct_p.dim_labels[ct_p.dim_names[1]]
            
            m_r = m_d0.index('ogółem') if 'ogółem' in m_d0 else None
            m_c = m_d1.index('ogółem') if 'ogółem' in m_d1 else None
            p_r = p_d0.index('ogółem') if 'ogółem' in p_d0 else None
            p_c = p_d1.index('ogółem') if 'ogółem' in p_d1 else None
            
            if m_r is not None and m_c is not None:
                m_total = tbl_m[m_r, m_c]
                print(f"  M_pop__age_sex ogółem total (2002): {m_total:.0f} (at [{m_r},{m_c}])")
            if p_r is not None and p_c is not None:
                p_total = tbl_p[p_r, p_c]
                print(f"  P2137          ogółem total (2002): {p_total:.0f} (at [{p_r},{p_c}])")
            
            # Compare matching labels
            common_d0 = sorted(set(m_d0) & set(p_d0))
            common_d1 = sorted(set(m_d1) & set(p_d1))
            print(f"  Common row labels ({len(common_d0)}): {common_d0[:5]}...")
            print(f"  Common col labels ({len(common_d1)}): {common_d1}")
            
            # For common labels that match, check if values are identical
            mismatches = 0
            for rl in common_d0:
                for cl in common_d1:
                    mi = m_d0.index(rl)
                    mj = m_d1.index(cl)
                    pi = p_d0.index(rl)
                    pj = p_d1.index(cl)
                    mv = tbl_m[mi, mj]
                    pv = tbl_p[pi, pj]
                    if not (np.isnan(mv) and np.isnan(pv)):
                        if abs(mv - pv) > 0.5:
                            mismatches += 1
                            if mismatches <= 5:
                                print(f"    MISMATCH at ({rl}, {cl}): M={mv:.0f} vs P={pv:.0f}")
            print(f"  Total mismatches among common labels: {mismatches}")
        else:
            print(f"  M_pop__age_sex has 2002 data: {2002 in ct_m.years_with_data}")
            print(f"  P2137 has 2002 data:          {2002 in ct_p.years_with_data}")
    
    # If this gmina doesn't have M_pop__age_sex, try finding one that has both
    if not has_m:
        print("\n  This gmina lacks M_pop__age_sex. Searching for one with both...")
        for tid2, r2 in gminas.items():
            if 'M_pop__age_sex' in r2.cross_tables and 'P2137' in r2.cross_tables:
                ct_m2 = r2.cross_tables['M_pop__age_sex']
                ct_p2 = r2.cross_tables['P2137']
                if 2002 in ct_m2.years_with_data and 2002 in ct_p2.years_with_data:
                    print(f"  Found: {tid2} — {r2.name}")
                    print(f"\n  ── M_pop__age_sex ──")
                    print(f"  dim_names: {ct_m2.dim_names}, shape: {ct_m2.shape}")
                    for dn in ct_m2.dim_names:
                        print(f"  dim_labels['{dn}']: {ct_m2.dim_labels[dn]}")
                    
                    print(f"\n  ── P2137 ──")
                    print(f"  dim_names: {ct_p2.dim_names}, shape: {ct_p2.shape}")
                    for dn in ct_p2.dim_names:
                        print(f"  dim_labels['{dn}']: {ct_p2.dim_labels[dn]}")
                    
                    # Key differences
                    for dn in sorted(set(ct_m2.dim_names) | set(ct_p2.dim_names)):
                        lm = set(ct_m2.dim_labels.get(dn, []))
                        lp = set(ct_p2.dim_labels.get(dn, []))
                        only_m = lm - lp
                        only_p = lp - lm
                        if only_m or only_p:
                            print(f"\n  Dim '{dn}' differences:")
                            if only_m: print(f"    Only in M: {sorted(only_m)}")
                            if only_p: print(f"    Only in P: {sorted(only_p)}")
                    break

# ════════════════════════════════════════════════════════════════════════
# Q3. P2137 YEARS WITH DATA PER GMINA (10 random gminas)
# ════════════════════════════════════════════════════════════════════════
print(SEP)
print("Q3. P2137 YEARS WITH DATA PER GMINA (10 random gminas)")
print("=" * 100)

gminas_with_p2137 = [(tid, r) for tid, r in gminas.items() if 'P2137' in r.cross_tables]
print(f"\nTotal gminas with P2137 cross table: {len(gminas_with_p2137)} / {len(gminas)}")

random.seed(42)
sample = random.sample(gminas_with_p2137, min(10, len(gminas_with_p2137)))

print(f"\n{'TERYT':>10s}  {'Name':30s}  {'#Yrs':>4s}  Years with data (1995-2024 range)")
print("-" * 100)
for tid, r in sample:
    ct = r.cross_tables['P2137']
    yrs = ct.years_with_data
    yrs_in_range = [y for y in yrs if 1995 <= y <= 2024]
    # Show as compact representation
    yr_str = ','.join(str(y) for y in yrs_in_range)
    print(f"{tid:>10s}  {r.name:30s}  {len(yrs_in_range):4d}  {yr_str}")

# Coverage statistics across all gminas with P2137
print(f"\n  Coverage statistics across ALL {len(gminas_with_p2137)} gminas with P2137:")
year_coverage = Counter()
for tid, r in gminas_with_p2137:
    ct = r.cross_tables['P2137']
    for yr in ct.years_with_data:
        if 1995 <= yr <= 2024:
            year_coverage[yr] += 1

print(f"  {'Year':>6s}  {'Gminas with data':>16s}  {'Coverage%':>10s}")
for yr in range(1995, 2025):
    n = year_coverage.get(yr, 0)
    pct = n / len(gminas_with_p2137) * 100 if gminas_with_p2137 else 0
    marker = " <<<" if pct < 90 else ""
    print(f"  {yr:>6d}  {n:>16d}  {pct:>9.1f}%{marker}")

# ════════════════════════════════════════════════════════════════════════
# Q4. CENSUS CROSS TABLE SHAPES FOR A SINGLE GMINA
# ════════════════════════════════════════════════════════════════════════
print(SEP)
print("Q4. CENSUS CROSS TABLE SHAPES FOR A SINGLE GMINA")
print("=" * 100)

census_subjects_to_check = [
    ('P2887', '1988 hh'),
    ('P2884', '1988 age'),
    ('P2885', '1988 educ'),
    ('P2883', '1988 sex'),
    ('P2871', '2002 hh'),
    ('P2402', '2002 sex×educ'),
    ('P2114', '2002 age×sex'),
]

# Find a gmina that has ALL of these census subjects
target_census_gmina = None
for tid, r in gminas.items():
    has_all = True
    for subj, _ in census_subjects_to_check:
        if subj not in r.cross_tables:
            has_all = False
            break
        ct = r.cross_tables[subj]
        if not ct.years_with_data:
            has_all = False
            break
    if has_all:
        target_census_gmina = (tid, r)
        break

if target_census_gmina:
    tid, r = target_census_gmina
    print(f"\nSelected gmina: {tid} — {r.name} (kind={r.kind})")
    
    for subj, desc in census_subjects_to_check:
        ct = r.cross_tables[subj]
        print(f"\n  ── {subj} ({desc}) ──")
        print(f"  subject_name: {ct.subject_name}")
        print(f"  shape:        {ct.shape}")
        print(f"  ndim:         {ct.ndim}")
        print(f"  dim_names:    {ct.dim_names}")
        for dn in ct.dim_names:
            print(f"  dim_labels['{dn}']: {ct.dim_labels[dn]}")
        print(f"  years_with_data: {ct.years_with_data}")
else:
    print("  No gmina found with ALL census subjects!")
    # Show what's available
    for subj, desc in census_subjects_to_check:
        count = sum(1 for tid, r in gminas.items() 
                    if subj in r.cross_tables and r.cross_tables[subj].years_with_data)
        print(f"  {subj} ({desc}): available in {count} gminas")
    
    # Try finding one that has as many as possible
    best_tid = None
    best_count = 0
    for tid, r in gminas.items():
        count = sum(1 for subj, _ in census_subjects_to_check 
                    if subj in r.cross_tables and r.cross_tables[subj].years_with_data)
        if count > best_count:
            best_count = count
            best_tid = tid
    
    if best_tid:
        r = gminas[best_tid]
        print(f"\n  Best gmina with most subjects: {best_tid} — {r.name} ({best_count}/{len(census_subjects_to_check)})")
        for subj, desc in census_subjects_to_check:
            if subj in r.cross_tables and r.cross_tables[subj].years_with_data:
                ct = r.cross_tables[subj]
                print(f"\n  ── {subj} ({desc}) ──")
                print(f"  shape:     {ct.shape}")
                print(f"  dim_names: {ct.dim_names}")
                for dn in ct.dim_names:
                    print(f"  dim_labels['{dn}']: {ct.dim_labels[dn]}")
                print(f"  years_with_data: {ct.years_with_data}")
            else:
                print(f"\n  ── {subj} ({desc}) — NOT AVAILABLE ──")

# ════════════════════════════════════════════════════════════════════════
# Q5. HOW "OGÓŁEM" CELLS WORK IN CROSS TABLES
# ════════════════════════════════════════════════════════════════════════
print(SEP)
print("Q5. HOW 'OGÓŁEM' CELLS WORK IN 2D CROSS TABLES (P2137 age×sex)")
print("=" * 100)

if target_gmina:
    tid, r = target_gmina
    ct = r.cross_tables['P2137']
    
    # Pick a year with data
    test_year = 2020 if 2020 in ct.years_with_data else ct.years_with_data[-1]
    tbl = ct.tables[test_year]
    df = ct.get_as_dataframe(test_year)
    
    print(f"\n  Gmina: {tid} — {r.name}, year={test_year}")
    print(f"  dim_names: {ct.dim_names}")
    
    # Identify which dimension is which
    d0_name = ct.dim_names[0]
    d1_name = ct.dim_names[1]
    d0_labels = ct.dim_labels[d0_name]
    d1_labels = ct.dim_labels[d1_name]
    
    print(f"\n  Rows ({d0_name}):    {d0_labels}")
    print(f"  Columns ({d1_name}): {d1_labels}")
    
    print(f"\n  Is 'ogółem' a row label?    {'ogółem' in d0_labels}")
    print(f"  Is 'ogółem' a column label? {'ogółem' in d1_labels}")
    
    # Find ogółem indices
    try:
        ogolem_row_idx = d0_labels.index('ogółem')
    except ValueError:
        ogolem_row_idx = None
    try:
        ogolem_col_idx = d1_labels.index('ogółem')
    except ValueError:
        ogolem_col_idx = None
    
    print(f"  ogółem row index: {ogolem_row_idx}")
    print(f"  ogółem col index: {ogolem_col_idx}")
    
    print(f"\n  Full table:\n{df.to_string()}")
    
    # Check: does ogółem row = sum of all other rows?
    if ogolem_row_idx is not None:
        ogolem_row = tbl[ogolem_row_idx, :]
        other_rows_sum = np.nansum(tbl[[i for i in range(tbl.shape[0]) if i != ogolem_row_idx], :], axis=0)
        print(f"\n  ── Checking ogółem ROW (index={ogolem_row_idx}) ──")
        print(f"  ogółem row values:   {ogolem_row}")
        print(f"  Sum of ALL other rows: {other_rows_sum}")
        row_match = np.allclose(ogolem_row, other_rows_sum, atol=1, equal_nan=True)
        print(f"  ogółem row ≈ sum(ALL other rows)?  {row_match}")
        if not row_match:
            diff = ogolem_row - other_rows_sum
            print(f"  Differences: {diff}")
            # P2137 has OVERLAPPING age groups: 0-14 = 0-4+5-9+10-14, and 70+ = 70-74+75-79+80-84+85+
            # So we need to exclude these aggregate rows
            print(f"\n  NOTE: P2137 has OVERLAPPING age groups!")
            print(f"  '0-14' is the aggregate of '0-4' + '5-9' + '10-14'")
            print(f"  '70 i więcej' is the aggregate of '70-74' + '75-79' + '80-84' + '85 i więcej'")
            # Identify non-overlapping rows (exclude 0-14 and 70 i więcej)  
            overlap_labels = {'0-14', '70 i więcej'}
            non_overlap_idx = [i for i in range(tbl.shape[0]) 
                              if i != ogolem_row_idx and d0_labels[i] not in overlap_labels]
            non_overlap_sum = np.nansum(tbl[non_overlap_idx, :], axis=0)
            non_overlap_labels = [d0_labels[i] for i in non_overlap_idx]
            print(f"  Non-overlapping rows ({len(non_overlap_idx)}): {non_overlap_labels}")
            print(f"  Sum of non-overlapping rows: {non_overlap_sum}")
            row_match_fixed = np.allclose(ogolem_row, non_overlap_sum, atol=1, equal_nan=True)
            print(f"  ogółem row ≈ sum(non-overlapping rows)?  {row_match_fixed}")
            if not row_match_fixed:
                print(f"  Still differs by: {ogolem_row - non_overlap_sum}")
    
    # Check: does ogółem column = sum of all other columns?
    if ogolem_col_idx is not None:
        ogolem_col = tbl[:, ogolem_col_idx]
        other_cols_sum = np.nansum(tbl[:, [i for i in range(tbl.shape[1]) if i != ogolem_col_idx]], axis=1)
        print(f"\n  ── Checking ogółem COLUMN (index={ogolem_col_idx}) ──")
        print(f"  ogółem col values:     {ogolem_col}")
        print(f"  Sum of other columns:  {other_cols_sum}")
        col_match = np.allclose(ogolem_col, other_cols_sum, atol=1, equal_nan=True)
        print(f"  ogółem col ≈ sum(other cols)?  {col_match}")
        if not col_match:
            diff = ogolem_col - other_cols_sum
            print(f"  Differences: {diff}")
    
    # Grand total check
    if ogolem_row_idx is not None and ogolem_col_idx is not None:
        grand_total_cell = tbl[ogolem_row_idx, ogolem_col_idx]
        inner_sum = np.nansum(tbl[[i for i in range(tbl.shape[0]) if i != ogolem_row_idx], :]
                               [:, [j for j in range(tbl.shape[1]) if j != ogolem_col_idx]])
        print(f"\n  ── Grand total consistency ──")
        print(f"  Cell [ogółem, ogółem] = {grand_total_cell:.0f}")
        print(f"  Sum of inner cells (excl. ogółem row & col) = {inner_sum:.0f}")
        print(f"  Match: {abs(grand_total_cell - inner_sum) < 1}")

# ════════════════════════════════════════════════════════════════════════
# Q6. M_pop__educ COVERAGE
# ════════════════════════════════════════════════════════════════════════
print(SEP)
print("Q6. M_pop__educ COVERAGE AND SAMPLE CROSS TABLES")
print("=" * 100)

# Count records with M_pop__educ
records_with_mpopeduc = []
for tid, r in records.items():
    if 'M_pop__educ' in r.cross_tables:
        ct = r.cross_tables['M_pop__educ']
        if ct.years_with_data:
            records_with_mpopeduc.append((tid, r, ct))

print(f"\n  Total records with M_pop__educ cross table: {len(records_with_mpopeduc)}")

# Level distribution
level_counts = Counter()
for tid, r, ct in records_with_mpopeduc:
    level_counts[r.level] += 1
print(f"  By level: {dict(sorted(level_counts.items()))}")

# Year coverage
all_years_educ = Counter()
for tid, r, ct in records_with_mpopeduc:
    for yr in ct.years_with_data:
        all_years_educ[yr] += 1

print(f"\n  Year coverage across all records with M_pop__educ:")
for yr in sorted(all_years_educ.keys()):
    print(f"    {yr}: {all_years_educ[yr]} records")

# Show labels from first occurrence
if records_with_mpopeduc:
    tid0, r0, ct0 = records_with_mpopeduc[0]
    print(f"\n  ── Labels (from {tid0} — {r0.name}) ──")
    print(f"  dim_names: {ct0.dim_names}")
    print(f"  shape:     {ct0.shape}")
    for dn in ct0.dim_names:
        print(f"  dim_labels['{dn}']: {ct0.dim_labels[dn]}")

# Find a gmina with M_pop__educ at 1988 and/or 2002
educ_sample_gmina = None
for tid, r, ct in records_with_mpopeduc:
    if r.level == 6:
        yrs = ct.years_with_data
        if 1988 in yrs or 2002 in yrs:
            educ_sample_gmina = (tid, r, ct)
            break

if not educ_sample_gmina:
    # try any record
    for tid, r, ct in records_with_mpopeduc:
        yrs = ct.years_with_data
        if 1988 in yrs or 2002 in yrs:
            educ_sample_gmina = (tid, r, ct)
            break

if educ_sample_gmina:
    tid, r, ct = educ_sample_gmina
    print(f"\n  ── Sample cross tables from {tid} — {r.name} (level={r.level}) ──")
    print(f"  years_with_data: {ct.years_with_data}")
    
    for yr in [1988, 2002]:
        if yr in ct.years_with_data:
            print(f"\n  === Year {yr} ===")
            df_yr = ct.get_as_dataframe(yr)
            print(df_yr.to_string())
        else:
            print(f"\n  === Year {yr} — NO DATA ===")
else:
    print("\n  No record found with M_pop__educ at 1988 or 2002")

# ════════════════════════════════════════════════════════════════════════
# Q7. BDL P2137 FOR VOIVODESHIP RECORDS
# ════════════════════════════════════════════════════════════════════════
print(SEP)
print("Q7. BDL P2137 DATA FOR VOIVODESHIP RECORDS (level=2)")
print("=" * 100)

# Check which voivodeship records have P2137
woj_records = {tid: r for tid, r in records.items() if r.level == 2}
print(f"\n  Total voivodeship (level=2) records: {len(woj_records)}")

woj_with_p2137 = []
for tid, r in woj_records.items():
    if 'P2137' in r.cross_tables:
        ct = r.cross_tables['P2137']
        if ct.years_with_data:
            woj_with_p2137.append((tid, r, ct))

print(f"  Voivodeships with P2137 cross table: {len(woj_with_p2137)}")

# Show a few
for tid, r, ct in sorted(woj_with_p2137, key=lambda x: x[0])[:5]:
    print(f"    {tid}: {r.name:25s} shape={ct.shape} years={ct.years_with_data[:3]}...{ct.years_with_data[-3:]}")

# Focus on voivodeship 0200000 (Dolnośląskie)
woj_tid = '0200000'
test_year = 2020

print(f"\n  ── Detailed analysis for {woj_tid} ──")

if woj_tid in records:
    woj_r = records[woj_tid]
    print(f"  Name: {woj_r.name}, level={woj_r.level}")
    
    if 'P2137' in woj_r.cross_tables:
        woj_ct = woj_r.cross_tables['P2137']
        woj_tbl = woj_ct.tables.get(test_year)
        
        if woj_tbl is not None and not np.all(np.isnan(woj_tbl)):
            # Find ogółem indices properly
            w_d0 = woj_ct.dim_labels[woj_ct.dim_names[0]]
            w_d1 = woj_ct.dim_labels[woj_ct.dim_names[1]]
            w_r = w_d0.index('ogółem') if 'ogółem' in w_d0 else 0
            w_c = w_d1.index('ogółem') if 'ogółem' in w_d1 else 0
            woj_total = woj_tbl[w_r, w_c] if woj_tbl.ndim == 2 else woj_tbl.flat[0]
            print(f"  P2137 ogółem ({test_year}): {woj_total:.0f} (at [{w_r},{w_c}])")
            print(f"  P2137 shape: {woj_ct.shape}")
            
            # Sum all gminas within this voivodeship (woj code = 02)
            gmina_sum = 0
            gmina_count = 0
            gmina_missing = 0
            gmina_details = []
            
            for tid2, r2 in gminas.items():
                if tid2[:2] == '02':  # Same voivodeship
                    if 'P2137' in r2.cross_tables:
                        ct2 = r2.cross_tables['P2137']
                        tbl2 = ct2.tables.get(test_year)
                        if tbl2 is not None and not np.all(np.isnan(tbl2)):
                            g_d0 = ct2.dim_labels[ct2.dim_names[0]]
                            g_d1 = ct2.dim_labels[ct2.dim_names[1]]
                            g_r = g_d0.index('ogółem') if 'ogółem' in g_d0 else 0
                            g_c = g_d1.index('ogółem') if 'ogółem' in g_d1 else 0
                            val = tbl2[g_r, g_c] if tbl2.ndim == 2 else tbl2.flat[0]
                            if not np.isnan(val):
                                gmina_sum += val
                                gmina_count += 1
                                gmina_details.append((tid2, val))
                            else:
                                gmina_missing += 1
                        else:
                            gmina_missing += 1
                    else:
                        gmina_missing += 1
            
            print(f"\n  Sum of ALL gminas in woj 02 (P2137 ogółem, {test_year}):")
            print(f"    Gminas with data: {gmina_count}")
            print(f"    Gminas missing:   {gmina_missing}")
            print(f"    Sum of gminas:    {gmina_sum:.0f}")
            print(f"    Voivodeship:      {woj_total:.0f}")
            diff = gmina_sum - woj_total
            pct = diff / woj_total * 100 if woj_total != 0 else 0
            print(f"    Difference:       {diff:.0f} ({pct:.2f}%)")
            
            if abs(pct) < 1:
                print(f"    ✓ Hierarchical aggregation holds (within 1%)")
            else:
                print(f"    ⚠ Hierarchical aggregation does NOT hold cleanly")
                
            # Also show the voivodeship cross table for 2020 (first few rows)
            print(f"\n  Voivodeship P2137 cross table ({test_year}) — first 5 rows:")
            df_woj = woj_ct.get_as_dataframe(test_year)
            print(df_woj.head(5).to_string())
        else:
            print(f"  No data for year {test_year}")
    else:
        print(f"  No P2137 cross table")
else:
    print(f"  Record {woj_tid} not found!")

print(SEP)
print("ANALYSIS COMPLETE")
print("=" * 100)
