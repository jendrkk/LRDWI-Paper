#!/usr/bin/env python3
"""
Supplementary analysis: investigate units and label ordering issues.
"""
import sys, os, pickle, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(os.path.expanduser(
    "~/Documents/Studium Volkswirschaftslehre/3. Semester/"
    "Long-run dynamics of wealth inequalities/Paper/Data/Geospatial/geoteryt_O.pkl"
))
sys.path.insert(0, str(DB_PATH.parents[2] / 'local_repo' / 'LRDWI-Paper' / 'Code' / 'tools'))
from geoTERYT_db import load_complete_database
db = load_complete_database(DB_PATH, verbose=False)
records = db._records
gminas = {tid: r for tid, r in records.items() if r.level == 6}

SEP = "\n" + "="*90

# ── A. Inspect P2137 and P4253 cross-table dimensions and labels ──
print(SEP)
print("A. CROSS-TABLE DIMENSION LABELS (P2137 vs P4253)")
print("="*90)

for subj in ['P2137', 'P4253', 'P2114', 'P2884', 'P2883', 'P2885', 'P2887',
             'P2871', 'P2402', 'P4287', 'P4315']:
    # Find one example
    for tid, r in gminas.items():
        if subj in r.cross_tables:
            ct = r.cross_tables[subj]
            if ct.years_with_data:
                print(f"\n  {subj} (from {tid}, {r.name}):")
                print(f"    dim_names: {ct.dim_names}")
                print(f"    shape: {ct.shape}")
                for dn in ct.dim_names:
                    print(f"    {dn}: {ct.dim_labels[dn]}")
                yr = ct.years_with_data[0]
                tbl = ct.tables[yr]
                # Show ogółem location
                for i, dn in enumerate(ct.dim_names):
                    labels = ct.dim_labels[dn]
                    for j, lab in enumerate(labels):
                        if 'ogółem' in lab.lower() or 'ogólne' in lab.lower() or 'ogółem' in lab:
                            print(f"    → ogółem in dim {dn} at index {j}")
                # Print the ogółem total
                if ct.ndim == 2:
                    # Find ogółem indices
                    og_idx = []
                    for dn in ct.dim_names:
                        labels = ct.dim_labels[dn]
                        found = False
                        for j, lab in enumerate(labels):
                            if 'ogółem' in lab.lower():
                                og_idx.append(j)
                                found = True
                                break
                        if not found:
                            og_idx.append(0)
                    print(f"    Value at ogółem indices {og_idx}: {tbl[tuple(og_idx)]:.0f}")
                    print(f"    Value at [0,0]: {tbl[0,0]:.0f}")
                elif ct.ndim == 1:
                    labels = ct.dim_labels[ct.dim_names[0]]
                    for j, lab in enumerate(labels):
                        if 'ogółem' in lab.lower():
                            print(f"    Value at ogółem [{j}]: {tbl[j]:.0f}")
                    print(f"    Value at [0]: {tbl[0]:.0f}")
                break

# ── B. P2137: what units? Check actual values ──
print(SEP)
print("B. P2137 UNITS CHECK")
print("="*90)

# Sample a known-size gmina
tid_sample = '0201011'
r = records.get(tid_sample)
if r and 'P2137' in r.cross_tables:
    ct = r.cross_tables['P2137']
    print(f"\n  {tid_sample} ({r.name}):")
    print(f"  pop (2020 if available): {r.pop.get(pd.Timestamp(2020,1,1), 'N/A')}")
    if 2020 in ct.years_with_data:
        tbl = ct.tables[2020]
        print(f"  P2137 full cross-table for 2020:")
        df = ct.get_as_dataframe(2020)
        print(df)

# ── C. Check the P4253 structure ──
print(SEP)
print("C. P4253 STRUCTURE CHECK")
print("="*90)

for tid, r in gminas.items():
    if 'P4253' in r.cross_tables:
        ct = r.cross_tables['P4253']
        if ct.years_with_data:
            print(f"\n  {tid} ({r.name}):")
            df = ct.get_as_dataframe(2021)
            print(f"  dim_names: {ct.dim_names}, shape: {ct.shape}")
            print(df)
            # Also print P2137 for same gmina for 2021
            if 'P2137' in r.cross_tables:
                ct2 = r.cross_tables['P2137']
                if 2021 in ct2.years_with_data:
                    df2 = ct2.get_as_dataframe(2021)
                    print(f"\n  P2137 for same gmina at 2021:")
                    print(df2)
            break

# ── D. Hierarchical aggregation: dig into the issue ──
print(SEP)
print("D. HIERARCHICAL AGGREGATION DETAILED CHECK")
print("="*90)

# Dolnośląskie - what levels exist?
print("\n  Records with teryt starting with 02:")
by_level = defaultdict(int)
for tid, r in records.items():
    if tid[:2] == '02':
        by_level[r.level] += 1
print(f"  Level distribution: {dict(sorted(by_level.items()))}")

# The voivodeship is level=2, level=5 is powiat, level=6 is gmina
# In BDL, P2137 gmina data are in 100s or 1000s? Let's check

woj_r = records['0200000']
ct_woj = woj_r.cross_tables.get('P2137')
if ct_woj:
    print(f"\n  Voivodeship P2137 (2020):")
    df_woj = ct_woj.get_as_dataframe(2020)
    print(df_woj)

# Check if P2137 for voivodeship might be in thousands
# Sum of all gminas:
gmina_totals = {}
for tid, r in gminas.items():
    if tid[:2] == '02' and 'P2137' in r.cross_tables:
        ct = r.cross_tables['P2137']
        if 2020 in ct.years_with_data:
            tbl = ct.tables[2020]
            # Find ogółem indices
            og = []
            for dn in ct.dim_names:
                labels = ct.dim_labels[dn]
                idx = next((j for j, l in enumerate(labels) if 'ogółem' in l.lower()), 0)
                og.append(idx)
            gmina_totals[tid] = tbl[tuple(og)]

print(f"\n  Number of gminas with P2137 data for 2020 in woj 02: {len(gmina_totals)}")
print(f"  Sum of gmina ogółem (at proper ogółem index): {sum(gmina_totals.values()):.0f}")

# Also check just the [0,0] sums vs proper ogółem sums
og_sum_00 = 0
og_sum_proper = 0
for tid, r in gminas.items():
    if tid[:2] == '02' and 'P2137' in r.cross_tables:
        ct = r.cross_tables['P2137']
        if 2020 in ct.years_with_data:
            tbl = ct.tables[2020]
            og_sum_00 += tbl[0, 0]
            # proper
            og = []
            for dn in ct.dim_names:
                labels = ct.dim_labels[dn]
                idx = next((j for j, l in enumerate(labels) if 'ogółem' in l.lower()), 0)
                og.append(idx)
            og_sum_proper += tbl[tuple(og)]

print(f"  Sum at [0,0]: {og_sum_00:.0f}")
print(f"  Sum at ogółem indices: {og_sum_proper:.0f}")
if ct_woj:
    tbl_woj = ct_woj.tables[2020]
    og_woj = []
    for dn in ct_woj.dim_names:
        labels = ct_woj.dim_labels[dn]
        idx = next((j for j, l in enumerate(labels) if 'ogółem' in l.lower()), 0)
        og_woj.append(idx)
    print(f"  Voivodeship at ogółem indices: {tbl_woj[tuple(og_woj)]:.0f}")
    print(f"  Voivodeship at [0,0]: {tbl_woj[0, 0]:.0f}")

# ── E. H_age_sex: check teryt_ids mapping to old voivodeships ──
print(SEP)
print("E. H_age_sex TERYT_ID ↔ OLD_WOJ_ID MAPPING")
print("="*90)

for tid, r in records.items():
    if 'H_age_sex' in r.cross_tables:
        ct = r.cross_tables['H_age_sex']
        if ct.years_with_data:
            print(f"  {tid}: {r.name}, level={r.level}, old_woj_id={r.old_woj_id}")

# ── F. P2137 vs P2114 at 2002: refine to use proper ogółem ──
print(SEP)
print("F. P2137 vs P2114 at 2002 (with proper ogółem)")  
print("="*90)

comparisons = []
for tid, r in gminas.items():
    p2137_val = None
    p2114_val = None
    if 'P2137' in r.cross_tables:
        ct = r.cross_tables['P2137']
        if 2002 in ct.years_with_data:
            tbl = ct.tables[2002]
            og = []
            for dn in ct.dim_names:
                labels = ct.dim_labels[dn]
                idx = next((j for j, l in enumerate(labels) if 'ogółem' in l.lower()), 0)
                og.append(idx)
            p2137_val = tbl[tuple(og)]
    if 'P2114' in r.cross_tables:
        ct = r.cross_tables['P2114']
        if 2002 in ct.years_with_data:
            tbl = ct.tables[2002]
            og = []
            for dn in ct.dim_names:
                labels = ct.dim_labels[dn]
                idx = next((j for j, l in enumerate(labels) if 'ogółem' in l.lower()), 0)
                og.append(idx)
            p2114_val = tbl[tuple(og)]
    if p2137_val is not None and p2114_val is not None and not np.isnan(p2137_val) and not np.isnan(p2114_val):
        comparisons.append((tid, p2137_val, p2114_val))

print(f"  Gminas with both P2137 and P2114 (proper ogółem) @ 2002: {len(comparisons)}")
if comparisons:
    diffs = [(tid, abs(a-b), a, b) for tid, a, b in comparisons]
    diffs.sort(key=lambda x: x[1])
    n_exact = sum(1 for _, d, _, _ in diffs if d < 1)
    n_close = sum(1 for _, d, _, _ in diffs if d < 100)
    mean_pct = np.mean([abs(a-b)/b*100 for tid, a, b in comparisons if b > 0])
    print(f"  Exact matches: {n_exact}, < 100 diff: {n_close}, mean pct diff: {mean_pct:.2f}%")
    print(f"  Sample (largest diffs):")
    for tid, d, a, b in diffs[-10:]:
        print(f"    {tid}: P2137={a:.0f}, P2114={b:.0f}, diff={d:.0f} ({d/b*100 if b else 0:.1f}%)")

# ── G. Pop data: check units for pop ──
print(SEP)
print("G. POP DATA UNITS")
print("="*90)

# Record.pop might be actual population while P2137 is in hundreds
# Let's check the ratio
for tid, r in list(gminas.items())[:10]:
    if r.pop.notna().any() and 'P2137' in r.cross_tables:
        ct = r.cross_tables['P2137']
        for yr in [2020, 2015, 2010]:
            ts = pd.Timestamp(yr, 1, 1)
            if yr in ct.years_with_data and ts in r.pop.index and pd.notna(r.pop[ts]):
                tbl = ct.tables[yr]
                og = []
                for dn in ct.dim_names:
                    labels = ct.dim_labels[dn]
                    idx = next((j for j, l in enumerate(labels) if 'ogółem' in l.lower()), 0)
                    og.append(idx)
                val = tbl[tuple(og)]
                ratio = r.pop[ts] / val if val > 0 else 0
                print(f"  {tid} ({r.name}), {yr}: pop={r.pop[ts]:.0f}, P2137_og={val:.0f}, ratio={ratio:.3f}")
                break

# ── H. Further check on H_age_sex: are teryt_ids same as old_woj_ids? ──
print(SEP)
print("H. OLD WOJ MAPPING: DIRECT TERYT_ID MATCH")
print("="*90)

# The old_woj_ids stored are strings like '5100000'. The H_age_sex teryt_ids are also '5100000' etc.
# So the mapping is: old_woj_id IS the teryt_id of the H_age_sex record.
h_age_sex_tids = set()
for tid, r in records.items():
    if 'H_age_sex' in r.cross_tables:
        ct = r.cross_tables['H_age_sex']
        if ct.years_with_data:
            h_age_sex_tids.add(tid)

# Check if old_woj_ids are in h_age_sex_tids
ref_ids = set()
for tid, r in gminas.items():
    if r.old_woj_id is not None:
        ref_ids.add(str(r.old_woj_id))

print(f"  H_age_sex teryt_ids: {sorted(h_age_sex_tids)}")
print(f"  old_woj_ids as strings: {sorted(ref_ids)[:10]}...")
print(f"  Overlap: {len(ref_ids & h_age_sex_tids)} / {len(ref_ids)}")
print(f"  Missing from H_age_sex: {sorted(ref_ids - h_age_sex_tids)[:5]}...")

print(SEP)
print("SUPPLEMENTARY ANALYSIS COMPLETE")
