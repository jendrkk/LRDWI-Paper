#!/usr/bin/env python3
"""
Deep analysis of GeoTERYT database: data dependencies, cross-variable links, pitfalls.
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import warnings
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

# Helper: get all level-6 records
gminas = {tid: r for tid, r in records.items() if r.level == 6}
print(f"Level-6 (gmina) records: {len(gminas)}")

SEPARATOR = "\n" + "=" * 90

# ════════════════════════════════════════════════════════════════════════
# 1. HISTORICAL TERYT CODES IMPACT ASSESSMENT
# ════════════════════════════════════════════════════════════════════════
print(SEPARATOR)
print("1. HISTORICAL TERYT CODES IMPACT ASSESSMENT")
print("=" * 90)

# 1a. How many level=6 records have historical_codes with more than 1 entry?
hist_counts = {tid: len(r.historical_codes) for tid, r in gminas.items()}
multi_hist = {tid: n for tid, n in hist_counts.items() if n > 1}
print(f"\n1a. Level-6 records with >1 historical_codes: {len(multi_hist)} / {len(gminas)}")
print(f"    Distribution of historical_codes count:")
cnt = Counter(hist_counts.values())
for k in sorted(cnt.keys()):
    print(f"      {k} codes: {cnt[k]} gminas")

# 1b. Census subjects: data vs no-data for current teryt_ids
census_subjects = {
    1988: ['P2887', 'P2884', 'P2885', 'P2883'],
    2002: ['P2871', 'P2402', 'P2114'],
    2021: ['P4287', 'P4315', 'P4253'],
}

print(f"\n1b-c. Census subject data coverage at gmina level (current teryt_id):")

# Build a quick lookup: which teryt_ids exist as records in the DB
all_teryt_ids_set = set(records.keys())

for census_year, subjects in census_subjects.items():
    print(f"\n  ── Census {census_year} ──")
    for subj in subjects:
        has_data = []
        no_data = []
        for tid, r in gminas.items():
            # Check cross_tables first, then data dict
            found = False
            if subj in r.cross_tables:
                ct = r.cross_tables[subj]
                tbl = ct.tables.get(census_year)
                if tbl is not None and not np.all(np.isnan(tbl)):
                    found = True
            if not found:
                # Check DataSeries
                for key, ds in r.data.items():
                    if key[1] == subj:
                        val = ds.values
                        if isinstance(val, pd.Series):
                            ts = pd.Timestamp(year=census_year, month=1, day=1)
                            if ts in val.index and pd.notna(val[ts]):
                                found = True
                                break
                        elif isinstance(val, np.ndarray):
                            idx = census_year - 1988
                            if 0 <= idx < len(val) and not np.isnan(val[idx]):
                                found = True
                                break
            if found:
                has_data.append(tid)
            else:
                no_data.append(tid)

        # 1c. For no-data gminas: how many have a historical_code that IS a teryt_id of a record with data?
        resolvable = 0
        for tid in no_data:
            r = gminas[tid]
            for hc in r.historical_codes:
                if hc == tid:
                    continue
                if hc in records:
                    hr = records[hc]
                    h_found = False
                    if subj in hr.cross_tables:
                        ct = hr.cross_tables[subj]
                        tbl = ct.tables.get(census_year)
                        if tbl is not None and not np.all(np.isnan(tbl)):
                            h_found = True
                    if not h_found:
                        for key, ds in hr.data.items():
                            if key[1] == subj:
                                val = ds.values
                                if isinstance(val, pd.Series):
                                    ts = pd.Timestamp(year=census_year, month=1, day=1)
                                    if ts in val.index and pd.notna(val[ts]):
                                        h_found = True
                                        break
                    if h_found:
                        resolvable += 1
                        break  # one match is enough

        print(f"    {subj}: has_data={len(has_data)}, no_data={len(no_data)}, "
              f"resolvable_via_hist={resolvable}")

# ════════════════════════════════════════════════════════════════════════
# 2. CROSS-VARIABLE CONSISTENCY CHECKS
# ════════════════════════════════════════════════════════════════════════
print(SEPARATOR)
print("2. CROSS-VARIABLE CONSISTENCY CHECKS")
print("=" * 90)

# 2a. Compare P2137 ogółem totals with census subjects at census years
# P2137 is age×sex cross table; ogółem total is at index [0,0] typically
# Census subjects to compare: P2114 (2002), P3304 (?), P4253 (2021)
# Let's check P2114 at 2002, P4253 at 2021 

census_compare = [
    (2002, 'P2114'),
    (2021, 'P4253'),
]

print("\n2a. P2137 vs census subject ogółem totals:")
for cyr, csubj in census_compare:
    comparisons = []
    for tid, r in gminas.items():
        # Get P2137 ogółem for census year
        p2137_val = None
        if 'P2137' in r.cross_tables:
            ct = r.cross_tables['P2137']
            tbl = ct.tables.get(cyr)
            if tbl is not None and not np.all(np.isnan(tbl)):
                # ogółem should be first element in each dimension (index 0)
                p2137_val = tbl[0, 0] if tbl.ndim == 2 else tbl.flat[0]

        # Get census subject ogółem
        census_val = None
        if csubj in r.cross_tables:
            ct = r.cross_tables[csubj]
            tbl = ct.tables.get(cyr)
            if tbl is not None and not np.all(np.isnan(tbl)):
                census_val = tbl[0, 0] if tbl.ndim >= 2 else (tbl[0] if tbl.ndim == 1 else tbl.flat[0])
        if not census_val:
            # Try DataSeries
            for key, ds in r.data.items():
                if key[1] == csubj:
                    val = ds.values
                    cats = ds.categories if hasattr(ds, 'categories') else {}
                    # Only take ogółem
                    is_ogolem = all(v.lower() == 'ogółem' for v in cats.values()) if cats else True
                    if is_ogolem and isinstance(val, pd.Series):
                        ts = pd.Timestamp(year=cyr, month=1, day=1)
                        if ts in val.index and pd.notna(val[ts]):
                            census_val = val[ts]
                            break

        if p2137_val is not None and census_val is not None and not np.isnan(p2137_val) and not np.isnan(census_val):
            comparisons.append((tid, p2137_val, census_val))

    print(f"\n  {csubj} @ {cyr}: {len(comparisons)} gminas with both P2137 and {csubj}")
    if comparisons:
        sample = comparisons[:50]
        matches = 0
        mismatches = []
        for tid, p_val, c_val in sample:
            if abs(p_val - c_val) < 1:
                matches += 1
            else:
                mismatches.append((tid, p_val, c_val, abs(p_val - c_val)))
        print(f"    Of first 50: {matches} match exactly, {len(mismatches)} differ")
        if mismatches:
            print(f"    First 10 mismatches:")
            for tid, pv, cv, diff in mismatches[:10]:
                pct = diff / cv * 100 if cv != 0 else float('inf')
                print(f"      {tid}: P2137={pv:.0f}, {csubj}={cv:.0f}, diff={diff:.0f} ({pct:.1f}%)")

# 2b. Compare record.pop with P2137 ogółem
print("\n2b. record.pop vs P2137 ogółem cross-table totals:")

# First – what units does pop contain in? Let's sample a few to understand
print("  Sampling pop values vs P2137 ogółem to understand units:")
sampled = 0
for tid, r in gminas.items():
    if sampled >= 5:
        break
    if hasattr(r, 'pop') and isinstance(r.pop, pd.Series) and r.pop.notna().any():
        if 'P2137' in r.cross_tables:
            ct = r.cross_tables['P2137']
            for yr in ct.years_with_data:
                ts = pd.Timestamp(year=yr, month=1, day=1)
                if ts in r.pop.index and pd.notna(r.pop[ts]):
                    tbl = ct.tables[yr]
                    ogolem = tbl[0, 0] if tbl.ndim == 2 else tbl.flat[0]
                    if not np.isnan(ogolem):
                        print(f"    {tid} ({r.name}): year={yr}, pop={r.pop[ts]:.0f}, P2137_ogółem={ogolem:.0f} (×1000)")
                        sampled += 1
                        break

# Now do the full comparison after understanding units
pop_comparisons = defaultdict(list)
for tid, r in gminas.items():
    pop_years = {}
    if hasattr(r, 'pop') and isinstance(r.pop, pd.Series):
        for ts in r.pop.dropna().index:
            pop_years[ts.year] = r.pop[ts]
    if 'P2137' in r.cross_tables:
        ct = r.cross_tables['P2137']
        for yr in ct.years_with_data:
            if yr in pop_years:
                tbl = ct.tables[yr]
                ogolem = tbl[0, 0] if tbl.ndim == 2 else tbl.flat[0]
                if not np.isnan(ogolem):
                    pop_comparisons[yr].append((tid, pop_years[yr], ogolem))

if pop_comparisons:
    all_years = sorted(pop_comparisons.keys())
    print(f"  Years with overlapping pop & P2137 data: {all_years}")
    for yr in [all_years[0], all_years[len(all_years)//2], all_years[-1]]:
        pairs = pop_comparisons[yr]
        n_match = sum(1 for _, pv, ov in pairs if abs(pv - ov) < 1)
        n_close = sum(1 for _, pv, ov in pairs if abs(pv - ov) < 10)
        diffs = [abs(pv - ov) for _, pv, ov in pairs]
        print(f"  Year {yr}: {len(pairs)} gminas, exact match={n_match}, "
              f"<10 diff={n_close}, max_diff={max(diffs):.0f}, mean_diff={np.mean(diffs):.1f}")
else:
    print("  No overlapping years found between pop and P2137")

# ════════════════════════════════════════════════════════════════════════
# 3. INFORMATION OVERLAP MATRIX
# ════════════════════════════════════════════════════════════════════════
print(SEPARATOR)
print("3. INFORMATION OVERLAP MATRIX")
print("=" * 90)

# For each record, collect which subjects have data and for which years
subject_records = defaultdict(set)  # subject -> set of teryt_ids
subject_years = defaultdict(set)    # subject -> set of years with data

for tid, r in records.items():
    # Cross tables
    for subj_id, ct in r.cross_tables.items():
        yrs = ct.years_with_data
        if yrs:
            subject_records[subj_id].add(tid)
            subject_years[subj_id].update(yrs)
    # DataSeries
    for key, ds in r.data.items():
        src, subj_id, var_id = key
        val = ds.values
        if isinstance(val, pd.Series):
            non_nan_years = {ts.year for ts in val.dropna().index}
        else:
            non_nan_years = set()
        if non_nan_years:
            subject_records[subj_id].add(tid)
            subject_years[subj_id].update(non_nan_years)

all_subjects = sorted(subject_records.keys())
print(f"\nSubjects found: {len(all_subjects)}")
for s in all_subjects:
    yrs = sorted(subject_years[s])
    yr_str = f"{yrs[0]}-{yrs[-1]}" if yrs else "none"
    print(f"  {s}: {len(subject_records[s])} records, years {yr_str} ({len(yrs)} years)")

# Pairwise overlap
print(f"\nPairwise subject overlap (records with BOTH):")
# Focus on key subjects
key_subjects = ['P2137', 'P2884', 'P2883', 'P2885', 'P2887', 'P2871', 'P2402', 'P2114',
                'P4287', 'P4315', 'P4253', 'P2350', 'P4092', 'H_age_sex', 'H_educ_age']
key_subjects = [s for s in key_subjects if s in subject_records]

print(f"  {'':20s}", end='')
for s2 in key_subjects:
    print(f"{s2:>10s}", end='')
print()
for s1 in key_subjects:
    print(f"  {s1:20s}", end='')
    for s2 in key_subjects:
        overlap = len(subject_records[s1] & subject_records[s2])
        print(f"{overlap:10d}", end='')
    print()

# Year overlap for key pairs
print(f"\n  Year overlaps for select pairs:")
pairs_to_check = [
    ('P2137', 'P2884'), ('P2137', 'P2114'), ('P2137', 'P4253'),
    ('P2884', 'P2883'), ('P2884', 'P2885'), ('P2137', 'H_age_sex'),
    ('P2350', 'P4092'), ('H_educ_age', 'P2884'),
]
for s1, s2 in pairs_to_check:
    if s1 in subject_years and s2 in subject_years:
        yr_overlap = sorted(subject_years[s1] & subject_years[s2])
        print(f"    {s1} ∩ {s2}: {len(yr_overlap)} years: {yr_overlap}")

# ════════════════════════════════════════════════════════════════════════
# 4. OLD VOIVODESHIP MAPPING COMPLETENESS
# ════════════════════════════════════════════════════════════════════════
print(SEPARATOR)
print("4. OLD VOIVODESHIP MAPPING COMPLETENESS")
print("=" * 90)

gminas_with_old_woj = {tid: r for tid, r in gminas.items() if r.old_woj_id is not None}
gminas_without_old_woj = {tid: r for tid, r in gminas.items() if r.old_woj_id is None}
print(f"\n  Gminas with old_woj set: {len(gminas_with_old_woj)} / {len(gminas)}")
print(f"  Gminas missing old_woj: {len(gminas_without_old_woj)}")

if gminas_without_old_woj:
    print(f"  Missing gminas (first 20):")
    for tid, r in list(gminas_without_old_woj.items())[:20]:
        print(f"    {tid}: {r.name} (valid: {sorted(r.years_valid)[:3]}...)")

# Count per old voivodeship
old_woj_counts = Counter()
old_woj_names = {}
for tid, r in gminas_with_old_woj.items():
    old_woj_counts[r.old_woj_id] += 1
    old_woj_names[r.old_woj_id] = r.old_woj

print(f"\n  Old voivodeship gmina counts ({len(old_woj_counts)} voivodeships):")
for wid in sorted(old_woj_counts.keys()):
    print(f"    {str(wid):>5s} ({old_woj_names.get(wid, '?'):25s}): {old_woj_counts[wid]} gminas")

# Check H_age_sex coverage
print(f"\n  H_age_sex coverage check:")
h_age_sex_tids = set()
for tid, r in records.items():
    if 'H_age_sex' in r.cross_tables:
        ct = r.cross_tables['H_age_sex']
        if ct.years_with_data:
            h_age_sex_tids.add(tid)
print(f"    Records with H_age_sex data: {len(h_age_sex_tids)}")
print(f"    H_age_sex teryt_ids: {sorted(h_age_sex_tids)[:10]}... (showing first 10)")

# The old_woj_ids referenced by gminas
referenced_old_woj_ids = set(old_woj_counts.keys())
# The H_age_sex records are at level=2 (old voivodeships). Their teryt_ids might be indices like 01-49
# or 2-digit codes. Let's check.
h_levels = set()
for tid in h_age_sex_tids:
    if tid in records:
        h_levels.add(records[tid].level)
print(f"    H_age_sex record levels: {h_levels}")

# Map old_woj_id to H_age_sex records
# The H_age_sex records might use old_woj_id as their teryt_id or some other mapping
# Let's check if old_woj_ids are covered
h_old_woj_ids = set()
for tid in h_age_sex_tids:
    r = records[tid]
    if r.old_woj_id is not None:
        h_old_woj_ids.add(r.old_woj_id)
print(f"    old_woj_ids of H_age_sex records: {sorted(h_old_woj_ids)}")
print(f"    old_woj_ids referenced by gminas: {sorted(referenced_old_woj_ids)}")
missing_in_h = referenced_old_woj_ids - h_old_woj_ids
covered_in_h = referenced_old_woj_ids & h_old_woj_ids
print(f"    Covered: {len(covered_in_h)}, Missing from H_age_sex: {len(missing_in_h)}")
if missing_in_h:
    print(f"    Missing IDs: {sorted(missing_in_h)}")
    for mid in sorted(missing_in_h):
        print(f"      {str(mid):>5s}: {old_woj_names.get(mid, '?')} ({old_woj_counts[mid]} gminas)")

# ════════════════════════════════════════════════════════════════════════
# 5. POPULATION DATA COVERAGE POST-HISTORICAL-TERYT FIX
# ════════════════════════════════════════════════════════════════════════
print(SEPARATOR)
print("5. POPULATION DATA COVERAGE")
print("=" * 90)

# Valid gminas per year
valid_per_year = Counter()
pop_per_year = Counter()
for tid, r in gminas.items():
    for yr in r.years_valid:
        valid_per_year[yr] += 1
    if hasattr(r, 'pop') and isinstance(r.pop, pd.Series):
        for ts in r.pop.dropna().index:
            pop_per_year[ts.year] += 1

all_years_sorted = sorted(set(valid_per_year.keys()) | set(pop_per_year.keys()))
print(f"\n  Year | Valid gminas | With pop | Coverage")
print(f"  {'─'*50}")
for yr in all_years_sorted:
    v = valid_per_year.get(yr, 0)
    p = pop_per_year.get(yr, 0)
    frac = p / v * 100 if v > 0 else 0
    marker = " <<<" if yr in [1988, 1995, 2002] else ""
    print(f"  {yr}   {v:6d}       {p:6d}     {frac:5.1f}%{marker}")

# ════════════════════════════════════════════════════════════════════════
# 6. P2350/P4092 (VOIVODESHIP EDUCATION) COVERAGE
# ════════════════════════════════════════════════════════════════════════
print(SEPARATOR)
print("6. P2350/P4092 VOIVODESHIP EDUCATION COVERAGE")
print("=" * 90)

for subj in ['P2350', 'P4092']:
    print(f"\n  {subj}:")
    tids_with_data = []
    for tid, r in records.items():
        has = False
        if subj in r.cross_tables:
            ct = r.cross_tables[subj]
            if ct.years_with_data:
                has = True
                tids_with_data.append((tid, r.name, r.level, ct.years_with_data))
        if not has:
            for key, ds in r.data.items():
                if key[1] == subj:
                    val = ds.values
                    if isinstance(val, pd.Series) and val.notna().any():
                        yrs = sorted([ts.year for ts in val.dropna().index])
                        tids_with_data.append((tid, r.name, r.level, yrs))
                        break

    print(f"    Records with data: {len(tids_with_data)}")
    for tid, name, level, yrs in sorted(tids_with_data, key=lambda x: x[0]):
        print(f"      {tid} (L{level}): {name:30s} years: {yrs}")

# ════════════════════════════════════════════════════════════════════════
# 7. MARGINAL CONSISTENCY ACROSS LEVELS
# ════════════════════════════════════════════════════════════════════════
print(SEPARATOR)
print("7. MARGINAL CONSISTENCY: VOIVODESHIP vs SUM OF GMINAS")
print("=" * 90)

# Pick voivodeship 0200000 (Dolnośląskie)
woj_tid = '0200000'
test_year = 2020

print(f"\n  Testing voivodeship {woj_tid} for year {test_year}")

if woj_tid in records:
    woj_r = records[woj_tid]
    print(f"  Voivodeship: {woj_r.name} (level={woj_r.level})")

    # Get voivodeship P2137 ogółem
    woj_ogolem = None
    if 'P2137' in woj_r.cross_tables:
        ct = woj_r.cross_tables['P2137']
        tbl = ct.tables.get(test_year)
        if tbl is not None and not np.all(np.isnan(tbl)):
            woj_ogolem = tbl[0, 0] if tbl.ndim == 2 else tbl.flat[0]
            print(f"  Voivodeship P2137 ogółem ({test_year}): {woj_ogolem:.0f}")
            print(f"  Voivodeship P2137 dim_names: {ct.dim_names}")
            print(f"  Voivodeship P2137 shape: {ct.shape}")
        else:
            print(f"  Voivodeship has no P2137 data for {test_year}")
    else:
        print(f"  Voivodeship has no P2137 cross table")

    # Sum all gminas in this voivodeship
    gmina_sum = 0
    gmina_count = 0
    gmina_missing = 0
    for tid, r in gminas.items():
        if tid[:2] == '02':  # Same voivodeship code
            if 'P2137' in r.cross_tables:
                ct = r.cross_tables['P2137']
                tbl = ct.tables.get(test_year)
                if tbl is not None and not np.all(np.isnan(tbl)):
                    val = tbl[0, 0] if tbl.ndim == 2 else tbl.flat[0]
                    if not np.isnan(val):
                        gmina_sum += val
                        gmina_count += 1
                    else:
                        gmina_missing += 1
                else:
                    gmina_missing += 1
            else:
                gmina_missing += 1

    print(f"  Sum of gmina P2137 ogółem: {gmina_sum:.0f} ({gmina_count} gminas, {gmina_missing} missing)")
    if woj_ogolem is not None:
        diff = gmina_sum - woj_ogolem
        pct = diff / woj_ogolem * 100 if woj_ogolem != 0 else float('inf')
        print(f"  Difference: {diff:.0f} ({pct:.2f}%)")
        if abs(pct) > 1:
            print(f"  ⚠ WARNING: Hierarchical aggregation does NOT hold cleanly!")
        else:
            print(f"  ✓ Hierarchical aggregation holds (within 1%)")
else:
    print(f"  Record {woj_tid} not found!")

# Also test for a powiat level
print(f"\n  Also testing at powiat level for voivodeship 02:")
powiat_sum = 0
powiat_count = 0
for tid, r in records.items():
    if tid[:2] == '02' and r.level == 5:  # powiat level
        if 'P2137' in r.cross_tables:
            ct = r.cross_tables['P2137']
            tbl = ct.tables.get(test_year)
            if tbl is not None and not np.all(np.isnan(tbl)):
                val = tbl[0, 0] if tbl.ndim == 2 else tbl.flat[0]
                if not np.isnan(val):
                    powiat_sum += val
                    powiat_count += 1
print(f"  Sum of powiat P2137 ogółem: {powiat_sum:.0f} ({powiat_count} powiats)")

# ════════════════════════════════════════════════════════════════════════
# 8. H_educ_age 3D CROSS TABLE STRUCTURE
# ════════════════════════════════════════════════════════════════════════
print(SEPARATOR)
print("8. H_educ_age 3D CROSS TABLE STRUCTURE")
print("=" * 90)

# Find H_educ_age
h_educ_age_found = False
for tid, r in records.items():
    if 'H_educ_age' in r.cross_tables:
        ct = r.cross_tables['H_educ_age']
        print(f"\n  Found H_educ_age on record {tid} ({r.name}, level={r.level})")
        print(f"  Subject ID: {ct.subject_id}")
        print(f"  Subject name: {ct.subject_name}")
        print(f"  ndim: {ct.ndim}")
        print(f"  dim_names: {ct.dim_names}")
        print(f"  shape: {ct.shape}")
        print(f"  year_range: {ct.year_range[0]}-{ct.year_range[-1]}")
        print(f"  years_with_data: {ct.years_with_data}")
        print(f"  years_missing: {ct.years_missing[:5]}... ({len(ct.years_missing)} total)")

        for dim_name in ct.dim_names:
            labels = ct.dim_labels[dim_name]
            print(f"\n  Dimension '{dim_name}' ({len(labels)} labels):")
            for i, lab in enumerate(labels):
                print(f"    [{i}] {lab}")

        # Print actual data for 1988
        if 1988 in ct.years_with_data:
            tbl = ct.tables[1988]
            print(f"\n  Data for 1988 (shape={tbl.shape}):")
            print(f"  Total (ogółem × ogółem × ogółem if 3D): {tbl[0,0,0] if tbl.ndim==3 else tbl[0,0] if tbl.ndim==2 else tbl[0]:.0f}")
            # Print as DataFrame
            df = ct.get_as_dataframe(1988)
            print(f"\n  DataFrame representation (first 30 rows, first 10 cols):")
            with pd.option_context('display.max_rows', 30, 'display.max_columns', 10, 'display.width', 120):
                print(df.iloc[:30, :10])

        print(f"\n  Bridge potential: This tensor links EDUCATION AND AGE, allowing")
        print(f"  cross-dimensional estimation. Can serve as seed for IPF across")
        print(f"  education×age when only marginals are available at gmina level.")
        h_educ_age_found = True

if not h_educ_age_found:
    print("  H_educ_age NOT found in any record!")
    # Check if it's in DataSeries instead
    for tid, r in records.items():
        for key in r.data:
            if 'H_educ_age' in str(key):
                print(f"  Found as DataSeries: {key} on {tid}")
                break

print(SEPARATOR)
print("ANALYSIS COMPLETE")
print("=" * 90)
