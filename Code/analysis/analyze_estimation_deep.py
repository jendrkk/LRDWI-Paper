"""
Deep follow-up analysis: targeted investigation of issues found in round 1.
"""
import sys, os
import numpy as np
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Code', 'tools'))

from geoTERYT_db import (
    load_complete_database,
    LEVEL_GMINA, LEVEL_POWIAT, LEVEL_VOIVODESHIP,
    RODZ_AGGREGATION_SET,
)

DATA_ROOT = os.path.join(REPO, '..', '..', 'Data', 'Geospatial')

print("Loading databases...")
db_e = load_complete_database(os.path.join(DATA_ROOT, 'geoteryt_E.pkl'))
db_o = load_complete_database(os.path.join(DATA_ROOT, 'geoteryt_O.pkl'))

E_SIDS = [
    'E_age_sex_2000', 'E_age_sex_1990',
    'E_educ_2000', 'E_educ_1990',
    'E_educ_sex_2000', 'E_educ_sex_1990',
    'E_hh_size_2000', 'E_hh_size_1990',
]

# ============================================================================
# DEEP 1: Check is_observed flag on CrossTables at all levels
# ============================================================================
print("\n" + "="*70)
print("DEEP 1: is_observed flags at different admin levels")
print("="*70)

for level, level_name in [(LEVEL_GMINA, 'GMINA'), (LEVEL_POWIAT, 'POWIAT'), 
                           (LEVEL_VOIVODESHIP, 'VOIVODESHIP')]:
    print(f"\n  {level_name} level:")
    for e_sid in E_SIDS:
        n_with = 0
        n_observed = 0
        n_not_observed = 0
        n_no_attr = 0
        for tid, rec in db_e._records.items():
            if rec.level != level:
                continue
            if level == LEVEL_GMINA and rec.rodz not in RODZ_AGGREGATION_SET:
                continue
            ct = rec.cross_tables.get(e_sid)
            if ct is None:
                continue
            n_with += 1
            if hasattr(ct, 'is_observed'):
                if ct.is_observed:
                    n_observed += 1
                else:
                    n_not_observed += 1
            else:
                n_no_attr += 1
        print(f"    {e_sid:<24s}: has_ct={n_with:>5d}, observed={n_observed:>5d}, "
              f"NOT_observed={n_not_observed:>5d}, no_attr={n_no_attr:>5d}")

# Also check old voivodeships (level might vary)
print(f"\n  OLD VOIVODESHIPS:")
for e_sid in E_SIDS:
    n_with = 0
    n_observed = 0
    n_not_observed = 0
    n_no_attr = 0
    for tid, rec in db_e._records.items():
        if not hasattr(rec, 'old_woj') or rec.old_woj is None:
            continue
        # Old voivodeships have a specific pattern
        if rec.level == LEVEL_VOIVODESHIP:
            continue
        if rec.level == LEVEL_POWIAT:
            continue
        if rec.level == LEVEL_GMINA:
            continue
        ct = rec.cross_tables.get(e_sid)
        if ct is None:
            continue
        n_with += 1
        if hasattr(ct, 'is_observed'):
            if ct.is_observed:
                n_observed += 1
            else:
                n_not_observed += 1
        else:
            n_no_attr += 1
    if n_with > 0:
        print(f"    {e_sid:<24s}: has_ct={n_with:>5d}, observed={n_observed:>5d}, "
              f"NOT_observed={n_not_observed:>5d}, no_attr={n_no_attr:>5d}")

# Check what levels exist in the DB
print(f"\n  All levels in DB:")
level_counts = defaultdict(int)
for tid, rec in db_e._records.items():
    level_counts[rec.level] += 1
for lev, cnt in sorted(level_counts.items()):
    print(f"    level={lev}: {cnt} records")


# ============================================================================
# DEEP 2: Wesoła extreme spike investigation  
# ============================================================================
print("\n" + "="*70)
print("DEEP 2: Wesoła (1412031) age_sex_1990 spike investigation")
print("="*70)

rec_e = db_e._records.get('1412031')
rec_o = db_o._records.get('1412031')

if rec_e:
    print(f"  Name: {rec_e.name}, level: {rec_e.level}, rodz: {rec_e.rodz}")
    ct = rec_e.cross_tables.get('E_age_sex_1990')
    if ct:
        print(f"  E_age_sex_1990 tables available for years: {sorted(ct.tables.keys())}")
        for y in range(1999, 2003):
            tbl = ct.tables.get(y)
            if tbl is not None:
                print(f"    year {y}: shape={tbl.shape}, total={np.nansum(tbl):.0f}")
        # Also check 2002 specifically
        tbl_2002 = ct.tables.get(2002)
        if tbl_2002 is not None:
            print(f"    year 2002: total={np.nansum(tbl_2002):.0f}, "
                  f"max_cell={np.nanmax(tbl_2002):.0f}, non-zero={np.count_nonzero(tbl_2002)}")
    
    # Check M_ data for this gmina
    if rec_o:
        for m_sid in ['M_age_sex', 'M_age_1990']:
            m_ct = rec_o.cross_tables.get(m_sid)
            if m_ct:
                print(f"  Original {m_sid}: years={m_ct.years_with_data}")
                for y in [2001, 2002]:
                    tbl = m_ct.tables.get(y)
                    if tbl is not None:
                        print(f"    year {y}: total={np.nansum(tbl):.0f}")

# Also check if Wesoła is part of Warsaw (merged gmina)
print(f"\n  Wesoła parent info:")
if rec_e:
    for yr_key in [1999, 2002]:
        parent = getattr(rec_e, 'parent_id', None)
        print(f"    parent_id: {parent}")
    # Check if this TID had boundary changes
    print(f"    children_ids keys: {list(rec_e.children_ids.keys()) if rec_e.children_ids else 'none'}")


# ============================================================================
# DEEP 3: educ_2000 census modification — WHY is scaling modifying census data?
# ============================================================================
print("\n" + "="*70)
print("DEEP 3: educ_2000 census data modification analysis")
print("="*70)

# Pick a specific gmina and compare M_ and E_ in detail
sample_gminas = []
for tid, rec in db_e._records.items():
    if rec.level == LEVEL_GMINA and rec.rodz in RODZ_AGGREGATION_SET:
        ct_e = rec.cross_tables.get('E_educ_2000')
        if ct_e is None:
            continue
        rec_o_check = db_o._records.get(tid)
        if rec_o_check:
            m_ct = rec_o_check.cross_tables.get('M_educ_2000')
            if m_ct and 2002 in m_ct.years_with_data:
                sample_gminas.append(tid)
    if len(sample_gminas) >= 5:
        break

for tid in sample_gminas:
    rec_e_g = db_e._records[tid]
    rec_o_g = db_o._records[tid]
    
    m_ct = rec_o_g.cross_tables.get('M_educ_2000')
    e_ct = rec_e_g.cross_tables.get('E_educ_2000')
    
    print(f"\n  {tid} ({rec_e_g.name}):")
    
    for y in [2002, 2011, 2021]:
        m_tbl = m_ct.tables.get(y)
        e_tbl = e_ct.tables.get(y) if e_ct else None
        
        if m_tbl is not None and e_tbl is not None:
            m_total = np.nansum(m_tbl)
            e_total = np.nansum(e_tbl)
            ratios = np.where(m_tbl > 0, e_tbl / m_tbl, np.nan)
            print(f"    yr={y}: M_total={m_total:.0f}, E_total={e_total:.0f}, "
                  f"diff={100*(e_total-m_total)/m_total:+.2f}%, "
                  f"cell_ratio_range=[{np.nanmin(ratios):.4f}, {np.nanmax(ratios):.4f}]")
        elif m_tbl is not None:
            print(f"    yr={y}: M_total={np.nansum(m_tbl):.0f}, E=missing")
        elif e_tbl is not None:
            print(f"    yr={y}: M=missing, E_total={np.nansum(e_tbl):.0f}")

# Check: does the voivodeship have M_educ_2000 data?
print(f"\n  Voivodeship M_educ_2000 data availability:")
for tid, rec in db_o._records.items():
    if rec.level != LEVEL_VOIVODESHIP:
        continue
    m_ct = rec.cross_tables.get('M_educ_2000')
    if m_ct and m_ct.years_with_data:
        print(f"    voiv {tid} ({rec.name}): M_educ_2000 years={m_ct.years_with_data[:5]}...")


# ============================================================================
# DEEP 4: 2001→2002 spike pattern in educ_2000 — is it boundary-change related?
# ============================================================================
print("\n" + "="*70)
print("DEEP 4: 2001→2002 educ_2000 drops — pattern analysis")
print("="*70)

spike_gminas = []
for tid, rec in db_e._records.items():
    if rec.level != LEVEL_GMINA or rec.rodz not in RODZ_AGGREGATION_SET:
        continue
    
    ct = rec.cross_tables.get('E_educ_2000')
    if ct is None:
        continue
    
    t_2001 = ct.tables.get(2001)
    t_2002 = ct.tables.get(2002)
    
    if t_2001 is None or t_2002 is None:
        continue
    
    s1 = np.nansum(t_2001)
    s2 = np.nansum(t_2002)
    
    if s1 > 100 and s2 > 100:
        pct = (s2 - s1) / s1 * 100
        if abs(pct) > 15:
            spike_gminas.append((abs(pct), pct, tid, rec.name, s1, s2, rec.rodz))

spike_gminas.sort(reverse=True)

print(f"  Gminas with >15% 2001→2002 change in E_educ_2000: {len(spike_gminas)}")
print(f"\n  Top 30 spikes:")
for idx, (_, pct, tid, name, s1, s2, rodz) in enumerate(spike_gminas[:30]):
    # Get voivodeship
    voiv_code = tid[:2]
    print(f"    {tid} {name:25s} rodz={rodz} voiv={voiv_code}: "
          f"{s1:.0f}→{s2:.0f} ({pct:+.1f}%)")

# Check if these gminas are disproportionately from one voivodeship
voiv_dist = defaultdict(int)
for _, _, tid, _, _, _, _ in spike_gminas:
    voiv_dist[tid[:2]] += 1
print(f"\n  Voivodeship distribution of spikes:")
for voiv, cnt in sorted(voiv_dist.items(), key=lambda x: -x[1])[:10]:
    print(f"    voiv {voiv}: {cnt} gminas")


# ============================================================================
# DEEP 5: Hierarchical inconsistency — what's going on at powiat level?
# ============================================================================
print("\n" + "="*70)
print("DEEP 5: Hierarchical inconsistency at powiat level (top examples)")
print("="*70)

# Analyze the worst cases in detail
for e_sid in ['E_age_sex_2000', 'E_educ_2000']:
    print(f"\n  --- {e_sid} ---")
    worst = []
    
    for tid, rec in db_e._records.items():
        if rec.level != LEVEL_POWIAT:
            continue
        
        ct_pow = rec.cross_tables.get(e_sid)
        if ct_pow is None:
            continue
        
        # Pick a recent year
        for y in [2021, 2020, 2019]:
            pow_tbl = ct_pow.tables.get(y)
            if pow_tbl is None or np.all(np.isnan(pow_tbl)):
                continue
            
            child_tids = rec.children_ids.get(1999, [])
            child_gminas = [
                c for c in child_tids
                if c in db_e._records and db_e._records[c].rodz in RODZ_AGGREGATION_SET
            ]
            
            child_sum = np.zeros_like(pow_tbl)
            n_children_with_data = 0
            for c_tid in child_gminas:
                c_ct = db_e._records[c_tid].cross_tables.get(e_sid)
                if c_ct is None:
                    continue
                c_tbl = c_ct.tables.get(y)
                if c_tbl is not None and not np.all(np.isnan(c_tbl)):
                    child_sum += np.nan_to_num(c_tbl, nan=0.0)
                    n_children_with_data += 1
            
            pow_total = np.nansum(pow_tbl)
            child_total = np.nansum(child_sum)
            
            if pow_total > 0:
                pct_diff = (child_total - pow_total) / pow_total * 100
                if abs(pct_diff) > 0.5:
                    worst.append((abs(pct_diff), pct_diff, tid, rec.name, y, 
                                  pow_total, child_total, len(child_gminas), n_children_with_data))
            break  # only one year per powiat
    
    worst.sort(reverse=True)
    for idx, (_, pct, tid, name, y, pt, ct_val, n_ch, n_ch_data) in enumerate(worst[:10]):
        print(f"    {tid} {name:25s} yr={y}: pow={pt:.0f} Σgm={ct_val:.0f} "
              f"diff={pct:+.1f}% (children={n_ch}, with_data={n_ch_data})")


# ============================================================================
# DEEP 6: Check if synthetic 2011 anchor exists for educ_2000
# ============================================================================
print("\n" + "="*70)
print("DEEP 6: Does M_educ_2000 have synthetic 2011 data at gmina level?")
print("="*70)

has_2011 = 0
has_2002 = 0
has_2021 = 0
total_gminas = 0

for tid, rec in db_o._records.items():
    if rec.level != LEVEL_GMINA or rec.rodz not in RODZ_AGGREGATION_SET:
        continue
    total_gminas += 1
    m_ct = rec.cross_tables.get('M_educ_2000')
    if m_ct is None:
        continue
    yrs = set(m_ct.years_with_data)
    if 2002 in yrs:
        has_2002 += 1
    if 2011 in yrs:
        has_2011 += 1
    if 2021 in yrs:
        has_2021 += 1

print(f"  Total rodz 1,2,3 gminas: {total_gminas}")
print(f"  With M_educ_2000 2002: {has_2002}")
print(f"  With M_educ_2000 2011: {has_2011}")
print(f"  With M_educ_2000 2021: {has_2021}")

# For voivodeships:
print(f"\n  At voivodeship level:")
for tid, rec in db_o._records.items():
    if rec.level != LEVEL_VOIVODESHIP:
        continue
    for m_sid in ['M_educ_2000', 'M_educ_1990', 'M_educ_sex_2000', 'M_educ_sex_1990',
                  'M_hh_size_2000', 'M_hh_size_1990']:
        m_ct = rec.cross_tables.get(m_sid)
        if m_ct and m_ct.years_with_data:
            print(f"    voiv {tid[:4]}.. {m_sid}: years={m_ct.years_with_data}")
            break
    else:
        continue
    break  # just show first voivodeship


# ============================================================================
# DEEP 7: Data availability for powiats, voivs in the original DB (M_ subjects)
# ============================================================================
print("\n" + "="*70)
print("DEEP 7: M_ data availability at powiat & voivodeship level")
print("="*70)

for level, level_name in [(LEVEL_POWIAT, 'POWIAT'), (LEVEL_VOIVODESHIP, 'VOIV')]:
    print(f"\n  {level_name} level:")
    for m_sid in ['M_age_sex', 'M_age_1990', 'M_educ_2000', 'M_educ_1990', 
                  'M_educ_sex_2000', 'M_educ_sex_1990', 'M_hh_size_2000', 'M_hh_size_1990']:
        n_with = 0
        example_years = set()
        for tid, rec in db_o._records.items():
            if rec.level != level:
                continue
            m_ct = rec.cross_tables.get(m_sid)
            if m_ct and m_ct.years_with_data:
                n_with += 1
                example_years.update(m_ct.years_with_data[:5])
        yr_str = sorted(example_years)[:10] if example_years else []
        print(f"    {m_sid:<24s}: {n_with:>4d} units, sample_years={yr_str}")


# ============================================================================
# DEEP 8: Check _aggregate_to_parents behavior — powiat E_ = Σ gmina E_?
# ============================================================================
print("\n" + "="*70)
print("DEEP 8: Is powiat E_ exactly Σ gmina E_? (testing aggregate_to_parents)")
print("="*70)

for e_sid in ['E_age_sex_2000', 'E_educ_2000']:
    print(f"\n  --- {e_sid} ---")
    exact_match = 0
    close_match = 0
    mismatch = 0
    total = 0
    
    for tid, rec in db_e._records.items():
        if rec.level != LEVEL_POWIAT:
            continue
        
        ct_pow = rec.cross_tables.get(e_sid)
        if ct_pow is None:
            continue
        
        pow_tbl = ct_pow.tables.get(2010)
        if pow_tbl is None:
            continue
        
        child_tids = rec.children_ids.get(1999, [])
        child_gminas = [c for c in child_tids
                       if c in db_e._records and db_e._records[c].rodz in RODZ_AGGREGATION_SET]
        
        sum_tbl = np.zeros_like(pow_tbl)
        for c_tid in child_gminas:
            c_ct = db_e._records[c_tid].cross_tables.get(e_sid)
            if c_ct:
                c_tbl = c_ct.tables.get(2010)
                if c_tbl is not None:
                    sum_tbl += np.nan_to_num(c_tbl, nan=0.0)
        
        total += 1
        if np.allclose(pow_tbl, sum_tbl, atol=0.01):
            exact_match += 1
        elif np.allclose(pow_tbl, sum_tbl, rtol=0.01):
            close_match += 1
        else:
            mismatch += 1
            if mismatch <= 3:
                pt = np.nansum(pow_tbl)
                st = np.nansum(sum_tbl)
                print(f"    MISMATCH {tid} ({rec.name}): pow={pt:.0f} Σgm={st:.0f} "
                      f"diff={100*(st-pt)/pt:+.2f}%")
    
    print(f"    yr=2010: total={total}, exact_match={exact_match}, "
          f"close={close_match}, mismatch={mismatch}")


print("\n\n" + "="*70)
print("DEEP ANALYSIS COMPLETE")
print("="*70)
