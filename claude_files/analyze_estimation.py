"""
Deep analysis of estimation results in geoteryt_E.pkl
Examines: spline behavior, observed vs estimated, hierarchical consistency,
plausibility of predictions, spikes/drops, and parent-level data preservation.
"""
import sys, os
import numpy as np
import pandas as pd
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Code', 'tools'))

from geoTERYT_db import (
    load_complete_database,
    LEVEL_GMINA, LEVEL_POWIAT, LEVEL_VOIVODESHIP,
    RODZ_AGGREGATION_SET,
)

DATA_ROOT = os.path.join(REPO, '..', '..', 'Data', 'Geospatial')

print("Loading geoteryt_E.pkl...")
db_e = load_complete_database(os.path.join(DATA_ROOT, 'geoteryt_E.pkl'))
print(f"  {len(db_e._records)} records loaded")

print("\nLoading geoteryt_O.pkl (pre-estimation)...")
db_o = load_complete_database(os.path.join(DATA_ROOT, 'geoteryt_O.pkl'))
print(f"  {len(db_o._records)} records loaded")

E_SIDS = [
    'E_age_sex_2000', 'E_age_sex_1990',
    'E_educ_2000', 'E_educ_1990',
    'E_educ_sex_2000', 'E_educ_sex_1990',
    'E_hh_size_2000', 'E_hh_size_1990',
]

E_TO_ANCHOR = {
    'E_age_sex_2000':  ['M_age_sex'],
    'E_age_sex_1990':  ['M_age_sex', 'M_age_1990'],
    'E_educ_2000':     ['M_educ_2000'],
    'E_educ_1990':     ['M_educ_1990'],
    'E_educ_sex_2000': ['M_educ_sex_2000'],
    'E_educ_sex_1990': ['M_educ_sex_1990'],
    'E_hh_size_2000':  ['M_hh_size_2000'],
    'E_hh_size_1990':  ['M_hh_size_1990'],
}

YEAR_RANGE_2000 = list(range(1999, 2026))
YEAR_RANGE_1990 = list(range(1986, 2003))

def get_years(e_sid):
    return YEAR_RANGE_2000 if '2000' in e_sid else YEAR_RANGE_1990

# ============================================================================
# ANALYSIS 1: Parent-level data — observed in O vs estimated in E
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 1: Parent E_ data vs original M_ data (observed)")
print("="*70)

for e_sid in E_SIDS:
    anchors = E_TO_ANCHOR[e_sid]
    print(f"\n--- {e_sid} ---")
    
    # Check voivodeships
    voivs_with_M = 0
    voivs_with_E = 0
    voivs_mismatch = 0
    total_year_checks = 0
    
    for tid, rec_e in db_e._records.items():
        if rec_e.level != LEVEL_VOIVODESHIP:
            continue
        
        e_ct = rec_e.cross_tables.get(e_sid)
        if e_ct is None:
            continue
        voivs_with_E += 1
        
        # Check original DB for M_ data
        rec_o = db_o._records.get(tid)
        if rec_o is None:
            continue
        
        has_any_M = False
        for m_sid in anchors:
            m_ct = rec_o.cross_tables.get(m_sid)
            if m_ct is None:
                continue
            m_years = m_ct.years_with_data
            if m_years:
                has_any_M = True
                voivs_with_M += 1
                
                # Compare: does E_ match M_ at observed years?
                for y in m_years:
                    if y not in get_years(e_sid):
                        continue
                    total_year_checks += 1
                    m_tbl = m_ct.tables[y]
                    e_tbl = e_ct.tables.get(y)
                    if e_tbl is None:
                        continue
                    
                    # Check if shapes match
                    if m_tbl.shape != e_tbl.shape:
                        # Different shapes — skip comparison
                        continue
                    
                    m_total = np.nansum(m_tbl)
                    e_total = np.nansum(e_tbl)
                    if m_total > 0:
                        pct_diff = abs(e_total - m_total) / m_total * 100
                        if pct_diff > 0.1:
                            voivs_mismatch += 1
                            if voivs_mismatch <= 3:
                                print(f"  MISMATCH voiv {tid} ({rec_e.name}) year={y}: "
                                      f"M_total={m_total:.0f} E_total={e_total:.0f} "
                                      f"diff={pct_diff:.2f}%")
                break  # only need to check first anchor
    
    print(f"  Voivs with E_: {voivs_with_E}, with M_: {voivs_with_M}")
    print(f"  Year-level checks: {total_year_checks}, mismatches (>0.1%): {voivs_mismatch}")


# ============================================================================
# ANALYSIS 2: Spline interpolation plausibility — detect spikes/drops
# ============================================================================
print("\n\n" + "="*70)
print("ANALYSIS 2: Temporal plausibility — detect spikes/drops in E_ data")
print("="*70)

def detect_temporal_anomalies(tables_dict, years, threshold_pct=20.0):
    """Detect year-over-year changes exceeding threshold_pct in total."""
    anomalies = []
    prev_total = None
    prev_year = None
    for y in sorted(years):
        tbl = tables_dict.get(y)
        if tbl is None:
            prev_total = None
            continue
        total = np.nansum(tbl)
        if total < 1:
            prev_total = None
            continue
        if prev_total is not None and prev_total > 0:
            pct_change = (total - prev_total) / prev_total * 100
            if abs(pct_change) > threshold_pct:
                anomalies.append((prev_year, y, pct_change, prev_total, total))
        prev_total = total
        prev_year = y
    return anomalies

spike_counts = defaultdict(int)
worst_spikes = []

for tid, rec in db_e._records.items():
    if rec.level != LEVEL_GMINA or rec.rodz not in RODZ_AGGREGATION_SET:
        continue
    
    for e_sid in E_SIDS:
        ct = rec.cross_tables.get(e_sid)
        if ct is None:
            continue
        years = get_years(e_sid)
        anomalies = detect_temporal_anomalies(ct.tables, years, threshold_pct=25.0)
        for a in anomalies:
            spike_counts[e_sid] += 1
            worst_spikes.append((abs(a[2]), e_sid, tid, rec.name, a))

# Sort worst
worst_spikes.sort(reverse=True)

print(f"\nSpike counts (>25% year-over-year change) per subject:")
for e_sid in E_SIDS:
    print(f"  {e_sid:<25s}: {spike_counts[e_sid]:>6d}")

print(f"\nTop 20 worst spikes:")
for i, (pct, e_sid, tid, name, (y1, y2, chg, t1, t2)) in enumerate(worst_spikes[:20]):
    print(f"  {e_sid:<22s} {tid} {name:20s} {y1}→{y2}: "
          f"{t1:.0f}→{t2:.0f} ({chg:+.1f}%)")


# ============================================================================
# ANALYSIS 3: Shape analysis — do interpolated values overshoot/undershoot?
# ============================================================================
print("\n\n" + "="*70)
print("ANALYSIS 3: Cubic spline overshooting (non-monotone between anchors)")
print("="*70)

# For E_educ_2000 and E_hh_size_2000 (few anchors: 2002, 2011, 2021)
# Check if interpolated years between anchors go above/below both anchors

import random
random.seed(42)

# Sample 200 gminas
gmina_tids = [
    tid for tid, rec in db_e._records.items()
    if rec.level == LEVEL_GMINA and rec.rodz in RODZ_AGGREGATION_SET
]
sample_tids = random.sample(gmina_tids, min(200, len(gmina_tids)))

overshoot_counts = defaultdict(int)
undershoot_counts = defaultdict(int)
checked_counts = defaultdict(int)

for tid in sample_tids:
    rec_e = db_e._records[tid]
    rec_o = db_o._records.get(tid)
    if rec_o is None:
        continue
    
    for e_sid in ['E_educ_2000', 'E_hh_size_2000', 'E_educ_sex_2000']:
        ct_e = rec_e.cross_tables.get(e_sid)
        if ct_e is None:
            continue
        
        # Find anchor years from M_ subject
        anchors_sids = E_TO_ANCHOR[e_sid]
        anchor_years = set()
        for m_sid in anchors_sids:
            m_ct = rec_o.cross_tables.get(m_sid)
            if m_ct:
                anchor_years.update(m_ct.years_with_data)
        
        anchor_years = sorted(y for y in anchor_years if y in get_years(e_sid))
        if len(anchor_years) < 2:
            continue
        
        # Check between consecutive anchors
        for ai in range(len(anchor_years) - 1):
            y1, y2 = anchor_years[ai], anchor_years[ai + 1]
            t1 = np.nansum(ct_e.tables.get(y1, np.array([np.nan])))
            t2 = np.nansum(ct_e.tables.get(y2, np.array([np.nan])))
            if t1 < 1 or t2 < 1:
                continue
            
            lo, hi = min(t1, t2), max(t1, t2)
            checked_counts[e_sid] += 1
            
            for y in range(y1 + 1, y2):
                t = np.nansum(ct_e.tables.get(y, np.array([np.nan])))
                if t > hi * 1.05:
                    overshoot_counts[e_sid] += 1
                    break
                if t < lo * 0.95:
                    undershoot_counts[e_sid] += 1
                    break

print(f"\nOvershoot/undershoot for subjects with few anchors (5% tolerance):")
for e_sid in ['E_educ_2000', 'E_hh_size_2000', 'E_educ_sex_2000']:
    checked = checked_counts[e_sid]
    over = overshoot_counts[e_sid]
    under = undershoot_counts[e_sid]
    print(f"  {e_sid:<25s}: checked={checked:>5d}, overshoot={over:>5d} ({100*over/(checked+1):.1f}%), "
          f"undershoot={under:>5d} ({100*under/(checked+1):.1f}%)")


# ============================================================================
# ANALYSIS 4: Hierarchical consistency at powiat level
# ============================================================================
print("\n\n" + "="*70)
print("ANALYSIS 4: Hierarchical consistency — E_ powiat vs Σ E_ gminas")
print("="*70)

from geoTERYT_db import filter_aggregation_children

inconsistency_counts = defaultdict(int)
total_checks = defaultdict(int)
worst_inconsistencies = []

for tid, rec in db_e._records.items():
    if rec.level != LEVEL_POWIAT:
        continue
    
    for e_sid in E_SIDS:
        ct_pow = rec.cross_tables.get(e_sid)
        if ct_pow is None:
            continue
        
        years = get_years(e_sid)
        
        # Get children
        child_tids = rec.children_ids.get(1999, [])
        if not child_tids:
            continue
        
        # Filter to rodz 1,2,3
        child_gminas = [
            c for c in child_tids
            if c in db_e._records and db_e._records[c].rodz in RODZ_AGGREGATION_SET
        ]
        
        for y in years:
            pow_tbl = ct_pow.tables.get(y)
            if pow_tbl is None or np.all(np.isnan(pow_tbl)):
                continue
            
            total_checks[e_sid] += 1
            
            # Sum children
            child_sum = np.zeros_like(pow_tbl)
            for c_tid in child_gminas:
                c_ct = db_e._records[c_tid].cross_tables.get(e_sid)
                if c_ct is None:
                    continue
                c_tbl = c_ct.tables.get(y)
                if c_tbl is not None and not np.all(np.isnan(c_tbl)):
                    child_sum += np.nan_to_num(c_tbl, nan=0.0)
            
            pow_total = np.nansum(pow_tbl)
            child_total = np.nansum(child_sum)
            
            if pow_total > 0:
                pct_diff = abs(child_total - pow_total) / pow_total * 100
                if pct_diff > 0.1:
                    inconsistency_counts[e_sid] += 1
                    if len(worst_inconsistencies) < 10:
                        worst_inconsistencies.append(
                            (pct_diff, e_sid, tid, rec.name, y, pow_total, child_total)
                        )

print(f"\nHierarchical consistency check (E_ powiat vs Σ E_ gminas):")
for e_sid in E_SIDS:
    t = total_checks[e_sid]
    i = inconsistency_counts[e_sid]
    print(f"  {e_sid:<25s}: checks={t:>6d}, inconsistent(>0.1%)={i:>5d}")

if worst_inconsistencies:
    print(f"\nSample inconsistencies:")
    for pct, e_sid, tid, name, year, pt, ct in sorted(worst_inconsistencies, reverse=True)[:5]:
        print(f"  {e_sid} {tid} {name} yr={year}: powiat={pt:.0f} Σgminas={ct:.0f} diff={pct:.2f}%")


# ============================================================================
# ANALYSIS 5: Observed data preservation at gmina level
# ============================================================================
print("\n\n" + "="*70)
print("ANALYSIS 5: Census year data preservation (E_ vs M_ at gmina level)")
print("="*70)

census_preservation = defaultdict(lambda: {'match': 0, 'mismatch': 0, 'details': []})

for tid in sample_tids[:100]:
    rec_e = db_e._records[tid]
    rec_o = db_o._records.get(tid)
    if rec_o is None:
        continue
    
    for e_sid in E_SIDS:
        ct_e = rec_e.cross_tables.get(e_sid)
        if ct_e is None:
            continue
        
        for m_sid in E_TO_ANCHOR[e_sid]:
            m_ct = rec_o.cross_tables.get(m_sid)
            if m_ct is None:
                continue
            
            for y in m_ct.years_with_data:
                if y not in get_years(e_sid):
                    continue
                
                m_tbl = m_ct.tables[y]
                e_tbl = ct_e.tables.get(y)
                if e_tbl is None:
                    continue
                if m_tbl.shape != e_tbl.shape:
                    continue
                
                m_total = np.nansum(m_tbl)
                e_total = np.nansum(e_tbl)
                
                if m_total > 0:
                    pct_diff = abs(e_total - m_total) / m_total * 100
                    if pct_diff < 0.01:
                        census_preservation[e_sid]['match'] += 1
                    else:
                        census_preservation[e_sid]['mismatch'] += 1
                        if len(census_preservation[e_sid]['details']) < 3:
                            census_preservation[e_sid]['details'].append(
                                (tid, rec_e.name, y, m_total, e_total, pct_diff, m_sid)
                            )

print(f"\nCensus data preservation at gmina level (100 sampled gminas):")
for e_sid in E_SIDS:
    d = census_preservation[e_sid]
    total = d['match'] + d['mismatch']
    if total == 0:
        print(f"  {e_sid:<25s}: no overlapping data")
        continue
    print(f"  {e_sid:<25s}: match={d['match']:>5d}, mismatch={d['mismatch']:>5d} "
          f"({100*d['mismatch']/total:.1f}%)")
    for detail in d['details'][:2]:
        tid, name, y, m_t, e_t, pct, m_sid = detail
        print(f"    {tid} {name} yr={y}: M({m_sid})={m_t:.0f} E={e_t:.0f} diff={pct:.2f}%")


# ============================================================================
# ANALYSIS 6: Anchor years and interpolation coverage
# ============================================================================
print("\n\n" + "="*70)
print("ANALYSIS 6: Anchor years per subject (what data do gminas actually have?)")
print("="*70)

for e_sid in E_SIDS:
    anchors = E_TO_ANCHOR[e_sid]
    year_counts = defaultdict(int)
    n_gminas = 0
    anchor_count_dist = defaultdict(int)
    
    for tid in sample_tids[:200]:
        rec_o = db_o._records.get(tid)
        if rec_o is None:
            continue
        if rec_o.level != LEVEL_GMINA or rec_o.rodz not in RODZ_AGGREGATION_SET:
            continue
        
        n_anchors_for_this = 0
        for m_sid in anchors:
            m_ct = rec_o.cross_tables.get(m_sid)
            if m_ct is None:
                continue
            for y in m_ct.years_with_data:
                if y in get_years(e_sid):
                    year_counts[y] += 1
                    n_anchors_for_this += 1
        
        n_gminas += 1
        anchor_count_dist[n_anchors_for_this] += 1
    
    print(f"\n  {e_sid}:")
    print(f"    Gminas sampled: {n_gminas}")
    print(f"    Anchor count distribution: ", end="")
    for k in sorted(anchor_count_dist.keys()):
        print(f"{k}anchors={anchor_count_dist[k]} ", end="")
    print()
    
    if year_counts:
        sorted_yc = sorted(year_counts.items())
        # Show which years have data
        years_str = [f"{y}:{c}" for y, c in sorted_yc if c > 10]
        print(f"    Top years: {', '.join(years_str[:15])}")


# ============================================================================
# ANALYSIS 7: Category-level anomalies
# ============================================================================
print("\n\n" + "="*70)
print("ANALYSIS 7: Negative values and NaN in E_ data")
print("="*70)

neg_counts = defaultdict(int)
nan_counts = defaultdict(int)
total_tables = defaultdict(int)

for tid, rec in db_e._records.items():
    if rec.level != LEVEL_GMINA or rec.rodz not in RODZ_AGGREGATION_SET:
        continue
    
    for e_sid in E_SIDS:
        ct = rec.cross_tables.get(e_sid)
        if ct is None:
            continue
        for y in get_years(e_sid):
            tbl = ct.tables.get(y)
            if tbl is None:
                continue
            if np.all(np.isnan(tbl)):
                nan_counts[e_sid] += 1
            elif np.any(tbl < -0.01):
                neg_counts[e_sid] += 1
            total_tables[e_sid] += 1

print(f"\nNegative/NaN tables at gmina level:")
for e_sid in E_SIDS:
    t = total_tables[e_sid]
    print(f"  {e_sid:<25s}: total={t:>8d}, negative={neg_counts[e_sid]:>5d}, "
          f"all-NaN={nan_counts[e_sid]:>5d}")


print("\n\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
