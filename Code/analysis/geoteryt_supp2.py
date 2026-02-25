#!/usr/bin/env python3
"""Supplementary checks with proper ogółem indexing."""
import sys, os, numpy as np, pandas as pd
from pathlib import Path
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

def get_ogolem(ct, yr):
    """Get the ogółem total from a cross table for a given year."""
    tbl = ct.tables.get(yr)
    if tbl is None or np.all(np.isnan(tbl)):
        return None
    og = []
    for d in ct.dim_names:
        idx = next((j for j, l in enumerate(ct.dim_labels[d]) if 'ogółem' in l.lower()), 0)
        og.append(idx)
    v = tbl[tuple(og)]
    return v if not np.isnan(v) else None

# ── F. P2137 vs P2114 at 2002 ──
print("F. P2137 vs P2114 at 2002 (proper ogółem)")
comps = []
for tid, r in gminas.items():
    a = get_ogolem(r.cross_tables['P2137'], 2002) if 'P2137' in r.cross_tables else None
    b = get_ogolem(r.cross_tables['P2114'], 2002) if 'P2114' in r.cross_tables else None
    if a is not None and b is not None:
        comps.append((tid, a, b))
n_exact = sum(1 for _, a, b in comps if abs(a - b) < 1)
n_1pct = sum(1 for _, a, b in comps if abs(a - b) / max(b, 1) * 100 < 1)
pct_diffs = [abs(a - b) / b * 100 for _, a, b in comps if b > 0]
print(f"  N={len(comps)}, exact_match={n_exact}, within_1%={n_1pct}")
print(f"  mean_pct_diff={np.mean(pct_diffs):.2f}%, median={np.median(pct_diffs):.2f}%, max={max(pct_diffs):.2f}%")
top = sorted(comps, key=lambda x: abs(x[1] - x[2]), reverse=True)[:5]
for tid, a, b in top:
    print(f"    {tid}: P2137={a:.0f}, P2114={b:.0f}, diff={abs(a - b):.0f} ({abs(a - b) / b * 100:.1f}%)")

# P2137 vs P4253 at 2021 (proper ogółem)
print("\n  P2137 vs P4253 at 2021 (proper ogółem)")
comps2 = []
for tid, r in gminas.items():
    a = get_ogolem(r.cross_tables['P2137'], 2021) if 'P2137' in r.cross_tables else None
    b = get_ogolem(r.cross_tables['P4253'], 2021) if 'P4253' in r.cross_tables else None
    if a is not None and b is not None:
        comps2.append((tid, a, b))
n_exact2 = sum(1 for _, a, b in comps2 if abs(a - b) < 1)
n_1pct2 = sum(1 for _, a, b in comps2 if abs(a - b) / max(b, 1) * 100 < 1)
pct_diffs2 = [abs(a - b) / b * 100 for _, a, b in comps2 if b > 0]
print(f"  N={len(comps2)}, exact_match={n_exact2}, within_1%={n_1pct2}")
print(f"  mean_pct_diff={np.mean(pct_diffs2):.2f}%, median={np.median(pct_diffs2):.2f}%, max={max(pct_diffs2):.2f}%")
top2 = sorted(comps2, key=lambda x: abs(x[1] - x[2]), reverse=True)[:5]
for tid, a, b in top2:
    print(f"    {tid}: P2137={a:.0f}, P4253={b:.0f}, diff={abs(a - b):.0f} ({abs(a - b) / b * 100:.1f}%)")

# ── G. Pop vs P2137 ──
print("\nG. Pop vs P2137 (proper ogółem) - ratio analysis")
ratios = []
for tid, r in gminas.items():
    if not (r.pop.notna().any() and 'P2137' in r.cross_tables):
        continue
    ct = r.cross_tables['P2137']
    for yr in ct.years_with_data:
        ts = pd.Timestamp(yr, 1, 1)
        if ts in r.pop.index and pd.notna(r.pop[ts]):
            v = get_ogolem(ct, yr)
            if v and v > 0:
                ratios.append((tid, yr, r.pop[ts], v, r.pop[ts] / v))
                break
if ratios:
    rs = [r[-1] for r in ratios]
    print(f"  N={len(ratios)} gminas sampled")
    print(f"  Ratio pop/P2137_og: mean={np.mean(rs):.4f}, std={np.std(rs):.4f}, min={min(rs):.4f}, max={max(rs):.4f}")
    print(f"  Sample:")
    for tid, yr, pop, og, ratio in ratios[:8]:
        print(f"    {tid} ({records[tid].name}), {yr}: pop={pop:.0f}, P2137_og={og:.0f}, ratio={ratio:.4f}")

# ── H. Hierarchical aggregation ──
print("\nH. Hierarchical aggregation: voivodeship vs sum of gminas (proper ogółem)")
for wcode, wname in [('02', 'Dolnoslaskie'), ('04', 'Kuj-Pom'), ('06', 'Lubelskie'),
                     ('10', 'Lodzkie'), ('14', 'Mazowieckie'), ('22', 'Pomorskie'),
                     ('24', 'Slaskie'), ('30', 'Wielkopolskie'), ('32', 'Zachodniopom')]:
    wid = wcode + '00000'
    if wid in records and 'P2137' in records[wid].cross_tables:
        wv = get_ogolem(records[wid].cross_tables['P2137'], 2020)
        if wv is None:
            continue
        gs = 0
        cnt = 0
        for tid, r in gminas.items():
            if tid[:2] == wcode and 'P2137' in r.cross_tables:
                v = get_ogolem(r.cross_tables['P2137'], 2020)
                if v:
                    gs += v
                    cnt += 1
        ratio = gs / wv
        print(f"  {wid} ({records[wid].name:25s}): woj={wv:.0f}, sum_gminas({cnt})={gs:.0f}, ratio={ratio:.4f}")

# Also check children hierarchy
print("\n  Direct children of woj 0200000:")
woj_r = records['0200000']
print(f"  Children: {len(woj_r.children_ids)}")
for cid in woj_r.children_ids[:5]:
    cr = records.get(cid)
    if cr:
        print(f"    {cid}: {cr.name} (level={cr.level})")

# ── I. Merged subjects ──
print("\nI. Merged (M_) subjects in database:")
m_subjects = set()
for tid in records:
    for s in records[tid].cross_tables:
        if s.startswith('M_'):
            m_subjects.add(s)
for subj in sorted(m_subjects):
    ct_count = sum(1 for tid in records if subj in records[tid].cross_tables and records[tid].cross_tables[subj].years_with_data)
    # Get one example
    for tid in records:
        if subj in records[tid].cross_tables:
            ct = records[tid].cross_tables[subj]
            if ct.years_with_data:
                print(f"  {subj}: dims={ct.dim_names}, shape={ct.shape}, years={ct.years_with_data[:5]}..., records={ct_count}")
                for dn in ct.dim_names:
                    print(f"    {dn}: {ct.dim_labels[dn]}")
                break

print("\nDone.")
