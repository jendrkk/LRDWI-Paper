"""Investigate the GeoTERYT database for Warsaw, Mazowieckie, and Wałbrzych issues."""
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/local_repo/LRDWI-Paper/Code/tools")

out = open("/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/local_repo/LRDWI-Paper/Code/analysis/_investigate_output.txt", "w")

def write(msg):
    out.write(msg + "\n")
    out.flush()

pkl_paths = [
    "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/Geospatial/geoteryt_E.pkl",
    "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/Geospatial/geoteryt_complete_final.pkl",
    "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/Geospatial/geoteryt_v5.pkl",
]

db = None
for path in pkl_paths:
    try:
        write(f"Trying to load: {path}")
        from geoTERYT_db import load_complete_database
        db = load_complete_database(path, verbose=False)
        write(f"  SUCCESS - loaded {len(db._records)} records")
        break
    except Exception as e:
        write(f"  Failed: {type(e).__name__}: {e}")

if db is None:
    write("Trying raw pickle load...")
    for path in pkl_paths:
        try:
            with open(path, 'rb') as f:
                db = pickle.load(f)
            write(f"  Raw pickle SUCCESS from {path} - type: {type(db)}")
            if isinstance(db, dict):
                write(f"  Keys: {list(db.keys())[:20]}")
            break
        except Exception as e:
            write(f"  Failed: {e}")

if db is None:
    write("Could not load any database!")
    out.close()
    sys.exit(1)

# ==============================================================================
# 1. Investigate Warsaw
# ==============================================================================
write("\n" + "="*80)
write("SECTION 1: WARSAW INVESTIGATION")
write("="*80)

warsaw_codes = {
    '1465000': 'Warsaw powiat (current)',
    '14650011': 'Could be Warsaw urban gmina 8-digit',
    '1465001': 'Warsaw urban gmina (current, 7-digit)',
    '1431000': 'Warsaw powiat (pre-2001)',
    '1431001': 'Warsaw urban gmina (pre-2001)',
    '1400000': 'Mazowieckie voivodeship',
    '1300000': 'Warsaw agglomeration NUTS-2',
    '1500000': 'Regional Masovian NUTS-2',
}

for code, desc in warsaw_codes.items():
    code_padded = code.zfill(7)
    rec = db.get_by_teryt_id(code_padded)
    if rec:
        write(f"\n{code_padded} ({desc}): {rec.name}")
        write(f"  Level: {rec.level}, Kind: {rec.kind}, Rodz: {rec.rodz}")
        yrs = sorted(rec.years_valid)
        write(f"  Years valid: {yrs[:5]}...{yrs[-5:] if len(yrs) > 5 else ''}")
        write(f"  Historical codes: {rec.historical_codes}")
        cby_items = list(rec.code_by_year.items())
        write(f"  Code by year (first 10): {dict(cby_items[:10])}")
        write(f"  Past TERYT IDs: {rec.past_teryt_ids}")
        write(f"  Past levels: {rec.past_levels}")
        write(f"  Past kinds: {rec.past_kinds}")
        write(f"  Has changes: {rec.has_changes}, N changes: {rec.n_changes}")
        write(f"  Old woj: {rec.old_woj}, Old woj id: {rec.old_woj_id}")

        ch_keys = sorted([k for k in rec.children_ids.keys() if isinstance(k, int)])
        write(f"  Children ID year keys: {ch_keys[:15]}...")
        for yr in [1986, 1988, 1995, 1999, 2000, 2001, 2002, 2003, 2010, 2020, 2024]:
            ch = rec.children_ids.get(yr, [])
            if ch:
                write(f"    Children {yr}: {ch}")

        p_keys = sorted([k for k in rec.parent_id.keys() if isinstance(k, int)])
        write(f"  Parent ID year keys: {p_keys[:15]}...")
        for yr in [1986, 1988, 1995, 1999, 2000, 2001, 2002, 2003, 2010, 2020, 2024]:
            p = rec.parent_id.get(yr)
            if p:
                write(f"    Parent {yr}: {p}")

        write(f"  Subjects: {rec.list_subjects()[:15]}")
        write(f"  Cross tables: {rec.list_cross_tables()}")
        write(f"  N data series: {rec.n_data_series}")

        e_tables = [ct_id for ct_id in rec.list_cross_tables() if ct_id.startswith('E_')]
        write(f"  Estimated cross tables: {e_tables}")

        for ct_id in rec.list_cross_tables():
            ct = rec.get_cross_table(ct_id)
            if ct:
                write(f"\n  Cross table {ct_id}:")
                write(f"    Shape: {ct.shape}, Dim names: {ct.dim_names}")
                write(f"    Dim labels: {ct.dim_labels}")
                write(f"    Observed years: {sorted(ct.observed_years)}")
                write(f"    Years with data: {ct.years_with_data}")
                write(f"    Year-by-year sums:")
                for yr in range(1986, 2025):
                    t = ct.get_table(yr)
                    if t is not None and not np.all(np.isnan(t)):
                        total = np.nansum(t)
                        flat = t.flatten()
                        write(f"      {yr}: sum={total:.1f}, vals={np.array2string(flat[:10], precision=1, suppress_small=True)}")
    else:
        write(f"\n{code_padded} ({desc}): NOT FOUND in database")

# ==============================================================================
# 2. Mazowieckie education data
# ==============================================================================
write("\n" + "="*80)
write("SECTION 2: MAZOWIECKIE / NUTS SPLIT EDUCATION DATA")
write("="*80)

maz_codes = ['1400000', '1300000', '1500000']
for code in maz_codes:
    rec = db.get_by_teryt_id(code.zfill(7))
    if rec:
        write(f"\n{code}: {rec.name}, Level: {rec.level}, Kind: {rec.kind}")
        write(f"  Years valid: {sorted(rec.years_valid)[:5]}...{sorted(rec.years_valid)[-5:]}")
        write(f"  Children keys: {sorted([k for k in rec.children_ids.keys() if isinstance(k, int)])[:10]}")
        for yr in [1999, 2000, 2001, 2002, 2020]:
            ch = rec.children_ids.get(yr, [])
            write(f"  Children {yr}: {len(ch)} items -> {ch[:10]}")

        for ct_id in rec.list_cross_tables():
            ct = rec.get_cross_table(ct_id)
            if ct:
                obs = sorted(ct.observed_years)
                dwdata = ct.years_with_data
                write(f"  CT {ct_id}: dims={ct.dim_names}, shape={ct.shape}, observed={obs}, data_years_count={len(dwdata)}")
                # Show sums for key years
                for yr in [1988, 1995, 2000, 2002, 2005, 2010, 2015, 2020]:
                    t = ct.get_table(yr)
                    if t is not None and not np.all(np.isnan(t)):
                        write(f"    {yr}: sum={np.nansum(t):.0f}")
    else:
        write(f"\n{code}: NOT FOUND")

# ==============================================================================
# 3. Wałbrzych
# ==============================================================================
write("\n" + "="*80)
write("SECTION 3: WAŁBRZYCH")
write("="*80)

walbrzych_records = db.search_by_name("wałbrzych")
write(f"\nRecords matching 'wałbrzych': {len(walbrzych_records)}")
for rec in walbrzych_records:
    write(f"\n  {rec.teryt_id}: {rec.name} ({rec.name_dod})")
    write(f"    Level: {rec.level}, Kind: {rec.kind}, Rodz: {rec.rodz}")
    yrs = sorted(rec.years_valid)
    write(f"    Years valid: {yrs[:5]}...{yrs[-5:]}")
    write(f"    Past levels: {rec.past_levels}")
    write(f"    Past kinds: {rec.past_kinds}")
    write(f"    Historical codes: {rec.historical_codes}")
    write(f"    Code by year: {rec.code_by_year}")
    write(f"    Changes: {rec.changes}")

    for yr in [1999, 2002, 2003, 2010, 2012, 2013, 2020]:
        ch = rec.children_ids.get(yr, [])
        p = rec.parent_id.get(yr)
        if ch or p:
            write(f"    Year {yr}: parent={p}, children={ch}")

    for ct_id in rec.list_cross_tables():
        ct = rec.get_cross_table(ct_id)
        if ct:
            write(f"    CT {ct_id}: observed={sorted(ct.observed_years)}, data_years={len(ct.years_with_data)}")

# ==============================================================================
# 4. Check education coverage summary
# ==============================================================================
write("\n" + "="*80)
write("SECTION 4: EDUCATION DATA COVERAGE SUMMARY")
write("="*80)

gminas_with_educ_ct = 0
gminas_total = 0

for tid, rec in db._records.items():
    if rec.level == 6 and rec.rodz in ('1', '2', '3'):
        gminas_total += 1
        has_educ = any('educ' in ct_id.lower() for ct_id in rec.list_cross_tables())
        if has_educ:
            gminas_with_educ_ct += 1

write(f"Total gminas (rodz 1,2,3): {gminas_total}")
write(f"With education cross tables: {gminas_with_educ_ct}")

# ==============================================================================
# 5. Check country record and old voivodeships
# ==============================================================================
write("\n" + "="*80)
write("SECTION 5: COUNTRY RECORD AND OLD VOIVODESHIPS")
write("="*80)

root = db.get_by_teryt_id('0000000')
if root:
    write(f"Country record found: {root.name}")
    write(f"  Children keys: {sorted(root.children_ids.keys())[:15]}")
    for key in ['old', 'nuts']:
        ch = root.children_ids.get(key, [])
        write(f"  Children '{key}': {ch}")
    for yr in [1999, 2002, 2020]:
        ch = root.children_ids.get(yr, [])
        write(f"  Children {yr}: {ch}")

# Check old voivodships existence
old_woj_ids = [f'{i:02d}00000' for i in range(51, 100, 2)]
found_old = []
for oid in old_woj_ids:
    rec = db.get_by_teryt_id(oid)
    if rec:
        found_old.append(f"{oid}:{rec.name}")
write(f"\nOld voivodeships found: {len(found_old)}")
if found_old:
    for f in found_old[:5]:
        write(f"  {f}")

out.close()
print("Investigation complete. Output saved.")
