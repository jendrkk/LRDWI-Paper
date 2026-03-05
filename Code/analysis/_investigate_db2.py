"""Part 2: Deeper investigation of Warsaw education data."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/local_repo/LRDWI-Paper/Code/tools")
from geoTERYT_db import load_complete_database

out = open("/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/local_repo/LRDWI-Paper/Code/analysis/_investigate_output2.txt", "w")
def write(msg):
    out.write(msg + "\n")
    out.flush()

db = load_complete_database("/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/Geospatial/geoteryt_E.pkl", verbose=False)

# 1. Check Warsaw 1431001 M_educ_1990 cross table
write("=== Warsaw gmina 1431001 - M_educ_1990 ===")
rec = db.get_by_teryt_id('1431001')
if rec:
    ct = rec.get_cross_table('M_educ_1990')
    if ct:
        write(f"  Shape: {ct.shape}, Dim: {ct.dim_names}, Labels: {ct.dim_labels}")
        write(f"  Years with data: {ct.years_with_data}")
        write(f"  Observed years: {sorted(ct.observed_years)}")
        for yr in ct.years_with_data:
            t = ct.get_table(yr)
            write(f"  {yr}: {np.nansum(t):.0f} -> {t.flatten()}")
    else:
        write("  M_educ_1990 NOT FOUND!")

    # Check M_educ_2000 on 1431001
    ct2 = rec.get_cross_table('M_educ_2000')
    if ct2:
        write(f"\n  M_educ_2000 shape: {ct2.shape}, years: {ct2.years_with_data}")
    else:
        write(f"\n  M_educ_2000: NOT FOUND on 1431001!")

    # Check what P2402 has (2002 census education by sex)
    ct3 = rec.get_cross_table('P2402')
    if ct3:
        write(f"\n  P2402 shape: {ct3.shape}, years: {ct3.years_with_data}")
        for yr in ct3.years_with_data:
            t = ct3.get_table(yr)
            write(f"  P2402 {yr}: sum={np.nansum(t):.0f}")
    else:
        write(f"\n  P2402: NOT FOUND on 1431001!")

    # Check P4315 (2021 census education)
    ct4 = rec.get_cross_table('P4315')
    if ct4:
        write(f"\n  P4315 shape: {ct4.shape}, years: {ct4.years_with_data}")
    else:
        write(f"\n  P4315: NOT FOUND on 1431001!")

# 2. Check Warsaw powiat 1465000 - M_educ_2000
write("\n=== Warsaw powiat 1465000 - M_educ_2000 ===")
rec2 = db.get_by_teryt_id('1465000')
if rec2:
    ct = rec2.get_cross_table('M_educ_2000')
    if ct:
        write(f"  Shape: {ct.shape}, Years: {ct.years_with_data}")
        write(f"  Observed: {sorted(ct.observed_years)}")
        for yr in ct.years_with_data:
            t = ct.get_table(yr)
            write(f"  {yr}: sum={np.nansum(t):.0f}")
    else:
        write("  M_educ_2000 NOT FOUND!")

# 3. Check E_educ_2000 on 1465000 vs 1431001 to verify data path
write("\n=== Comparing E_educ_2000 on 1465000 (powiat) vs its child ===")
rec_pow = db.get_by_teryt_id('1465000')
if rec_pow:
    ct_p = rec_pow.get_cross_table('E_educ_2000')
    if ct_p:
        write(f"  1465000 E_educ_2000: observed={sorted(ct_p.observed_years)}, years={ct_p.years_with_data[:5]}...")
        for yr in [2002, 2005, 2010, 2015, 2020]:
            t = ct_p.get_table(yr)
            if t is not None and not np.all(np.isnan(t)):
                write(f"  1465000 {yr}: sum={np.nansum(t):.0f}, vals={t.flatten()}")
    
# 4. Check parents of 1431001 across years - is it always under 1431000?
write("\n=== Parent chain of Warsaw gmina 1431001 ===")
rec_gm = db.get_by_teryt_id('1431001')
if rec_gm:
    for yr in range(1986, 2025):
        p = rec_gm.parent_id.get(yr)
        if p:
            write(f"  {yr}: parent = {p}")

# 5. BIG FINDING: Check whether 1431001 has M_educ_2000 or E_educ_2000
write("\n=== Does 1431001 have E_educ_2000? ===")
if rec_gm:
    ct_e = rec_gm.get_cross_table('E_educ_2000')
    if ct_e:
        write(f"  YES! E_educ_2000: observed={sorted(ct_e.observed_years)}, years={ct_e.years_with_data}")
    else:
        write("  NO E_educ_2000 on 1431001!")
    
    ct_m = rec_gm.get_cross_table('M_educ_2000')
    if ct_m:
        write(f"  M_educ_2000: years={ct_m.years_with_data}")
    else:
        write("  NO M_educ_2000 on 1431001!")

# 6. Check Wałbrzych powiat/gmina detail
write("\n=== Wałbrzych details ===")
walb_records = db.search_by_name("wałbrzych")
for r in walb_records:
    write(f"\n{r.teryt_id}: {r.name} ({r.name_dod}), level={r.level}, rodz={r.rodz}")
    ct_e2 = r.get_cross_table('E_educ_2000')
    if ct_e2:
        write(f"  E_educ_2000: observed={sorted(ct_e2.observed_years)}, years_count={len(ct_e2.years_with_data)}")
        write(f"  Years with data: {ct_e2.years_with_data}")
    else:
        write("  NO E_educ_2000")
    
    ct_m2 = r.get_cross_table('M_educ_2000')
    if ct_m2:
        write(f"  M_educ_2000: years={ct_m2.years_with_data}, observed={sorted(ct_m2.observed_years)}")
    else:
        write("  NO M_educ_2000")

# 7. Check 1300000 / 1500000 - do they have children properly?
write("\n=== NUTS split 1300000 / 1500000 details ===")
for code in ['1300000', '1500000']:
    rec = db.get_by_teryt_id(code)
    if rec:
        write(f"\n{code}: {rec.name}, level={rec.level}")
        write(f"  years_valid: {sorted(rec.years_valid)[:5] if rec.years_valid else 'EMPTY'}")
        write(f"  children keys: {sorted([k for k in rec.children_ids.keys() if isinstance(k, int)])[:5]}")
        ch_2002 = rec.children_ids.get(2002, [])
        ch_2020 = rec.children_ids.get(2020, [])
        write(f"  children 2002 ({len(ch_2002)}): includes powiat IDs ending in 000")
        # Check if powiats or gminas
        powiats = [c for c in ch_2002 if c[4:] == '000']
        gminas = [c for c in ch_2002 if c[4:] != '000']
        write(f"    powiats: {powiats[:5]}")
        write(f"    gminas: {gminas[:5]}")
        
        # Check cross tables
        for ct_name in rec.list_cross_tables():
            ct = rec.get_cross_table(ct_name)
            if ct:
                write(f"  CT {ct_name}: years={len(ct.years_with_data)}")

# 8. Critical: Check if 1465000 has E_educ_2000 and where data comes from
write("\n=== Does 1465000 (powiat) get E_educ_2000 from child 1465011 or from aggregation? ===")
rec_pow = db.get_by_teryt_id('1465000')
if rec_pow:
    # Check children for 2002
    ch_2002 = rec_pow.children_ids.get(2002, [])
    write(f"  1465000 children 2002: {ch_2002}")
    for cid in ch_2002:
        crec = db.get_by_teryt_id(cid)
        if crec:
            ct_c = crec.get_cross_table('E_educ_2000')
            if ct_c:
                write(f"    {cid} ({crec.name}): E_educ_2000 years={ct_c.years_with_data[:5]}...")
            else:
                write(f"    {cid} ({crec.name}): NO E_educ_2000")

out.close()
print("Done. Output at _investigate_output2.txt")
