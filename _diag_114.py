"""Diagnostic: why Group 114 is not plotted."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent
DATA_ROOT = REPO.parent.parent / 'Data' / 'replication_package'
sys.path.insert(0, str(REPO / 'Code' / 'tools'))
import pandas as pd
import geopandas as gpd
import inequality_analyzers as ia

lis_groups = pd.read_csv(DATA_ROOT / 'LIS' / 'LIS_Groups.csv')
region_gdf = gpd.read_file(DATA_ROOT / 'geometry' / 'geometry.gpkg', layer='new_voiv_groups')

lis_groups_a = ia.LISAnalyzer(lis_groups, income_type='hitotalnet', deflator_col='deflator_2023')

YEAR = 1999
lis_yr = lis_groups_a.regional_panel(metrics=['median'])
lis_yr = lis_yr[lis_yr['year'] == YEAR]

g = region_gdf[['id']].copy()
g['v'] = g['id'].map(lis_yr.set_index('region')['median'])
n = g[g['v'].isna()]

print(f"Null count: {len(n)}")
print(f"Null IDs: {n['id'].tolist()}")
print(f"gdf id type: {type(region_gdf['id'].iloc[0]).__name__}")
print(f"lis region type: {type(lis_yr['region'].iloc[0]).__name__}")
print(f"gdf sample: {region_gdf['id'].iloc[0]!r}")
print(f"lis sample: {lis_yr['region'].iloc[0]!r}")
print(f"Group 114 in gdf: {'Group 114' in region_gdf['id'].values}")
print(f"Group 114 in lis: {'Group 114' in lis_yr['region'].values}")

g114 = region_gdf[region_gdf['id'] == 'Group 114']
print(f"G114 geom rows: {len(g114)}")
if len(g114) > 0:
    print(f"G114 geom null: {g114.geometry.isna().any()}")
    print(f"G114 geom empty: {g114.geometry.is_empty.any()}")

gdf_set = set(region_gdf['id'])
lis_set = set(lis_yr['region'])
print(f"In gdf not LIS: {gdf_set - lis_set}")
print(f"In LIS not gdf: {lis_set - gdf_set}")
print(f"Total gdf: {len(gdf_set)}, Total LIS: {len(lis_set)}")

# Check geometry area and drawing order for Group 114
print(f"\n=== Group 114 detail ===")
g114_idx = region_gdf.index[region_gdf['id'] == 'Group 114'].tolist()
print(f"G114 index position: {g114_idx}")
print(f"G114 area: {g114.geometry.area.values}")
print(f"G114 name: {g114['name'].values}")

# Check for overlapping geometries
from shapely.ops import unary_union
g114_geom = g114.geometry.values[0]
overlaps = []
for idx, row in region_gdf.iterrows():
    if row['id'] != 'Group 114' and row.geometry.intersects(g114_geom):
        overlap_area = row.geometry.intersection(g114_geom).area
        if overlap_area > 0:
            overlaps.append((row['id'], row['name'], idx, overlap_area))
print(f"\nGroups overlapping Group 114: {len(overlaps)}")
for oid, oname, oidx, oarea in overlaps[:10]:
    print(f"  {oid} ({oname}), index={oidx}, overlap_area={oarea:.6f}")

# Check if the plot draws NaN on top
print(f"\n=== Drawing order check ===")
gdf_check = region_gdf.copy()
gdf_check['val'] = gdf_check['id'].map(lis_yr.set_index('region')['median'])
has_val = gdf_check[gdf_check['val'].notna()]
no_val = gdf_check[gdf_check['val'].isna()]
print(f"Rows with values: {len(has_val)}")
print(f"Rows without values (NaN): {len(no_val)}")
# Check if any NaN row comes AFTER Group 114 in index and overlaps with it
g114_idx_val = g114_idx[0]
later_nan = no_val[no_val.index > g114_idx_val]
later_overlapping = [r['id'] for _, r in later_nan.iterrows() if r.geometry.intersects(g114_geom)]
print(f"NaN rows drawn AFTER Group 114 that overlap it: {later_overlapping[:5]}")
