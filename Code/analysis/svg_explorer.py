import xml.etree.ElementTree as ET
from pathlib import Path

geo_root = Path("/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/Geospatial")

# Parse file_1
file_1_path = geo_root / "Poland_administrative_divisions_comparison_map_(1999_and_pre-1999).svg"
tree = ET.parse(file_1_path)
root = tree.getroot()

print("="*60)
print("FILE 1 STRUCTURE:")
print("="*60)
print(f"Root tag: {root.tag}")
print(f"Children count: {len(list(root))}")

# Count path elements and their styles
paths = root.findall('.//{http://www.w3.org/2000/svg}path')
print(f"\nTotal path elements: {len(paths)}")

# Analyze stroke-widths
stroke_widths = {}
fills = {}
for p in paths:
    style = p.get('style', '')
    # Extract stroke-width
    if 'stroke-width:' in style:
        sw = style.split('stroke-width:')[1].split(';')[0]
        stroke_widths[sw] = stroke_widths.get(sw, 0) + 1
    # Extract fill
    if 'fill:' in style:
        f = style.split('fill:')[1].split(';')[0]
        fills[f] = fills.get(f, 0) + 1
    elif p.get('fill'):
        f = p.get('fill')
        fills[f] = fills.get(f, 0) + 1

print("\nStroke widths found:")
for sw, cnt in sorted(stroke_widths.items()):
    print(f"  {sw}: {cnt} elements")

print("\nFill colors (first 30):")
for f, cnt in sorted(fills.items(), key=lambda x: -x[1])[:30]:
    print(f"  {f}: {cnt} elements")

# Look for groups
groups = root.findall('.//{http://www.w3.org/2000/svg}g')
print(f"\nTotal groups: {len(groups)}")
for g in groups[:10]:
    g_id = g.get('id', 'no-id')
    label = g.get('{http://www.inkscape.org/namespaces/inkscape}label', 'no-label')
    children = len(list(g))
    print(f"  Group: id={g_id}, label={label}, children={children}")

print("\n" + "="*60)
print("FILE 2 STRUCTURE:")
print("="*60)

# Parse file_2
file_2_path = geo_root / "Poland_administrative_divisions_(1999).svg"
tree2 = ET.parse(file_2_path)
root2 = tree2.getroot()

print(f"Root tag: {root2.tag}")
print(f"Children count: {len(list(root2))}")

# Count path elements
paths2 = root2.findall('.//{http://www.w3.org/2000/svg}path')
print(f"\nTotal path elements: {len(paths2)}")

# Analyze stroke-widths
stroke_widths2 = {}
fills2 = {}
for p in paths2:
    style = p.get('style', '')
    if 'stroke-width:' in style:
        sw = style.split('stroke-width:')[1].split(';')[0]
        stroke_widths2[sw] = stroke_widths2.get(sw, 0) + 1
    if 'fill:' in style:
        f = style.split('fill:')[1].split(';')[0]
        fills2[f] = fills2.get(f, 0) + 1
    elif p.get('fill'):
        f = p.get('fill')
        fills2[f] = fills2.get(f, 0) + 1

print("\nStroke widths found:")
for sw, cnt in sorted(stroke_widths2.items()):
    print(f"  {sw}: {cnt} elements")

print("\nFill colors (first 30):")
for f, cnt in sorted(fills2.items(), key=lambda x: -x[1])[:30]:
    print(f"  {f}: {cnt} elements")

# Look for groups
groups2 = root2.findall('.//{http://www.w3.org/2000/svg}g')
print(f"\nTotal groups: {len(groups2)}")
for g in groups2[:20]:
    g_id = g.get('id', 'no-id')
    label = g.get('{http://www.inkscape.org/namespaces/inkscape}label', 'no-label')
    children = len(list(g))
    print(f"  Group: id={g_id}, label={label}, children={children}")
