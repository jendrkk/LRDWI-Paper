#!/usr/bin/env python3
"""Analyze the new_map.svg structure - focus on the 50-path group"""
import xml.etree.ElementTree as ET
from pathlib import Path
import re

# Find repo root
repo_root = Path(__file__).parent.parent.parent
svg_path = repo_root / 'new_map.svg'

print(f"Analyzing: {svg_path}")

tree = ET.parse(svg_path)
root = tree.getroot()
ns = {'svg': 'http://www.w3.org/2000/svg'}

# Find the 50-path group (fill=none, stroke=none, fill-opacity=.4)
groups = root.findall('.//svg:g', ns)
target_group = None
for g in groups:
    fill = g.get('fill', 'N/A')
    stroke = g.get('stroke', 'N/A')
    fill_opacity = g.get('fill-opacity', 'N/A')
    paths = g.findall('.//svg:path', ns)
    
    if len(paths) == 50:  # The 50-path group
        target_group = g
        print(f"Found target group: fill={fill}, stroke={stroke}, fill-opacity={fill_opacity}")
        print(f"Number of paths: {len(paths)}")
        
        # Analyze each path
        print("\nPath analysis (all 50):")
        single_m_count = 0
        for i, p in enumerate(paths):
            d = p.get('d', '')
            m_count = len(re.findall(r'[Mm]', d))
            if m_count == 1:
                single_m_count += 1
            # Just show path index, length, and M count
            print(f"  Path {i:2d}: len={len(d):5d}, M={m_count}")
        
        print(f"\nTotal single-M paths (complete polygons): {single_m_count}")
        print(f"Expected: 49 voivodeships + 1 outer border = 50")
        
        # Check transforms
        transforms = set()
        for p in paths:
            t = p.get('transform', 'NONE')
            transforms.add(t)
        print(f"\nTransforms: {transforms}")
