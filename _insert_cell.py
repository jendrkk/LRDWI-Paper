import json

NB_PATH = 'Analysis/05_groups_2000.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)

new_source = [
    '# \u2500\u2500 Map: Theil index & contributions \u2014 LIS only (YEAR) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n',
    '# LIS regional metrics\n',
    'lis_reg = lis_groups_a.regional_panel(metrics=["theil"])\n',
    'lis_reg = lis_reg[lis_reg["year"] == YEAR].copy()\n',
    '# LIS contributions\n',
    'lis_ctr = lis_contrib.copy()\n',
    '\n',
    '# Plot definitions: (data, col, label, cmap, diverging)\n',
    'plot_defs = [\n',
    '    (lis_reg, "theil",                "Theil index",          "YlOrRd",   False),\n',
    '    (lis_ctr, "between_contribution", "Between contribution", "RdYlGn_r", True),\n',
    '    (lis_ctr, "within_contribution",  "Within contribution",  "RdYlGn_r", True),\n',
    ']\n',
    '\n',
    'fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n',
    'fig.subplots_adjust(wspace=0.05)\n',
    '\n',
    'for col, (data, val_col, label, cmap_name, diverging) in enumerate(plot_defs):\n',
    '    vals = data.set_index("region" if "region" in data.columns else "group")[val_col]\n',
    '\n',
    '    if diverging:\n',
    '        vabs = max(abs(vals.min()), abs(vals.max()))\n',
    '        norm = mpl.colors.TwoSlopeNorm(vmin=-vabs * 0.5, vcenter=0, vmax=vabs)\n',
    '    else:\n',
    '        norm = mpl.colors.Normalize(vmin=vals.min(), vmax=vals.max())\n',
    '    cmap_map = mpl.colormaps[cmap_name]\n',
    '\n',
    '    gdf = region_gdf.copy()\n',
    '    gdf["value"] = gdf["id"].map(vals)\n',
    '    gdf.plot(column="value", ax=axes[col], legend=False, cmap=cmap_map, norm=norm,\n',
    '             edgecolor="white", linewidth=0.3)\n',
    '    axes[col].set_title(f"LIS \u2014 {label}", fontsize=11)\n',
    '    axes[col].axis("off")\n',
    '\n',
    '    sm = plt.cm.ScalarMappable(cmap=cmap_map, norm=norm)\n',
    '    fig.colorbar(sm, ax=axes[col], orientation="horizontal", fraction=0.046, pad=0.04)\n',
    '\n',
    'plt.savefig(PLOT_ROOT / f"theil_contributions_map_lis_groups_{YEAR}.png", bbox_inches="tight")\n',
    'plt.show()',
]

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': new_source,
}

# Find the insertion point: after the CBOS+LIS map panel, before markdown
insert_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'theil_contributions_map_cbos_vs_lis_groups' in src:
        insert_idx = i + 1

if insert_idx is not None:
    nb['cells'].insert(insert_idx, new_cell)
    with open(NB_PATH, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Cell inserted at index {insert_idx}. Total cells: {len(nb["cells"])}')
else:
    print('ERROR: Could not find insertion point')
