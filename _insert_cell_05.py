"""Insert LIS-only Theil map cell into notebook 05."""
import json, pathlib

NB = pathlib.Path("Analysis/05_groups_2000.ipynb")
nb = json.loads(NB.read_text())

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "lis_theil_map_only",
    "metadata": {},
    "outputs": [],
    "source": [
        "# \u2500\u2500 Map: Theil index & contributions \u2014 LIS only (YEAR) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "# LIS regional metrics\n",
        "lis_reg_lis = lis_groups_a.regional_panel(metrics=['theil'])\n",
        "lis_reg_lis = lis_reg_lis[lis_reg_lis['year'] == YEAR].copy()\n",
        "# LIS contributions\n",
        "lis_ctr_lis = lis_contrib.copy()\n",
        "\n",
        "# Plot definitions: (data, col, label, cmap, diverging)\n",
        "plot_defs = [\n",
        "    (lis_reg_lis, 'theil',                'Theil index',          'YlOrRd',   False),\n",
        "    (lis_ctr_lis, 'between_contribution', 'Between contribution', 'RdYlGn_r', True),\n",
        "    (lis_ctr_lis, 'within_contribution',  'Within contribution',  'RdYlGn_r', True),\n",
        "]\n",
        "\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n",
        "fig.subplots_adjust(wspace=0.05)\n",
        "\n",
        "for col_idx, (data, val_col, label, cmap_name, diverging) in enumerate(plot_defs):\n",
        "    vals = data.set_index('region' if 'region' in data.columns else 'group')[val_col]\n",
        "\n",
        "    if diverging:\n",
        "        vabs = max(abs(vals.min()), abs(vals.max()))\n",
        "        norm = mpl.colors.TwoSlopeNorm(vmin=-vabs * 0.5, vcenter=0, vmax=vabs)\n",
        "    else:\n",
        "        norm = mpl.colors.Normalize(vmin=vals.min(), vmax=vals.max())\n",
        "    cmap_map = mpl.colormaps[cmap_name]\n",
        "\n",
        "    gdf_lis = region_gdf.copy()\n",
        "    gdf_lis['value'] = gdf_lis['id'].map(vals)\n",
        "    gdf_lis.plot(column='value', ax=axes[col_idx], legend=False, cmap=cmap_map, norm=norm,\n",
        "                 edgecolor='white', linewidth=0.3)\n",
        "    axes[col_idx].set_title(f'LIS \u2014 {label}', fontsize=11)\n",
        "    axes[col_idx].axis('off')\n",
        "\n",
        "    sm = plt.cm.ScalarMappable(cmap=cmap_map, norm=norm)\n",
        "    fig.colorbar(sm, ax=axes[col_idx], orientation='horizontal', fraction=0.046, pad=0.04)\n",
        "\n",
        "plt.savefig(PLOT_ROOT / f\"theil_contributions_map_lis_groups_{YEAR}.png\", bbox_inches='tight')\n",
        "plt.show()"
    ]
}

# Find the CBOS+LIS map panel cell
target_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'theil_contributions_map_cbos_vs_lis_groups' in src:
        target_idx = i + 1

if target_idx is None:
    raise RuntimeError("Could not find CBOS+LIS map panel cell")

nb['cells'].insert(target_idx, new_cell)

# Write back with same formatting
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Inserted at index {target_idx}. Total cells: {len(nb['cells'])}")
