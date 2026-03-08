"""
Diagnostic script to understand income question patterns in CBOS SPSS files.
Run from the analysis directory.
"""
import pyreadstat as prs
import numpy as np
import os
from pathlib import Path

path_spss = Path("/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/CBOS SPSS")
list_dir = [f for f in os.listdir(path_spss) if f.endswith(".sav")]
list_dir.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

print(f"Total files: {len(list_dir)}")

# Load a sample of files and inspect income-related columns
sample_indices = list(range(0,5)) + list(range(50,55)) + list(range(100,105)) + list(range(150,155)) + list(range(200,205)) + list(range(250,255)) + list(range(300,min(305,len(list_dir))))
sample_indices = [i for i in sample_indices if i < len(list_dir)]

output_lines = []

for idx in sample_indices:
    file = list_dir[idx]
    df, meta = prs.read_sav(str(path_spss / file))
    
    output_lines.append(f"\n{'='*100}")
    output_lines.append(f"FILE [{idx:3d}]: {file}  (cols={len(meta.column_names)})")
    output_lines.append(f"{'='*100}")
    
    for col_idx, (code, label) in enumerate(zip(meta.column_names, meta.column_labels)):
        label_lower = label.lower() if isinstance(label, str) else ""
        if any(kw in label_lower for kw in ['dochod', 'dochód', 'zarobk', 'kwot', 'przedział', 'ile wynos']):
            vtype = meta.original_variable_types.get(code, "?")
            has_labels = code in meta.variable_value_labels
            n_labels = len(meta.variable_value_labels[code]) if has_labels else 0
            label_vals = ""
            if has_labels:
                vals = meta.variable_value_labels[code]
                # Show first 3 labels
                label_vals = str(dict(list(vals.items())[:3]))
            output_lines.append(f"  [{col_idx:3d}] {code:15s} type={str(vtype):8s} n_labels={n_labels:3d}  | {label}")
            if label_vals:
                output_lines.append(f"        Labels: {label_vals}")

output = "\n".join(output_lines)
print(output[:50000])  # First 50K chars

# Save full output
with open("_income_diagnostic_output.txt", "w", encoding="utf-8") as f:
    f.write(output)
print(f"\nFull output saved to _income_diagnostic_output.txt ({len(output_lines)} lines)")
