"""
Deeper diagnostic for income columns - focus on all files, classify patterns.
"""
import pyreadstat as prs
import numpy as np
import os
from pathlib import Path
import re

path_spss = Path("/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/CBOS SPSS")
list_dir = [f for f in os.listdir(path_spss) if f.endswith(".sav")]
list_dir.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

print(f"Total files: {len(list_dir)}")

output = []

# Classification keywords for income columns
# We want: personal income (numeric + categorical) and household per capita income (numeric + categorical)

for file_idx, file in enumerate(list_dir):
    df, meta = prs.read_sav(str(path_spss / file))
    
    year = int(file[-8:-4])
    
    # Find ALL columns with "dochod" or related keywords in labels that are in the M (metryczka) section
    income_candidates = []
    
    for col_idx, (code, label) in enumerate(zip(meta.column_names, meta.column_labels)):
        label_lower = label.lower() if isinstance(label, str) else ""
        code_lower = code.lower() if isinstance(code, str) else ""
        
        # Focus on metryczka income questions - they typically start with M or q4xx or are near end
        is_income_q = False
        
        # Check if label mentions income in the metryczka context
        if 'ile wynos' in label_lower and 'dochod' in label_lower:
            is_income_q = True
        elif 'dochody na 1 osob' in label_lower or 'dochody na jedną osob' in label_lower:
            is_income_q = True
        elif ('pana(i) miesięczne dochody' in label_lower or 'pana(i) dochody' in label_lower) and 'osob' not in label_lower:
            is_income_q = True
        elif 'dochód na jedną osobę' in label_lower:
            is_income_q = True
        elif 'kwot' in label_lower and col_idx > 0:
            # Check if previous column was an income question
            prev_label = meta.column_labels[col_idx-1].lower() if col_idx > 0 else ""
            if 'dochod' in prev_label and ('ile wynos' in prev_label or 'kwot' in prev_label or 'przedział' in prev_label):
                is_income_q = True
        elif 'przedział' in label_lower and 'dochod' in label_lower:
            is_income_q = True
        elif code_lower.startswith('m') and 'dochod' in label_lower and ('ile' in label_lower or 'wynos' in label_lower):
            is_income_q = True
        elif code_lower.startswith('q4') and 'dochod' in label_lower and ('ile' in label_lower or 'wynos' in label_lower):
            is_income_q = True
            
        if is_income_q:
            vtype = str(meta.original_variable_types.get(code, "?"))
            has_labels = code in meta.variable_value_labels
            n_labels = len(meta.variable_value_labels[code]) if has_labels else 0
            
            # Classify: personal vs household
            is_personal = ('pana(i) miesięczne dochody' in label_lower or 'pana(i) dochody' in label_lower) and 'osob' not in label_lower and '1 osob' not in label_lower and 'jedną' not in label_lower
            is_household_pc = '1 osob' in label_lower or 'jedną osob' in label_lower or 'na osobę' in label_lower
            
            # If neither personal nor household, check if it's total household
            is_total_hh = 'łączne dochody' in label_lower or 'dochody w pana' in label_lower
            
            # Classify: numeric vs categorical
            # Numeric: large range values (F4.0, F5.0 with no labels or few labels like 9991, 99991)
            # Categorical: has value labels mapping to text descriptions of ranges
            is_kwota = 'kwot' in label_lower  # explicit amount column
            is_przedzial = 'przedział' in label_lower  # explicit category column
            
            income_candidates.append({
                'idx': col_idx,
                'code': code,
                'label': label,
                'vtype': vtype,
                'n_labels': n_labels,
                'is_personal': is_personal,
                'is_household_pc': is_household_pc,
                'is_total_hh': is_total_hh,
                'is_kwota': is_kwota,
                'is_przedzial': is_przedzial,
            })
    
    if len(income_candidates) > 0:
        output.append(f"\n[{file_idx:3d}] {file} (year={year}, cols={len(meta.column_names)}, income_cols={len(income_candidates)})")
        for c in income_candidates:
            flags = []
            if c['is_personal']: flags.append('PERSONAL')
            if c['is_household_pc']: flags.append('HH_PC')
            if c['is_total_hh']: flags.append('TOTAL_HH')
            if c['is_kwota']: flags.append('KWOTA')
            if c['is_przedzial']: flags.append('PRZEDZIAL')
            flag_str = ", ".join(flags) if flags else "UNCLASSIFIED"
            output.append(f"    [{c['idx']:3d}] {c['code']:15s} type={c['vtype']:8s} n_labs={c['n_labels']:3d}  [{flag_str:20s}]  | {c['label'][:120]}")
    else:
        output.append(f"\n[{file_idx:3d}] {file} (year={year}) -- NO INCOME COLUMNS FOUND")

result = "\n".join(output)
print(result[:80000])

with open("_income_deep_diagnostic.txt", "w", encoding="utf-8") as f:
    f.write(result)
print(f"\nWritten to _income_deep_diagnostic.txt")
