"""
income_imputer.py
=================
Parse Polish income bracket labels from CBOS surveys and impute
a point estimate from the categorical interval.

Handles four thresholded columns (OLD/NEW × personal/household):
  income_hh_OLD_T, income_hh_NEW_T, income_p_OLD_T, income_p_NEW_T

Label formats encountered (examples):
  Range:      '1.801 - 2.100 tys. zł.', '451 – 700tys. zł',
              'Od 1001 do 1500 złotych', 'od 751 do 1000 złotych',
              'Od 650 do 999 złotych'
  Lower-only: '300 tys. zł i mniej', 'do 450 tys. zł',
              'Nie więcej niż 750 tys. zł', 'do 500 złotych',
              'Do 649 złotych'
  Upper-only: 'powyżej 2.200 tys. zł.', '4.501 tys. zł i więcej',
              'Powyżej 1500 złotych', '2600 złotych i więcej'
  Non-income: 'trudno powiedzieć', 'Odmowa odpowiedzi', 'NIE DOTYCZY',
              'BRAK DANYCH / Odmowa odpowiedzi'
  Technical:  'ND: q171!=4 or q171!=5', 'q127!=4 or q127!=5'
              (always coded -1)

Imputation rules:
  (a, b)   → (a + b) / 2          midpoint
  (0, a]   → a * (2/3)            shifted toward zero for gamma-like dist.
  [a, ∞)   → a * (3/2)            Pareto-tail approximation (α ≈ 3)
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches numbers like 2.200 (thousands notation), 1.801, 450, 1500, 750 etc.
_NUM = r'(\d+(?:\.\d+)?)'

# Compiled patterns (case-insensitive)
# 1) Range:  "X - Y tys. zł" / "X – Y tys. zł" / "Od X do Y złotych"
_RANGE_PAT = re.compile(
    rf'(?:od\s+)?{_NUM}\s*(?:-|–|do)\s*{_NUM}\s*(?:tys\.?\s*)?(?:zł\.?|złot)',
    re.IGNORECASE
)
# 2) Upper-bound only: "do X ...", "nie więcej niż X ...", "X ... i mniej"
_UPPER_PAT = re.compile(
    rf'(?:do\s+{_NUM}|nie\s+więcej\s+niż\s+{_NUM}|{_NUM}\s*(?:tys\.?\s*)?(?:zł\.?|złot)[^\d]*i\s+mniej)',
    re.IGNORECASE
)
# 3) Lower-bound only: "powyżej X ...", "X ... i więcej"
_LOWER_PAT = re.compile(
    rf'(?:powyżej\s+{_NUM}|{_NUM}\s*(?:tys\.?\s*)?(?:zł\.?|złot)?[^\d]*i\s+więcej)',
    re.IGNORECASE
)
# 4) Technical / filter labels (contain =, !=, <>, etc.)
_TECH_PAT = re.compile(r'[!=<>]|^ND:', re.IGNORECASE)

# Words indicating non-response / not-applicable
_NON_INCOME_WORDS = [
    'trudno powiedzieć', 'trudno powiedziec',
    'odmowa odpowiedzi', 'brak danych',
    'nie dotyczy', 'inna odpow',
    'odpowiedź zbyt', 'odpowiedź nie na temat',
]


def _parse_number(s: str) -> float:
    """
    Parse a Polish-format number string.
    '2.200' → 2200.0   (dot as thousands separator when digits after dot ≥ 3)
    '450'   → 450.0
    '3.75'  → 3.75     (true decimal: 1-2 digits after dot)
    """
    s = s.strip()
    parts = s.split('.')
    if len(parts) == 2 and len(parts[1]) >= 3:
        # Thousands separator: '2.200' → '2200'
        return float(parts[0] + parts[1])
    return float(s)


def _has_tys(label: str) -> bool:
    """Return True if the label uses 'tys.' (thousands of old PLN)."""
    return 'tys' in label.lower()


def parse_bracket_label(label: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse a single bracket label string.

    Returns (lower_bound, upper_bound) in base monetary units.
    - Both None → label is not a parseable income bracket (non-response etc.)
    - lower=None, upper=value → open-bottom bracket [0, value]
    - lower=value, upper=None → open-top bracket [value, ∞)
    - both set → closed bracket [lower, upper]

    The 'tys.' multiplier (×1000) is applied automatically.
    """
    if pd.isna(label):
        return None, None

    lbl = str(label).strip()
    if not lbl:
        return None, None

    lbl_lower = lbl.lower()

    # Non-income text labels
    for w in _NON_INCOME_WORDS:
        if w in lbl_lower:
            return None, None

    # Technical / filter labels
    if _TECH_PAT.search(lbl):
        return None, None

    multiplier = 1000.0 if _has_tys(lbl) else 1.0

    # Try range first (most common)
    m = _RANGE_PAT.search(lbl)
    if m:
        lo = _parse_number(m.group(1)) * multiplier
        hi = _parse_number(m.group(2)) * multiplier
        return lo, hi

    # Try upper-bound only ("do X", "nie więcej niż X", "X i mniej")
    m = _UPPER_PAT.search(lbl)
    if m:
        val_str = m.group(1) or m.group(2) or m.group(3)
        val = _parse_number(val_str) * multiplier
        return None, val

    # Try lower-bound only ("powyżej X", "X i więcej")
    m = _LOWER_PAT.search(lbl)
    if m:
        val_str = m.group(1) or m.group(2)
        val = _parse_number(val_str) * multiplier
        return val, None

    # If we get here, couldn't parse — treat as missing
    return None, None


def impute_from_bracket(
    lower: Optional[float],
    upper: Optional[float],
) -> float:
    """
    Impute a point estimate from bracket bounds.

    Rules:
      (a, b)   → (a + b) / 2        midpoint
      (0, a]   → a * (2/3)           gamma-shifted lower bracket
      [a, ∞)   → a * (3/2)           Pareto tail (α ≈ 3)
      None     → NaN
    """
    if lower is None and upper is None:
        return np.nan
    if lower is None:
        # Open-bottom bracket: (0, upper]
        return upper * (2.0 / 3.0)
    if upper is None:
        # Open-top bracket: [lower, ∞)
        return lower
    # Closed bracket
    return (lower + upper) / 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def impute_income_column(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
) -> tuple[pd.Series, list[int]]:
    """
    Impute income point estimates from a thresholded (categorical) income
    column and its label column.

    Parameters
    ----------
    df : DataFrame
        Must contain `value_col` (numeric category code) and `label_col`
        (string bracket description).
    value_col : str
        Column name for the numeric category code (e.g. 'income_hh_OLD_T').
    label_col : str
        Column name for the label (e.g. 'income_hh_OLD_T_L').

    Returns
    -------
    imputed : pd.Series
        Same index as df, with imputed float income values (or NaN).
    manual_check_indices : list[int]
        DataFrame indices where category code is present and ≥ 0
        but label is empty string → needs manual verification.
    """
    imputed = pd.Series(np.nan, index=df.index, dtype=float)
    manual_check = []

    for idx in df.index:
        cat_val = df.at[idx, value_col]
        lbl_val = df.at[idx, label_col]

        # Missing category → NaN
        if pd.isna(cat_val):
            continue

        cat_num = float(cat_val)

        # Technical filter code (always -1) → NaN
        if cat_num < 0:
            continue

        # Non-missing code but empty/missing label
        if pd.isna(lbl_val) or str(lbl_val).strip() == '':
            manual_check.append(idx)
            continue

        # Parse bracket and impute
        lo, hi = parse_bracket_label(str(lbl_val))
        imputed.at[idx] = impute_from_bracket(lo, hi)

    return imputed, manual_check


def impute_all_income_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """
    Impute all four thresholded income columns, adding _IM suffix columns.

    Processes:
      income_hh_OLD_T → income_hh_OLD_IM
      income_hh_NEW_T → income_hh_NEW_IM
      income_p_OLD_T  → income_p_OLD_IM
      income_p_NEW_T  → income_p_NEW_IM

    Returns
    -------
    df : DataFrame
        Input DataFrame with 4 new _IM columns added.
    manual_checks : dict
        {col_name: [list of indices needing manual check]} for each column.
    """
    col_pairs = [
        ('income_hh_OLD_T', 'income_hh_OLD_T_L', 'income_hh_OLD_IM'),
        ('income_hh_NEW_T', 'income_hh_NEW_T_L', 'income_hh_NEW_IM'),
        ('income_p_OLD_T',  'income_p_OLD_T_L',  'income_p_OLD_IM'),
        ('income_p_NEW_T',  'income_p_NEW_T_L',  'income_p_NEW_IM'),
    ]

    manual_checks = {}

    for val_col, lbl_col, im_col in col_pairs:
        if val_col not in df.columns or lbl_col not in df.columns:
            print(f"  ⚠ Skipping {val_col}: column not found in DataFrame")
            df[im_col] = np.nan
            manual_checks[val_col] = []
            continue

        imputed, mc = impute_income_column(df, val_col, lbl_col)
        df[im_col] = imputed
        manual_checks[val_col] = mc

        n_total = len(df)
        n_imputed = imputed.notna().sum()
        n_nan = imputed.isna().sum()
        n_manual = len(mc)
        print(f"  {val_col} → {im_col}: "
              f"{n_imputed} imputed, {n_nan} NaN, {n_manual} need manual check")

    return df, manual_checks
