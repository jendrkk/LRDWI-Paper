"""
regional_weighting.py
Raking (Iterative Proportional Fitting) for CBOS survey data.

Produces 8 weight columns:
  weight_VOIV, weight_VOIV_500, weight_VOIV_100_500, weight_VOIV_100
  weight_VOIV_h, weight_VOIV_500_h, weight_VOIV_100_500_h, weight_VOIV_100_h

Individual weights (no _h) rake on: age×sex, educ×sex, pop_class.
Household weights (_h)  rake on: age×sex, educ×sex, pop_class, hh_size.

Old voivodeships (1990-1998): age_1990_L, educ_1990_L, hh_size_1990_L
New voivodeships (1999-2017): age_2000_L, educ_2000_L, hh_size_2000_L
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Label-mapping constants
# ---------------------------------------------------------------------------

# Map cs_prob (1-5) to the pop_class key(s) in the CT JSON.
# cs_prob=3 covers the two medium-city classes which are sometimes split.
CS_PROB_TO_POP_CLASS: dict[float, list[str]] = {
    1.0: ["wieś"],
    2.0: ["miasto do 20 000"],
    3.0: ["miasto od 20 001 do 50 000", "miasto od 50 001 do 100 000"],
    4.0: ["miasto od 100 001 do 500 000"],
    5.0: ["miasto 500 001 i więcej"],
}

# educ_1990_L values in the df → JSON category in E_educ_sex_1990
EDUC_1990_REMAP: dict[str, str] = {
    "podstawowe": "gimnazjalne, podstawowe i niższe",
    "podstawowe nieukończone i bez wykształcenia": "gimnazjalne, podstawowe i niższe",
    "zasadnicze zawodowe": "zasadnicze zawodowe",
    "średnie": "średnie",
    "wyższe": "wyższe",
}

# sex_L in df → JSON sex label
SEX_TO_JSON: dict[str, str] = {
    "Kobieta": "kobiety",
    "Mężczyzna": "mężczyźni",
}


# ---------------------------------------------------------------------------
# Core IPF / raking function
# ---------------------------------------------------------------------------

def rake_ipf(
    weights: np.ndarray,
    margins: list[list[tuple[np.ndarray, float]]],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Iterative Proportional Fitting (raking).

    Parameters
    ----------
    weights  : 1-D float array of starting weights (length N).
    margins  : list of margins. Each margin is a list of (mask, target) pairs,
               where mask is a boolean array (length N) selecting observations
               in one category and target is the desired weighted total for
               that category.
    max_iter : maximum number of full passes over all margins.
    tol      : convergence criterion (max relative weight change per pass).

    Returns
    -------
    Adjusted weight array (length N).
    """
    w = weights.astype(float).copy()

    for _iteration in range(max_iter):
        prev = w.copy()

        for margin in margins:
            for mask, target in margin:
                if not mask.any() or target <= 0.0:
                    continue
                current = (w * mask).sum()
                if current > 1e-12:
                    w[mask] *= target / current

        denom = np.maximum(np.abs(prev), 1e-10)
        if np.max(np.abs(w - prev) / denom) < tol:
            break

    return w


# ---------------------------------------------------------------------------
# Margin builders
# ---------------------------------------------------------------------------

def _build_age_sex_margin(
    df_sub: pd.DataFrame,
    age_col: str,
    ct: dict,
) -> list[tuple[np.ndarray, float]]:
    """Build (mask, target) pairs for the age × sex margin."""
    margin = []
    for key, target in ct.items():
        if "×" not in key:
            continue
        parts = [p.strip() for p in key.split("×", 1)]
        age_label, sex_json = parts[0], parts[1]
        if age_label == "ogółem" or sex_json == "ogółem":
            continue
        sex_df = "Kobieta" if sex_json == "kobiety" else "Mężczyzna"
        mask = (
            (df_sub[age_col] == age_label) & (df_sub["sex_L"] == sex_df)
        ).values
        if mask.any():
            margin.append((mask, float(target)))
    return margin


def _build_educ_sex_margin(
    df_sub: pd.DataFrame,
    educ_col: str,
    ct: dict,
    educ_remap: dict | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Build (mask, target) pairs for the educ × sex margin."""
    if educ_remap:
        educ_series = df_sub[educ_col].map(lambda x: educ_remap.get(x, x))
    else:
        educ_series = df_sub[educ_col]

    margin = []
    for key, target in ct.items():
        if "×" not in key:
            continue
        parts = [p.strip() for p in key.split("×", 1)]
        educ_label, sex_json = parts[0], parts[1]
        if educ_label == "ogółem" or sex_json == "ogółem":
            continue
        sex_df = "Kobieta" if sex_json == "kobiety" else "Mężczyzna"
        mask = (
            (educ_series == educ_label) & (df_sub["sex_L"] == sex_df)
        ).values
        if mask.any():
            margin.append((mask, float(target)))
    return margin


def _build_pop_class_margin(
    df_sub: pd.DataFrame,
    pop_class_ct: dict,
) -> list[tuple[np.ndarray, float]]:
    """
    Build (mask, target) pairs for the pop_class margin.

    Only meaningful when the group has >1 city-size class.
    cs_prob=3 is mapped to the sum of 'od 20k–50k' and 'od 50k–100k' targets.
    """
    if len(pop_class_ct) <= 1:
        return []

    margin = []
    for cs_val in sorted(df_sub["cs_prob"].dropna().unique()):
        target_keys = CS_PROB_TO_POP_CLASS.get(float(cs_val), [])
        target = sum(pop_class_ct.get(k, 0.0) for k in target_keys)
        if target <= 0.0:
            continue
        mask = (df_sub["cs_prob"] == cs_val).values
        if mask.any():
            margin.append((mask, float(target)))
    return margin


def _build_hh_size_margin(
    df_sub: pd.DataFrame,
    hh_col: str,
    hh_ct: dict,
) -> list[tuple[np.ndarray, float]]:
    """Build (mask, target) pairs for the household-size margin."""
    margin = []
    for hh_label, target in hh_ct.items():
        if hh_label == "ogółem":
            continue
        if float(target) <= 0.0:
            continue
        mask = (df_sub[hh_col] == hh_label).values
        if mask.any():
            margin.append((mask, float(target)))
    return margin


# ---------------------------------------------------------------------------
# Per-group raking (one year, one weight column)
# ---------------------------------------------------------------------------

def _rake_single_group(
    df_group: pd.DataFrame,
    weight_col: str,
    ct: dict,
    age_col: str,
    educ_col: str,
    hh_col: str,
    is_household: bool,
    is_old: bool,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Run IPF for one sub-group (voivodeship or G_VOIV group).

    Returns an array of adjusted weights aligned with df_group's index.
    """
    age_sex_key  = "E_age_sex_1990"  if is_old else "E_age_sex_2000"
    educ_sex_key = "E_educ_sex_1990" if is_old else "E_educ_sex_2000"
    hh_key       = "E_hh_size_1990"  if is_old else "E_hh_size_2000"
    educ_remap   = EDUC_1990_REMAP   if is_old else None

    margins: list[list[tuple[np.ndarray, float]]] = []

    # 1. age × sex
    age_sex_ct = ct.get(age_sex_key, {})
    if age_sex_ct:
        m = _build_age_sex_margin(df_group, age_col, age_sex_ct)
        if m:
            margins.append(m)

    # 2. educ × sex
    educ_sex_ct = ct.get(educ_sex_key, {})
    if educ_sex_ct:
        m = _build_educ_sex_margin(df_group, educ_col, educ_sex_ct, educ_remap)
        if m:
            margins.append(m)

    # 3. pop_class (only if multiple classes present)
    pop_class_ct = ct.get("pop_class", {})
    if pop_class_ct:
        m = _build_pop_class_margin(df_group, pop_class_ct)
        if m:
            margins.append(m)

    # 4. hh_size (household weights only)
    if is_household:
        hh_ct = ct.get(hh_key, {})
        if hh_ct:
            m = _build_hh_size_margin(df_group, hh_col, hh_ct)
            if m:
                margins.append(m)

    if not margins:
        return df_group[weight_col].values.astype(float)

    start_w = df_group[weight_col].values.astype(float)
    return rake_ipf(start_w, margins, max_iter=max_iter, tol=tol)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_regional_weights(
    df: pd.DataFrame,
    json_root: str | Path,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute raking weights for all 8 weight columns and write them into *df*.

    Parameters
    ----------
    df        : The survey dataframe (already containing weight_VOIV etc. as
                initial values equal to the original survey weights).
    json_root : Path to the directory that contains CBOS_old_voiv/,
                CBOS_new_voiv/, CBOS_U_old_voiv_dict/, CBOS_U_new_voiv_dict/.
    max_iter  : Maximum IPF iterations per group.
    tol       : IPF convergence tolerance.
    verbose   : Print progress.

    Returns
    -------
    df with updated weight columns (in-place modification + return).
    """
    json_root = Path(json_root)

    # ------------------------------------------------------------------
    # Load CT JSON files
    # ------------------------------------------------------------------
    if verbose:
        print("Loading CT JSON files …")

    with open(json_root / "CBOS_old_voiv" / "CBOS_all_years.json",
              encoding="utf-8") as f:
        ct_old_voiv: dict = json.load(f)

    with open(json_root / "CBOS_new_voiv" / "CBOS_all_years.json",
              encoding="utf-8") as f:
        ct_new_voiv: dict = json.load(f)

    with open(json_root / "CBOS_U_old_voiv_dict" / "CBOS_all_groups.json",
              encoding="utf-8") as f:
        ct_old_groups: dict = json.load(f)

    with open(json_root / "CBOS_U_new_voiv_dict" / "CBOS_all_groups.json",
              encoding="utf-8") as f:
        ct_new_groups: dict = json.load(f)

    # ------------------------------------------------------------------
    # CT lookup helpers
    # ------------------------------------------------------------------

    def voiv_ct(voiv_name: str, year_str: str, is_old: bool) -> dict | None:
        try:
            src = ct_old_voiv if is_old else ct_new_voiv
            return src[year_str][voiv_name][year_str]
        except (KeyError, TypeError):
            return None

    def group_ct(group_name: str, year_str: str, is_old: bool) -> dict | None:
        try:
            src = ct_old_groups if is_old else ct_new_groups
            return src[group_name][year_str]
        except (KeyError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Weight-column configurations
    # ------------------------------------------------------------------
    # Each entry: (individual_col, household_col, group_col_for_df, use_group_ct)
    # group_col_for_df=None means group by voivodeship
    configs = [
        ("weight_VOIV",         "weight_VOIV_h",         None,              False),
        ("weight_VOIV_500",     "weight_VOIV_500_h",     "G_VOIV_500",      True),
        ("weight_VOIV_100_500", "weight_VOIV_100_500_h", "G_VOIV_100_500",  True),
        ("weight_VOIV_100",     "weight_VOIV_100_h",     "G_VOIV_100",      True),
    ]

    years = range(1990, 2018)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for ind_col, hh_col_w, g_col, use_group in configs:
        if verbose:
            print(f"\n=== {ind_col} / {hh_col_w} ===")

        for is_household in (False, True):
            weight_col = hh_col_w if is_household else ind_col

            for year in years:
                year_str = str(year)
                is_old = year <= 1998

                # Subset to this year
                year_mask = df["survey_year"] == year
                if not year_mask.any():
                    continue

                # Period-specific column names
                age_col  = "age_1990_L"     if is_old else "age_2000_L"
                educ_col = "educ_1990_L"    if is_old else "educ_2000_L"
                hh_col   = "hh_size_1990_L" if is_old else "hh_size_2000_L"

                # Grouping column
                if not use_group:
                    grp_col = "location_old_L" if is_old else "location_new_L"
                else:
                    grp_col = g_col  # e.g. 'G_VOIV_500'

                # CT lookup function for this period
                def _ct_lookup(name: str, _is_old: bool = is_old,
                               _use_group: bool = use_group) -> dict | None:
                    if _use_group:
                        return group_ct(name, year_str, _is_old)
                    else:
                        return voiv_ct(name, year_str, _is_old)

                df_year = df.loc[year_mask]
                updated_weights = df.loc[year_mask, weight_col].copy()

                for grp_val, df_group in df_year.groupby(grp_col, dropna=True):
                    ct = _ct_lookup(grp_val)
                    if ct is None:
                        if verbose:
                            warnings.warn(
                                f"No CT for {weight_col}, year={year}, "
                                f"group={grp_val!r}"
                            )
                        continue

                    new_w = _rake_single_group(
                        df_group, weight_col, ct,
                        age_col, educ_col, hh_col,
                        is_household, is_old,
                        max_iter=max_iter, tol=tol,
                    )
                    updated_weights.loc[df_group.index] = new_w

                df.loc[year_mask, weight_col] = updated_weights.values

                if verbose:
                    print(
                        f"  {'HH' if is_household else 'Ind'} "
                        f"{year}: done  "
                        f"(mean={df.loc[year_mask, weight_col].mean():.4f}, "
                        f"min={df.loc[year_mask, weight_col].min():.4f}, "
                        f"max={df.loc[year_mask, weight_col].max():.4f})"
                    )

    return df
