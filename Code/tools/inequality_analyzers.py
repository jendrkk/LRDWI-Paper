"""
Inequality Analyzers — Comprehensive toolkit for income inequality analysis.

Three main classes:
    CBOSAnalyzer       — works with CBOS microdata (individual survey records)
    LISAnalyzer        — works with LIS pre-computed regional/group aggregates
    LISCountryAnalyzer — works with LIS country-level subgroup data

Supports: Gini, Palma, Theil (with full decomposition), Atkinson, MLD,
income shares, percentile ratios, percentile levels, demographic
decomposition, regional analysis, spatial group analysis.

All CBOS computations use survey weights by default for population-representative
estimates. Regional/spatial analyses pool 12 monthly surveys into yearly batches
to increase sample size.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Optional, Dict, Tuple, List, Union
import warnings


# ══════════════════════════════════════════════════════════════════════════════
# CORE WEIGHTED STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def weighted_mean(values, weights):
    """Weighted arithmetic mean."""
    v, w = _clean_pair(values, weights)
    if len(v) == 0 or w.sum() == 0:
        return np.nan
    return np.average(v, weights=w)


def weighted_median(values, weights):
    """Weighted median (50th percentile)."""
    return weighted_quantile(values, weights, 0.5)


def weighted_quantile(values, weights, q):
    """
    Weighted quantile using cumulative weight interpolation.

    Parameters
    ----------
    values : array-like
    weights : array-like — non-negative weights.
    q : float — quantile in [0, 1].
    """
    v, w = _clean_pair(values, weights)
    if len(v) == 0 or w.sum() == 0:
        return np.nan
    idx = np.argsort(v)
    sv, sw = v[idx], w[idx]
    cum = np.cumsum(sw)
    cutoff = q * cum[-1]
    pos = np.searchsorted(cum, cutoff)
    return sv[min(pos, len(sv) - 1)]


def weighted_percentiles(values, weights, percentiles=(10, 25, 50, 75, 90, 99)):
    """Return dict of weighted percentiles: {f'p{p}': value}."""
    return {f'p{p}': weighted_quantile(values, weights, p / 100) for p in percentiles}


# ── Inequality indices ────────────────────────────────────────────────────────

def weighted_gini(values, weights):
    """
    Weighted Gini coefficient.

    Uses the covariance formula:  G = (2 * cov(y, F(y))) / mean(y)
    where F(y) is the weighted cumulative distribution.
    """
    v, w = _clean_pair(values, weights)
    if len(v) < 2 or w.sum() == 0:
        return np.nan
    idx = np.argsort(v)
    sv, sw = v[idx], w[idx]
    cum_w = np.cumsum(sw)
    total_w = cum_w[-1]
    if total_w == 0:
        return np.nan
    # Cumulative population share (midpoint)
    F = (cum_w - sw / 2) / total_w
    mu = np.sum(sv * sw) / total_w
    if mu <= 0:
        return np.nan
    cov = np.sum(sw * sv * F) / total_w - mu * np.sum(sw * F) / total_w
    return 2 * cov / mu


def weighted_theil_t(values, weights):
    """
    Weighted Theil T index (GE(1), Generalized Entropy with alpha=1).

    T = (1/N_w) * sum_i [ w_i * (y_i / mu) * ln(y_i / mu) ]

    Only defined for strictly positive incomes.
    """
    v, w = _clean_pair(values, weights, positive_only=True)
    if len(v) < 2 or w.sum() == 0:
        return np.nan
    mu = np.average(v, weights=w)
    if mu <= 0:
        return np.nan
    ratio = v / mu
    return np.sum(w * ratio * np.log(ratio)) / w.sum()


def weighted_theil_l(values, weights):
    """
    Weighted Theil L index (GE(0), Mean Log Deviation / MLD).

    L = (1/N_w) * sum_i [ w_i * ln(mu / y_i) ]

    Only defined for strictly positive incomes.
    """
    v, w = _clean_pair(values, weights, positive_only=True)
    if len(v) < 2 or w.sum() == 0:
        return np.nan
    mu = np.average(v, weights=w)
    if mu <= 0:
        return np.nan
    return np.sum(w * np.log(mu / v)) / w.sum()


def weighted_atkinson(values, weights, epsilon=0.5):
    """
    Weighted Atkinson index.

    For epsilon != 1:
        A = 1 - (1/mu) * [ (1/N_w) * sum_i w_i * y_i^(1-e) ]^(1/(1-e))
    For epsilon == 1:
        A = 1 - exp( (1/N_w) * sum_i w_i * ln(y_i) ) / mu

    Parameters
    ----------
    epsilon : float
        Inequality aversion parameter. Higher = more sensitive to bottom.
        Common values: 0.5, 1.0, 2.0
    """
    v, w = _clean_pair(values, weights, positive_only=True)
    if len(v) < 2 or w.sum() == 0:
        return np.nan
    mu = np.average(v, weights=w)
    if mu <= 0:
        return np.nan
    total_w = w.sum()
    if abs(epsilon - 1.0) < 1e-10:
        log_mean = np.sum(w * np.log(v)) / total_w
        return 1.0 - np.exp(log_mean) / mu
    else:
        power_mean = (np.sum(w * v ** (1 - epsilon)) / total_w) ** (1 / (1 - epsilon))
        return 1.0 - power_mean / mu


def weighted_palma(values, weights):
    """
    Weighted Palma ratio: income share of top 10% / income share of bottom 40%.
    """
    v, w = _clean_pair(values, weights)
    if len(v) == 0 or w.sum() == 0:
        return np.nan
    idx = np.argsort(v)
    sv, sw = v[idx], w[idx]
    cum_w = np.cumsum(sw)
    total_w = cum_w[-1]
    weighted_inc = sv * sw
    # Bottom 40%
    b40_mask = cum_w <= 0.4 * total_w
    b40_income = weighted_inc[b40_mask].sum()
    if b40_mask.sum() < len(sv):
        boundary = b40_mask.sum()
        frac = (0.4 * total_w - (cum_w[boundary - 1] if boundary > 0 else 0)) / sw[boundary]
        b40_income += frac * weighted_inc[boundary]
    # Top 10%
    t10_mask = cum_w > 0.9 * total_w
    t10_income = weighted_inc[t10_mask].sum()
    if t10_mask.sum() < len(sv):
        boundary = len(sv) - t10_mask.sum() - 1
        if boundary >= 0:
            frac = (cum_w[boundary] - 0.9 * total_w) / sw[boundary]
            t10_income += frac * weighted_inc[boundary]
    if b40_income <= 0:
        return np.nan
    return t10_income / b40_income


def weighted_income_shares(values, weights, groups=None):
    """
    Compute income shares for distributional groups.

    Parameters
    ----------
    groups : dict, optional
        {name: (lower_pct, upper_pct)}. Default matches LIS format:
        Bottom_50, P50_90, Top_10, Top_1

    Returns
    -------
    dict : {f'share_{name}': share, name: mean_income, f'N_{name}': count, ...}
    """
    if groups is None:
        groups = {
            'Bottom_50': (0.0, 0.5),
            'P50_90': (0.5, 0.9),
            'Top_10': (0.9, 1.0),
            'Top_1': (0.99, 1.0),
        }
    v, w = _clean_pair(values, weights)
    if len(v) == 0 or w.sum() == 0:
        return {k: np.nan for g in groups for k in (f'share_{g}', g, f'N_{g}', f'Nw_{g}')}

    idx = np.argsort(v)
    sv, sw = v[idx], w[idx]
    cum_w = np.cumsum(sw)
    total_w = cum_w[-1]
    total_income = np.sum(sv * sw)

    result = {}
    for name, (lo, hi) in groups.items():
        mask = (cum_w / total_w > lo) & (cum_w / total_w <= hi)
        if lo == 0:
            mask = cum_w / total_w <= hi
        group_income = np.sum(sv[mask] * sw[mask])
        group_weight = sw[mask].sum()
        result[f'share_{name}'] = group_income / total_income if total_income > 0 else np.nan
        result[name] = np.average(sv[mask], weights=sw[mask]) if group_weight > 0 else np.nan
        result[f'N_{name}'] = int(mask.sum())
        result[f'Nw_{name}'] = group_weight
    return result


def weighted_percentile_ratios(values, weights):
    """Compute standard percentile ratios: p90/p10, p90/p50, p50/p10."""
    pcts = weighted_percentiles(values, weights, (10, 50, 90))
    p10, p50, p90 = pcts['p10'], pcts['p50'], pcts['p90']
    return {
        'p90p10': p90 / p10 if p10 > 0 else np.nan,
        'p90p50': p90 / p50 if p50 > 0 else np.nan,
        'p50p10': p50 / p10 if p10 > 0 else np.nan,
    }


# ── Theil decomposition ──────────────────────────────────────────────────────

def theil_decomposition(values, weights, group_labels):
    """
    Full Theil T decomposition into between-group and within-group components.

    T_total = T_between + T_within
    T_between = sum_k [ s_k * ln(mu_k / mu) ]
    T_within  = sum_k [ s_k * T_k ]

    where s_k = (sum w_i*y_i in group k) / (sum w_i*y_i total)  [income share]
          mu_k = weighted mean income in group k
          T_k  = Theil T within group k

    Parameters
    ----------
    values : array-like — individual incomes.
    weights : array-like — survey weights.
    group_labels : array-like — group membership for each observation.

    Returns
    -------
    dict with keys:
        total_theil, between, within,
        group_theil (dict), group_share (dict), group_mean (dict),
        group_N (dict), overall_mean, between_pct, within_pct
    """
    v, w, g = _clean_triple(values, weights, group_labels, positive_only=True)
    if len(v) < 2:
        return _empty_decomposition()

    mu = np.average(v, weights=w)
    total_wincome = np.sum(v * w)

    groups = np.unique(g)
    g_theil, g_share, g_mean, g_N = {}, {}, {}, {}

    for grp in groups:
        mask = g == grp
        gv, gw = v[mask], w[mask]
        if len(gv) == 0 or gw.sum() == 0:
            g_theil[grp] = 0.0
            g_share[grp] = 0.0
            g_mean[grp] = 0.0
            g_N[grp] = 0.0
            continue
        g_mean[grp] = np.average(gv, weights=gw)
        g_share[grp] = np.sum(gv * gw) / total_wincome
        g_N[grp] = gw.sum()
        g_theil[grp] = weighted_theil_t(gv, gw)
        if np.isnan(g_theil[grp]):
            g_theil[grp] = 0.0

    between = sum(
        g_share[grp] * np.log(g_mean[grp] / mu)
        for grp in groups if g_share[grp] > 0 and g_mean[grp] > 0
    )
    within = sum(g_share[grp] * g_theil[grp] for grp in groups)
    total = between + within

    return {
        'total_theil': total,
        'between': between,
        'within': within,
        'between_pct': between / total * 100 if total > 0 else np.nan,
        'within_pct': within / total * 100 if total > 0 else np.nan,
        'group_theil': g_theil,
        'group_share': g_share,
        'group_mean': g_mean,
        'group_N': g_N,
        'overall_mean': mu,
    }


def theil_group_contributions(values, weights, group_labels):
    """
    Compute each group's contribution to the total Theil T index.

    Each group g contributes:
        between:  s_g * ln(mu_g / mu)
        within:   s_g * T_g
        total:    between + within

    where s_g = income share = (sum w_i*y_i in g) / (sum w_i*y_i total).

    Returns
    -------
    pd.DataFrame with columns: group, income_share, group_mean, group_theil,
        group_N, between_contribution, within_contribution,
        total_contribution, contribution_pct
    """
    dec = theil_decomposition(values, weights, group_labels)
    if np.isnan(dec['total_theil']):
        return pd.DataFrame()
    mu = dec['overall_mean']
    total = dec['total_theil']
    rows = []
    for grp in sorted(dec['group_share'].keys()):
        s_g = dec['group_share'][grp]
        mu_g = dec['group_mean'][grp]
        T_g = dec['group_theil'][grp]
        N_g = dec['group_N'][grp]
        b_g = s_g * np.log(mu_g / mu) if s_g > 0 and mu_g > 0 else 0.0
        w_g = s_g * T_g
        t_g = b_g + w_g
        rows.append({
            'group': grp,
            'income_share': s_g,
            'group_mean': mu_g,
            'group_theil': T_g,
            'group_N': N_g,
            'between_contribution': b_g,
            'within_contribution': w_g,
            'total_contribution': t_g,
            'contribution_pct': t_g / total * 100 if total > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def mld_decomposition(values, weights, group_labels):
    """
    Theil L (MLD / GE(0)) decomposition into between-group and within-group.

    L_between = sum_k [ n_k * ln(mu / mu_k) ]
    L_within  = sum_k [ n_k * L_k ]

    where n_k = (sum w_i in group k) / (sum w_i total)  [population share]
    """
    v, w, g = _clean_triple(values, weights, group_labels, positive_only=True)
    if len(v) < 2:
        return _empty_decomposition()

    mu = np.average(v, weights=w)
    total_w = w.sum()

    groups = np.unique(g)
    g_mld, g_pop_share, g_mean, g_N = {}, {}, {}, {}

    for grp in groups:
        mask = g == grp
        gv, gw = v[mask], w[mask]
        if len(gv) == 0 or gw.sum() == 0:
            g_mld[grp] = 0.0
            g_pop_share[grp] = 0.0
            g_mean[grp] = 0.0
            g_N[grp] = 0.0
            continue
        g_mean[grp] = np.average(gv, weights=gw)
        g_pop_share[grp] = gw.sum() / total_w
        g_N[grp] = gw.sum()
        g_mld[grp] = weighted_theil_l(gv, gw)
        if np.isnan(g_mld[grp]):
            g_mld[grp] = 0.0

    between = sum(
        g_pop_share[grp] * np.log(mu / g_mean[grp])
        for grp in groups if g_pop_share[grp] > 0 and g_mean[grp] > 0
    )
    within = sum(g_pop_share[grp] * g_mld[grp] for grp in groups)
    total = between + within

    return {
        'total_mld': total,
        'between': between,
        'within': within,
        'between_pct': between / total * 100 if total > 0 else np.nan,
        'within_pct': within / total * 100 if total > 0 else np.nan,
        'group_mld': g_mld,
        'group_share': g_pop_share,
        'group_mean': g_mean,
        'group_N': g_N,
        'overall_mean': mu,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_array(x):
    """Convert to numpy float64 array."""
    if isinstance(x, pd.Series):
        return x.values.astype(float)
    return np.asarray(x, dtype=float)


def _clean_pair(values, weights, positive_only=False):
    """Remove NaN/Inf from paired arrays. Optionally keep only positive values."""
    v, w = _to_array(values), _to_array(weights)
    mask = np.isfinite(v) & np.isfinite(w)
    if positive_only:
        mask &= v > 0
    return v[mask], w[mask]


def _clean_triple(values, weights, labels, positive_only=False):
    """Clean values/weights/labels triple."""
    v, w = _to_array(values), _to_array(weights)
    g = np.asarray(labels)
    mask = np.isfinite(v) & np.isfinite(w)
    if positive_only:
        mask &= v > 0
    return v[mask], w[mask], g[mask]


def _empty_decomposition():
    return {
        'total_theil': np.nan, 'between': np.nan, 'within': np.nan,
        'between_pct': np.nan, 'within_pct': np.nan,
        'group_theil': {}, 'group_share': {}, 'group_mean': {}, 'group_N': {},
        'overall_mean': np.nan,
    }


def compute_all_metrics(values, weights):
    """
    Compute a full suite of inequality metrics for a single distribution.
    Returns a flat dict matching LIS-style column naming.

    Output keys: N_total, Nw_total, mean, median, p10, p25, p75, p90, p99,
                 p90p10, p90p50, p50p10, gini, theil, mld, palma,
                 atkinson_05, atkinson_1, atkinson_2,
                 share_Bottom_50, share_P50_90, share_Top_10, share_Top_1,
                 Bottom_50, P50_90, Top_10, Top_1  (group means),
                 N_Bottom_50, ..., Nw_Bottom_50, ...
    """
    v, w = _clean_pair(values, weights)
    n = len(v)
    nw = w.sum()

    if n == 0:
        return {k: np.nan for k in _ALL_METRIC_KEYS}

    result = {
        'N_total': n,
        'Nw_total': nw,
        'mean': weighted_mean(v, w),
        'median': weighted_median(v, w),
    }
    result.update(weighted_percentiles(v, w))
    result.update(weighted_percentile_ratios(v, w))
    result['gini'] = weighted_gini(v, w)
    result['theil'] = weighted_theil_t(v, w)
    result['mld'] = weighted_theil_l(v, w)
    result['palma'] = weighted_palma(v, w)
    result['atkinson_05'] = weighted_atkinson(v, w, epsilon=0.5)
    result['atkinson_1'] = weighted_atkinson(v, w, epsilon=1.0)
    result['atkinson_2'] = weighted_atkinson(v, w, epsilon=2.0)
    result.update(weighted_income_shares(v, w))
    return result


_ALL_METRIC_KEYS = [
    'N_total', 'Nw_total', 'mean', 'median',
    'p10', 'p25', 'p50', 'p75', 'p90', 'p99',
    'p90p10', 'p90p50', 'p50p10',
    'gini', 'theil', 'mld', 'palma',
    'atkinson_05', 'atkinson_1', 'atkinson_2',
    'share_Bottom_50', 'share_P50_90', 'share_Top_10', 'share_Top_1',
    'Bottom_50', 'P50_90', 'Top_10', 'Top_1',
    'N_Bottom_50', 'N_P50_90', 'N_Top_10', 'N_Top_1',
    'Nw_Bottom_50', 'Nw_P50_90', 'Nw_Top_10', 'Nw_Top_1',
]


# ══════════════════════════════════════════════════════════════════════════════
# CBOS ANALYZER — microdata analysis
# ══════════════════════════════════════════════════════════════════════════════

class CBOSAnalyzer:
    """
    Comprehensive inequality analyzer for CBOS survey microdata.

    Supports country-level, regional, spatial, and demographic analyses.
    All computations use survey weights for population-representative estimates.

    Parameters
    ----------
    df : pd.DataFrame
        CBOS survey data (CBOS_survey.csv).
    income_col : str
        Income column to analyze.  Default 'income_hh_imputed' (complete, no NaN).
        Alternative: 'income_p_imputed' for per-capita income.
    weight_col : str
        Survey weight column.  Choose based on analysis level:
        - 'weight_VOIV_NORM' : voivodeship-calibrated (default)
        - 'weight_MACRO_NORM': macroregion-calibrated (1999+ only)
        - 'weight_VOIV'      : un-normalized voivodeship weights
    deflator_col : str or None
        Deflator column for real income. None = nominal.
        e.g. 'deflator_2017' for 2017 PLN prices.
    year_col, month_col, file_col : str
        Column names for time identifiers.

    Example
    -------
    >>> cbos = pd.read_csv('CBOS_survey.csv', low_memory=False)
    >>> ca = CBOSAnalyzer(cbos, deflator_col='deflator_2017')
    >>> ts = ca.country_timeseries(freq='annual')
    >>> theil = ca.regional_theil_timeseries(region_col='location_new_L')
    """

    DEFAULTS = {
        'income_col': 'income_hh_imputed',
        'weight_col': 'weight_VOIV_NORM',
        'year_col': 'survey_year',
        'month_col': 'survey_month',
        'file_col': 'survey_file',
        'hh_size_col': 'household_size',
    }

    def __init__(self, df: pd.DataFrame,
                 income_col: str = 'income_hh_imputed',
                 weight_col: str = 'weight_VOIV_NORM',
                 deflator_col: Optional[str] = None,
                 year_col: str = 'survey_year',
                 month_col: str = 'survey_month',
                 file_col: str = 'survey_file',
                 hh_size_col: str = 'household_size'):

        self.df = df
        self.income_col = income_col
        self.weight_col = weight_col
        self.deflator_col = deflator_col
        self.year_col = year_col
        self.month_col = month_col
        self.file_col = file_col
        self.hh_size_col = hh_size_col

        # Precompute analysis columns (optionally deflated)
        self._income_key = '_income_analysis'
        self.df[self._income_key] = pd.to_numeric(self.df[income_col], errors='coerce')
        if deflator_col is not None:
            defl = pd.to_numeric(self.df[deflator_col], errors='coerce')
            self.df[self._income_key] = self.df[self._income_key] * defl

        self._weight_key = '_weight_analysis'
        self.df[self._weight_key] = pd.to_numeric(self.df[weight_col], errors='coerce')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_income_weights(self, mask=None):
        """Return (income_array, weight_array) for rows matching mask."""
        sub = self.df if mask is None else self.df.loc[mask]
        return sub[self._income_key].values, sub[self._weight_key].values

    def _years(self, year_range=None):
        """Sorted unique years, optionally filtered."""
        years = sorted(self.df[self.year_col].dropna().unique().astype(int))
        if year_range is not None:
            years = [y for y in years if year_range[0] <= y <= year_range[1]]
        return years

    def _monthly_periods(self, year_range=None):
        """Return sorted list of (year, month) tuples."""
        sub = self.df[[self.year_col, self.month_col]].dropna()
        pairs = sub.drop_duplicates().values.astype(int)
        pairs = sorted(map(tuple, pairs))
        if year_range is not None:
            pairs = [(y, m) for y, m in pairs if year_range[0] <= y <= year_range[1]]
        return pairs

    # ══════════════════════════════════════════════════════════════════════════
    # COUNTRY-LEVEL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def country_timeseries(self, freq: str = 'annual', year_range=None) -> pd.DataFrame:
        """
        Compute full inequality metrics at country level over time.

        Parameters
        ----------
        freq : str
            'annual'  — pool all months within each year (larger sample).
            'monthly' — compute separately for each survey month.
        year_range : tuple of (start_year, end_year) or None

        Returns
        -------
        pd.DataFrame indexed by date, columns = all metric keys.
        """
        rows = []
        if freq == 'annual':
            for year in self._years(year_range):
                mask = self.df[self.year_col] == year
                inc, w = self._get_income_weights(mask)
                metrics = compute_all_metrics(inc, w)
                metrics['year'] = year
                metrics['date'] = pd.Timestamp(year, 7, 1)
                rows.append(metrics)
        else:
            for year, month in self._monthly_periods(year_range):
                mask = (self.df[self.year_col] == year) & (self.df[self.month_col] == month)
                inc, w = self._get_income_weights(mask)
                metrics = compute_all_metrics(inc, w)
                metrics['year'] = year
                metrics['month'] = month
                metrics['date'] = pd.Timestamp(year, int(month), 1)
                rows.append(metrics)

        result = pd.DataFrame(rows)
        if len(result) > 0:
            result.set_index('date', inplace=True)
        return result

    def country_monthly_within_year(self, year: int) -> pd.DataFrame:
        """
        Monthly inequality metrics for a single year — for intra-year dynamics.
        """
        rows = []
        year_mask = self.df[self.year_col] == year
        for month in sorted(self.df.loc[year_mask, self.month_col].dropna().unique().astype(int)):
            mask = year_mask & (self.df[self.month_col] == month)
            inc, w = self._get_income_weights(mask)
            metrics = compute_all_metrics(inc, w)
            metrics['month'] = month
            rows.append(metrics)
        return pd.DataFrame(rows).set_index('month') if rows else pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════════
    # REGIONAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def regional_metrics(self, year: int, region_col: str,
                         regions: Optional[List] = None,
                         region_id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Full inequality metrics for each region in a given year.
        Uses yearly batches (all months pooled) for sufficient sample size.

        Parameters
        ----------
        year : int
        region_col : str
            e.g. 'location_new_L', 'location_old_L', 'macroregion'
        regions : list or None — restrict to these regions.
        region_id_col : str or None
            Column to use as region identifier (e.g. 'teryt_id_VOIV').
            If provided and present in data, a 'region_id' column is added.
        """
        mask_year = self.df[self.year_col] == year
        sub = self.df.loc[mask_year].dropna(subset=[region_col])
        if regions is not None:
            sub = sub[sub[region_col].isin(regions)]

        rows = []
        for region in sorted(sub[region_col].unique()):
            rmask = sub[region_col] == region
            inc = sub.loc[rmask, self._income_key].values
            w = sub.loc[rmask, self._weight_key].values
            metrics = compute_all_metrics(inc, w)
            metrics['region'] = region
            if region_id_col is not None and region_id_col in sub.columns:
                id_vals = sub.loc[rmask, region_id_col].dropna()
                metrics['region_id'] = id_vals.iloc[0] if len(id_vals) > 0 else np.nan
            else:
                metrics['region_id'] = np.nan
            metrics['year'] = year
            rows.append(metrics)

        return pd.DataFrame(rows)

    def regional_timeseries(self, region_col: str,
                            regions: Optional[List] = None,
                            year_range=None,
                            region_id_col: Optional[str] = None) -> pd.DataFrame:
        """Inequality metrics per region per year (panel data)."""
        rows = []
        for year in self._years(year_range):
            yearly = self.regional_metrics(year, region_col, regions, region_id_col=region_id_col)
            if len(yearly) > 0:
                rows.append(yearly)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def regional_theil_decomposition(self, year: int, region_col: str,
                                      regions: Optional[List] = None) -> dict:
        """
        Theil T between-region / within-region decomposition for a single year.
        """
        mask = self.df[self.year_col] == year
        sub = self.df.loc[mask].dropna(subset=[region_col])
        if regions is not None:
            sub = sub[sub[region_col].isin(regions)]
        inc = sub[self._income_key].values
        w = sub[self._weight_key].values
        g = sub[region_col].values
        return theil_decomposition(inc, w, g)

    def regional_theil_timeseries(self, region_col: str,
                                   regions: Optional[List] = None,
                                   year_range=None) -> pd.DataFrame:
        """Between/within Theil decomposition over time for regions."""
        rows = []
        for year in self._years(year_range):
            result = self.regional_theil_decomposition(year, region_col, regions)
            row = {
                'year': year,
                'total_theil': result['total_theil'],
                'between': result['between'],
                'within': result['within'],
                'between_pct': result['between_pct'],
                'within_pct': result['within_pct'],
                'overall_mean': result['overall_mean'],
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def theil_contributions(self, year: int, group_col: str,
                            groups: Optional[List] = None,
                            region_id_col: Optional[str] = None) -> pd.DataFrame:
        """Per-group contributions to total Theil T for a single year."""
        mask = self.df[self.year_col] == year
        sub = self.df.loc[mask].dropna(subset=[group_col])
        if groups is not None:
            sub = sub[sub[group_col].isin(groups)]
        inc = sub[self._income_key].values
        w = sub[self._weight_key].values
        g = sub[group_col].values
        result = theil_group_contributions(inc, w, g)
        if len(result) > 0:
            result['year'] = year
            if region_id_col is not None and region_id_col in sub.columns:
                id_map = {}
                for grp in result['group'].unique():
                    id_vals = sub.loc[sub[group_col] == grp, region_id_col].dropna()
                    id_map[grp] = id_vals.iloc[0] if len(id_vals) > 0 else np.nan
                result['region_id'] = result['group'].map(id_map)
        return result

    def theil_contribution_timeseries(self, group_col: str,
                                      groups: Optional[List] = None,
                                      year_range=None,
                                      region_id_col: Optional[str] = None) -> pd.DataFrame:
        """Per-group Theil contributions over time (panel: year x group)."""
        frames = []
        for year in self._years(year_range):
            yearly = self.theil_contributions(year, group_col, groups,
                                             region_id_col=region_id_col)
            if len(yearly) > 0:
                frames.append(yearly)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def regional_mld_decomposition(self, year: int, region_col: str,
                                    regions: Optional[List] = None) -> dict:
        """MLD (Theil L / GE(0)) decomposition by region for a single year."""
        mask = self.df[self.year_col] == year
        sub = self.df.loc[mask].dropna(subset=[region_col])
        if regions is not None:
            sub = sub[sub[region_col].isin(regions)]
        inc = sub[self._income_key].values
        w = sub[self._weight_key].values
        g = sub[region_col].values
        return mld_decomposition(inc, w, g)

    def regional_mld_timeseries(self, region_col: str,
                                 regions: Optional[List] = None,
                                 year_range=None) -> pd.DataFrame:
        """MLD between/within decomposition over time."""
        rows = []
        for year in self._years(year_range):
            result = self.regional_mld_decomposition(year, region_col, regions)
            row = {
                'year': year,
                'total_mld': result.get('total_mld', np.nan),
                'between': result['between'],
                'within': result['within'],
                'between_pct': result['between_pct'],
                'within_pct': result['within_pct'],
                'overall_mean': result['overall_mean'],
            }
            rows.append(row)
        return pd.DataFrame(rows)

    # ══════════════════════════════════════════════════════════════════════════
    # SPATIAL GROUP ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def spatial_metrics(self, year: int, group_col: str,
                        groups: Optional[List] = None,
                        region_id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Inequality metrics per spatial group for a single year.

        Parameters
        ----------
        group_col : str
            Spatial group column, e.g.:
            - 'G_VOIV_500'      : voiv x (city>500k vs rest)
            - 'G_VOIV_100'      : voiv x (city>100k vs rest)
            - 'G_VOIV_100_500'  : voiv x (city>500k, 100-500k, rest)
            - 'G_MACRO_500'     : macroregion x (city>500k vs rest)
            - 'G_MACRO_100'     : macroregion x (city>100k vs rest)
            - 'G_MACRO_100_500' : macroregion x (city>500k, 100-500k, rest)
            - 'cs_L'            : city size categories only (no region)
        region_id_col : str or None
        """
        return self.regional_metrics(year, group_col, groups, region_id_col=region_id_col)

    def spatial_timeseries(self, group_col: str,
                           groups: Optional[List] = None,
                           year_range=None,
                           region_id_col: Optional[str] = None) -> pd.DataFrame:
        """Inequality metrics per spatial group over time."""
        return self.regional_timeseries(group_col, groups, year_range, region_id_col=region_id_col)

    def spatial_theil_decomposition(self, year: int, group_col: str,
                                     groups: Optional[List] = None) -> dict:
        """Theil T decomposition across spatial groups for a single year."""
        return self.regional_theil_decomposition(year, group_col, groups)

    def spatial_theil_timeseries(self, group_col: str,
                                  groups: Optional[List] = None,
                                  year_range=None) -> pd.DataFrame:
        """Theil between/within spatial groups over time."""
        return self.regional_theil_timeseries(group_col, groups, year_range)

    # ══════════════════════════════════════════════════════════════════════════
    # DEMOGRAPHIC ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def demographic_metrics(self, year: int, demo_col: str,
                            categories: Optional[List] = None,
                            region_id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Full inequality metrics for each demographic category in a given year.

        Parameters
        ----------
        demo_col : str
            - 'sex_L'         : Male / Female
            - 'educ_1990_L'   : 5-category education (old division, full coverage)
            - 'educ_2000_L'   : 5-category education (new division, finer for 2000+)
            - 'age_1990_L'    : 10-year age bins (7 groups)
            - 'age_2000_L'    : 5-year age bins (14 groups)
            - 'hh_size_1990_L': household size (4 groups)
            - 'hh_size_2000_L': household size (5 groups)
            - 'sol_L'         : self-assessed standard of living
        categories : list or None — restrict to these categories.
        region_id_col : str or None
        """
        return self.regional_metrics(year, demo_col, categories, region_id_col=region_id_col)

    def demographic_timeseries(self, demo_col: str,
                               categories: Optional[List] = None,
                               year_range=None,
                               region_id_col: Optional[str] = None) -> pd.DataFrame:
        """Inequality metrics per demographic group over time (panel structure)."""
        return self.regional_timeseries(demo_col, categories, year_range, region_id_col=region_id_col)

    def demographic_theil_decomposition(self, year: int, demo_col: str,
                                         categories: Optional[List] = None) -> dict:
        """
        Theil T decomposition between/within demographic groups.

        Answers: how much of total inequality is explained by between-group
        differences (e.g. education premium) vs within-group variation?
        """
        return self.regional_theil_decomposition(year, demo_col, categories)

    def demographic_theil_timeseries(self, demo_col: str,
                                      categories: Optional[List] = None,
                                      year_range=None) -> pd.DataFrame:
        """Theil between/within by demographic dimension over time."""
        return self.regional_theil_timeseries(demo_col, categories, year_range)

    def demographic_mld_decomposition(self, year: int, demo_col: str,
                                       categories: Optional[List] = None) -> dict:
        """MLD decomposition between/within demographic groups."""
        return self.regional_mld_decomposition(year, demo_col, categories)

    def demographic_mld_timeseries(self, demo_col: str,
                                    categories: Optional[List] = None,
                                    year_range=None) -> pd.DataFrame:
        """MLD between/within by demographic dimension over time."""
        return self.regional_mld_timeseries(demo_col, categories, year_range)

    # ══════════════════════════════════════════════════════════════════════════
    # TWO-LEVEL DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════════════

    def nested_theil_decomposition(self, year: int,
                                    outer_col: str, inner_col: str,
                                    outer_groups: Optional[List] = None) -> dict:
        """
        Two-level (nested) Theil decomposition.

        Example: outer = region, inner = city_size within region.

        Total = Between_outer + Within_outer
        Within_outer = sum_k s_k * (Between_inner_k + Within_inner_k)

        Returns
        -------
        dict with: total_theil, between_outer, within_outer,
                   outer_result, inner_results (dict per outer group)
        """
        mask = self.df[self.year_col] == year
        sub = self.df.loc[mask].dropna(subset=[outer_col, inner_col])
        if outer_groups is not None:
            sub = sub[sub[outer_col].isin(outer_groups)]

        inc = sub[self._income_key].values
        w = sub[self._weight_key].values
        g_outer = sub[outer_col].values

        outer_result = theil_decomposition(inc, w, g_outer)

        inner_results = {}
        for grp in np.unique(g_outer):
            gmask = g_outer == grp
            g_inner = sub.loc[sub[outer_col] == grp, inner_col].values
            gv, gw = inc[gmask], w[gmask]
            valid = np.isfinite(gv) & np.isfinite(gw) & (gv > 0)
            if valid.sum() >= 2:
                inner_results[grp] = theil_decomposition(gv[valid], gw[valid], g_inner[valid])
            else:
                inner_results[grp] = _empty_decomposition()

        return {
            'total_theil': outer_result['total_theil'],
            'between_outer': outer_result['between'],
            'within_outer': outer_result['within'],
            'outer_result': outer_result,
            'inner_results': inner_results,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # CONVERGENCE / DISPERSION ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def sigma_convergence(self, region_col: str,
                          metric: str = 'mean',
                          year_range=None) -> pd.DataFrame:
        """
        Sigma-convergence: cross-sectional dispersion of regional means over time.

        Declining CV implies sigma-convergence (regions becoming more similar).
        """
        panel = self.regional_timeseries(region_col, year_range=year_range)
        rows = []
        for year, group in panel.groupby('year'):
            vals = group[metric].dropna().values
            if len(vals) < 2:
                continue
            rows.append({
                'year': year,
                'cv': np.std(vals) / np.mean(vals) if np.mean(vals) != 0 else np.nan,
                'sd': np.std(vals),
                'mean_of_metric': np.mean(vals),
                'n_regions': len(vals),
            })
        return pd.DataFrame(rows)

    def beta_convergence(self, region_col: str,
                         start_year: int, end_year: int,
                         metric: str = 'mean') -> pd.DataFrame:
        """
        Beta-convergence: do poorer regions grow faster?

        Returns a DataFrame ready for scatter plot + OLS regression.
        """
        initial = self.regional_metrics(start_year, region_col)
        final = self.regional_metrics(end_year, region_col)

        merged = initial[['region', metric]].merge(
            final[['region', metric]],
            on='region', suffixes=('_initial', '_final')
        )
        merged['growth_rate'] = (
            (merged[f'{metric}_final'] - merged[f'{metric}_initial'])
            / merged[f'{metric}_initial']
        )
        merged['log_initial'] = np.log(merged[f'{metric}_initial'].clip(lower=1e-10))
        return merged

    # ══════════════════════════════════════════════════════════════════════════
    # GROWTH INCIDENCE
    # ══════════════════════════════════════════════════════════════════════════

    def growth_incidence_curve(self, start_year: int, end_year: int,
                                n_quantiles: int = 20) -> pd.DataFrame:
        """
        Growth incidence curve: income growth rate by quantile.

        Shows how the gains of economic growth are distributed across
        the income distribution. Key tool from pro-poor growth literature.
        """
        quantile_points = np.linspace(0, 1, n_quantiles + 1)

        mask_start = self.df[self.year_col] == start_year
        mask_end = self.df[self.year_col] == end_year

        inc_s, w_s = self._get_income_weights(mask_start)
        inc_e, w_e = self._get_income_weights(mask_end)

        rows = []
        for i in range(n_quantiles):
            lo, hi = quantile_points[i], quantile_points[i + 1]
            mid = (lo + hi) / 2

            q_start = weighted_quantile(inc_s, w_s, mid)
            q_end = weighted_quantile(inc_e, w_e, mid)

            growth = (q_end - q_start) / q_start if q_start > 0 else np.nan

            rows.append({
                'quantile': mid,
                'quantile_pct': mid * 100,
                'start_income': q_start,
                'end_income': q_end,
                'growth_rate': growth,
                'growth_pct': growth * 100 if not np.isnan(growth) else np.nan,
            })

        return pd.DataFrame(rows)

    # ══════════════════════════════════════════════════════════════════════════
    # POLARIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def polarization_index(self, year: int, alpha: float = 1.0) -> float:
        """
        Esteban-Ray polarization index (simplified).

        Measures clustering of the distribution around distinct income levels.
        Uses quintiles as groups.

        P = sum_i sum_j  n_i^(1+alpha) * n_j * |mu_i - mu_j|
        """
        mask = self.df[self.year_col] == year
        inc, w = self._get_income_weights(mask)
        v, w_clean = _clean_pair(inc, w)
        if len(v) < 2:
            return np.nan

        n_groups = 5
        q_points = np.linspace(0, 1, n_groups + 1)
        idx = np.argsort(v)
        sv, sw = v[idx], w_clean[idx]
        cum_w = np.cumsum(sw)
        total_w = cum_w[-1]

        group_means = []
        group_shares = []
        for i in range(n_groups):
            lo_w = q_points[i] * total_w
            hi_w = q_points[i + 1] * total_w
            mask_g = (cum_w > lo_w) & (cum_w <= hi_w)
            if i == 0:
                mask_g = cum_w <= hi_w
            gv, gw = sv[mask_g], sw[mask_g]
            if gw.sum() > 0:
                group_means.append(np.average(gv, weights=gw))
                group_shares.append(gw.sum() / total_w)
            else:
                group_means.append(0)
                group_shares.append(0)

        P = 0.0
        for i in range(n_groups):
            for j in range(n_groups):
                P += (group_shares[i] ** (1 + alpha)) * group_shares[j] * abs(group_means[i] - group_means[j])
        return P

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def country_subgroup_metrics(self, year: int, demo_col: str,
                                  categories: Optional[List] = None,
                                  weight_col: Optional[str] = None) -> pd.DataFrame:
        """
        Inequality metrics per demographic subgroup at country level.

        Unlike demographic_metrics (which uses the analyzer's weight_col),
        this method allows specifying a different weight column — typically
        the original survey 'weight' for country-level comparability with LIS.

        Parameters
        ----------
        year : int
        demo_col : str — e.g. 'sex_L', 'educ_1990_L', 'age_1990_L'
        categories : list or None
        weight_col : str or None
            Weight column to use. None = use the analyzer's default weight.
            Use 'weight' for original survey weights (LIS comparability).
        """
        mask_year = self.df[self.year_col] == year
        sub = self.df.loc[mask_year].dropna(subset=[demo_col])
        if categories is not None:
            sub = sub[sub[demo_col].isin(categories)]

        inc_col = self._income_key
        w_col = weight_col if weight_col is not None else self._weight_key
        if weight_col is not None and weight_col != self._weight_key:
            w_vals = pd.to_numeric(sub[weight_col], errors='coerce').values
        else:
            w_vals = None  # use precomputed

        rows = []
        for cat in sorted(sub[demo_col].unique()):
            cmask = sub[demo_col] == cat
            inc = sub.loc[cmask, inc_col].values
            w = w_vals[cmask.values] if w_vals is not None else sub.loc[cmask, self._weight_key].values
            metrics = compute_all_metrics(inc, w)
            metrics['subgroup'] = cat
            metrics['year'] = year
            rows.append(metrics)
        return pd.DataFrame(rows)

    def country_subgroup_timeseries(self, demo_col: str,
                                    categories: Optional[List] = None,
                                    year_range=None,
                                    weight_col: Optional[str] = None) -> pd.DataFrame:
        """
        Subgroup metrics over time at country level.
        See country_subgroup_metrics for parameters.
        """
        rows = []
        for year in self._years(year_range):
            yearly = self.country_subgroup_metrics(year, demo_col, categories, weight_col=weight_col)
            if len(yearly) > 0:
                rows.append(yearly)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def summary(self) -> dict:
        """Quick summary of the loaded dataset."""
        years = self._years()
        return {
            'n_observations': len(self.df),
            'year_range': (min(years), max(years)) if years else None,
            'n_years': len(years),
            'income_col': self.income_col,
            'weight_col': self.weight_col,
            'deflator_col': self.deflator_col,
            'income_non_null': self.df[self._income_key].notna().sum(),
            'weight_non_null': self.df[self._weight_key].notna().sum(),
        }

    def available_regions(self, region_col: str, year: Optional[int] = None) -> list:
        """List unique regions available (optionally for a specific year)."""
        sub = self.df if year is None else self.df[self.df[self.year_col] == year]
        return sorted(sub[region_col].dropna().unique())


# ══════════════════════════════════════════════════════════════════════════════
# LIS ANALYZER — pre-computed aggregates
# ══════════════════════════════════════════════════════════════════════════════

class LISAnalyzer:
    """
    Analyzer for LIS pre-computed regional/group aggregates.

    LIS data has columns like 'pitotalnet_gini', 'hitotal_theil', etc.
    This class provides easy access with optional deflation.

    Parameters
    ----------
    df : pd.DataFrame
        LIS data (LIS_Voiv.csv or LIS_Groups.csv).
    income_type : str
        Income variable prefix:
        - 'pitotalnet'      : Person total net income (closest to CBOS income_p)
        - 'pitotalnet_pos'  : Person total net income, positive only
        - 'pilab_pens'      : Person labor + pension income
        - 'hitotalnet'      : Household total net income (closest to CBOS income_hh)
        - 'hitotalnet_pos'  : Household total net income, positive only
        - 'hilab_pens'      : Household labor + pension income
    deflator_col : str or None
        Deflator column for real values, e.g. 'deflator_2017'.
    region_col : str
        Column identifying regions/groups.
    year_col : str

    Example
    -------
    >>> lis_voiv = pd.read_csv('LIS_Voiv.csv')
    >>> la = LISAnalyzer(lis_voiv, income_type='pitotalnet')
    >>> panel = la.regional_panel()
    >>> gini = la.get_metric('gini')
    """

    METRIC_SUFFIXES = [
        'N_total', 'Nw_total', 'mean', 'median',
        'p10', 'p25', 'p75', 'p90', 'p99',
        'p90p10', 'p90p50', 'p50p10',
        'gini', 'theil', 'palma',
        'N_Bottom_50', 'Nw_Bottom_50', 'Bottom_50',
        'N_P50_90', 'Nw_P50_90', 'P50_90',
        'N_Top_10', 'Nw_Top_10', 'Top_10',
        'N_Top_1', 'Nw_Top_1', 'Top_1',
        'share_Bottom_50', 'share_P50_90', 'share_Top_10', 'share_Top_1',
    ]

    def __init__(self, df: pd.DataFrame,
                 income_type: str = 'pitotalnet',
                 deflator_col: Optional[str] = None,
                 region_col: str = 'region',
                 year_col: str = 'year'):

        self.df = df.copy()
        self.income_type = income_type
        self.deflator_col = deflator_col
        self.region_col = region_col
        self.year_col = year_col

        # Build column mapping: short name -> actual column
        self._col_map = {}
        for suffix in self.METRIC_SUFFIXES:
            full_col = f'{income_type}_{suffix}'
            if full_col in self.df.columns:
                self._col_map[suffix] = full_col

        # Convert monetary columns to numeric, divide by 12 (annual → monthly),
        # and deflate
        monetary_keys = ['mean', 'median', 'p10', 'p25', 'p75', 'p90', 'p99',
                         'Bottom_50', 'P50_90', 'Top_10', 'Top_1']
        for key in monetary_keys:
            if key in self._col_map:
                self.df[self._col_map[key]] = pd.to_numeric(
                    self.df[self._col_map[key]], errors='coerce'
                ) / 12  # LIS income is annual; convert to monthly for CBOS comparability
                if deflator_col is not None and deflator_col in self.df.columns:
                    defl = pd.to_numeric(self.df[deflator_col], errors='coerce')
                    self.df[self._col_map[key]] = self.df[self._col_map[key]] * defl

        # Convert non-monetary to numeric
        for key, col in self._col_map.items():
            if key not in monetary_keys:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def _col(self, metric_suffix: str) -> str:
        """Get full column name for a metric suffix."""
        if metric_suffix in self._col_map:
            return self._col_map[metric_suffix]
        raise KeyError(f"Metric '{metric_suffix}' not found for income type "
                       f"'{self.income_type}'. Available: {list(self._col_map.keys())}")

    def get_metric(self, metric: str) -> pd.DataFrame:
        """
        Get a single metric as a year x region pivot table.

        Parameters
        ----------
        metric : str — e.g. 'gini', 'theil', 'mean', 'share_Top_10', 'p90p10'

        Returns
        -------
        pd.DataFrame : rows = years, columns = regions.
        """
        col = self._col(metric)
        return self.df.pivot_table(values=col, index=self.year_col, columns=self.region_col)

    def regional_panel(self, metrics: Optional[List[str]] = None,
                       region_id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Long-form panel: year x region with selected metrics.

        Parameters
        ----------
        metrics : list of str or None — metric suffixes. None = all available.
        region_id_col : str or None
            Column to include as region identifier (e.g. 'region_id').
        """
        if metrics is None:
            metrics = list(self._col_map.keys())

        cols = [self.year_col, self.region_col]
        if region_id_col is not None and region_id_col in self.df.columns:
            cols.append(region_id_col)
        rename = {}
        for m in metrics:
            if m in self._col_map:
                cols.append(self._col_map[m])
                rename[self._col_map[m]] = m

        result = self.df[cols].copy()
        result.rename(columns=rename, inplace=True)
        return result

    def country_timeseries(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Country-level time series by averaging across regions
        (weighted by Nw_total where possible).

        Note: for inequality indices, this is a population-weighted average
        across regions — NOT the same as computing from pooled microdata.
        """
        if metrics is None:
            metrics = list(self._col_map.keys())

        sum_metrics = {'N_total', 'Nw_total',
                       'N_Bottom_50', 'Nw_Bottom_50',
                       'N_P50_90', 'Nw_P50_90',
                       'N_Top_10', 'Nw_Top_10',
                       'N_Top_1', 'Nw_Top_1'}

        nw_col = self._col_map.get('Nw_total')

        rows = []
        for year, group in self.df.groupby(self.year_col):
            row = {'year': year}
            if nw_col is not None:
                nw = pd.to_numeric(group[nw_col], errors='coerce')
            else:
                nw = pd.Series(np.ones(len(group)))

            for m in metrics:
                if m not in self._col_map:
                    continue
                vals = pd.to_numeric(group[self._col_map[m]], errors='coerce')
                valid = vals.notna() & nw.notna()
                if valid.sum() == 0:
                    row[m] = np.nan
                elif m in sum_metrics:
                    row[m] = vals[valid].sum()
                else:
                    row[m] = np.average(vals[valid], weights=nw[valid])
            rows.append(row)

        return pd.DataFrame(rows).set_index('year')

    def regions(self) -> list:
        """List available regions."""
        return sorted(self.df[self.region_col].dropna().unique())

    def years(self) -> list:
        """List available years."""
        return sorted(self.df[self.year_col].dropna().unique())

    def available_metrics(self) -> list:
        """List available metric suffixes for current income type."""
        return list(self._col_map.keys())

    def summary(self) -> dict:
        """Quick summary."""
        return {
            'income_type': self.income_type,
            'n_regions': len(self.regions()),
            'year_range': (min(self.years()), max(self.years())),
            'n_metrics': len(self._col_map),
            'available_metrics': self.available_metrics(),
            'deflator': self.deflator_col,
        }

    def theil_contributions(self, year: int,
                            region_id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Per-region contributions to total Theil T from pre-computed stats.
        Uses Nw_total, mean, and theil per region.

        Parameters
        ----------
        year : int
        region_id_col : str or None
            Column to include as region identifier (e.g. 'region_id').
        """
        sub = self.df[self.df[self.year_col] == year].copy()
        if sub.empty:
            return pd.DataFrame()
        nw_col = self._col_map.get('Nw_total')
        mean_col = self._col_map.get('mean')
        theil_col = self._col_map.get('theil')
        if nw_col is None or mean_col is None:
            return pd.DataFrame()
        sub = sub.dropna(subset=[nw_col, mean_col])
        nw = sub[nw_col].values
        mu_g = sub[mean_col].values
        pos = (nw > 0) & (mu_g > 0)
        if pos.sum() < 2:
            return pd.DataFrame()
        sub_pos = sub[pos].copy()
        nw, mu_g = nw[pos], mu_g[pos]
        mu = (nw * mu_g).sum() / nw.sum()
        s_g = (nw * mu_g) / (nw * mu_g).sum()
        T_g = sub_pos[theil_col].values if theil_col is not None else np.zeros(len(sub_pos))
        b_g = s_g * np.log(mu_g / mu)
        w_g = s_g * T_g
        t_g = b_g + w_g
        total = t_g.sum()
        result = pd.DataFrame({
            'group': sub_pos[self.region_col].values,
            'income_share': s_g,
            'group_mean': mu_g,
            'group_theil': T_g,
            'group_N': nw,
            'between_contribution': b_g,
            'within_contribution': w_g,
            'total_contribution': t_g,
            'contribution_pct': t_g / total * 100 if total > 0 else np.nan,
            'year': year,
        })
        if region_id_col is not None and region_id_col in sub_pos.columns:
            result['region_id'] = sub_pos[region_id_col].values
        return result

    def theil_contribution_timeseries(self, year_range=None,
                                      region_id_col: Optional[str] = None) -> pd.DataFrame:
        """Per-region Theil contributions over time."""
        years = self.years()
        if year_range is not None:
            years = [y for y in years if year_range[0] <= y <= year_range[1]]
        frames = []
        for year in years:
            yearly = self.theil_contributions(year, region_id_col=region_id_col)
            if len(yearly) > 0:
                frames.append(yearly)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# LIS COUNTRY ANALYZER — country-level subgroup data
# ══════════════════════════════════════════════════════════════════════════════

class LISCountryAnalyzer:
    """
    Analyzer for LIS country-level data with demographic subgroups.

    Expects LIS_Country_full.csv with a 'subgroup' column containing:
    - 'households' / 'persons' for aggregate totals
    - 'h_*' subgroups (household-head-based) with hitotalnet_* data
    - 'p_*' subgroups (person-based) with pitotalnet_* data

    Parameters
    ----------
    df : pd.DataFrame
        LIS_Country_full.csv data.
    income_type : str
        Income variable prefix. Must match the unit of analysis:
        - 'pitotalnet'      : Person-level (for 'persons' / 'p_*' subgroups)
        - 'pitotalnet_pos'  : Person-level, positive only
        - 'pilab_pens'      : Person labor + pension
        - 'hitotalnet'      : Household-level (for 'households' / 'h_*' subgroups)
        - 'hitotalnet_pos'  : Household-level, positive only
        - 'hilab_pens'      : Household labor + pension
    deflator_col : str or None
    subgroup_col : str
    year_col : str
    """

    METRIC_SUFFIXES = LISAnalyzer.METRIC_SUFFIXES

    # Mapping: subgroup prefix category -> list of subgroup name patterns
    SUBGROUP_CATEGORIES = {
        'sex':      ['sex_kobiety', 'sex_mezczyzni'],
        'edu':      ['edu_'],
        'age':      ['age_'],
        'hhsize':   ['hhsize_'],
        'popclass': ['popclass_'],
    }

    def __init__(self, df: pd.DataFrame,
                 income_type: str = 'pitotalnet',
                 deflator_col: Optional[str] = None,
                 subgroup_col: str = 'subgroup',
                 year_col: str = 'year'):

        self.df = df.copy()
        self.income_type = income_type
        self.deflator_col = deflator_col
        self.subgroup_col = subgroup_col
        self.year_col = year_col

        # Determine unit prefix for filtering subgroups
        self._unit_prefix = 'p_' if income_type.startswith('pi') else 'h_'
        self._aggregate_row = 'persons' if income_type.startswith('pi') else 'households'

        # Build column mapping (same as LISAnalyzer)
        self._col_map = {}
        for suffix in self.METRIC_SUFFIXES:
            full_col = f'{income_type}_{suffix}'
            if full_col in self.df.columns:
                self._col_map[suffix] = full_col

        # Convert monetary columns to numeric, divide by 12 (annual → monthly),
        # and deflate
        monetary_keys = ['mean', 'median', 'p10', 'p25', 'p75', 'p90', 'p99',
                         'Bottom_50', 'P50_90', 'Top_10', 'Top_1']
        for key in monetary_keys:
            if key in self._col_map:
                self.df[self._col_map[key]] = pd.to_numeric(
                    self.df[self._col_map[key]], errors='coerce'
                ) / 12  # LIS income is annual; convert to monthly for CBOS comparability
                if deflator_col is not None and deflator_col in self.df.columns:
                    defl = pd.to_numeric(self.df[deflator_col], errors='coerce')
                    self.df[self._col_map[key]] = self.df[self._col_map[key]] * defl

        for key, col in self._col_map.items():
            if key not in monetary_keys:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def _col(self, metric_suffix: str) -> str:
        if metric_suffix in self._col_map:
            return self._col_map[metric_suffix]
        raise KeyError(f"Metric '{metric_suffix}' not found for income type "
                       f"'{self.income_type}'. Available: {list(self._col_map.keys())}")

    def aggregate_timeseries(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Country-level aggregate time series (from 'households' or 'persons' rows).

        Returns one row per year with selected metrics.
        """
        if metrics is None:
            metrics = list(self._col_map.keys())

        sub = self.df[self.df[self.subgroup_col] == self._aggregate_row].copy()
        cols = [self.year_col]
        rename = {}
        for m in metrics:
            if m in self._col_map:
                cols.append(self._col_map[m])
                rename[self._col_map[m]] = m
        result = sub[cols].copy()
        result.rename(columns=rename, inplace=True)
        return result.set_index(self.year_col).sort_index()

    def subgroup_metrics(self, year: int,
                         category: Optional[str] = None,
                         subgroups: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get metrics for demographic subgroups in a given year.

        Parameters
        ----------
        year : int
        category : str or None
            Filter by subgroup category: 'sex', 'edu', 'age', 'hhsize', 'popclass'.
            If None, returns all subgroups matching the income type's unit prefix.
        subgroups : list of str or None
            Explicit list of subgroup names to include.
        """
        sub = self.df[self.df[self.year_col] == year].copy()

        # Filter to relevant unit prefix (p_ or h_) subgroups
        mask = sub[self.subgroup_col].str.startswith(self._unit_prefix)
        if category is not None and category in self.SUBGROUP_CATEGORIES:
            patterns = self.SUBGROUP_CATEGORIES[category]
            cat_mask = pd.Series(False, index=sub.index)
            for pat in patterns:
                cat_mask |= sub[self.subgroup_col].str.contains(
                    self._unit_prefix + pat, regex=False
                )
            mask = mask & cat_mask
        sub = sub[mask]

        if subgroups is not None:
            sub = sub[sub[self.subgroup_col].isin(subgroups)]

        cols = [self.subgroup_col]
        rename = {}
        for m in self._col_map:
            cols.append(self._col_map[m])
            rename[self._col_map[m]] = m
        result = sub[cols].copy()
        result.rename(columns=rename, inplace=True)
        result['year'] = year
        result.rename(columns={self.subgroup_col: 'subgroup'}, inplace=True)
        return result.reset_index(drop=True)

    def subgroup_timeseries(self, category: Optional[str] = None,
                            subgroups: Optional[List[str]] = None,
                            year_range=None) -> pd.DataFrame:
        """
        Subgroup metrics over time (panel: year × subgroup).
        """
        years = sorted(self.df[self.year_col].dropna().unique())
        if year_range is not None:
            years = [y for y in years if year_range[0] <= y <= year_range[1]]
        rows = []
        for year in years:
            yearly = self.subgroup_metrics(year, category, subgroups)
            if len(yearly) > 0:
                rows.append(yearly)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def available_subgroups(self, category: Optional[str] = None,
                            year: Optional[int] = None) -> list:
        """
        List subgroups available for the current income type's unit.
        Optionally filter by category and/or year.
        """
        sub = self.df if year is None else self.df[self.df[self.year_col] == year]
        mask = sub[self.subgroup_col].str.startswith(self._unit_prefix)
        if category is not None and category in self.SUBGROUP_CATEGORIES:
            patterns = self.SUBGROUP_CATEGORIES[category]
            cat_mask = pd.Series(False, index=sub.index)
            for pat in patterns:
                cat_mask |= sub[self.subgroup_col].str.contains(
                    self._unit_prefix + pat, regex=False
                )
            mask = mask & cat_mask
        return sorted(sub.loc[mask, self.subgroup_col].unique())

    def years(self) -> list:
        return sorted(self.df[self.year_col].dropna().unique())

    def available_metrics(self) -> list:
        return list(self._col_map.keys())

    def summary(self) -> dict:
        return {
            'income_type': self.income_type,
            'unit': self._unit_prefix.rstrip('_'),
            'aggregate_row': self._aggregate_row,
            'n_subgroups': len(self.available_subgroups()),
            'year_range': (min(self.years()), max(self.years())),
            'n_metrics': len(self._col_map),
            'deflator': self.deflator_col,
        }

    def theil_contributions(self, year: int,
                            category: Optional[str] = None,
                            subgroups: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Per-subgroup contributions to total Theil T from pre-computed stats.
        Uses Nw_total, mean, and theil per subgroup.
        """
        sg = self.subgroup_metrics(year, category, subgroups)
        if sg.empty or 'mean' not in sg.columns or 'Nw_total' not in sg.columns:
            return pd.DataFrame()
        sg = sg.dropna(subset=['mean', 'Nw_total'])
        nw = sg['Nw_total'].values
        mu_g = sg['mean'].values
        pos = (nw > 0) & (mu_g > 0)
        if pos.sum() < 2:
            return pd.DataFrame()
        sg_pos = sg[pos].copy()
        nw, mu_g = nw[pos], mu_g[pos]
        mu = (nw * mu_g).sum() / nw.sum()
        s_g = (nw * mu_g) / (nw * mu_g).sum()
        T_g = sg_pos['theil'].values if 'theil' in sg_pos.columns else np.zeros(len(sg_pos))
        b_g = s_g * np.log(mu_g / mu)
        w_g = s_g * T_g
        t_g = b_g + w_g
        total = t_g.sum()
        return pd.DataFrame({
            'group': sg_pos['subgroup'].values,
            'income_share': s_g,
            'group_mean': mu_g,
            'group_theil': T_g,
            'group_N': nw,
            'between_contribution': b_g,
            'within_contribution': w_g,
            'total_contribution': t_g,
            'contribution_pct': t_g / total * 100 if total > 0 else np.nan,
            'year': year,
        })

    def theil_contribution_timeseries(self, category: Optional[str] = None,
                                      subgroups: Optional[List[str]] = None,
                                      year_range=None) -> pd.DataFrame:
        """Per-subgroup Theil contributions over time."""
        years = sorted(self.df[self.year_col].dropna().unique())
        if year_range is not None:
            years = [y for y in years if year_range[0] <= y <= year_range[1]]
        frames = []
        for year in years:
            yearly = self.theil_contributions(year, category, subgroups)
            if len(yearly) > 0:
                frames.append(yearly)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TOOLS — align CBOS and LIS results
# ══════════════════════════════════════════════════════════════════════════════

def compare_cbos_lis(cbos_ts: pd.DataFrame, lis_ts: pd.DataFrame,
                     metrics: List[str] = None,
                     on: str = 'year') -> pd.DataFrame:
    """
    Merge CBOS and LIS time series for side-by-side comparison.

    Parameters
    ----------
    cbos_ts : pd.DataFrame — from CBOSAnalyzer.country_timeseries().
    lis_ts : pd.DataFrame — from LISAnalyzer.country_timeseries().
    metrics : list of str — which metrics to compare.
    on : str — merge key (default 'year').

    Returns
    -------
    pd.DataFrame with columns: year, metric_cbos, metric_lis, metric_diff, metric_ratio
    """
    if metrics is None:
        metrics = ['gini', 'theil', 'palma', 'mean', 'median',
                   'share_Bottom_50', 'share_Top_10', 'share_Top_1',
                   'p90p10']

    cbos = cbos_ts.reset_index() if on not in cbos_ts.columns else cbos_ts.copy()
    lis = lis_ts.reset_index() if on not in lis_ts.columns else lis_ts.copy()

    common = [m for m in metrics if m in cbos.columns and m in lis.columns]

    cbos_sub = cbos[[on] + common].copy()
    lis_sub = lis[[on] + common].copy()

    merged = cbos_sub.merge(lis_sub, on=on, suffixes=('_cbos', '_lis'), how='outer')

    for m in common:
        cb, li = f'{m}_cbos', f'{m}_lis'
        merged[f'{m}_diff'] = merged[cb] - merged[li]
        merged[f'{m}_ratio'] = merged[cb] / merged[li]

    return merged.sort_values(on)


def compare_regional(cbos_panel: pd.DataFrame, lis_panel: pd.DataFrame,
                     metric: str = 'gini',
                     region_col_cbos: str = 'region',
                     region_col_lis: str = 'region',
                     year_col: str = 'year',
                     region_map: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compare a metric across CBOS and LIS at regional level.

    Parameters
    ----------
    cbos_panel : pd.DataFrame — from CBOSAnalyzer.regional_timeseries()
    lis_panel : pd.DataFrame — from LISAnalyzer.regional_panel()
    metric : str
    region_map : dict or None — {cbos_region_name: lis_region_name}

    Returns
    -------
    pd.DataFrame : year, region, metric_cbos, metric_lis, diff
    """
    cbos = cbos_panel[[year_col, region_col_cbos, metric]].copy()
    cbos.columns = [year_col, 'region', f'{metric}_cbos']

    lis = lis_panel[[year_col, region_col_lis, metric]].copy()
    lis.columns = [year_col, 'region', f'{metric}_lis']

    if region_map:
        cbos['region'] = cbos['region'].map(region_map).fillna(cbos['region'])

    merged = cbos.merge(lis, on=[year_col, 'region'], how='outer')
    merged[f'{metric}_diff'] = merged[f'{metric}_cbos'] - merged[f'{metric}_lis']

    return merged.sort_values([year_col, 'region'])

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY AND PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def assign_geometries(df: pd.DataFrame, geo_df: gpd.GeoDataFrame, merge_on: str = 'region_id') -> pd.DataFrame:
    """
    Assign geometries to a DataFrame by merging on a shared column.

    Parameters
    ----------
    df : pd.DataFrame with a merge_on column.
    geo_df : gpd.GeoDataFrame with 'id' and 'geometry' columns.
    merge_on : str — column in df to merge with 'id' in geo_df.

    Returns
    -------
    pd.DataFrame with an added 'geometry' column from geo_df.
    """
    merged = df.merge(geo_df[['id', 'geometry']], left_on=merge_on, right_on='id', how='left')
    return merged