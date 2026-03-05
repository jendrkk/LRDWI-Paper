"""
Demographic Estimator Module — v5.1 Architecture & Infrastructure
=================================================================

Standalone module for all numerical estimation logic that produces
estimated cross tables for Polish administrative divisions (gminas).

Separation of concerns:
  - geoTERYT_db.py  = storage / retrieval / data handling
  - demographic_estimator.py = numerical estimation

The estimation follows a three-layer pipeline for each
(variable type × prediction section) combination:

  Layer 1: Temporal seed generation via log-linear interpolation
  Layer 2: Marginal fitting via multi-dimensional IPF
  Layer 3: Hierarchical consistency enforcement via Gurobi QP

Estimation results are stored as E_ prefix subjects to keep
observed M_ subjects untouched.

Author: Jedrzej Slowinski and Claude Opus 4.6
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

try:
    import ipfn as _ipfn_module
    IPFN_AVAILABLE = True
except ImportError:
    IPFN_AVAILABLE = False
    warnings.warn("ipfn package not available — IPF-based marginal fitting disabled.")

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    warnings.warn("gurobipy not available — Gurobi QP solver disabled; "
                  "will use iterative IPF fallback for hierarchical consistency.")

if TYPE_CHECKING:
    from geoTERYT_db import (
        GeoTERYTDatabase, TERYTRecord, CrossTable, DataSeries,
    )


# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================

# Numerical parameters
EPSILON = 1e-10          # Additive smoothing for log-space & zero-cell handling
IPF_MAX_ITER = 1000      # Maximum IPF iterations
IPF_CONVERGENCE = 1e-6   # IPF convergence threshold

# Prediction sections
PREDICTION_1990_RANGE = range(1986, 2003)   # 1986–2002 inclusive
PREDICTION_2000_RANGE = range(1999, 2026)   # 1999–2025 inclusive

# Census anchor years
CENSUS_YEARS = [1988, 2002, 2011, 2021]

# Aggregation rules — ONLY these rodz types are summed when building
# parent totals from children.  Never include 4, 5, 8, 9.
RODZ_AGGREGATION_SET = {'1', '2', '3'}

# ── Subject-to-variable-type mapping ──
# Each estimation pipeline operates on a (variable_type, prediction_section) pair.
# This mapping defines which M_ subjects serve as anchors.

ANCHOR_SUBJECTS = {
    # variable_type: {prediction_section: {'anchor_subjects': [...], 'marginal_subjects': [...]}}
    'age_sex': {
        '2000': {
            'anchor_subjects': ['M_age_sex'],
            'marginal_subjects': ['M_age_sex'],
        },
        '1990': {
            'anchor_subjects': ['M_age_sex', 'M_age_1990'],
            'marginal_subjects': ['H_age_sex'],
        },
    },
    'educ': {
        '2000': {
            'anchor_subjects': ['M_educ_2000'],
            'marginal_subjects': ['P2350', 'P4092'],
        },
        '1990': {
            'anchor_subjects': ['M_educ_1990'],
            'marginal_subjects': ['H_sex_educ'],
        },
    },
    'educ_sex': {
        '2000': {
            'anchor_subjects': ['M_educ_sex_2000'],
            'marginal_subjects': ['P2350', 'P4092'],
        },
        '1990': {
            'anchor_subjects': ['M_educ_sex_1990'],
            'marginal_subjects': ['H_sex_educ'],
        },
    },
    'hh_size': {
        '2000': {
            'anchor_subjects': ['M_hh_size_2000'],
            'marginal_subjects': [],
        },
        '1990': {
            'anchor_subjects': ['M_hh_size_1990'],
            'marginal_subjects': [],
        },
    },
    'age_educ': {
        '2000': {
            'anchor_subjects': ['M_pop__age_educ'],
            'marginal_subjects': [],   # uses E_age_sex & E_educ as cross-constraints
        },
    },
}

# E_ prefix output subject names
E_SUBJECT_NAMES = {
    ('age_sex',   '2000'): 'E_age_sex_2000',
    ('age_sex',   '1990'): 'E_age_sex_1990',
    ('educ',      '2000'): 'E_educ_2000',
    ('educ',      '1990'): 'E_educ_1990',
    ('educ_sex',  '2000'): 'E_educ_sex_2000',
    ('educ_sex',  '1990'): 'E_educ_sex_1990',
    ('hh_size',   '2000'): 'E_hh_size_2000',
    ('hh_size',   '1990'): 'E_hh_size_1990',
    ('age_educ',  '2000'): 'E_age_educ_2000',
}


# ==============================================================================
# HELPER:  Aggregation children
# ==============================================================================

def _get_aggregation_children(
    record: 'TERYTRecord',
    db: 'GeoTERYTDatabase',
    year: int = 1999,
) -> List[str]:
    """Return the child teryt_ids whose data should be summed to produce
    the parent record's total.

    Parameters
    ----------
    record : TERYTRecord
        The parent unit.
    db : GeoTERYTDatabase
        The database instance.
    year : int
        The year for which to retrieve the hierarchy (default 1999).
        Falls back to nearest available year if exact match not found.

    Rules
    -----
    - Powiat  (teryt[-1] == '0'): children with rodz ∈ {1, 2, 3},
      with encompassing-child deduplication (Warsaw 1999–2001).
      Falls back to including rodz=8 (city districts) if no
      rodz ∈ {1,2,3} children are found.
    - Voivodeship (teryt[2:] == '00000'): all powiats + all direct
      gminas with rodz ∈ {1, 2, 3}
    - Country (teryt == '0000000'): always uses new voivodeships (16)
      for estimation aggregation, regardless of the year hierarchy
      (which may point to old voivodeships for pre-1995).
    - NEVER include rodz 4, 5, 9 in aggregation sums.
    """
    from geoTERYT_db import filter_aggregation_children
    tid = record.teryt_id

    # Country level — always use the 16 new voivodeships for estimation
    if tid == '0000000':
        children = record.get_children(year)
        # If children include old voivodeships (IDs >= '5100000'),
        # fall back to new voivodeships from a post-reform year
        if children and any(c[:2] > '50' for c in children if len(c) == 7):
            children = record.get_children(max(year, 1999))
        return children

    # Voivodeship level
    if tid[2:] == '00000':
        children = []
        for child_tid in record.get_children(year):
            child = db._records.get(child_tid)
            if child is None:
                continue
            if child.level == 5:            # powiat
                children.append(child_tid)
            elif child.level == 6:          # gmina
                rodz = child_tid[-1]
                if rodz in RODZ_AGGREGATION_SET:
                    children.append(child_tid)
        return children

    # Powiat level (last digit '0')
    if tid[-1] == '0':
        raw_children = record.get_children(year)
        raw = [
            child_tid for child_tid in raw_children
            if child_tid[-1] in RODZ_AGGREGATION_SET
        ]
        filtered = filter_aggregation_children(raw, year, db._records)

        # If no rodz ∈ {1,2,3} children found, try rodz=8
        # (Warsaw city districts post-reorganisation)
        if not filtered:
            raw_8 = [
                child_tid for child_tid in raw_children
                if child_tid[-1] == '8'
            ]
            if raw_8:
                return raw_8

        return filtered

    # Anything else (gmina, sub-division) — no children to aggregate
    return []


# ==============================================================================
# PROVENANCE CROSS TABLE
# ==============================================================================

class ProvenanceMask:
    """Boolean cross table marking observed (True) vs estimated (False) cells.

    One mask per (E_subject, teryt_id) pair.  Shape matches the E_ subject's
    CrossTable dimensions.
    """

    __slots__ = ('subject_id', 'dim_names', 'dim_labels', '_masks')

    def __init__(self, subject_id: str, dim_names: List[str],
                 dim_labels: Dict[str, List[str]],
                 year_range: List[int] | None = None):
        self.subject_id = subject_id
        self.dim_names = dim_names
        self.dim_labels = dim_labels

        shape = tuple(len(dim_labels[d]) for d in dim_names)
        if year_range is None:
            from geoTERYT_db import YEAR_RANGE_FULL
            year_range = YEAR_RANGE_FULL
        # All cells start as False (= estimated)
        self._masks: Dict[int, np.ndarray] = {
            yr: np.zeros(shape, dtype=bool) for yr in year_range
        }

    def mark_observed(self, year: int, idx: tuple | None = None):
        """Mark one cell or all cells for *year* as directly observed."""
        if year not in self._masks:
            return
        if idx is None:
            self._masks[year][:] = True
        else:
            self._masks[year][idx] = True

    def mark_all_estimated(self, year: int):
        """Mark every cell for *year* as estimated."""
        if year in self._masks:
            self._masks[year][:] = False

    def is_observed(self, year: int, idx: tuple | None = None) -> bool | np.ndarray:
        """Query provenance for a single cell or the whole table."""
        mask = self._masks.get(year)
        if mask is None:
            return False
        if idx is None:
            return mask.copy()
        return bool(mask[idx])

    def fraction_observed(self, year: int) -> float:
        """Fraction of cells that are directly observed for *year*."""
        mask = self._masks.get(year)
        if mask is None:
            return 0.0
        return float(np.mean(mask))


# ==============================================================================
# DEMOGRAPHIC ESTIMATOR
# ==============================================================================

class DemographicEstimator:
    """Top-level estimator that orchestrates the three-layer pipeline.

    Parameters
    ----------
    db : GeoTERYTDatabase
        The database providing all observed data and receiving estimation
        results.
    verbose : bool
        Print progress messages.
    """

    def __init__(self, db: 'GeoTERYTDatabase', verbose: bool = True):
        self.db = db
        self.verbose = verbose

        # Store provenance masks:  E_subject_id -> teryt_id -> ProvenanceMask
        self.provenance: Dict[str, Dict[str, ProvenanceMask]] = {}

        # Track which estimation pipelines have been run
        self._completed: Set[Tuple[str, str]] = set()

        # Performance: skip DataSeries creation for E_ subjects by default
        # (only CrossTables are needed for the estimation pipeline)
        self.create_data_series = False

        self._log(f"DemographicEstimator initialised  "
                  f"(Gurobi={'YES' if GUROBI_AVAILABLE else 'NO'}, "
                  f"IPFN={'YES' if IPFN_AVAILABLE else 'NO'})")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Upstream data validation & cleaning
    # ------------------------------------------------------------------

    # Known TERYT merger artifacts: (teryt_id, subject_id, year_to_nan)
    _KNOWN_DATA_ANOMALIES: List[Tuple[str, str, int]] = [
        # Wesoła absorbed into Warsaw in 2002 — M_age_sex has a
        # spurious spike at 2002 when population drops to near-zero
        ('1412031', 'M_age_sex', 2002),
        ('1412031', 'M_age_1990', 2002),
    ]

    def _validate_upstream_data(self):
        """Scan M_ cross tables for extreme YoY changes and clean known
        anomalies before estimation.

        1. NaN-out known data anomalies (TERYT merger artifacts).
        2. Flag any M_ value with >1000 % year-over-year change for
           audit (logged but not automatically corrected).
        """
        self._log("\n── Upstream data validation ──")

        # ── Step 1: clean known anomalies ──
        n_cleaned = 0
        for tid, sid, bad_year in self._KNOWN_DATA_ANOMALIES:
            rec = self.db._records.get(tid)
            if rec is None:
                continue
            ct = rec.cross_tables.get(sid)
            if ct is None:
                continue
            tbl = ct.tables.get(bad_year)
            if tbl is None or np.all(np.isnan(tbl)):
                continue
            ct.tables[bad_year] = np.full(ct.shape, np.nan)
            n_cleaned += 1
            self._log(
                f"  Cleaned: {tid} / {sid} / {bad_year} "
                f"(known TERYT merger artifact)"
            )
        self._log(f"  Known anomalies cleaned: {n_cleaned}")

        # ── Step 2: scan for extreme YoY changes ──
        m_subjects = [
            s for s in ['M_age_sex', 'M_educ_2000', 'M_educ_1990',
                        'M_educ_sex_2000', 'M_educ_sex_1990',
                        'M_hh_size_2000', 'M_hh_size_1990',
                        'M_age_1990']
        ]

        n_flagged = 0
        for tid, rec in self.db._records.items():
            if rec.level != 6:  # only scan gminas
                continue
            for sid in m_subjects:
                ct = rec.cross_tables.get(sid)
                if ct is None:
                    continue
                data_years = ct.years_with_data
                if len(data_years) < 2:
                    continue
                for i in range(1, len(data_years)):
                    y_prev, y_cur = data_years[i - 1], data_years[i]
                    t_prev = ct.tables.get(y_prev)
                    t_cur = ct.tables.get(y_cur)
                    if t_prev is None or t_cur is None:
                        continue
                    sum_prev = np.nansum(np.abs(t_prev))
                    sum_cur = np.nansum(np.abs(t_cur))
                    if sum_prev < 1.0:
                        continue
                    ratio = sum_cur / sum_prev
                    if ratio > 11.0 or ratio < 1.0 / 11.0:
                        if n_flagged < 20:
                            self._log(
                                f"  ⚠  {tid}/{sid}: "
                                f"YoY {y_prev}→{y_cur} "
                                f"ratio={ratio:.1f}x"
                            )
                        n_flagged += 1

        if n_flagged > 20:
            self._log(
                f"  … and {n_flagged - 20} more flagged"
            )
        self._log(f"  Extreme YoY changes flagged: {n_flagged}")
        self._log("── Validation complete ──\n")

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_all(self):
        """Run every estimation pipeline in the correct dependency order.

        Order: age×sex → educ → educ_sex → hh_size → age_educ
        Within each: Prediction2000 first, then Prediction1990.
        """
        # Pre-estimation data cleaning
        self._validate_upstream_data()

        pipeline_order = [
            ('age_sex',  '2000'),
            ('age_sex',  '1990'),
            ('educ',     '2000'),
            ('educ',     '1990'),
            ('educ_sex', '2000'),
            ('educ_sex', '1990'),
            ('hh_size',  '2000'),
            ('hh_size',  '1990'),
            ('age_educ', '2000'),
        ]
        for var_type, section in pipeline_order:
            self.run_pipeline(var_type, section)

    def run_pipeline(self, variable_type: str, prediction_section: str):
        """Run a single (variable_type × prediction_section) pipeline.

        This is the top-level dispatcher that calls the variable-specific
        ``estimate_*`` methods implemented in work chunks C–D.
        """
        key = (variable_type, prediction_section)
        if key in self._completed:
            self._log(f"  ⏭  {key} already completed — skipping")
            return

        e_sid = E_SUBJECT_NAMES.get(key)
        if e_sid is None:
            raise ValueError(f"Unknown pipeline: {key}")

        self._log(f"\n{'='*60}")
        self._log(f"  PIPELINE: {variable_type} / Prediction{prediction_section}")
        self._log(f"  Output subject: {e_sid}")
        self._log(f"{'='*60}")

        # Dispatch to the variable-specific method
        method_name = f"_estimate_{variable_type}_{prediction_section}"
        method = getattr(self, method_name, None)
        if method is None:
            self._log(f"  ⚠  Method {method_name}() not yet implemented — skipping")
            return

        method(e_sid)
        self._completed.add(key)
        self._log(f"  ✓  {e_sid} complete")

    # ------------------------------------------------------------------
    # E_ subject storage helpers
    # ------------------------------------------------------------------

    def _store_estimated_cross_table(
        self,
        teryt_id: str,
        e_subject_id: str,
        year: int,
        table: np.ndarray,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
        is_observed: bool = False,
    ):
        """Store a single estimated cross table into the database.

        Creates the CrossTable on the record if it doesn't exist yet,
        and updates the provenance mask.
        """
        record = self.db._records.get(teryt_id)
        if record is None:
            return

        # Ensure the CrossTable exists on the record
        if e_subject_id not in record.cross_tables:
            from geoTERYT_db import CrossTable, YEAR_RANGE_FULL
            ct = CrossTable(
                subject_id=e_subject_id,
                dim_names=dim_names,
                dim_labels=dim_labels,
                subject_name=e_subject_id,
                year_range=YEAR_RANGE_FULL,
            )
            record.cross_tables[e_subject_id] = ct

        record.cross_tables[e_subject_id].set_table(year, table)

        # Track observed years on the CrossTable itself
        if is_observed:
            record.cross_tables[e_subject_id].observed_years.add(year)
        else:
            record.cross_tables[e_subject_id].observed_years.discard(year)

        # Store DataSeries counterparts (optional, slow for large runs)
        if self.create_data_series:
            self._store_estimated_data_series(
                record, e_subject_id, year, table, dim_names, dim_labels
            )

        # Update provenance
        prov_dict = self.provenance.setdefault(e_subject_id, {})
        if teryt_id not in prov_dict:
            from geoTERYT_db import YEAR_RANGE_FULL
            prov_dict[teryt_id] = ProvenanceMask(
                e_subject_id, dim_names, dim_labels, YEAR_RANGE_FULL
            )
        if is_observed:
            prov_dict[teryt_id].mark_observed(year)
        else:
            prov_dict[teryt_id].mark_all_estimated(year)

    def _store_estimated_data_series(
        self,
        record: 'TERYTRecord',
        e_subject_id: str,
        year: int,
        table: np.ndarray,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
    ):
        """Create / update DataSeries entries for an E_ subject on a record."""
        from geoTERYT_db import DataSeries, DATETIME_INDEX_FULL

        ts = pd.Timestamp(year, 1, 1)
        ndim = len(dim_names)

        if ndim == 1:
            labels_n1 = dim_labels[dim_names[0]]
            for i, lbl in enumerate(labels_n1):
                var_id = f'E{i+1:04d}'
                mkey = ('Estimated', e_subject_id, var_id)
                if mkey not in record.data:
                    record.data[mkey] = DataSeries(
                        source_type='Estimated',
                        subject_id=e_subject_id,
                        variable_id=var_id,
                        subject_name=e_subject_id,
                        categories={dim_names[0]: lbl},
                    )
                record.data[mkey].values[ts] = float(table[i])

        elif ndim == 2:
            labels_n1 = dim_labels[dim_names[0]]
            labels_n2 = dim_labels[dim_names[1]]
            counter = 1
            for i, lbl1 in enumerate(labels_n1):
                for j, lbl2 in enumerate(labels_n2):
                    var_id = f'E{counter:04d}'
                    mkey = ('Estimated', e_subject_id, var_id)
                    if mkey not in record.data:
                        record.data[mkey] = DataSeries(
                            source_type='Estimated',
                            subject_id=e_subject_id,
                            variable_id=var_id,
                            subject_name=e_subject_id,
                            categories={dim_names[0]: lbl1, dim_names[1]: lbl2},
                        )
                    record.data[mkey].values[ts] = float(table[i, j])
                    counter += 1

    # ------------------------------------------------------------------
    # Layer 1: Temporal seed generation (log-linear interpolation)
    # ------------------------------------------------------------------

    def _generate_seeds(
        self,
        teryt_ids: List[str],
        source_subject_id: str,
        year_range: range,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
        exclude_ogolem: bool = True,
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Generate seed cross tables via log-linear interpolation.

        For each territorial unit, collects all years with observed data
        from *source_subject_id*, then interpolates in log-space to fill
        every year in *year_range*.

        Algorithm (per unit):
          1. Collect anchor years (years with non-NaN cross table data).
          2. Additive smoothing: T_c ← T_c + ε for log-space safety.
          3. Transform to log-space: log(T_c) per anchor year.
          4. ≥3 anchors → natural cubic spline per cell.
             2 anchors  → linear interpolation in log-space.
             1 anchor   → constant seed.
          5. Exponentiate back: T̂(t) = exp(spline(t)).
          6. Years outside anchor range: use nearest anchor (no extrapolation).
          7. If *exclude_ogolem*: recompute ogółem rows/columns as sums
             of non-ogółem cells after interpolation.

        Parameters
        ----------
        teryt_ids : list of str
            TERYT IDs to generate seeds for.
        source_subject_id : str
            Merged subject to read observed cross tables from (e.g. 'M_age_1990').
        year_range : range
            Target years for the seeds.
        dim_names : list of str
            Dimension names of the cross table (e.g. ['n1'] or ['n1','n2']).
        dim_labels : dict
            Dimension → ordered list of labels.
        exclude_ogolem : bool
            If True (default), exclude ogółem rows/columns from interpolation
            and recompute them as sums afterwards.

        Returns
        -------
        dict : teryt_id → {year → np.ndarray}
        """
        results: Dict[str, Dict[int, np.ndarray]] = {}
        ndim = len(dim_names)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)

        # Identify ogółem indices per dimension
        ogolem_idx = {}
        non_ogolem_slices = {}
        for di, dname in enumerate(dim_names):
            labels = dim_labels[dname]
            og_i = None
            for li, lbl in enumerate(labels):
                if lbl.lower() == 'ogółem':
                    og_i = li
                    break
            ogolem_idx[di] = og_i
            non_ogolem_slices[di] = [i for i in range(len(labels)) if i != og_i] if og_i is not None else list(range(len(labels)))

        # Build the "core" shape (excluding ogółem dimensions)
        if exclude_ogolem:
            core_shape = tuple(len(non_ogolem_slices[di]) for di in range(ndim))
        else:
            core_shape = full_shape

        n_generated = 0
        n_skipped = 0

        for tid in teryt_ids:
            record = self.db._records.get(tid)
            if record is None:
                n_skipped += 1
                continue

            ct = record.cross_tables.get(source_subject_id)
            if ct is None:
                n_skipped += 1
                continue

            # Shape safety: skip records whose CrossTable shape differs
            if ct.shape != full_shape:
                n_skipped += 1
                continue

            # 1. Collect anchor years with observed data
            anchor_years = []
            anchor_tables = []
            for yr in sorted(ct.tables.keys()):
                tbl = ct.tables[yr]
                if tbl is not None and not np.all(np.isnan(tbl)):
                    # Extract core (non-ogółem) cells
                    if exclude_ogolem and ndim == 1:
                        core = tbl[non_ogolem_slices[0]]
                    elif exclude_ogolem and ndim == 2:
                        core = tbl[np.ix_(non_ogolem_slices[0], non_ogolem_slices[1])]
                    else:
                        core = tbl
                    # Only use if all core cells are non-NaN
                    if not np.any(np.isnan(core)):
                        anchor_years.append(yr)
                        anchor_tables.append(core.copy())

            if len(anchor_years) == 0:
                n_skipped += 1
                continue

            # 2. Additive smoothing for log-space
            # 2. Additive smoothing for log-space safety (ensure all > 0)
            smoothed = [np.maximum(t, 0.0) + EPSILON for t in anchor_tables]

            # 3. Log-transform (all values guaranteed positive)
            log_tables = [np.log(t) for t in smoothed]

            # 4-6. Interpolate / extrapolate
            seed_tables: Dict[int, np.ndarray] = {}

            if len(anchor_years) == 1:
                # Constant seed: use the single anchor for all years
                base = anchor_tables[0]
                for yr in year_range:
                    seed_tables[yr] = base.copy()

            elif len(anchor_years) <= 3:
                # ≤3 anchors: linear interpolation in log-space
                # (geometric interpolation — no overshoot possible)
                for yr in year_range:
                    if yr <= anchor_years[0]:
                        seed_tables[yr] = anchor_tables[0].copy()
                    elif yr >= anchor_years[-1]:
                        seed_tables[yr] = anchor_tables[-1].copy()
                    else:
                        # Find bracketing anchors
                        for ai in range(len(anchor_years) - 1):
                            if anchor_years[ai] <= yr <= anchor_years[ai + 1]:
                                y1, y2 = anchor_years[ai], anchor_years[ai + 1]
                                lt1, lt2 = log_tables[ai], log_tables[ai + 1]
                                frac = (yr - y1) / (y2 - y1)
                                log_interp = lt1 * (1 - frac) + lt2 * frac
                                seed_tables[yr] = np.exp(log_interp)
                                break

            elif len(anchor_years) <= 10:
                # 4-10 anchors: PCHIP (shape-preserving monotone cubic)
                ay = np.array(anchor_years, dtype=float)
                flat_logs = np.array([lt.ravel() for lt in log_tables])
                n_cells = flat_logs.shape[1]
                splines = []
                for ci in range(n_cells):
                    splines.append(PchipInterpolator(ay, flat_logs[:, ci]))

                for yr in year_range:
                    if yr < anchor_years[0]:
                        seed_tables[yr] = anchor_tables[0].copy()
                    elif yr > anchor_years[-1]:
                        seed_tables[yr] = anchor_tables[-1].copy()
                    else:
                        interp_flat = np.array([sp(yr) for sp in splines])
                        seed_tables[yr] = np.exp(interp_flat).reshape(core_shape)

            else:
                # >10 anchors: natural cubic spline per cell
                ay = np.array(anchor_years, dtype=float)
                flat_logs = np.array([lt.ravel() for lt in log_tables])  # (n_anchors, n_cells)
                n_cells = flat_logs.shape[1]
                splines = []
                for ci in range(n_cells):
                    splines.append(CubicSpline(ay, flat_logs[:, ci], bc_type='natural'))

                for yr in year_range:
                    if yr < anchor_years[0]:
                        seed_tables[yr] = anchor_tables[0].copy()
                    elif yr > anchor_years[-1]:
                        seed_tables[yr] = anchor_tables[-1].copy()
                    else:
                        interp_flat = np.array([sp(yr) for sp in splines])
                        seed_tables[yr] = np.exp(interp_flat).reshape(core_shape)

            # 7. Recompute ogółem and assemble full-shape table
            year_seeds: Dict[int, np.ndarray] = {}
            for yr, core_tbl in seed_tables.items():
                # Ensure non-negativity
                core_tbl = np.maximum(core_tbl, 0.0)

                if exclude_ogolem:
                    full_tbl = self._assemble_with_ogolem(
                        core_tbl, ndim, full_shape, ogolem_idx, non_ogolem_slices
                    )
                else:
                    full_tbl = core_tbl
                year_seeds[yr] = full_tbl
            results[tid] = year_seeds
            n_generated += 1

        self._log(f"    Seeds generated: {n_generated}/{len(teryt_ids)} units "
                  f"(skipped {n_skipped})")
        return results

    @staticmethod
    def _assemble_with_ogolem(
        core: np.ndarray,
        ndim: int,
        full_shape: tuple,
        ogolem_idx: Dict[int, Optional[int]],
        non_ogolem_slices: Dict[int, List[int]],
    ) -> np.ndarray:
        """Insert a core (ogółem-excluded) array into a full-shape array
        and compute ogółem rows/columns as sums of non-ogółem cells."""
        full = np.zeros(full_shape, dtype=float)

        if ndim == 1:
            og_i = ogolem_idx[0]
            noi = non_ogolem_slices[0]
            for ci, fi in enumerate(noi):
                full[fi] = core[ci]
            if og_i is not None:
                full[og_i] = core.sum()
        elif ndim == 2:
            og_i0 = ogolem_idx[0]
            og_i1 = ogolem_idx[1]
            noi0 = non_ogolem_slices[0]
            noi1 = non_ogolem_slices[1]
            for ci0, fi0 in enumerate(noi0):
                for ci1, fi1 in enumerate(noi1):
                    full[fi0, fi1] = core[ci0, ci1]
            # Row sums → ogółem column
            if og_i1 is not None:
                for fi0 in noi0:
                    full[fi0, og_i1] = sum(full[fi0, fi1] for fi1 in noi1)
            # Column sums → ogółem row
            if og_i0 is not None:
                for fi1 in noi1:
                    full[og_i0, fi1] = sum(full[fi0, fi1] for fi0 in noi0)
            # Grand total
            if og_i0 is not None and og_i1 is not None:
                full[og_i0, og_i1] = sum(full[fi0, fi1] for fi0 in noi0 for fi1 in noi1)
        else:
            # 3D+ fallback: just return the core (no ogółem recomputation)
            return core
        return full

    # ------------------------------------------------------------------
    # Layer 2: Marginal fitting via multi-dimensional IPF
    # ------------------------------------------------------------------

    def _fit_marginals_ipf(
        self,
        seed: np.ndarray,
        marginals: List[Tuple[np.ndarray, List[int]]],
        max_iter: int = IPF_MAX_ITER,
        convergence: float = IPF_CONVERGENCE,
    ) -> np.ndarray:
        """Fit a seed table to known marginals via N-dimensional IPF.

        Uses the ``ipfn`` package (numpy backend) when available, otherwise
        falls back to a manual iterative proportional fitting implementation.

        Parameters
        ----------
        seed : np.ndarray
            Initial estimate (any shape, must be non-negative).
        marginals : list of (target_array, dimensions)
            Each element ``(target, dims)`` specifies a marginal constraint.
            ``target`` is the desired marginal totals along ``dims``.
            ``dims`` is the list of dimension indices that the marginal
            corresponds to (i.e. the dimensions that remain after summing).
        max_iter : int
            Maximum iterations (default: IPF_MAX_ITER).
        convergence : float
            Convergence threshold (default: IPF_CONVERGENCE).

        Returns
        -------
        np.ndarray : IPF-adjusted table (same shape as seed).
        """
        if len(marginals) == 0:
            return seed.copy()

        # Ensure positive seed for IPF stability
        result = seed.copy().astype(float)
        result = np.maximum(result, EPSILON)

        if IPFN_AVAILABLE:
            # Use ipfn package
            try:
                from ipfn import ipfn as ipfn_cls
                aggregates = [m[0].astype(float) for m in marginals]
                dimensions = [m[1] for m in marginals]
                ipf = ipfn_cls.ipfn(
                    result, aggregates, dimensions,
                    convergence_rate=convergence,
                    max_iteration=max_iter,
                    rate_tolerance=1e-8,
                )
                result = ipf.iteration()
                return result
            except Exception as e:
                warnings.warn(f"ipfn failed ({e}), falling back to manual IPF")

        # Manual IPF fallback
        for iteration in range(max_iter):
            max_change = 0.0
            for target, dims in marginals:
                # Compute current marginal along `dims`
                sum_axes = tuple(
                    ax for ax in range(result.ndim) if ax not in dims
                )
                if len(sum_axes) == 0:
                    current = result.copy()
                else:
                    current = result.sum(axis=sum_axes)

                # Compute scaling factors
                with np.errstate(divide='ignore', invalid='ignore'):
                    factors = np.where(current > EPSILON,
                                       target / current,
                                       1.0)

                # Apply factors by broadcasting
                # Reshape factors to broadcast over the summed dimensions
                shape_for_broadcast = [1] * result.ndim
                for d in dims:
                    shape_for_broadcast[d] = result.shape[d]
                factors_broad = factors.reshape(shape_for_broadcast)
                result *= factors_broad

                change = float(np.max(np.abs(factors - 1.0)))
                max_change = max(max_change, change)

            if max_change < convergence:
                break

        return result

    def _scale_gminas_to_parent(
        self,
        gmina_seeds: Dict[str, np.ndarray],
        parent_table: np.ndarray,
        exclude_ogolem: bool = True,
        dim_names: List[str] = None,
        dim_labels: Dict[str, List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Scale gmina-level seed tables so they sum to a known parent total.

        Implements item 17:
          1. Aggregate gmina seeds → parent total.
          2. Compute cell-wise scaling: r_ij = parent_ij / aggregated_ij.
          3. Apply r_ij uniformly to each gmina.

        Parameters
        ----------
        gmina_seeds : dict
            teryt_id → np.ndarray (same shape, full table including ogółem).
        parent_table : np.ndarray
            The known parent-level cross table (e.g. voivodeship total).
        exclude_ogolem : bool
            If True, only use non-ogółem cells for scaling and recompute
            ogółem afterwards.
        dim_names, dim_labels : optional
            Needed if exclude_ogolem=True to identify ogółem positions.

        Returns
        -------
        dict : teryt_id → scaled np.ndarray
        """
        if not gmina_seeds:
            return {}

        example = next(iter(gmina_seeds.values()))
        full_shape = example.shape
        ndim = example.ndim

        # Identify ogółem indices
        ogolem_idx = {}
        non_ogolem_slices = {}
        if exclude_ogolem and dim_names and dim_labels:
            for di, dname in enumerate(dim_names):
                labels = dim_labels[dname]
                og_i = None
                for li, lbl in enumerate(labels):
                    if lbl.lower() == 'ogółem':
                        og_i = li
                        break
                ogolem_idx[di] = og_i
                non_ogolem_slices[di] = [i for i in range(len(labels)) if i != og_i] if og_i is not None else list(range(len(labels)))

        # Sum all gmina seeds
        aggregated = np.zeros(full_shape, dtype=float)
        for tbl in gmina_seeds.values():
            aggregated += np.nan_to_num(tbl, nan=0.0)

        # Compute scaling factors
        with np.errstate(divide='ignore', invalid='ignore'):
            factors = np.where(aggregated > EPSILON,
                               parent_table / aggregated,
                               1.0)

        # Apply to each gmina
        scaled = {}
        for tid, tbl in gmina_seeds.items():
            s = tbl * factors
            # Ensure non-negativity
            s = np.maximum(s, 0.0)
            # Recompute ogółem if requested
            if exclude_ogolem and dim_names and dim_labels:
                s = self._assemble_with_ogolem(
                    self._extract_core(s, ndim, ogolem_idx, non_ogolem_slices),
                    ndim, full_shape, ogolem_idx, non_ogolem_slices
                )
            scaled[tid] = s

        return scaled

    @staticmethod
    def _extract_core(
        full: np.ndarray,
        ndim: int,
        ogolem_idx: Dict[int, Optional[int]],
        non_ogolem_slices: Dict[int, List[int]],
    ) -> np.ndarray:
        """Extract the core (non-ogółem) cells from a full table."""
        if ndim == 1:
            return full[non_ogolem_slices[0]]
        elif ndim == 2:
            return full[np.ix_(non_ogolem_slices[0], non_ogolem_slices[1])]
        return full

    # ------------------------------------------------------------------
    # Temporal smoothing for Layer 2 scaling factors
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_factors_temporal(
        factor_series: Dict[int, np.ndarray],
        halflife: int = 3,
    ) -> Dict[int, np.ndarray]:
        """Smooth per-year scaling factor arrays using exponential moving avg.

        Demographic structure evolves slowly.  When Layer 2 applies
        per-year scaling factors to match voivodeship marginals, the
        raw factors can jump from year to year (with noise in the
        underlying data, source switches, etc.).  This produces the
        sudden jumps the user has flagged.

        The fix: smooth the factor time-series using a symmetric
        exponential kernel centred on each year.  Census years where
        ``factor == 1.0`` (no scaling) are excluded from the kernel.

        Parameters
        ----------
        factor_series : dict  year → np.ndarray (factors, shape may vary)
        halflife : int
            Halflife in years for the exponential kernel (default 3).

        Returns
        -------
        dict  year → np.ndarray (smoothed factors)
        """
        if len(factor_series) <= 1:
            return dict(factor_series)   # nothing to smooth

        sorted_years = sorted(factor_series.keys())
        shape = next(iter(factor_series.values())).shape

        # Stack factors into array  (n_years × *shape)
        stacked = np.array([factor_series[y] for y in sorted_years])
        n_years = stacked.shape[0]

        # Build symmetric exponential kernel weights
        result = np.empty_like(stacked)
        decay = np.log(2) / max(halflife, 1)

        for i in range(n_years):
            # weights: exp(-decay * |Δt|)
            weights = np.exp(-decay * np.abs(
                np.array(sorted_years) - sorted_years[i]
            ))
            # Normalise along time axis
            w_sum = weights.sum()
            w = weights / w_sum

            # Weighted average of factors along time axis
            # w has shape (n_years,), stacked has shape (n_years, *shape)
            # Broadcast: reshape w to (n_years, 1, 1, ...) for N-D factors
            w_broad = w.reshape((-1,) + (1,) * len(shape))
            result[i] = (stacked * w_broad).sum(axis=0)

        smoothed = {yr: result[i] for i, yr in enumerate(sorted_years)}
        return smoothed

    def _layer2_voiv_scaling_smoothed(
        self,
        seeds: Dict[str, Dict[int, np.ndarray]],
        source_sid: str,
        year_range: range,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
        observed_years_per_gmina: Dict[str, Set[int]],
        factor_guard: float = 3.0,
    ) -> int:
        """Voivodeship marginal scaling with temporal smoothing.

        Fix 44: Growth-rate-deviation approach.
        ----------------------------------------
        Instead of computing ``factors = voiv / aggregate`` (which forces
        all gminas to match voivodeship-level *proportions*), we compute
        the *deviation* between voivodeship temporal growth and the
        aggregate's Layer 1 growth relative to a shared census baseline:

            voiv_growth = voiv[year]  / voiv[closest_census]
            agg_growth  = agg[year]  / agg[closest_census]
            factors     = voiv_growth / agg_growth

        Properties:
        - At census years (year == closest_census): factors ≈ 1.0.
        - Between censuses: factors capture how each *category* at the
          voivodeship level evolved differently than the aggregate of
          Layer 1 interpolations — typically a few percent.
        - Each gmina's own census-derived category proportions are
          *preserved*, with only small adjustments for deviations.
        - No coverage-gap inflation (ratios of same-source values).

        Returns
        -------
        int : number of voivodeship-year combinations scaled.
        """
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ndim = len(dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        voiv_tids = self._get_voivodeships()
        n_scaled = 0

        for voiv_tid in voiv_tids:
            # ── Phase 1: collect voiv & aggregate data for all years ──
            voiv_data: Dict[int, np.ndarray] = {}
            agg_data: Dict[int, np.ndarray] = {}
            gminas_per_year: Dict[int, Dict[str, np.ndarray]] = {}

            for year in year_range:
                voiv_tbl = self._get_observed_table(
                    voiv_tid, source_sid, year,
                )
                if voiv_tbl is None or voiv_tbl.shape != full_shape:
                    continue

                gminas = self._collect_voivodeship_gminas(voiv_tid, year)
                aggregated = np.zeros(full_shape, dtype=float)
                gmina_tables: Dict[str, np.ndarray] = {}
                for gid in gminas:
                    if gid not in seeds or year not in seeds[gid]:
                        continue
                    tbl = seeds[gid][year]
                    aggregated += np.nan_to_num(tbl, nan=0.0)
                    obs_yrs = observed_years_per_gmina.get(gid, set())
                    if year not in obs_yrs:
                        gmina_tables[gid] = tbl

                if np.sum(aggregated) < EPSILON:
                    continue

                voiv_data[year] = voiv_tbl
                agg_data[year] = aggregated
                gminas_per_year[year] = gmina_tables

            if not voiv_data:
                continue

            # ── Find census-year baselines ──
            census_candidates = sorted(
                y for y in CENSUS_YEARS
                if y in voiv_data and y in agg_data
            )
            if not census_candidates:
                # Fallback: use the year with smallest coverage gap
                best_yr = min(
                    voiv_data.keys(),
                    key=lambda y: abs(
                        np.nansum(voiv_data[y])
                        / max(np.nansum(agg_data[y]), EPSILON) - 1.0
                    ),
                )
                census_candidates = [best_yr]

            # ── Phase 2: compute growth-rate-deviation factors ──
            raw_factors: Dict[int, np.ndarray] = {}
            for year in voiv_data:
                closest_cy = min(
                    census_candidates, key=lambda cy: abs(year - cy),
                )
                voiv_base = voiv_data[closest_cy]
                agg_base = agg_data[closest_cy]
                voiv_current = voiv_data[year]
                agg_current = agg_data[year]

                with np.errstate(divide='ignore', invalid='ignore'):
                    voiv_growth = np.where(
                        voiv_base > EPSILON,
                        voiv_current / voiv_base, 1.0,
                    )
                    agg_growth = np.where(
                        agg_base > EPSILON,
                        agg_current / agg_base, 1.0,
                    )
                    factors = np.where(
                        agg_growth > EPSILON,
                        voiv_growth / agg_growth, 1.0,
                    )
                factors = np.where(np.isnan(factors), 1.0, factors)

                # Guard: skip if any cell deviates too far from 1.0
                max_dev = float(np.max(np.abs(factors - 1.0)))
                if max_dev > (factor_guard - 1.0):
                    self._log(
                        f"    ⚠  Skipping {voiv_tid} y={year}: "
                        f"extreme growth-rate deviation {max_dev:.2f}"
                    )
                    continue

                raw_factors[year] = factors

            if not raw_factors:
                continue

            # ── Phase 3: smooth factors temporally ──
            smoothed = self._smooth_factors_temporal(raw_factors, halflife=3)

            # ── Phase 4: apply smoothed factors ──
            for year, factors in smoothed.items():
                gmina_tables = gminas_per_year.get(year)
                if not gmina_tables:
                    continue

                for gid, tbl in gmina_tables.items():
                    scaled = np.maximum(tbl * factors, 0.0)
                    core = self._extract_core(
                        scaled, ndim, ogolem_idx, non_ogolem_slices,
                    )
                    seeds[gid][year] = self._assemble_with_ogolem(
                        core, ndim, full_shape,
                        ogolem_idx, non_ogolem_slices,
                    )
                n_scaled += 1

        return n_scaled

    def _layer2_national_scaling_smoothed(
        self,
        seeds: Dict[str, Dict[int, np.ndarray]],
        source_sid: str,
        year_range: range,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
        observed_years_per_gmina: Dict[str, Set[int]],
    ) -> int:
        """National marginal scaling with temporal smoothing.

        Fix 44: growth-rate-deviation approach (same logic as
        ``_layer2_voiv_scaling_smoothed``).

        Returns
        -------
        int : number of national-year combinations scaled.
        """
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ndim = len(dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        country_tid = '0000000'
        n_scaled = 0

        # ── Phase 1: collect country & aggregate data for all years ──
        country_data: Dict[int, np.ndarray] = {}
        agg_data: Dict[int, np.ndarray] = {}
        gminas_per_year: Dict[int, List[str]] = {}

        for year in year_range:
            country_tbl = self._get_observed_table(
                country_tid, source_sid, year,
            )
            if country_tbl is None or country_tbl.shape != full_shape:
                continue

            total_agg = np.zeros(full_shape, dtype=float)
            gminas_in_year: List[str] = []
            for tid in seeds:
                if year not in seeds[tid]:
                    continue
                total_agg += np.nan_to_num(seeds[tid][year], nan=0.0)
                obs_yrs = observed_years_per_gmina.get(tid, set())
                if year not in obs_yrs:
                    gminas_in_year.append(tid)

            if np.sum(total_agg) < EPSILON:
                continue

            country_data[year] = country_tbl
            agg_data[year] = total_agg
            gminas_per_year[year] = gminas_in_year

        if not country_data:
            return 0

        # ── Interpolate country_data for gap years ──
        # When the national constraint is sparse (e.g., H_sex_educ only
        # available 1986-88, 1991-94), linearly interpolate cell-by-cell
        # to fill all years in year_range that have aggregated gmina data.
        available_years = sorted(country_data.keys())
        if len(available_years) >= 2:
            all_years_with_agg = sorted(
                y for y in year_range
                if y not in country_data and y in agg_data
            )
            for gap_yr in all_years_with_agg:
                # Find bracketing years
                lower = [y for y in available_years if y <= gap_yr]
                upper = [y for y in available_years if y >= gap_yr]
                if lower and upper:
                    y0, y1 = lower[-1], upper[0]
                    if y0 == y1:
                        interp_tbl = country_data[y0].copy()
                    else:
                        alpha = (gap_yr - y0) / (y1 - y0)
                        interp_tbl = (
                            (1 - alpha) * country_data[y0]
                            + alpha * country_data[y1]
                        )
                elif lower:
                    # Extrapolate forward from last available
                    interp_tbl = country_data[lower[-1]].copy()
                elif upper:
                    # Extrapolate backward from first available
                    interp_tbl = country_data[upper[0]].copy()
                else:
                    continue
                country_data[gap_yr] = interp_tbl
                # These gminas were not originally in gminas_per_year,
                # so build the list now.
                gminas_in_year = [
                    tid for tid in seeds
                    if gap_yr in seeds[tid]
                    and gap_yr not in observed_years_per_gmina.get(tid, set())
                ]
                gminas_per_year[gap_yr] = gminas_in_year

        # ── Find census-year baselines ──
        census_candidates = sorted(
            y for y in CENSUS_YEARS
            if y in country_data and y in agg_data
        )
        if not census_candidates:
            best_yr = min(
                country_data.keys(),
                key=lambda y: abs(
                    np.nansum(country_data[y])
                    / max(np.nansum(agg_data[y]), EPSILON) - 1.0
                ),
            )
            census_candidates = [best_yr]

        # ── Phase 2: compute growth-rate-deviation factors ──
        raw_factors: Dict[int, np.ndarray] = {}
        for year in country_data:
            closest_cy = min(
                census_candidates, key=lambda cy: abs(year - cy),
            )
            country_base = country_data[closest_cy]
            agg_base = agg_data[closest_cy]
            country_current = country_data[year]
            agg_current = agg_data[year]

            with np.errstate(divide='ignore', invalid='ignore'):
                country_growth = np.where(
                    country_base > EPSILON,
                    country_current / country_base, 1.0,
                )
                agg_growth = np.where(
                    agg_base > EPSILON,
                    agg_current / agg_base, 1.0,
                )
                factors = np.where(
                    agg_growth > EPSILON,
                    country_growth / agg_growth, 1.0,
                )
            factors = np.where(np.isnan(factors), 1.0, factors)
            raw_factors[year] = factors

        # ── Phase 3: smooth factors temporally ──
        smoothed = self._smooth_factors_temporal(raw_factors, halflife=3)

        # ── Phase 4: apply smoothed factors ──
        for year, factors in smoothed.items():
            gminas = gminas_per_year.get(year)
            if not gminas:
                continue

            for tid in gminas:
                scaled = np.maximum(seeds[tid][year] * factors, 0.0)
                core = self._extract_core(
                    scaled, ndim, ogolem_idx, non_ogolem_slices,
                )
                seeds[tid][year] = self._assemble_with_ogolem(
                    core, ndim, full_shape,
                    ogolem_idx, non_ogolem_slices,
                )
            n_scaled += 1

        return n_scaled

    def _layer2_educ_sex_marginal_smoothed(
        self,
        seeds: Dict[str, Dict[int, np.ndarray]],
        source_sid: str,
        marginal_sid: str,
        year_range: range,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
        observed_years_per_gmina: Dict[str, Set[int]],
        factor_guard: float = 3.0,
    ) -> int:
        """Voivodeship education-marginal scaling with temporal smoothing.

        Fix 44: growth-rate-deviation approach.
        ----------------------------------------
        Instead of ``educ_factors[cat] = voiv[cat] / agg[cat]`` (which
        pushes every gmina's proportions toward the voivodeship average),
        compute the *deviation* of voivodeship temporal growth from the
        aggregate's Layer 1 growth relative to a census-year baseline.

        Returns
        -------
        int : number of voivodeship-year combinations scaled.
        """
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ndim = len(dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        voiv_tids = self._get_voivodeships()
        n_scaled = 0

        # Get 1D education dimensions for marginal matching
        try:
            educ_1d_names, educ_1d_labels = self._get_subject_dimensions(
                marginal_sid,
            )
            educ_1d_shape = tuple(
                len(educ_1d_labels[d]) for d in educ_1d_names
            )
        except ValueError:
            self._log(f"    ⚠  {marginal_sid} not found — skipping L2")
            return 0

        # Label mapping: M_educ_2000 index → M_educ_sex_2000 row
        educ_dim_name = dim_names[0]
        educ_labels_2d = dim_labels[educ_dim_name]
        educ_labels_1d = educ_1d_labels[educ_1d_names[0]]
        educ_2d_to_1d: Dict[int, int] = {}
        for i2d, lbl2d in enumerate(educ_labels_2d):
            for i1d, lbl1d in enumerate(educ_labels_1d):
                if lbl2d == lbl1d:
                    educ_2d_to_1d[i2d] = i1d
                    break

        sex_non_og = non_ogolem_slices.get(
            1, list(range(full_shape[1]))
        )
        n_educ = len(educ_labels_2d)

        for voiv_tid in voiv_tids:
            # ── Phase 1: collect voiv & aggregate educ data ──
            voiv_educ_data: Dict[int, np.ndarray] = {}   # year → 1D voiv
            agg_educ_data: Dict[int, np.ndarray] = {}    # year → 1D agg
            gminas_per_year: Dict[int, Dict[str, np.ndarray]] = {}

            for year in year_range:
                voiv_tbl = self._get_observed_table(
                    voiv_tid, marginal_sid, year,
                )
                if voiv_tbl is None or voiv_tbl.shape != educ_1d_shape:
                    continue

                gminas = self._collect_voivodeship_gminas(voiv_tid, year)
                agg_educ = np.zeros(n_educ, dtype=float)
                gmina_tables: Dict[str, np.ndarray] = {}
                for gid in gminas:
                    if gid not in seeds or year not in seeds[gid]:
                        continue
                    tbl = seeds[gid][year]
                    for ri in range(n_educ):
                        agg_educ[ri] += sum(
                            tbl[ri, si] for si in sex_non_og
                        )
                    obs_yrs = observed_years_per_gmina.get(gid, set())
                    if year not in obs_yrs:
                        gmina_tables[gid] = tbl

                if np.sum(agg_educ) < EPSILON:
                    continue

                voiv_educ_data[year] = voiv_tbl
                agg_educ_data[year] = agg_educ
                gminas_per_year[year] = gmina_tables

            if not voiv_educ_data:
                continue

            # ── Find census-year baselines ──
            census_candidates = sorted(
                y for y in CENSUS_YEARS
                if y in voiv_educ_data and y in agg_educ_data
            )
            if not census_candidates:
                best_yr = min(
                    voiv_educ_data.keys(),
                    key=lambda y: abs(
                        np.nansum(voiv_educ_data[y])
                        / max(np.nansum(agg_educ_data[y]), EPSILON) - 1.0
                    ),
                )
                census_candidates = [best_yr]

            # ── Phase 2: growth-rate-deviation factors per educ row ──
            raw_factors: Dict[int, np.ndarray] = {}
            for year in voiv_educ_data:
                closest_cy = min(
                    census_candidates, key=lambda cy: abs(year - cy),
                )
                voiv_base = voiv_educ_data[closest_cy]
                agg_base = agg_educ_data[closest_cy]
                voiv_current = voiv_educ_data[year]
                agg_current = agg_educ_data[year]

                educ_factors = np.ones(n_educ, dtype=float)
                for i2d, i1d in educ_2d_to_1d.items():
                    vb = float(voiv_base[i1d])
                    vc = float(voiv_current[i1d])
                    ab = float(agg_base[i2d])
                    ac = float(agg_current[i2d])

                    voiv_gr = vc / vb if vb > EPSILON else 1.0
                    agg_gr = ac / ab if ab > EPSILON else 1.0
                    educ_factors[i2d] = (
                        voiv_gr / agg_gr if agg_gr > EPSILON else 1.0
                    )

                # Guard
                max_dev = float(np.max(np.abs(educ_factors - 1.0)))
                if max_dev > (factor_guard - 1.0):
                    self._log(
                        f"    ⚠  Skipping {voiv_tid} y={year}: "
                        f"extreme educ growth deviation {max_dev:.2f}"
                    )
                    continue

                raw_factors[year] = educ_factors

            if not raw_factors:
                continue

            # ── Phase 3: smooth factors temporally ──
            smoothed = self._smooth_factors_temporal(raw_factors, halflife=3)

            # ── Phase 4: apply smoothed row factors ──
            for year, educ_factors in smoothed.items():
                gmina_tables = gminas_per_year.get(year)
                if not gmina_tables:
                    continue

                for gid, tbl in gmina_tables.items():
                    s = tbl.copy()
                    for ri in range(n_educ):
                        for si in range(full_shape[1]):
                            s[ri, si] *= educ_factors[ri]
                    s = np.maximum(s, 0.0)
                    core = self._extract_core(
                        s, ndim, ogolem_idx, non_ogolem_slices,
                    )
                    seeds[gid][year] = self._assemble_with_ogolem(
                        core, ndim, full_shape,
                        ogolem_idx, non_ogolem_slices,
                    )
                n_scaled += 1

        return n_scaled

    # ------------------------------------------------------------------
    # Census data preservation helpers
    # ------------------------------------------------------------------

    def _collect_observed_tables(
        self,
        seeds: Dict[str, Dict[int, np.ndarray]],
        source_sid: str,
        full_shape: tuple,
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Snapshot observed census tables before Layer 2 scaling.

        Returns a dict teryt_id → {year → original_table} for each
        gmina/year where M_ source data exists.  These can be used
        after scaling to restore census-year values.
        """
        observed: Dict[str, Dict[int, np.ndarray]] = {}
        for tid in seeds:
            rec = self.db._records.get(tid)
            if rec is None:
                continue
            ct = rec.cross_tables.get(source_sid)
            if ct is None or ct.shape != full_shape:
                continue
            for yr in ct.years_with_data:
                tbl = ct.tables[yr]
                if tbl is not None and not np.all(np.isnan(tbl)):
                    observed.setdefault(tid, {})[yr] = tbl.copy()
        return observed

    def _restore_census_data(
        self,
        seeds: Dict[str, Dict[int, np.ndarray]],
        observed: Dict[str, Dict[int, np.ndarray]],
        blend_weight: float = 1.0,
    ) -> int:
        """Restore census-year gmina tables, blending with scaled values.

        Instead of hard-overwriting (which creates discontinuities at
        census-year boundaries), we blend the original M_ value with the
        Layer-2-scaled value:

            restored = blend_weight * original + (1 - blend_weight) * scaled

        A blend_weight of 1.0 reproduces the old behaviour (hard restore).
        A blend_weight < 1.0 preserves some of the scaling adjustment,
        reducing spikes.

        Returns number of restored year-tables.
        """
        n_restored = 0
        for tid, year_tbls in observed.items():
            if tid not in seeds:
                continue
            for yr, orig_tbl in year_tbls.items():
                if yr in seeds[tid]:
                    scaled_tbl = seeds[tid][yr]
                    if blend_weight >= 1.0:
                        seeds[tid][yr] = orig_tbl.copy()
                    else:
                        seeds[tid][yr] = (
                            blend_weight * orig_tbl
                            + (1.0 - blend_weight) * scaled_tbl
                        )
                    n_restored += 1
        return n_restored

    # ------------------------------------------------------------------
    # Layer 3: Hierarchical consistency enforcement
    # ------------------------------------------------------------------

    def _enforce_hierarchy_gurobi(
        self,
        seeds: Dict[str, Dict[int, np.ndarray]],
        year: int,
        voivodeship_tid: str,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
        parent_tables: Dict[str, np.ndarray] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Enforce hierarchical consistency via Gurobi QP.

        For a given voivodeship and year, adjusts gmina-level estimates
        so that they sum to known parent totals while minimising a
        chi-squared-like objective.

        Decision variables: x_ij^g ≥ 0 for each gmina g (rodz ∈ {1,2,3})
        and cell (i,j) EXCLUDING ogółem.

        Objective:
            min Σ_g Σ_{i,j} (x_ij^g - x̂_ij^g)² / (x̂_ij^g + ε)

        Constraints:
          - Powiat aggregation: Σ_{g∈p} x_ij^g = X_ij^p (if known)
          - Non-negativity: x_ij^g ≥ 0
          - Optional: total pop consistency per gmina

        Parameters
        ----------
        seeds : dict
            teryt_id → {year → np.ndarray} (full tables including ogółem).
        year : int
            Calendar year.
        voivodeship_tid : str
            Voivodeship TERYT ID (e.g. '0200000').
        dim_names, dim_labels : dimension specification.
        parent_tables : dict or None
            Optional known parent-level tables. Keys are teryt_ids
            (powiats/voivodeship). Values are np.ndarrays.

        Returns
        -------
        dict : teryt_id → adjusted np.ndarray (full tables).
        """
        if not GUROBI_AVAILABLE:
            return self._enforce_hierarchy_ipf(
                seeds, year, voivodeship_tid, dim_names, dim_labels, parent_tables
            )

        voiv_rec = self.db._records.get(voivodeship_tid)
        if voiv_rec is None:
            return {tid: s.get(year, np.zeros(1)) for tid, s in seeds.items()}

        ndim = len(dim_names)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)

        # Identify ogółem
        ogolem_idx = {}
        non_ogolem_slices = {}
        for di, dname in enumerate(dim_names):
            labels = dim_labels[dname]
            og_i = None
            for li, lbl in enumerate(labels):
                if lbl.lower() == 'ogółem':
                    og_i = li
                    break
            ogolem_idx[di] = og_i
            non_ogolem_slices[di] = [i for i in range(len(labels)) if i != og_i] if og_i is not None else list(range(len(labels)))

        core_shape = tuple(len(non_ogolem_slices[di]) for di in range(ndim))
        n_core_cells = int(np.prod(core_shape))

        # Collect gminas within this voivodeship (rodz 1,2,3 only)
        # Two-level traverse: voiv → powiats → gminas
        all_gminas = []
        powiat_gminas: Dict[str, List[str]] = {}
        for child_tid in _get_aggregation_children(voiv_rec, self.db, year):
            child = self.db._records.get(child_tid)
            if child is None:
                continue
            if child.level == 5:     # powiat → get its gminas
                pgminas = _get_aggregation_children(child, self.db, year)
                for gid in pgminas:
                    all_gminas.append(gid)
                    powiat_gminas.setdefault(child_tid, []).append(gid)
            elif child.level == 6:   # direct gmina (city with powiat rights)
                all_gminas.append(child_tid)
                ptid = child_tid[:4] + '000'
                powiat_gminas.setdefault(ptid, []).append(child_tid)

        gmina_tids = [t for t in all_gminas if t in seeds and year in seeds[t]]

        if not gmina_tids:
            return {}

        # Prune powiat mapping to only gminas present in seeds
        powiat_gminas = {
            ptid: [g for g in gids if g in seeds and year in seeds[g]]
            for ptid, gids in powiat_gminas.items()
        }
        powiat_gminas = {k: v for k, v in powiat_gminas.items() if v}

        # Get seed values (core only)
        gmina_cores = {}
        for tid in gmina_tids:
            full_tbl = seeds[tid][year]
            core = self._extract_core(full_tbl, ndim, ogolem_idx, non_ogolem_slices)
            gmina_cores[tid] = core.ravel()

        try:
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            model = gp.Model(env=env)
            model.Params.OutputFlag = 0

            # Decision variables
            x = {}
            for tid in gmina_tids:
                for ci in range(n_core_cells):
                    x[tid, ci] = model.addVar(
                        lb=0.0, name=f"x_{tid}_{ci}"
                    )

            # Objective: chi-squared-like
            obj = gp.QuadExpr()
            for tid in gmina_tids:
                seed_flat = gmina_cores[tid]
                for ci in range(n_core_cells):
                    x_hat = float(seed_flat[ci])
                    weight = 1.0 / (x_hat + EPSILON)
                    obj += weight * (x[tid, ci] - x_hat) * (x[tid, ci] - x_hat)
            model.setObjective(obj, GRB.MINIMIZE)

            # Powiat aggregation constraints (if known)
            if parent_tables:
                for ptid, child_tids in powiat_gminas.items():
                    parent_tbl = parent_tables.get(ptid)
                    if parent_tbl is None:
                        continue
                    parent_core = self._extract_core(
                        parent_tbl, ndim, ogolem_idx, non_ogolem_slices
                    ).ravel()
                    valid_children = [t for t in child_tids if t in gmina_tids]
                    if not valid_children:
                        continue
                    for ci in range(n_core_cells):
                        model.addConstr(
                            gp.quicksum(x[t, ci] for t in valid_children) == parent_core[ci],
                            name=f"pow_{ptid}_c{ci}"
                        )

            # Total population constraint per gmina (if pop is known)
            ts = pd.Timestamp(year, 1, 1)
            for tid in gmina_tids:
                rec = self.db._records.get(tid)
                if rec is not None:
                    pop_val = rec.pop.get(ts, np.nan)
                    if pd.notna(pop_val) and pop_val > 0:
                        model.addConstr(
                            gp.quicksum(x[tid, ci] for ci in range(n_core_cells)) == pop_val,
                            name=f"pop_{tid}"
                        )

            model.optimize()

            if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
                result = {}
                for tid in gmina_tids:
                    core_vals = np.array([x[tid, ci].X for ci in range(n_core_cells)])
                    core_reshaped = core_vals.reshape(core_shape)
                    full_tbl = self._assemble_with_ogolem(
                        core_reshaped, ndim, full_shape, ogolem_idx, non_ogolem_slices
                    )
                    result[tid] = full_tbl
                return result
            else:
                self._log(f"    ⚠ Gurobi: status={model.status} for "
                         f"{voivodeship_tid}/{year} — falling back to IPF")
                return self._enforce_hierarchy_ipf(
                    seeds, year, voivodeship_tid, dim_names, dim_labels, parent_tables
                )

        except Exception as e:
            self._log(f"    ⚠ Gurobi error: {e} — falling back to IPF")
            return self._enforce_hierarchy_ipf(
                seeds, year, voivodeship_tid, dim_names, dim_labels, parent_tables
            )

    def _enforce_hierarchy_ipf(
        self,
        seeds: Dict[str, Dict[int, np.ndarray]],
        year: int,
        voivodeship_tid: str,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
        parent_tables: Dict[str, np.ndarray] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Enforce hierarchical consistency via iterated multi-level IPF.

        FALLBACK solver when Gurobi is unavailable.

        Algorithm:
          1. Aggregate gmina estimates to powiat level (rodz 1,2,3).
          2. If powiat data known → scale gminas within each powiat.
          3. Re-aggregate to voivodeship.
          4. If voivodeship data known → scale powiats, then gminas.
          5. Repeat until convergence.
        """
        voiv_rec = self.db._records.get(voivodeship_tid)
        if voiv_rec is None:
            return {}

        ndim = len(dim_names)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)

        # Collect gminas within this voivodeship (rodz 1,2,3 only)
        # Two-level traverse: voiv → powiats → gminas
        all_gminas = []
        powiat_gminas: Dict[str, List[str]] = {}
        for child_tid in _get_aggregation_children(voiv_rec, self.db, year):
            child = self.db._records.get(child_tid)
            if child is None:
                continue
            if child.level == 5:     # powiat → get its gminas
                pgminas = _get_aggregation_children(child, self.db, year)
                for gid in pgminas:
                    all_gminas.append(gid)
                    powiat_gminas.setdefault(child_tid, []).append(gid)
            elif child.level == 6:   # direct gmina (city with powiat rights)
                all_gminas.append(child_tid)
                ptid = child_tid[:4] + '000'
                powiat_gminas.setdefault(ptid, []).append(child_tid)

        gmina_tids = [t for t in all_gminas if t in seeds and year in seeds[t]]
        if not gmina_tids:
            return {}

        # Working copies
        current = {tid: seeds[tid][year].copy() for tid in gmina_tids}

        # Prune powiat mapping to only gminas present in seeds
        powiat_gminas = {
            ptid: [g for g in gids if g in seeds and year in seeds[g]]
            for ptid, gids in powiat_gminas.items()
        }
        powiat_gminas = {k: v for k, v in powiat_gminas.items() if v}

        for iteration in range(50):
            max_change = 0.0

            # Step 1-2: Powiat-level scaling
            if parent_tables:
                for ptid, child_tids in powiat_gminas.items():
                    parent_tbl = parent_tables.get(ptid)
                    if parent_tbl is None:
                        continue
                    valid = [t for t in child_tids if t in current]
                    if not valid:
                        continue

                    # Aggregate
                    agg = np.zeros(full_shape, dtype=float)
                    for t in valid:
                        agg += np.nan_to_num(current[t], nan=0.0)

                    # Scale
                    with np.errstate(divide='ignore', invalid='ignore'):
                        factors = np.where(agg > EPSILON,
                                          parent_tbl / agg,
                                          1.0)
                    max_change = max(max_change, float(np.max(np.abs(factors - 1.0))))

                    for t in valid:
                        current[t] = np.maximum(current[t] * factors, 0.0)

            # Step 3: Total population consistency per gmina
            ts = pd.Timestamp(year, 1, 1)
            for tid in gmina_tids:
                rec = self.db._records.get(tid)
                if rec is None:
                    continue
                pop_val = rec.pop.get(ts, np.nan)
                if pd.isna(pop_val) or pop_val <= 0:
                    continue
                tbl = current[tid]
                # Compute total from non-ogółem cells
                total = float(np.nansum(tbl))
                if total > EPSILON:
                    factor = pop_val / total
                    current[tid] = tbl * factor
                    change = abs(factor - 1.0)
                    max_change = max(max_change, change)

            if max_change < IPF_CONVERGENCE:
                break

        return current

    # ------------------------------------------------------------------
    # Estimation helpers (shared across variable-specific pipelines)
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_ogolem(
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
    ) -> Tuple[Dict[int, Optional[int]], Dict[int, List[int]]]:
        """Identify ogółem index and non-ogółem indices per dimension.

        Returns
        -------
        ogolem_idx : {dim_index: label_index_or_None}
        non_ogolem_slices : {dim_index: [non-ogółem label indices]}
        """
        ogolem_idx: Dict[int, Optional[int]] = {}
        non_ogolem_slices: Dict[int, List[int]] = {}
        for di, dname in enumerate(dim_names):
            labels = dim_labels[dname]
            og_i = None
            for li, lbl in enumerate(labels):
                if lbl.lower() == 'ogółem':
                    og_i = li
                    break
            ogolem_idx[di] = og_i
            non_ogolem_slices[di] = (
                [i for i in range(len(labels)) if i != og_i]
                if og_i is not None
                else list(range(len(labels)))
            )
        return ogolem_idx, non_ogolem_slices

    def _get_subject_dimensions(
        self, source_sid: str,
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Get dim_names and dim_labels from the first record that has
        a CrossTable for *source_sid*."""
        for rec in self.db._records.values():
            ct = rec.cross_tables.get(source_sid)
            if ct is not None:
                return ct.dim_names, ct.dim_labels
        raise ValueError(
            f"No records found with cross table subject '{source_sid}'"
        )

    def _get_all_gminas(self) -> List[str]:
        """Return sorted teryt_ids of all gminas with rodz ∈ {1,2,3}."""
        from geoTERYT_db import LEVEL_GMINA
        level_set = self.db._by_level.get(LEVEL_GMINA, set())
        return sorted(
            tid for tid in level_set if tid[-1] in RODZ_AGGREGATION_SET
        )

    def _get_voivodeships(self) -> List[str]:
        """Return sorted teryt_ids of all voivodeships (level 2)."""
        from geoTERYT_db import LEVEL_VOIVODESHIP
        return sorted(self.db._by_level.get(LEVEL_VOIVODESHIP, set()))

    def _collect_voivodeship_gminas(
        self, voiv_tid: str, year: int,
    ) -> List[str]:
        """Get all gminas (rodz 1,2,3) within a voivodeship for *year*.

        Two-level traverse: voiv → powiats → gminas, plus direct gminas.
        """
        voiv_rec = self.db._records.get(voiv_tid)
        if voiv_rec is None:
            return []
        gminas: List[str] = []
        for child_tid in _get_aggregation_children(voiv_rec, self.db, year):
            child = self.db._records.get(child_tid)
            if child is None:
                continue
            if child.level == 5:  # powiat
                for gid in _get_aggregation_children(child, self.db, year):
                    gminas.append(gid)
            elif child.level == 6:  # direct gmina (city-powiat)
                gminas.append(child_tid)
        return gminas

    def _get_observed_table(
        self, tid: str, source_sid: str, year: int,
    ) -> Optional[np.ndarray]:
        """Return a *copy* of the observed cross table, or ``None``."""
        rec = self.db._records.get(tid)
        if rec is None:
            return None
        ct = rec.cross_tables.get(source_sid)
        if ct is None:
            return None
        tbl = ct.tables.get(year)
        if tbl is None or np.all(np.isnan(tbl)):
            return None
        return tbl.copy()

    def _scale_table_to_pop(
        self,
        table: np.ndarray,
        pop: float,
        ogolem_idx: Dict[int, Optional[int]],
        non_ogolem_slices: Dict[int, List[int]],
        full_shape: tuple,
    ) -> np.ndarray:
        """Scale *table* so its grand total (sum of non-ogółem cells)
        equals *pop*, then recompute ogółem as sums."""
        ndim = len(full_shape)
        core = self._extract_core(table, ndim, ogolem_idx, non_ogolem_slices)
        current_total = float(np.nansum(core))
        if current_total > EPSILON and pop > 0:
            core = core * (pop / current_total)
        return self._assemble_with_ogolem(
            core, ndim, full_shape, ogolem_idx, non_ogolem_slices,
        )

    def _apply_residual_scaling(
        self,
        observed: Dict[str, np.ndarray],
        estimated: Dict[str, np.ndarray],
        parent_table: np.ndarray,
        ogolem_idx: Dict[int, Optional[int]],
        non_ogolem_slices: Dict[int, List[int]],
        full_shape: tuple,
        year: Optional[int] = None,
    ):
        """Scale estimated gmina tables so (obs + est) ≈ parent table.

        Modifies *estimated* dict **in place**.

        Algorithm:
          1. residual_core = parent_core − observed_core_sum.
          2. Clip residual to ≥ 0 per cell (conservative: never set
             estimated cells negative even if data is inconsistent).
          3. Scale each estimated gmina proportionally so that
             Σ estimated_core = residual_core.

        Falls back to pop-only scaling when residual is entirely ≤ 0.
        """
        ndim = len(full_shape)

        # Sum observed gmina core cells
        obs_core_sum = np.zeros(
            tuple(len(non_ogolem_slices[d]) for d in range(ndim)),
            dtype=float,
        )
        for tbl in observed.values():
            obs_core_sum += self._extract_core(
                np.nan_to_num(tbl, nan=0.0), ndim,
                ogolem_idx, non_ogolem_slices,
            )

        parent_core = self._extract_core(
            parent_table, ndim, ogolem_idx, non_ogolem_slices,
        )
        residual = parent_core - obs_core_sum

        if np.all(residual <= 0):
            # Observed exceeds parent → fall back to pop-only scaling
            if year is not None:
                ts = pd.Timestamp(year, 1, 1)
                for tid in estimated:
                    rec = self.db._records.get(tid)
                    if rec is not None:
                        pop = rec.pop.get(ts, np.nan)
                        if not np.isnan(pop) and pop > 0:
                            estimated[tid] = self._scale_table_to_pop(
                                estimated[tid], pop,
                                ogolem_idx, non_ogolem_slices, full_shape,
                            )
            return

        residual = np.maximum(residual, 0.0)

        # Sum estimated gmina core cells
        est_core_sum = np.zeros_like(residual)
        for tbl in estimated.values():
            est_core_sum += self._extract_core(
                np.nan_to_num(tbl, nan=0.0), ndim,
                ogolem_idx, non_ogolem_slices,
            )

        # Cell-wise scaling factors
        with np.errstate(divide='ignore', invalid='ignore'):
            factors = np.where(
                est_core_sum > EPSILON,
                residual / est_core_sum,
                1.0,
            )

        # Apply to each estimated gmina
        for tid in estimated:
            core = self._extract_core(
                estimated[tid], ndim, ogolem_idx, non_ogolem_slices,
            )
            scaled_core = np.maximum(core * factors, 0.0)
            estimated[tid] = self._assemble_with_ogolem(
                scaled_core, ndim, full_shape,
                ogolem_idx, non_ogolem_slices,
            )

    def _hybrid_scale_to_observed(
        self,
        aggregated: np.ndarray,
        parent_tid: str,
        source_sid: str,
        year: int,
        full_shape: tuple,
    ) -> Tuple[np.ndarray, bool]:
        """Scale an aggregated E_ table to match observed M_ total.

        Returns (scaled_table, has_observed) where has_observed is True
        if M_ anchor data was found and used for scaling.

        The aggregated cell *proportions* are preserved; only
        the overall magnitude is adjusted so that
        ``sum(result) == sum(M_observed)``.

        Fix 46: skip hybrid scaling when the scaling factor falls
        below 0.95, which indicates that M_ data likely has
        incomplete gmina coverage (children sum >> M_ total).
        """
        rec = self.db._records.get(parent_tid)
        if rec is None:
            return aggregated, False

        ct = rec.cross_tables.get(source_sid)
        if ct is None or ct.shape != full_shape:
            return aggregated, False

        m_tbl = ct.tables.get(year)
        if m_tbl is None or np.all(np.isnan(m_tbl)):
            return aggregated, False

        m_total = np.nansum(m_tbl)
        agg_total = np.nansum(aggregated)
        if agg_total <= EPSILON or m_total <= 0:
            return aggregated, False

        factor = m_total / agg_total
        # Fix 46: skip hybrid scaling when it would cause a
        # hierarchical consistency violation (powiat ≠ children sum).
        # Compute expected max_cell_diff after scaling and compare
        # against the HIER_TOL_PCT threshold from validation.
        HIER_TOL_PCT = 0.5
        scaled = np.maximum(aggregated * factor, 0.0)
        diff = np.abs(scaled - aggregated)
        max_diff = float(np.max(diff))
        total = float(np.sum(np.abs(scaled)))
        pct = 100 * max_diff / total if total > EPSILON else 0.0
        if pct > HIER_TOL_PCT:
            return aggregated, False
        return scaled, True

    def _aggregate_to_parents(
        self,
        e_sid: str,
        year_range: range,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
        voiv_tids: List[str],
        source_sid: Optional[str] = None,
    ):
        """Aggregate gmina-level E_ tables up to powiat and voivodeship.

        For each (year, voivodeship):
          powiat_table  = Σ gmina_table  for gminas with rodz ∈ {1,2,3}
          voiv_table    = Σ powiat_table + Σ direct-gmina_table

        When *source_sid* is provided, each aggregated parent table is
        scaled so that its total matches the observed M_ total at that
        administrative level (hybrid aggregation).  Cell proportions
        from the gmina aggregation are preserved; only the overall
        magnitude changes.
        """
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        n_pow = 0
        n_voiv = 0
        n_hybrid = 0

        for year in year_range:
            for voiv_tid in voiv_tids:
                voiv_rec = self.db._records.get(voiv_tid)
                if voiv_rec is None:
                    continue
                voiv_total = np.zeros(full_shape, dtype=float)
                has_voiv_data = False

                for child_tid in _get_aggregation_children(
                    voiv_rec, self.db, year
                ):
                    child = self.db._records.get(child_tid)
                    if child is None:
                        continue

                    if child.level == 5:  # powiat
                        powiat_total = np.zeros(full_shape, dtype=float)
                        has_pow_data = False
                        for gid in _get_aggregation_children(
                            child, self.db, year
                        ):
                            grec = self.db._records.get(gid)
                            if grec is None:
                                continue
                            gct = grec.cross_tables.get(e_sid)
                            if gct is None:
                                continue
                            gtbl = gct.tables.get(year)
                            if gtbl is not None and not np.all(np.isnan(gtbl)):
                                powiat_total += np.nan_to_num(gtbl, nan=0.0)
                                has_pow_data = True
                        if has_pow_data:
                            # Hybrid: scale to observed M_ total
                            pow_is_obs = False
                            if source_sid is not None:
                                powiat_total, pow_is_obs = (
                                    self._hybrid_scale_to_observed(
                                        powiat_total, child_tid,
                                        source_sid, year, full_shape,
                                    )
                                )
                                if pow_is_obs:
                                    n_hybrid += 1
                            self._store_estimated_cross_table(
                                child_tid, e_sid, year, powiat_total,
                                dim_names, dim_labels,
                                is_observed=pow_is_obs,
                            )
                            voiv_total += powiat_total
                            has_voiv_data = True
                            n_pow += 1

                    elif child.level == 6:  # direct gmina
                        gct = child.cross_tables.get(e_sid)
                        if gct is not None:
                            gtbl = gct.tables.get(year)
                            if gtbl is not None and not np.all(
                                np.isnan(gtbl)
                            ):
                                voiv_total += np.nan_to_num(gtbl, nan=0.0)
                                has_voiv_data = True

                if has_voiv_data:
                    # Hybrid: scale to observed M_ total
                    voiv_is_obs = False
                    if source_sid is not None:
                        voiv_total, voiv_is_obs = (
                            self._hybrid_scale_to_observed(
                                voiv_total, voiv_tid,
                                source_sid, year, full_shape,
                            )
                        )
                        if voiv_is_obs:
                            n_hybrid += 1
                    self._store_estimated_cross_table(
                        voiv_tid, e_sid, year, voiv_total,
                        dim_names, dim_labels,
                        is_observed=voiv_is_obs,
                    )
                    n_voiv += 1

        self._log(
            f"    Aggregated: {n_pow} powiat-years, {n_voiv} voiv-years"
            f" ({n_hybrid} hybrid-scaled to M_ observed)"
        )

    # ------------------------------------------------------------------
    # Log-linear interpolation (reusable core)
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_log_linear(
        anchor_years: List[int],
        anchor_tables: List[np.ndarray],
        year_range: range,
    ) -> Dict[int, np.ndarray]:
        """Interpolate arrays in log-space between anchor years.

        Parameters
        ----------
        anchor_years : list of int
            Sorted or unsorted years with known data.
        anchor_tables : list of np.ndarray
            Corresponding arrays (same shape, NON-ogółem core cells).
        year_range : range
            Target years.

        Returns
        -------
        dict : year → interpolated np.ndarray (same shape as inputs).
        """
        if not anchor_years:
            return {}

        # Sort
        pairs = sorted(zip(anchor_years, anchor_tables), key=lambda p: p[0])
        anchor_years = [p[0] for p in pairs]
        anchor_tables = [p[1] for p in pairs]

        smoothed = [np.maximum(t, 0.0) + EPSILON for t in anchor_tables]
        log_tables = [np.log(t) for t in smoothed]
        result: Dict[int, np.ndarray] = {}

        if len(anchor_years) == 1:
            for yr in year_range:
                result[yr] = anchor_tables[0].copy()

        elif len(anchor_years) <= 3:
            # ≤3 anchors: linear interpolation in log-space
            for yr in year_range:
                if yr <= anchor_years[0]:
                    result[yr] = anchor_tables[0].copy()
                elif yr >= anchor_years[-1]:
                    result[yr] = anchor_tables[-1].copy()
                else:
                    for ai in range(len(anchor_years) - 1):
                        if anchor_years[ai] <= yr <= anchor_years[ai + 1]:
                            y1, y2 = anchor_years[ai], anchor_years[ai + 1]
                            lt1, lt2 = log_tables[ai], log_tables[ai + 1]
                            f = (yr - y1) / (y2 - y1)
                            result[yr] = np.exp(lt1 * (1 - f) + lt2 * f)
                            break
        elif len(anchor_years) <= 10:
            # 4-10 anchors: PCHIP (shape-preserving monotone cubic)
            ay = np.array(anchor_years, dtype=float)
            flat = np.array([lt.ravel() for lt in log_tables])
            n_cells = flat.shape[1]
            core_shape = anchor_tables[0].shape
            splines = [
                PchipInterpolator(ay, flat[:, ci])
                for ci in range(n_cells)
            ]
            for yr in year_range:
                if yr < anchor_years[0]:
                    result[yr] = anchor_tables[0].copy()
                elif yr > anchor_years[-1]:
                    result[yr] = anchor_tables[-1].copy()
                else:
                    vals = np.array([sp(yr) for sp in splines])
                    result[yr] = np.exp(vals).reshape(core_shape)
        else:
            # >10 anchors: natural cubic spline per cell
            ay = np.array(anchor_years, dtype=float)
            flat = np.array([lt.ravel() for lt in log_tables])
            n_cells = flat.shape[1]
            core_shape = anchor_tables[0].shape
            splines = [
                CubicSpline(ay, flat[:, ci], bc_type='natural')
                for ci in range(n_cells)
            ]
            for yr in year_range:
                if yr < anchor_years[0]:
                    result[yr] = anchor_tables[0].copy()
                elif yr > anchor_years[-1]:
                    result[yr] = anchor_tables[-1].copy()
                else:
                    vals = np.array([sp(yr) for sp in splines])
                    result[yr] = np.exp(vals).reshape(core_shape)

        return result

    # ------------------------------------------------------------------
    # Grouped IPF for 1988 census disaggregation
    # ------------------------------------------------------------------

    def _get_1988_age_marginals(
        self,
        rec: 'TERYTRecord',
        group_idx_map: Dict[str, List[int]],
    ) -> Optional[Dict[str, float]]:
        """Extract 1988 age marginals from M_age_1990 or P2884.

        Returns a dict mapping 10yr group label → total count,
        or ``None`` if data unavailable.
        """
        # Try M_age_1990 first (has corrected ogółem)
        for sid in ('M_age_1990', 'P2884'):
            ct = rec.cross_tables.get(sid)
            if ct is None:
                continue
            tbl = ct.tables.get(1988)
            if tbl is None or np.all(np.isnan(tbl)):
                continue
            labels = ct.dim_labels[ct.dim_names[0]]
            result: Dict[str, float] = {}
            for grp_lbl in group_idx_map:
                idx = None
                for li, lbl in enumerate(labels):
                    if lbl == grp_lbl:
                        idx = li
                        break
                if idx is not None and not np.isnan(tbl[idx]):
                    result[grp_lbl] = float(tbl[idx])
            if len(result) == len(group_idx_map):
                return result
        return None

    def _get_1988_sex_marginals(
        self,
        rec: 'TERYTRecord',
        sex_labels: List[str],
    ) -> Optional[np.ndarray]:
        """Extract 1988 sex marginals from P2883 cross table.

        Returns an array of length ``len(sex_labels)`` with the
        non-ogółem sex values filled and ogółem = sum, or ``None``.
        """
        ct = rec.cross_tables.get('P2883')
        if ct is None:
            return None
        tbl = ct.tables.get(1988)
        if tbl is None or np.all(np.isnan(tbl)):
            return None
        ct_labels = ct.dim_labels[ct.dim_names[0]]

        out = np.full(len(sex_labels), np.nan)
        for i, sl in enumerate(sex_labels):
            if sl.lower() == 'ogółem':
                continue
            for j, cl in enumerate(ct_labels):
                if cl == sl:
                    out[i] = tbl[j]
                    break
        # Fill ogółem as sum of non-ogółem
        og_i = next(
            (i for i, l in enumerate(sex_labels) if l.lower() == 'ogółem'),
            None,
        )
        non_og = [v for i, v in enumerate(out) if i != og_i and not np.isnan(v)]
        if not non_og:
            return None
        if og_i is not None:
            out[og_i] = sum(non_og)

        if np.any(np.isnan(out)):
            return None
        return out

    @staticmethod
    def _grouped_ipf_age_sex(
        seed: np.ndarray,
        age_marginals: Dict[str, float],
        sex_marginals: np.ndarray,
        ogolem_idx: Dict[int, Optional[int]],
        non_ogolem_slices: Dict[int, List[int]],
        full_shape: tuple,
        group_idx_map: Dict[str, List[int]],
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> Optional[np.ndarray]:
        """Run grouped IPF to disaggregate census marginals.

        Fits *seed* (16×3 full table) so that:
          - grouped age sums match *age_marginals* (10yr totals)
          - column sums match *sex_marginals*
          - internal structure (5yr splits) comes from the seed.

        Parameters
        ----------
        seed : np.ndarray
            Old-voivodeship M_age_sex table (full shape incl. ogółem).
        age_marginals : dict
            10yr group label → target total (non-ogółem sex only).
        sex_marginals : np.ndarray
            Target per sex (full label array incl. ogółem).
        group_idx_map : dict
            10yr group label → list of FULL-TABLE row indices.

        Returns
        -------
        np.ndarray or None : adjusted full table, or None on failure.
        """
        ndim = len(full_shape)
        sex_non_og = non_ogolem_slices[1]
        age_non_og = non_ogolem_slices[0]

        # Work with a copy; ensure non-negative
        tbl = np.maximum(seed.copy(), EPSILON)

        for _it in range(max_iter):
            max_change = 0.0

            # ── Age group scaling ──
            for grp_lbl, row_indices in group_idx_map.items():
                target = age_marginals.get(grp_lbl, None)
                if target is None:
                    continue
                # Sum across sex (non-ogółem only) for these rows
                current = sum(
                    tbl[ri, si] for ri in row_indices for si in sex_non_og
                )
                if current > EPSILON:
                    factor = target / current
                    for ri in row_indices:
                        for si in sex_non_og:
                            tbl[ri, si] *= factor
                    max_change = max(max_change, abs(factor - 1.0))

            # ── Sex column scaling ──
            for si_core, si_full in enumerate(sex_non_og):
                target = sex_marginals[si_full]
                current = sum(tbl[ri, si_full] for ri in age_non_og)
                if current > EPSILON:
                    factor = target / current
                    for ri in age_non_og:
                        tbl[ri, si_full] *= factor
                    max_change = max(max_change, abs(factor - 1.0))

            if max_change < tol:
                break

        # Recompute ogółem
        og_age = ogolem_idx.get(0)
        og_sex = ogolem_idx.get(1)
        if og_sex is not None:
            for ri in age_non_og:
                tbl[ri, og_sex] = sum(tbl[ri, si] for si in sex_non_og)
        if og_age is not None:
            for si in sex_non_og:
                tbl[og_age, si] = sum(tbl[ri, si] for ri in age_non_og)
        if og_age is not None and og_sex is not None:
            tbl[og_age, og_sex] = sum(
                tbl[ri, si] for ri in age_non_og for si in sex_non_og
            )

        return tbl

    # ------------------------------------------------------------------
    # 2011 powiat disaggregation helper
    # ------------------------------------------------------------------

    def _disaggregate_2011_powiat_to_gmina(
        self,
        source_sid: str,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        """Disaggregate 2011 powiat-level cross tables to gmina level.

        For subjects where 2011 census data is at powiat level only
        (e.g. P3309 → M_educ_2000, P3420 → M_hh_size_2000), this
        method estimates gmina-level 2011 tables using the structure
        from 2002 and 2021 gmina-level data.

        Algorithm for each powiat with 2011 data:
          1. Collect gmina-level tables for 2002 and 2021.
          2. For each gmina, compute a geometric-mean share per core
             cell (average of 2002 and 2021 proportions in log-space).
          3. Distribute the powiat 2011 total across gminas according
             to these shares.
          4. Recompute ogółem.

        Parameters
        ----------
        source_sid : str
            The M_ subject (e.g. 'M_educ_2000', 'M_hh_size_2000').
        dim_names, dim_labels :
            Cross table dimension specification.

        Returns
        -------
        dict : gmina_teryt_id → full-shape np.ndarray (synthetic 2011)
        """
        ndim = len(dim_names)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        core_shape = tuple(
            len(non_ogolem_slices[d]) for d in range(ndim)
        )

        result: Dict[str, np.ndarray] = {}
        voiv_tids = self._get_voivodeships()
        n_disagg = 0

        for voiv_tid in voiv_tids:
            voiv_rec = self.db._records.get(voiv_tid)
            if voiv_rec is None:
                continue

            for pow_tid in _get_aggregation_children(
                voiv_rec, self.db, 2011
            ):
                pow_rec = self.db._records.get(pow_tid)
                if pow_rec is None or pow_rec.level != 5:
                    continue

                # ── powiat 2011 table ──
                pow_tbl = self._get_observed_table(
                    pow_tid, source_sid, 2011,
                )
                if pow_tbl is None or pow_tbl.shape != full_shape:
                    continue
                pow_core = self._extract_core(
                    np.maximum(pow_tbl, 0.0), ndim,
                    ogolem_idx, non_ogolem_slices,
                )

                # ── gminas in this powiat ──
                gmina_tids = _get_aggregation_children(
                    pow_rec, self.db, 2011,
                )
                if not gmina_tids:
                    continue

                # ── compute geometric-mean shares from 2002/2021 ──
                shares: Dict[str, np.ndarray] = {}
                for gid in gmina_tids:
                    cores_for_avg: List[np.ndarray] = []
                    for anchor_yr in (2002, 2021):
                        tbl = self._get_observed_table(
                            gid, source_sid, anchor_yr,
                        )
                        if tbl is not None and tbl.shape == full_shape:
                            core = self._extract_core(
                                tbl, ndim,
                                ogolem_idx, non_ogolem_slices,
                            )
                            if not np.any(np.isnan(core)):
                                cores_for_avg.append(
                                    np.maximum(core, 0.0) + EPSILON
                                )
                    if cores_for_avg:
                        log_mean = np.mean(
                            [np.log(c) for c in cores_for_avg],
                            axis=0,
                        )
                        shares[gid] = np.exp(log_mean)

                if not shares:
                    continue

                # ── distribute powiat total ──
                total_share = np.zeros(core_shape, dtype=float)
                for sh in shares.values():
                    total_share += sh

                for gid, share in shares.items():
                    with np.errstate(divide='ignore', invalid='ignore'):
                        frac = np.where(
                            total_share > EPSILON,
                            share / total_share,
                            1.0 / len(shares),
                        )
                    gmina_core = np.maximum(frac * pow_core, 0.0)
                    result[gid] = self._assemble_with_ogolem(
                        gmina_core, ndim, full_shape,
                        ogolem_idx, non_ogolem_slices,
                    )
                    n_disagg += 1

        self._log(
            f"    2011 powiat disaggregation: "
            f"{n_disagg} synthetic gmina tables"
        )
        return result

    # ------------------------------------------------------------------
    # Variable-specific pipelines
    # ------------------------------------------------------------------

    def _estimate_age_sex_2000(self, e_sid: str):
        """Age × sex estimation for Prediction2000 (1999–2025).

        Data landscape
        --------------
        - M_age_sex : merged from P2137 (BDL gmina 1995–2024) +
          H_age_sex (1986–1994, old voivodeships).
        - Consistent shape (16, 3) for all records.
        - Coverage: ~87–95 % of gminas per year.
        - Voivodeship-level M_age_sex available as constraint.

        Algorithm
        ---------
        1. Copy observed M_age_sex data directly (provenance=observed).
        2. For missing gminas: log-linear interpolation (Layer 1) from
           all available anchor years in the CrossTable.
        3. Scale estimated gminas so voivodeship residual is matched
           (Layer 3 light).
        4. Ensure per-gmina population consistency.
        5. Aggregate gmina results to powiat and voivodeship levels.
        """
        source_sid = 'M_age_sex'
        year_range = PREDICTION_2000_RANGE

        # ── Step 1: determine dimensions ──
        dim_names, dim_labels = self._get_subject_dimensions(source_sid)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        self._log(f"  Source: {source_sid}  shape={full_shape}")

        # ── Step 2: collect gminas ──
        all_gmina_tids = self._get_all_gminas()
        gminas_with_data = [
            tid for tid in all_gmina_tids
            if source_sid in self.db._records[tid].cross_tables
            and self.db._records[tid].cross_tables[source_sid].shape == full_shape
        ]
        n_shape_mismatch = sum(
            1 for tid in all_gmina_tids
            if source_sid in self.db._records[tid].cross_tables
            and self.db._records[tid].cross_tables[source_sid].shape != full_shape
        )
        self._log(
            f"  Gminas total: {len(all_gmina_tids)}, "
            f"with {source_sid}: {len(gminas_with_data)}"
            + (f" (skipped {n_shape_mismatch} with mismatched shape)"
               if n_shape_mismatch else "")
        )

        # ── Step 3: Layer 1 — seed generation ──
        self._log("  Layer 1: generating seeds (log-linear interpolation)…")
        seeds = self._generate_seeds(
            gminas_with_data, source_sid, year_range,
            dim_names, dim_labels, exclude_ogolem=True,
        )

        # ── Step 4: year-by-year processing ──
        voiv_tids = self._get_voivodeships()
        n_obs_total = 0
        n_est_total = 0

        for year in year_range:
            ts = pd.Timestamp(year, 1, 1)
            n_obs_yr = 0
            n_est_yr = 0

            for voiv_tid in voiv_tids:
                gminas = self._collect_voivodeship_gminas(voiv_tid, year)
                if not gminas:
                    continue

                # ── categorise observed / estimated ──
                observed: Dict[str, np.ndarray] = {}
                estimated: Dict[str, np.ndarray] = {}

                for tid in gminas:
                    tbl = self._get_observed_table(tid, source_sid, year)
                    if tbl is not None and tbl.shape == full_shape:
                        # Fix 45a: recompute ogółem from core cells
                        #   (source BDL may have stale ogółem)
                        core = self._extract_core(
                            tbl, len(dim_names),
                            ogolem_idx, non_ogolem_slices,
                        )
                        fixed_tbl = self._assemble_with_ogolem(
                            core, len(dim_names), full_shape,
                            ogolem_idx, non_ogolem_slices,
                        )
                        # Fix 45a+: scale to population when
                        #   sub-categories do not sum to pop
                        _rec = self.db._records.get(tid)
                        if _rec is not None:
                            _pop = _rec.pop.get(ts, np.nan)
                            if (
                                not np.isnan(_pop) and _pop > 0
                            ):
                                fixed_tbl = self._scale_table_to_pop(
                                    fixed_tbl, _pop,
                                    ogolem_idx,
                                    non_ogolem_slices,
                                    full_shape,
                                )
                        observed[tid] = fixed_tbl
                    elif tid in seeds and year in seeds[tid]:
                        estimated[tid] = seeds[tid][year].copy()

                # ── Layer 3 (light): residual scaling ──
                if estimated:
                    voiv_tbl = self._get_observed_table(
                        voiv_tid, source_sid, year,
                    )
                    if voiv_tbl is not None:
                        self._apply_residual_scaling(
                            observed, estimated, voiv_tbl,
                            ogolem_idx, non_ogolem_slices,
                            full_shape, year=year,
                        )
                    else:
                        # No voivodeship constraint → pop-only scaling
                        for tid in estimated:
                            rec = self.db._records.get(tid)
                            if rec is not None:
                                pop = rec.pop.get(ts, np.nan)
                                if not np.isnan(pop) and pop > 0:
                                    estimated[tid] = self._scale_table_to_pop(
                                        estimated[tid], pop,
                                        ogolem_idx, non_ogolem_slices,
                                        full_shape,
                                    )

                # ── store ──
                for tid, tbl in observed.items():
                    self._store_estimated_cross_table(
                        tid, e_sid, year, tbl,
                        dim_names, dim_labels, is_observed=True,
                    )
                    n_obs_yr += 1
                for tid, tbl in estimated.items():
                    self._store_estimated_cross_table(
                        tid, e_sid, year, tbl,
                        dim_names, dim_labels, is_observed=False,
                    )
                    n_est_yr += 1

            n_obs_total += n_obs_yr
            n_est_total += n_est_yr
            if year % 5 == 0 or year == year_range[-1]:
                self._log(
                    f"    {year}: {n_obs_yr} obs + {n_est_yr} est"
                )

        # ── Step 5: aggregate to powiat / voivodeship ──
        self._log("  Aggregating to powiat and voivodeship levels…")
        self._aggregate_to_parents(
            e_sid, year_range, dim_names, dim_labels, voiv_tids,
            source_sid=source_sid,
        )

        self._log(
            f"  Summary: {n_obs_total} observed + "
            f"{n_est_total} estimated cell-years stored"
        )

    def _estimate_age_sex_1990(self, e_sid: str):
        """Age × sex estimation for Prediction1990 (1986–2002).

        Data landscape
        --------------
        - M_age_sex (16×3): merged from P2137 (gmina 1995–2024) +
          H_age_sex (old voivodeships 1986–1994).
        - M_age_1990 (8,): census 1988 gmina (P2884), 10yr bins.
        - P2883 (3,): census 1988 gmina sex marginals.
        - H_age_sex on old voivodeships: 1986–1994.

        Algorithm
        ---------
        Phase A: Construct 1988 gmina-level age×sex (16×3) via grouped
                 IPF using H_age_sex old-voivodeship seed + P2884 age
                 marginals + P2883 sex marginals.
        Phase B: Collect anchor tables per gmina (1988 from Phase A +
                 M_age_sex 1995+ from BDL) and interpolate in log-space.
        Phase C: Layer 2 — scale gmina aggregates to match old
                 voivodeship M_age_sex totals for 1986–1994.
        Phase D: Store all results.
        """
        source_sid = 'M_age_sex'
        year_range = PREDICTION_1990_RANGE

        # ── Step 1: dimensions ──
        dim_names, dim_labels = self._get_subject_dimensions(source_sid)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        self._log(f"  Source: {source_sid}  shape={full_shape}")

        age_labels = dim_labels[dim_names[0]]
        sex_labels = dim_labels[dim_names[1]]

        # Mapping: 5yr M_age_sex bins → 10yr P2884 groups
        AGE_5_TO_10 = {
            '0-9':                ['0-4', '5-9'],
            '10-19':              ['10-14', '15-19'],
            '20-29':              ['20-24', '25-29'],
            '30-39':              ['30-34', '35-39'],
            '40-49':              ['40-44', '45-49'],
            '50-59':              ['50-54', '55-59'],
            '60 lat i więcej':    ['60-64', '65-69', '70 i więcej'],
        }
        # Build index map: 10yr label → list of FULL-TABLE row indices
        group_idx_map: Dict[str, List[int]] = {}
        for grp_lbl, sub_labels in AGE_5_TO_10.items():
            indices = [
                age_labels.index(sl) for sl in sub_labels
                if sl in age_labels
            ]
            if indices:
                group_idx_map[grp_lbl] = indices

        # ── Step 2: Phase A — construct 1988 tables ──
        self._log("  Phase A: constructing 1988 gmina age×sex via IPF…")
        all_gmina_tids = self._get_all_gminas()
        tables_1988: Dict[str, np.ndarray] = {}
        n_ok = 0
        n_skip = 0

        for tid in all_gmina_tids:
            rec = self.db._records.get(tid)
            if rec is None:
                n_skip += 1
                continue

            # --- old voivodeship seed ---
            old_voi_tid = getattr(rec, 'old_woj_id', None)
            if old_voi_tid is None:
                n_skip += 1
                continue
            old_voi_tid = str(old_voi_tid)
            old_voi_rec = self.db._records.get(old_voi_tid)
            if old_voi_rec is None:
                n_skip += 1
                continue
            voi_ct = old_voi_rec.cross_tables.get(source_sid)
            if voi_ct is None or voi_ct.shape != full_shape:
                n_skip += 1
                continue
            seed = voi_ct.tables.get(1988)
            if seed is None or np.all(np.isnan(seed)):
                n_skip += 1
                continue

            # --- P2884 age marginals (10yr bins) ---
            age_marg = self._get_1988_age_marginals(
                rec, group_idx_map,
            )
            if age_marg is None:
                n_skip += 1
                continue

            # --- P2883 sex marginals ---
            sex_marg = self._get_1988_sex_marginals(rec, sex_labels)
            if sex_marg is None:
                # fallback: use old voi sex ratio × gmina pop
                pop88 = rec.pop.get(pd.Timestamp(1988, 1, 1), np.nan)
                if np.isnan(pop88) or pop88 <= 0:
                    n_skip += 1
                    continue
                og_sex = ogolem_idx.get(1)  # sex ogółem index
                sex_non_og = non_ogolem_slices[1]
                voi_sex_sums = np.array(
                    [seed[:, si].sum() for si in sex_non_og],
                )
                voi_sex_total = voi_sex_sums.sum()
                if voi_sex_total <= 0:
                    n_skip += 1
                    continue
                sex_marg = voi_sex_sums * (pop88 / voi_sex_total)

            # --- grouped IPF ---
            result = self._grouped_ipf_age_sex(
                seed.copy(), age_marg, sex_marg,
                ogolem_idx, non_ogolem_slices,
                full_shape, group_idx_map,
            )
            if result is not None:
                # Scale to population for consistency
                pop88 = rec.pop.get(pd.Timestamp(1988, 1, 1), np.nan)
                if not np.isnan(pop88) and pop88 > 0:
                    result = self._scale_table_to_pop(
                        result, pop88,
                        ogolem_idx, non_ogolem_slices, full_shape,
                    )
                tables_1988[tid] = result
                n_ok += 1
            else:
                n_skip += 1

        self._log(
            f"    1988 IPF: {n_ok} OK, {n_skip} skipped"
        )

        # ── Step 3: Phase B — build seeds ──
        self._log("  Phase B: building seeds (log-linear interpolation)…")
        seeds: Dict[str, Dict[int, np.ndarray]] = {}
        core_shape = tuple(
            len(non_ogolem_slices[d]) for d in range(len(dim_names))
        )

        for tid in all_gmina_tids:
            rec = self.db._records.get(tid)
            if rec is None:
                continue

            anchor_years: List[int] = []
            anchor_cores: List[np.ndarray] = []

            # 1988 anchor from Phase A
            if tid in tables_1988:
                core88 = self._extract_core(
                    tables_1988[tid], len(dim_names),
                    ogolem_idx, non_ogolem_slices,
                )
                anchor_years.append(1988)
                anchor_cores.append(core88)

            # M_age_sex anchors (1995+ from BDL, if shape matches)
            ct = rec.cross_tables.get(source_sid)
            if ct is not None and ct.shape == full_shape:
                for yr in sorted(ct.tables.keys()):
                    tbl = ct.tables[yr]
                    if tbl is None or np.all(np.isnan(tbl)):
                        continue
                    core_t = self._extract_core(
                        tbl, len(dim_names),
                        ogolem_idx, non_ogolem_slices,
                    )
                    if not np.any(np.isnan(core_t)):
                        anchor_years.append(yr)
                        anchor_cores.append(core_t)

            if not anchor_years:
                continue

            # Interpolate
            interp = self._interpolate_log_linear(
                anchor_years, anchor_cores, year_range,
            )
            # Assemble full tables
            year_tables: Dict[int, np.ndarray] = {}
            for yr, core_tbl in interp.items():
                core_tbl = np.maximum(core_tbl, 0.0)
                year_tables[yr] = self._assemble_with_ogolem(
                    core_tbl, len(dim_names), full_shape,
                    ogolem_idx, non_ogolem_slices,
                )
            seeds[tid] = year_tables

        self._log(f"    Seeds: {len(seeds)} gminas")

        # ── Step 4: Phase C — Layer 2 (old voi constraints) ──
        # Fix 44: growth-rate-deviation — voivodeship data only
        # adjusts HOW FAST each cell grows, not WHAT the proportions are.
        # Census 1988 serves as baseline; factors ≈ 1.0 at baseline.
        self._log("  Phase C: old voivodeship marginal scaling (1986–1994)…")
        country_rec = self.db._records.get('0000000')
        old_voi_tids = (
            country_rec.children_ids.get('old', [])
            if country_rec is not None else []
        )

        BASELINE_YEAR = 1988
        factor_guard = 5.0

        # Pre-compute 1988 baselines per old voivodeship
        ov_baselines: Dict[str, tuple] = {}
        for ov_tid in old_voi_tids:
            ov_rec = self.db._records.get(ov_tid)
            if ov_rec is None:
                continue
            ov_ct = ov_rec.cross_tables.get(source_sid)
            if ov_ct is None or ov_ct.shape != full_shape:
                continue
            ov_tbl_base = ov_ct.tables.get(BASELINE_YEAR)
            if ov_tbl_base is None or np.all(np.isnan(ov_tbl_base)):
                continue
            children = ov_rec.get_children(BASELINE_YEAR)
            gminas_base = [
                g for g in children
                if g in seeds and BASELINE_YEAR in seeds[g]
            ]
            if not gminas_base:
                continue
            agg_base = np.zeros(full_shape, dtype=float)
            for g in gminas_base:
                agg_base += np.nan_to_num(
                    seeds[g][BASELINE_YEAR], nan=0.0,
                )
            ov_baselines[ov_tid] = (ov_tbl_base, agg_base)

        n_scaled = 0
        for year in range(1986, 1995):
            for ov_tid in old_voi_tids:
                if ov_tid not in ov_baselines:
                    continue
                ov_rec = self.db._records.get(ov_tid)
                if ov_rec is None:
                    continue
                ov_ct = ov_rec.cross_tables.get(source_sid)
                if ov_ct is None or ov_ct.shape != full_shape:
                    continue
                ov_tbl = ov_ct.tables.get(year)
                if ov_tbl is None or np.all(np.isnan(ov_tbl)):
                    continue

                children = ov_rec.get_children(year)
                gminas_in_ov = [
                    g for g in children
                    if g in seeds and year in seeds[g]
                ]
                if not gminas_in_ov:
                    continue

                agg = np.zeros(full_shape, dtype=float)
                for g in gminas_in_ov:
                    agg += np.nan_to_num(seeds[g][year], nan=0.0)

                # Growth-rate-deviation factors
                ov_tbl_base, agg_base = ov_baselines[ov_tid]
                with np.errstate(divide='ignore', invalid='ignore'):
                    voiv_growth = np.where(
                        ov_tbl_base > EPSILON,
                        ov_tbl / ov_tbl_base, 1.0,
                    )
                    agg_growth = np.where(
                        agg_base > EPSILON,
                        agg / agg_base, 1.0,
                    )
                    factors = np.where(
                        agg_growth > EPSILON,
                        voiv_growth / agg_growth, 1.0,
                    )
                # Guard against extreme factors
                factors = np.clip(
                    factors, 1.0 / factor_guard, factor_guard,
                )

                for g in gminas_in_ov:
                    seeds[g][year] = np.maximum(
                        seeds[g][year] * factors, 0.0,
                    )
                    # Recompute ogółem
                    core = self._extract_core(
                        seeds[g][year], len(dim_names),
                        ogolem_idx, non_ogolem_slices,
                    )
                    seeds[g][year] = self._assemble_with_ogolem(
                        core, len(dim_names), full_shape,
                        ogolem_idx, non_ogolem_slices,
                    )
                n_scaled += 1

        self._log(f"    Scaled {n_scaled} old-voi × year combinations")

        # ── Step 4b: Final pop consistency ──
        # After hierarchical scaling, re-scale each gmina to its
        # known population so that E_total = record.pop.
        for tid, year_tables in seeds.items():
            rec = self.db._records.get(tid)
            if rec is None:
                continue
            for yr in list(year_tables.keys()):
                ts = pd.Timestamp(yr, 1, 1)
                pop = rec.pop.get(ts, np.nan)
                if np.isnan(pop) or pop <= 0:
                    continue
                year_tables[yr] = self._scale_table_to_pop(
                    year_tables[yr], pop,
                    ogolem_idx, non_ogolem_slices, full_shape,
                )

        # ── Step 5: Phase D — store results ──
        self._log("  Phase D: storing results…")
        n_stored = 0
        for tid, year_tables in seeds.items():
            for year, tbl in year_tables.items():
                if year not in year_range:
                    continue
                # Mark as observed if M_age_sex has data for this year
                is_obs = False
                rec = self.db._records.get(tid)
                if rec is not None:
                    sct = rec.cross_tables.get(source_sid)
                    if sct is not None and sct.shape == full_shape:
                        stbl = sct.tables.get(year)
                        if stbl is not None and not np.all(np.isnan(stbl)):
                            is_obs = True
                self._store_estimated_cross_table(
                    tid, e_sid, year, tbl,
                    dim_names, dim_labels, is_observed=is_obs,
                )
                n_stored += 1

        # ── aggregate to parent levels ──
        voiv_tids = self._get_voivodeships()
        self._log("  Aggregating to powiat and voivodeship levels…")
        self._aggregate_to_parents(
            e_sid, year_range, dim_names, dim_labels, voiv_tids,
            source_sid=source_sid,
        )

        self._log(f"  Summary: {n_stored} cell-years for {len(seeds)} gminas")

    def _estimate_educ_2000(self, e_sid: str):
        """Education estimation for Prediction2000 (1999–2025).

        Data landscape
        --------------
        - M_educ_2000: shape (5,), 1D, **NO ogółem label**.
          Labels: wyższe, policealne…, średnie…, zasadnicze…,
                  gimnazjalne…
        - Census 2002 gmina (P2402), 2011 powiat (P3309),
          2021 gmina (P4315).
        - Voivodeship annual marginals: P2350 (1995–2020),
          P4092 (2010–2024) → stored in M_educ_2000 at level 2.

        Algorithm
        ---------
        1. Disaggregate 2011 powiat data to gmina level.
        2. Layer 1: Log-linear spline through anchors (2002, 2011-syn, 2021).
        3. Layer 2: Scale gmina aggregates per voivodeship to match
           M_educ_2000 voivodeship marginals for each year.
        4. Store results.
        5. Aggregate to powiat and voivodeship levels.
        """
        source_sid = 'M_educ_2000'
        year_range = PREDICTION_2000_RANGE

        # ── dimensions ──
        dim_names, dim_labels = self._get_subject_dimensions(source_sid)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        ndim = len(dim_names)
        self._log(f"  Source: {source_sid}  shape={full_shape}")

        # ── 2011 disaggregation ──
        self._log("  Disaggregating 2011 powiat data to gmina…")
        synthetic_2011 = self._disaggregate_2011_powiat_to_gmina(
            source_sid, dim_names, dim_labels,
        )

        # ── build seeds ──
        self._log("  Layer 1: building seeds (log-linear interpolation)…")
        all_gmina_tids = self._get_all_gminas()
        seeds: Dict[str, Dict[int, np.ndarray]] = {}
        n_generated = 0

        for tid in all_gmina_tids:
            rec = self.db._records.get(tid)
            if rec is None:
                continue

            anchor_years: List[int] = []
            anchor_cores: List[np.ndarray] = []

            ct = rec.cross_tables.get(source_sid)
            if ct is not None and ct.shape == full_shape:
                for yr in sorted(ct.tables.keys()):
                    tbl = ct.tables[yr]
                    if tbl is None or np.all(np.isnan(tbl)):
                        continue
                    core = self._extract_core(
                        tbl, ndim, ogolem_idx, non_ogolem_slices,
                    )
                    if not np.any(np.isnan(core)):
                        anchor_years.append(yr)
                        anchor_cores.append(core.copy())

            if tid in synthetic_2011 and 2011 not in anchor_years:
                syn_core = self._extract_core(
                    synthetic_2011[tid], ndim,
                    ogolem_idx, non_ogolem_slices,
                )
                if not np.any(np.isnan(syn_core)):
                    anchor_years.append(2011)
                    anchor_cores.append(syn_core.copy())

            if not anchor_years:
                continue

            interp = self._interpolate_log_linear(
                anchor_years, anchor_cores, year_range,
            )
            year_tables: Dict[int, np.ndarray] = {}
            for yr, core_tbl in interp.items():
                core_tbl = np.maximum(core_tbl, 0.0)
                year_tables[yr] = self._assemble_with_ogolem(
                    core_tbl, ndim, full_shape,
                    ogolem_idx, non_ogolem_slices,
                )
            seeds[tid] = year_tables
            n_generated += 1

        self._log(f"    Seeds: {n_generated} gminas")

        # ── Snapshot census data before scaling ──
        observed_census = self._collect_observed_tables(
            seeds, source_sid, full_shape,
        )

        # Fix 45b: protect synthetic 2011 from Layer 2 modification
        for tid, syn_tbl in synthetic_2011.items():
            if tid in seeds and 2011 in seeds[tid]:
                observed_census.setdefault(tid, {})[2011] = syn_tbl.copy()

        # ── Layer 2: voivodeship marginal scaling (temporally smoothed) ──
        self._log("  Layer 2: voivodeship marginal scaling (smoothed)…")
        voiv_tids = self._get_voivodeships()

        # Collect observed years per gmina to skip during scaling
        observed_years_per_gmina: Dict[str, Set[int]] = {}
        for tid, yr_tbls in observed_census.items():
            observed_years_per_gmina[tid] = set(yr_tbls.keys())

        n_scaled = self._layer2_voiv_scaling_smoothed(
            seeds, source_sid, year_range,
            dim_names, dim_labels,
            observed_years_per_gmina,
            factor_guard=3.0,
        )

        self._log(
            f"    Scaled {n_scaled} voivodeship-year combinations"
        )

        # ── Restore census data (belt-and-suspenders) ──
        n_restored = self._restore_census_data(seeds, observed_census)
        self._log(f"    Census data restored: {n_restored} gmina-years")

        # ── store ──
        self._log("  Storing results…")
        n_stored = 0
        for tid, year_tables in seeds.items():
            for year, tbl in year_tables.items():
                if year not in year_range:
                    continue
                is_obs = False
                rec = self.db._records.get(tid)
                if rec is not None:
                    sct = rec.cross_tables.get(source_sid)
                    if sct is not None and sct.shape == full_shape:
                        stbl = sct.tables.get(year)
                        if stbl is not None and not np.all(
                            np.isnan(stbl)
                        ):
                            is_obs = True
                self._store_estimated_cross_table(
                    tid, e_sid, year, tbl,
                    dim_names, dim_labels, is_observed=is_obs,
                )
                n_stored += 1

        # ── aggregate ──
        self._log("  Aggregating to powiat and voivodeship levels…")
        self._aggregate_to_parents(
            e_sid, year_range, dim_names, dim_labels, voiv_tids,
            source_sid=source_sid,
        )

        self._log(
            f"  Summary: {n_stored} cell-years for "
            f"{len(seeds)} gminas"
        )

    def _estimate_educ_1990(self, e_sid: str):
        """Education estimation for Prediction1990 (1986–2002).

        Data landscape
        --------------
        - M_educ_1990: shape (6,), 1D, WITH ogółem.
          Labels: ogółem, wyższe, średnie, zasadnicze zawodowe,
                  podstawowe, podstawowe nieukończone i bez wykształcenia
        - Census 1988 gmina (P2885 + P2884-derived ogółem + residual):
          all 6 categories already stored in M_educ_1990 during
          database construction.
        - Census 2002 gmina (P2402 sex=ogółem): 6 categories in
          M_educ_1990.
        - Country-level H_sex_educ (sex=ogółem extraction) → stored
          in M_educ_1990 at level 0 for 1986–1988 and 1991–1994.
        - ogółem = total population 15+ (NOT total population).

        Algorithm
        ---------
        1. Layer 1: Log-linear interpolation of M_educ_1990 gmina
           data (anchors typically 1988 and 2002).
        2. Layer 2: Scale national totals to match M_educ_1990
           country-level data for years where available (1986–88,
           1991–94).
        3. Store results.
        4. Aggregate to powiat and voivodeship levels.
        """
        source_sid = 'M_educ_1990'
        year_range = PREDICTION_1990_RANGE

        # ── dimensions ──
        dim_names, dim_labels = self._get_subject_dimensions(source_sid)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        ndim = len(dim_names)
        self._log(f"  Source: {source_sid}  shape={full_shape}")

        # ── collect gminas and generate seeds ──
        all_gmina_tids = self._get_all_gminas()
        gminas_with_data = [
            tid for tid in all_gmina_tids
            if tid in self.db._records
            and source_sid in self.db._records[tid].cross_tables
            and self.db._records[tid].cross_tables[source_sid].shape
            == full_shape
        ]
        self._log(
            f"  Gminas total: {len(all_gmina_tids)}, "
            f"with {source_sid}: {len(gminas_with_data)}"
        )

        self._log("  Layer 1: generating seeds (log-linear interpolation)…")
        seeds = self._generate_seeds(
            gminas_with_data, source_sid, year_range,
            dim_names, dim_labels, exclude_ogolem=True,
        )

        # ── Snapshot census data before scaling ──
        observed_census = self._collect_observed_tables(
            seeds, source_sid, full_shape,
        )
        observed_years_per_gmina: Dict[str, Set[int]] = {
            tid: set(yr_tbls.keys())
            for tid, yr_tbls in observed_census.items()
        }

        # ── Layer 2: national marginal scaling (temporally smoothed) ──
        # M_educ_1990 at country level (teryt 0000000) has H_sex_educ
        # data for 1986–88 and 1991–94.
        self._log("  Layer 2: national marginal scaling (smoothed)…")

        n_scaled = self._layer2_national_scaling_smoothed(
            seeds, source_sid, year_range,
            dim_names, dim_labels,
            observed_years_per_gmina,
        )

        self._log(
            f"    Scaled {n_scaled} national-year combinations"
        )

        # ── Restore census data (belt-and-suspenders) ──
        n_restored = self._restore_census_data(seeds, observed_census)
        self._log(f"    Census data restored: {n_restored} gmina-years")

        # ── store ──
        self._log("  Storing results…")
        n_stored = 0
        for tid, year_tables in seeds.items():
            for year, tbl in year_tables.items():
                if year not in year_range:
                    continue
                is_obs = False
                rec = self.db._records.get(tid)
                if rec is not None:
                    sct = rec.cross_tables.get(source_sid)
                    if sct is not None and sct.shape == full_shape:
                        stbl = sct.tables.get(year)
                        if stbl is not None and not np.all(
                            np.isnan(stbl)
                        ):
                            is_obs = True
                self._store_estimated_cross_table(
                    tid, e_sid, year, tbl,
                    dim_names, dim_labels, is_observed=is_obs,
                )
                n_stored += 1

        # ── aggregate ──
        voiv_tids = self._get_voivodeships()
        self._log("  Aggregating to powiat and voivodeship levels…")
        self._aggregate_to_parents(
            e_sid, year_range, dim_names, dim_labels, voiv_tids,
            source_sid=source_sid,
        )

        self._log(
            f"  Summary: {n_stored} cell-years for "
            f"{len(seeds)} gminas"
        )

    def _estimate_educ_sex_2000(self, e_sid: str):
        """Education × sex estimation for Prediction2000 (1999–2025).

        Data landscape
        --------------
        - M_educ_sex_2000: shape (5, 3), 2D.
          Educ dim: wyższe, policealne…, średnie…, zasadnicze…,
                    gimnazjalne…  (NO ogółem on educ dim)
          Sex dim:  ogółem, mężczyźni, kobiety
        - Census 2002 gmina (P2402), 2011 powiat (P3309),
          2021 gmina (P4315).
        - Voivodeship marginals: M_educ_2000 (1D, P2350/P4092)
          constrains education marginals (sum across sex).

        Algorithm
        ---------
        1. Disaggregate 2011 powiat data to gmina level.
        2. Layer 1: Log-linear spline through anchors.
        3. Layer 2: Scale education-marginal (row sums across sex) of
           gmina aggregates to match voivodeship M_educ_2000.
        4. Store results.
        5. Aggregate to parents.
        """
        source_sid = 'M_educ_sex_2000'
        year_range = PREDICTION_2000_RANGE

        # ── dimensions ──
        dim_names, dim_labels = self._get_subject_dimensions(source_sid)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        ndim = len(dim_names)
        self._log(f"  Source: {source_sid}  shape={full_shape}")

        # ── 2011 disaggregation ──
        self._log("  Disaggregating 2011 powiat data to gmina…")
        synthetic_2011 = self._disaggregate_2011_powiat_to_gmina(
            source_sid, dim_names, dim_labels,
        )

        # ── build seeds ──
        self._log("  Layer 1: building seeds (log-linear interpolation)…")
        all_gmina_tids = self._get_all_gminas()
        seeds: Dict[str, Dict[int, np.ndarray]] = {}
        n_generated = 0

        for tid in all_gmina_tids:
            rec = self.db._records.get(tid)
            if rec is None:
                continue

            anchor_years: List[int] = []
            anchor_cores: List[np.ndarray] = []

            ct = rec.cross_tables.get(source_sid)
            if ct is not None and ct.shape == full_shape:
                for yr in sorted(ct.tables.keys()):
                    tbl = ct.tables[yr]
                    if tbl is None or np.all(np.isnan(tbl)):
                        continue
                    core = self._extract_core(
                        tbl, ndim, ogolem_idx, non_ogolem_slices,
                    )
                    if not np.any(np.isnan(core)):
                        anchor_years.append(yr)
                        anchor_cores.append(core.copy())

            if tid in synthetic_2011 and 2011 not in anchor_years:
                syn_core = self._extract_core(
                    synthetic_2011[tid], ndim,
                    ogolem_idx, non_ogolem_slices,
                )
                if not np.any(np.isnan(syn_core)):
                    anchor_years.append(2011)
                    anchor_cores.append(syn_core.copy())

            if not anchor_years:
                continue

            interp = self._interpolate_log_linear(
                anchor_years, anchor_cores, year_range,
            )
            year_tables: Dict[int, np.ndarray] = {}
            for yr, core_tbl in interp.items():
                core_tbl = np.maximum(core_tbl, 0.0)
                year_tables[yr] = self._assemble_with_ogolem(
                    core_tbl, ndim, full_shape,
                    ogolem_idx, non_ogolem_slices,
                )
            seeds[tid] = year_tables
            n_generated += 1

        self._log(f"    Seeds: {n_generated} gminas")

        # ── Snapshot census data before scaling ──
        observed_census = self._collect_observed_tables(
            seeds, source_sid, full_shape,
        )

        # Fix 45b: protect synthetic 2011 from Layer 2 modification
        for tid, syn_tbl in synthetic_2011.items():
            if tid in seeds and 2011 in seeds[tid]:
                observed_census.setdefault(tid, {})[2011] = syn_tbl.copy()

        observed_years_per_gmina: Dict[str, Set[int]] = {
            tid: set(yr_tbls.keys())
            for tid, yr_tbls in observed_census.items()
        }

        # ── Layer 2: voivodeship education-marginal scaling (smoothed) ──
        # Use M_educ_2000 (1D, no ogółem) at voivodeship level to
        # constrain education marginals (sum across non-ogółem sex).
        self._log("  Layer 2: voivodeship education-marginal scaling (smoothed)…")
        marginal_sid = 'M_educ_2000'

        n_scaled = self._layer2_educ_sex_marginal_smoothed(
            seeds, source_sid, marginal_sid,
            year_range, dim_names, dim_labels,
            observed_years_per_gmina,
            factor_guard=3.0,
        )

        self._log(
            f"    Scaled {n_scaled} voivodeship-year combinations"
        )

        # ── Restore census data (belt-and-suspenders) ──
        n_restored = self._restore_census_data(seeds, observed_census)
        # Fix 45c: recompute ogółem in restored census tables
        #   (source data may have stale ogółem marginals)
        for _tid in observed_census:
            if _tid not in seeds:
                continue
            for _yr in observed_census[_tid]:
                if _yr in seeds[_tid]:
                    _core = self._extract_core(
                        seeds[_tid][_yr], ndim,
                        ogolem_idx, non_ogolem_slices,
                    )
                    seeds[_tid][_yr] = self._assemble_with_ogolem(
                        _core, ndim, full_shape,
                        ogolem_idx, non_ogolem_slices,
                    )
        self._log(f"    Census data restored: {n_restored} gmina-years")

        # ── store ──
        self._log("  Storing results…")
        n_stored = 0
        for tid, year_tables in seeds.items():
            for year, tbl in year_tables.items():
                if year not in year_range:
                    continue
                is_obs = False
                rec = self.db._records.get(tid)
                if rec is not None:
                    sct = rec.cross_tables.get(source_sid)
                    if sct is not None and sct.shape == full_shape:
                        stbl = sct.tables.get(year)
                        if stbl is not None and not np.all(
                            np.isnan(stbl)
                        ):
                            is_obs = True
                self._store_estimated_cross_table(
                    tid, e_sid, year, tbl,
                    dim_names, dim_labels, is_observed=is_obs,
                )
                n_stored += 1

        # ── aggregate ──
        voiv_tids = self._get_voivodeships()
        self._log("  Aggregating to powiat and voivodeship levels…")
        self._aggregate_to_parents(
            e_sid, year_range, dim_names, dim_labels, voiv_tids,
            source_sid=source_sid,
        )

        self._log(
            f"  Summary: {n_stored} cell-years for "
            f"{len(seeds)} gminas"
        )

    def _estimate_educ_sex_1990(self, e_sid: str):
        """Education × sex estimation for Prediction1990 (1986–2002).

        Data landscape
        --------------
        - M_educ_sex_1990: shape (6, 3), 2D (educ × sex).
          Educ labels: ogółem, wyższe, średnie, zasadnicze zawodowe,
                       podstawowe, podstawowe nieukończone…
          Sex labels:  ogółem, mężczyźni, kobiety
        - Census 2002 gmina (P2402 → M_educ_sex_1990).
        - Country-level H_sex_educ → M_educ_sex_1990 at level 0
          (1986–1988, 1991–1994).
        - NO 1988 gmina-level sex×educ joint data. Only M_educ_1990
          (1D educ) and P2883 (1D sex) at gmina level.

        Algorithm
        ---------
        Phase A: Construct 1988 gmina educ×sex via IPF:
          - Seed = M_educ_sex_1990 at country level (1988)
          - Fit educ marginals from M_educ_1990 gmina data (1988).
          - Sex distribution inherited from national structure.
        Phase B: Build seeds (1988 from Phase A + 2002 from
                 M_educ_sex_1990) via log-linear interpolation.
        Phase C: Layer 2 — scale national totals to match
                 M_educ_sex_1990 at country level for 1986–1994.
        Phase D: Store results.
        """
        source_sid = 'M_educ_sex_1990'
        educ_1d_sid = 'M_educ_1990'
        year_range = PREDICTION_1990_RANGE

        # ── dimensions ──
        dim_names, dim_labels = self._get_subject_dimensions(source_sid)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        ndim = len(dim_names)
        core_shape = tuple(
            len(non_ogolem_slices[d]) for d in range(ndim)
        )
        self._log(f"  Source: {source_sid}  shape={full_shape}")

        # Get 1D education dimensions for marginal matching
        try:
            educ_1d_names, educ_1d_labels = self._get_subject_dimensions(
                educ_1d_sid,
            )
            educ_1d_shape = tuple(
                len(educ_1d_labels[d]) for d in educ_1d_names
            )
            educ_1d_ogi, educ_1d_noi = self._identify_ogolem(
                educ_1d_names, educ_1d_labels,
            )
        except ValueError:
            self._log(
                "  ⚠  Cannot find M_educ_1990 dimensions — "
                "skipping Phase A"
            )
            educ_1d_shape = None

        # Build label mapping: 2D educ row → 1D educ core index
        educ_dim_name = dim_names[0]
        educ_labels_2d = dim_labels[educ_dim_name]
        educ_non_og_2d = non_ogolem_slices[0]

        label_2d_core_to_1d_core: Dict[int, int] = {}
        if educ_1d_shape is not None:
            educ_labels_1d = educ_1d_labels[educ_1d_names[0]]
            educ_non_og_1d = educ_1d_noi[0]
            for ci2d, fi2d in enumerate(educ_non_og_2d):
                lbl2d = educ_labels_2d[fi2d]
                for ci1d, fi1d in enumerate(educ_non_og_1d):
                    if educ_labels_1d[fi1d] == lbl2d:
                        label_2d_core_to_1d_core[ci2d] = ci1d
                        break

        # ── Phase A: construct 1988 gmina educ×sex via IPF ──
        self._log(
            "  Phase A: constructing 1988 gmina educ×sex via IPF…"
        )

        # National seed (country level, 1988)
        country_tid = '0000000'
        national_seed = self._get_observed_table(
            country_tid, source_sid, 1988,
        )
        national_seed_core: Optional[np.ndarray] = None
        if national_seed is not None and national_seed.shape == full_shape:
            national_seed_core = self._extract_core(
                national_seed, ndim, ogolem_idx, non_ogolem_slices,
            )
        else:
            self._log(
                "  ⚠  No national M_educ_sex_1990 seed for 1988"
            )

        all_gmina_tids = self._get_all_gminas()
        tables_1988: Dict[str, np.ndarray] = {}
        n_ok = 0
        n_skip = 0

        for tid in all_gmina_tids:
            rec = self.db._records.get(tid)
            if rec is None:
                n_skip += 1
                continue

            if (
                national_seed_core is None
                or educ_1d_shape is None
            ):
                n_skip += 1
                continue

            # gmina M_educ_1990 (1D) for 1988
            educ_1d_tbl = self._get_observed_table(
                tid, educ_1d_sid, 1988,
            )
            if (
                educ_1d_tbl is None
                or educ_1d_tbl.shape != educ_1d_shape
            ):
                n_skip += 1
                continue

            educ_1d_core = self._extract_core(
                educ_1d_tbl, 1, educ_1d_ogi, educ_1d_noi,
            )
            if np.any(np.isnan(educ_1d_core)):
                n_skip += 1
                continue

            # Build education-marginal target for IPF on core table.
            # educ_marginal[ci_2d] = sum across sex of core row ci_2d
            # → should match educ_1d_core[ci_1d] for matched labels.
            educ_marginal = np.zeros(
                len(educ_non_og_2d), dtype=float,
            )
            for ci2d, ci1d in label_2d_core_to_1d_core.items():
                if ci1d < len(educ_1d_core):
                    educ_marginal[ci2d] = max(
                        educ_1d_core[ci1d], 0.0,
                    )

            if np.sum(educ_marginal) <= EPSILON:
                n_skip += 1
                continue

            # IPF: fit national seed core to gmina educ marginals.
            # Marginal on dim 0 (education): row sums across sex
            # should match educ_marginal.
            seed_core = np.maximum(
                national_seed_core.copy(), EPSILON,
            )
            fitted = self._fit_marginals_ipf(
                seed_core,
                [(educ_marginal, [0])],
            )

            full_tbl = self._assemble_with_ogolem(
                fitted, ndim, full_shape,
                ogolem_idx, non_ogolem_slices,
            )
            tables_1988[tid] = full_tbl
            n_ok += 1

        self._log(f"    1988 IPF: {n_ok} OK, {n_skip} skipped")

        # ── Phase B: build seeds ──
        self._log(
            "  Phase B: building seeds (log-linear interpolation)…"
        )
        seeds: Dict[str, Dict[int, np.ndarray]] = {}

        for tid in all_gmina_tids:
            rec = self.db._records.get(tid)
            if rec is None:
                continue

            anchor_years: List[int] = []
            anchor_cores: List[np.ndarray] = []

            # 1988 from Phase A
            if tid in tables_1988:
                core88 = self._extract_core(
                    tables_1988[tid], ndim,
                    ogolem_idx, non_ogolem_slices,
                )
                anchor_years.append(1988)
                anchor_cores.append(core88)

            # M_educ_sex_1990 observed data (2002 from P2402)
            ct = rec.cross_tables.get(source_sid)
            if ct is not None and ct.shape == full_shape:
                for yr in sorted(ct.tables.keys()):
                    tbl = ct.tables[yr]
                    if tbl is None or np.all(np.isnan(tbl)):
                        continue
                    core = self._extract_core(
                        tbl, ndim,
                        ogolem_idx, non_ogolem_slices,
                    )
                    if not np.any(np.isnan(core)):
                        anchor_years.append(yr)
                        anchor_cores.append(core)

            if not anchor_years:
                continue

            interp = self._interpolate_log_linear(
                anchor_years, anchor_cores, year_range,
            )
            year_tables: Dict[int, np.ndarray] = {}
            for yr, core_tbl in interp.items():
                core_tbl = np.maximum(core_tbl, 0.0)
                year_tables[yr] = self._assemble_with_ogolem(
                    core_tbl, ndim, full_shape,
                    ogolem_idx, non_ogolem_slices,
                )
            seeds[tid] = year_tables

        self._log(f"    Seeds: {len(seeds)} gminas")

        # ── Phase C: Layer 2 — national marginal scaling (smoothed) ──
        self._log("  Phase C: national marginal scaling (smoothed)…")

        # Snapshot observed census data before scaling
        observed_census = self._collect_observed_tables(
            seeds, source_sid, full_shape,
        )
        observed_years_per_gmina: Dict[str, set] = {}
        for tid, yr_dict in observed_census.items():
            observed_years_per_gmina[tid] = set(yr_dict.keys())

        n_scaled = self._layer2_national_scaling_smoothed(
            seeds, source_sid, year_range,
            dim_names, dim_labels,
            observed_years_per_gmina,
        )

        # Restore census data that may have been affected
        n_restored = self._restore_census_data(seeds, observed_census)
        # Fix 45c: recompute ogółem in restored census tables
        #   (source data may have stale ogółem marginals)
        for _tid in observed_census:
            if _tid not in seeds:
                continue
            for _yr in observed_census[_tid]:
                if _yr in seeds[_tid]:
                    _core = self._extract_core(
                        seeds[_tid][_yr], ndim,
                        ogolem_idx, non_ogolem_slices,
                    )
                    seeds[_tid][_yr] = self._assemble_with_ogolem(
                        _core, ndim, full_shape,
                        ogolem_idx, non_ogolem_slices,
                    )
        self._log(
            f"    Scaled {n_scaled} national-year combinations"
            f" (restored {n_restored} observed year-tables)"
        )

        # ── Phase D: store results ──
        self._log("  Phase D: storing results…")
        n_stored = 0
        for tid, year_tables in seeds.items():
            for year, tbl in year_tables.items():
                if year not in year_range:
                    continue
                is_obs = False
                rec = self.db._records.get(tid)
                if rec is not None:
                    sct = rec.cross_tables.get(source_sid)
                    if sct is not None and sct.shape == full_shape:
                        stbl = sct.tables.get(year)
                        if stbl is not None and not np.all(
                            np.isnan(stbl)
                        ):
                            is_obs = True
                self._store_estimated_cross_table(
                    tid, e_sid, year, tbl,
                    dim_names, dim_labels, is_observed=is_obs,
                )
                n_stored += 1

        voiv_tids = self._get_voivodeships()
        self._log("  Aggregating to powiat and voivodeship levels…")
        self._aggregate_to_parents(
            e_sid, year_range, dim_names, dim_labels, voiv_tids,
            source_sid=source_sid,
        )

        self._log(
            f"  Summary: {n_stored} cell-years for "
            f"{len(seeds)} gminas"
        )

    def _estimate_hh_size_2000(self, e_sid: str):
        """Household size estimation for Prediction2000 (1999–2025).

        Data landscape
        --------------
        - M_hh_size_2000: shape (6,), 1D.
          Labels: ogółem, 1-osobowe, …, 5 i więcej-osobowe
        - Census 2002 (gmina, P2871), 2011 (powiat, P3420),
          2021 (gmina, P4287).
        - No annual marginals → Layer 2 is skipped.
        - ogółem = total households (NOT total population).

        Algorithm
        ---------
        1. Disaggregate 2011 powiat data to gmina level using
           2002/2021 gmina shares (geometric mean in log-space).
        2. Collect anchors: M_hh_size_2000 (2002/2021) + synthetic 2011.
        3. Layer 1: Log-linear spline interpolation through anchors.
        4. Store results.
        5. Aggregate to powiat and voivodeship levels.
        """
        source_sid = 'M_hh_size_2000'
        year_range = PREDICTION_2000_RANGE

        # ── Step 1: dimensions ──
        dim_names, dim_labels = self._get_subject_dimensions(source_sid)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        ndim = len(dim_names)
        self._log(f"  Source: {source_sid}  shape={full_shape}")

        # ── Step 2: 2011 powiat disaggregation ──
        self._log("  Disaggregating 2011 powiat data to gmina…")
        synthetic_2011 = self._disaggregate_2011_powiat_to_gmina(
            source_sid, dim_names, dim_labels,
        )

        # ── Step 3: build seeds (observed anchors + synthetic 2011) ──
        self._log("  Layer 1: building seeds (log-linear interpolation)…")
        all_gmina_tids = self._get_all_gminas()
        seeds: Dict[str, Dict[int, np.ndarray]] = {}
        n_generated = 0

        for tid in all_gmina_tids:
            rec = self.db._records.get(tid)
            if rec is None:
                continue

            anchor_years: List[int] = []
            anchor_cores: List[np.ndarray] = []

            # Observed anchors from M_hh_size_2000
            ct = rec.cross_tables.get(source_sid)
            if ct is not None and ct.shape == full_shape:
                for yr in sorted(ct.tables.keys()):
                    tbl = ct.tables[yr]
                    if tbl is None or np.all(np.isnan(tbl)):
                        continue
                    core = self._extract_core(
                        tbl, ndim, ogolem_idx, non_ogolem_slices,
                    )
                    if not np.any(np.isnan(core)):
                        anchor_years.append(yr)
                        anchor_cores.append(core.copy())

            # Synthetic 2011 (only if no observed 2011)
            if tid in synthetic_2011 and 2011 not in anchor_years:
                syn_core = self._extract_core(
                    synthetic_2011[tid], ndim,
                    ogolem_idx, non_ogolem_slices,
                )
                if not np.any(np.isnan(syn_core)):
                    anchor_years.append(2011)
                    anchor_cores.append(syn_core.copy())

            if not anchor_years:
                continue

            interp = self._interpolate_log_linear(
                anchor_years, anchor_cores, year_range,
            )
            year_tables: Dict[int, np.ndarray] = {}
            for yr, core_tbl in interp.items():
                core_tbl = np.maximum(core_tbl, 0.0)
                year_tables[yr] = self._assemble_with_ogolem(
                    core_tbl, ndim, full_shape,
                    ogolem_idx, non_ogolem_slices,
                )
            seeds[tid] = year_tables
            n_generated += 1

        self._log(f"    Seeds: {n_generated} gminas")

        # ── Step 4: store results ──
        self._log("  Storing results…")
        n_stored = 0
        for tid, year_tables in seeds.items():
            for year, tbl in year_tables.items():
                if year not in year_range:
                    continue
                is_obs = False
                rec = self.db._records.get(tid)
                if rec is not None:
                    sct = rec.cross_tables.get(source_sid)
                    if sct is not None and sct.shape == full_shape:
                        stbl = sct.tables.get(year)
                        if stbl is not None and not np.all(
                            np.isnan(stbl)
                        ):
                            is_obs = True
                self._store_estimated_cross_table(
                    tid, e_sid, year, tbl,
                    dim_names, dim_labels, is_observed=is_obs,
                )
                n_stored += 1

        # ── Step 5: aggregate to parents ──
        voiv_tids = self._get_voivodeships()
        self._log("  Aggregating to powiat and voivodeship levels…")
        self._aggregate_to_parents(
            e_sid, year_range, dim_names, dim_labels, voiv_tids,
            source_sid=source_sid,
        )

        self._log(
            f"  Summary: {n_stored} cell-years for "
            f"{len(seeds)} gminas"
        )

    def _estimate_hh_size_1990(self, e_sid: str):
        """Household size estimation for Prediction1990 (1986–2002).

        Data landscape
        --------------
        - M_hh_size_1990: shape (5,), 1D, level 6 (gmina only).
          Labels: ogółem, 1-osobowe, 2-osobowe, 3-4-osobowe,
                  5 i więcej-osobowe
        - Census 1988 (P2887) and 2002 (P2871) at gmina level.
        - No annual marginals → Layer 2 is skipped.
        - Note: ogółem = total households, NOT total population.
          Do NOT apply population scaling.

        Algorithm
        ---------
        1. Layer 1: Log-linear interpolation between 1988 and 2002
           gmina-level anchor points.
        2. Store results.
        3. Aggregate to powiat and voivodeship levels.
        """
        source_sid = 'M_hh_size_1990'
        year_range = PREDICTION_1990_RANGE

        # ── Step 1: dimensions ──
        dim_names, dim_labels = self._get_subject_dimensions(source_sid)
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels,
        )
        self._log(f"  Source: {source_sid}  shape={full_shape}")

        # ── Step 2: collect gminas ──
        all_gmina_tids = self._get_all_gminas()
        gminas_with_data = [
            tid for tid in all_gmina_tids
            if tid in self.db._records
            and source_sid in self.db._records[tid].cross_tables
            and self.db._records[tid].cross_tables[source_sid].shape
            == full_shape
        ]
        self._log(
            f"  Gminas total: {len(all_gmina_tids)}, "
            f"with {source_sid}: {len(gminas_with_data)}"
        )

        # ── Step 3: Layer 1 — seed generation ──
        self._log("  Layer 1: generating seeds (log-linear interpolation)…")
        seeds = self._generate_seeds(
            gminas_with_data, source_sid, year_range,
            dim_names, dim_labels, exclude_ogolem=True,
        )

        # ── Step 4: store results ──
        self._log("  Storing results…")
        n_stored = 0
        for tid, year_tables in seeds.items():
            for year, tbl in year_tables.items():
                if year not in year_range:
                    continue
                is_obs = False
                rec = self.db._records.get(tid)
                if rec is not None:
                    sct = rec.cross_tables.get(source_sid)
                    if sct is not None and sct.shape == full_shape:
                        stbl = sct.tables.get(year)
                        if stbl is not None and not np.all(np.isnan(stbl)):
                            is_obs = True
                self._store_estimated_cross_table(
                    tid, e_sid, year, tbl,
                    dim_names, dim_labels, is_observed=is_obs,
                )
                n_stored += 1

        # ── Step 5: aggregate to parents ──
        voiv_tids = self._get_voivodeships()
        self._log("  Aggregating to powiat and voivodeship levels…")
        self._aggregate_to_parents(
            e_sid, year_range, dim_names, dim_labels, voiv_tids,
            source_sid=source_sid,
        )

        self._log(
            f"  Summary: {n_stored} cell-years for "
            f"{len(seeds)} gminas"
        )

    def _estimate_age_educ_2000(self, e_sid: str):
        """Age × education estimation for Prediction2000.

        DEFERRED: Requires M_pop__age_educ merged subject which
        depends on P2403, P3311, P4320 data collection (not yet in
        database).  Also requires E_age_sex_2000 and E_educ_2000 to
        be estimated first (for cross-variable marginal constraints).
        See todo.md item 28 for full specification.
        """
        raise NotImplementedError(
            "age_educ_2000: source M_pop__age_educ not yet available. "
            "Needs P2403/P3311/P4320 data collection first."
        )

    # ------------------------------------------------------------------
    # Provenance queries
    # ------------------------------------------------------------------

    def get_provenance(
        self,
        e_subject_id: str,
        teryt_id: str,
        year: int,
    ) -> Optional[np.ndarray]:
        """Return the boolean provenance mask for one (subject, unit, year).

        True = directly observed, False = estimated.
        Returns None if no provenance is recorded.
        """
        prov = self.provenance.get(e_subject_id, {}).get(teryt_id)
        if prov is None:
            return None
        return prov.is_observed(year)

    def get_provenance_summary(self, e_subject_id: str) -> pd.DataFrame:
        """Fraction of observed cells per year for an E_ subject.

        Returns a DataFrame with index=year, columns=['n_units',
        'mean_frac_observed', 'min_frac_observed'].
        """
        prov_dict = self.provenance.get(e_subject_id, {})
        if not prov_dict:
            return pd.DataFrame()

        year_range = list(PREDICTION_2000_RANGE
                          if '2000' in e_subject_id
                          else PREDICTION_1990_RANGE)
        rows = []
        for yr in year_range:
            fracs = [pm.fraction_observed(yr) for pm in prov_dict.values()]
            if fracs:
                rows.append({
                    'year': yr,
                    'n_units': len(fracs),
                    'mean_frac_observed': float(np.mean(fracs)),
                    'min_frac_observed': float(np.min(fracs)),
                })
        return pd.DataFrame(rows).set_index('year')

    # ------------------------------------------------------------------
    # Diagnostics — work chunk E (v5.4)
    # ------------------------------------------------------------------

    def validate_results(self, e_subject_id: str) -> pd.DataFrame:
        """Run full consistency diagnostics for an E_ subject.

        Checks every (teryt_id, year) with E_ data for:
          1. Non-negativity — all cells >= 0
          2. Marginal consistency — ogółem row/column = sum of non-ogółem
          3. Hierarchical consistency — Σ children (rodz 1,2,3) ≈ parent
          4. Total population match — grand total ≈ record.pop
          5. Temporal smoothness — no abrupt year-over-year jumps (>20%)
          6. Sub-division consistency — rodz-3 table ≈ rodz-4 + rodz-5

        Returns
        -------
        pd.DataFrame
            Diagnostic report with columns:
            ['teryt_id', 'name', 'year', 'check', 'status', 'detail']
        """
        from geoTERYT_db import LEVEL_GMINA, LEVEL_POWIAT, LEVEL_VOIVODESHIP

        year_range = (PREDICTION_2000_RANGE if '2000' in e_subject_id
                      else PREDICTION_1990_RANGE)
        issues: list = []

        def _add(tid, name, yr, check, status, detail):
            issues.append({'teryt_id': tid, 'name': name, 'year': yr,
                           'check': check, 'status': status, 'detail': detail})

        # Collect all records that have E_ data
        e_records = {}
        for tid, rec in self.db._records.items():
            ct = rec.cross_tables.get(e_subject_id)
            if ct is not None and ct.years_with_data:
                e_records[tid] = rec

        if not e_records:
            self._log(f"  No records with {e_subject_id} data — nothing to validate")
            return pd.DataFrame(columns=['teryt_id', 'name', 'year',
                                         'check', 'status', 'detail'])

        # Get dimensions from first record
        sample_ct = next(iter(e_records.values())).cross_tables[e_subject_id]
        dim_names = sample_ct.dim_names
        dim_labels = sample_ct.dim_labels
        ndim = len(dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels
        )

        self._log(f"  Validating {e_subject_id}: {len(e_records)} records, "
                  f"shape={sample_ct.shape}")

        # ── 1. Non-negativity ──
        n_neg = 0
        for tid, rec in e_records.items():
            ct = rec.cross_tables[e_subject_id]
            for yr in year_range:
                tbl = ct.tables.get(yr)
                if tbl is None or np.all(np.isnan(tbl)):
                    continue
                neg_cells = np.sum(tbl[~np.isnan(tbl)] < -EPSILON)
                if neg_cells > 0:
                    _add(tid, rec.name, yr, 'non_negativity', 'FAIL',
                         f'{neg_cells} negative cells, min={np.nanmin(tbl):.4f}')
                    n_neg += 1
        self._log(f"    [1] Non-negativity: {n_neg} failures")

        # ── 2. Marginal consistency (ogółem = sum of sub-categories) ──
        n_marg_fail = 0
        MARG_TOL = 1.0
        for tid, rec in e_records.items():
            ct = rec.cross_tables[e_subject_id]
            for yr in year_range:
                tbl = ct.tables.get(yr)
                if tbl is None or np.all(np.isnan(tbl)):
                    continue
                # Check each dimension's ogółem
                for di in range(ndim):
                    og_i = ogolem_idx[di]
                    if og_i is None:
                        continue
                    noi = non_ogolem_slices[di]
                    if ndim == 1:
                        og_val = tbl[og_i]
                        sub_sum = np.nansum(tbl[noi])
                    elif ndim == 2:
                        if di == 0:
                            # ogółem row = sum of other rows, per column
                            og_row = tbl[og_i, :]
                            sub_rows = tbl[noi, :]
                            sub_sum = np.nansum(sub_rows, axis=0)
                            diff_arr = og_row - sub_sum
                            if np.all(np.isnan(diff_arr)):
                                continue
                            max_err = np.nanmax(np.abs(diff_arr))
                        else:
                            # ogółem column = sum of other columns, per row
                            og_col = tbl[:, og_i]
                            sub_cols = tbl[:, noi]
                            sub_sum = np.nansum(sub_cols, axis=1)
                            diff_arr = og_col - sub_sum
                            if np.all(np.isnan(diff_arr)):
                                continue
                            max_err = np.nanmax(np.abs(diff_arr))
                        if max_err > MARG_TOL:
                            _add(tid, rec.name, yr, 'marginal_consistency',
                                 'FAIL',
                                 f'dim={dim_names[di]} max_err={max_err:.4f}')
                            n_marg_fail += 1
                        continue
                    else:
                        continue  # Skip 3D+ for now
                    # 1D check
                    if abs(og_val - sub_sum) > MARG_TOL:
                        _add(tid, rec.name, yr, 'marginal_consistency',
                             'FAIL',
                             f'dim={dim_names[di]} og={og_val:.1f} '
                             f'sum={sub_sum:.1f} diff={og_val-sub_sum:.4f}')
                        n_marg_fail += 1
        self._log(f"    [2] Marginal consistency: {n_marg_fail} failures "
                  f"(tol={MARG_TOL})")

        # ── 3. Hierarchical consistency ──
        n_hier_fail = 0
        n_hier_checked = 0
        HIER_TOL_PCT = 0.5  # 0.5% tolerance

        # Check powiat = Σ children(rodz 1,2,3)
        powiat_tids = [tid for tid in e_records
                       if self.db._records[tid].level == LEVEL_POWIAT]
        for ptid in powiat_tids:
            prec = self.db._records[ptid]
            pct = prec.cross_tables.get(e_subject_id)
            if pct is None:
                continue
            for yr in year_range:
                ptbl = pct.tables.get(yr)
                if ptbl is None or np.all(np.isnan(ptbl)):
                    continue
                # Sum children
                children = _get_aggregation_children(prec, self.db, yr)
                child_sum = np.zeros_like(ptbl)
                n_children_with_data = 0
                for chtid in children:
                    chrec = self.db._records.get(chtid)
                    if chrec is None:
                        continue
                    chct = chrec.cross_tables.get(e_subject_id)
                    if chct is None:
                        continue
                    chtbl = chct.tables.get(yr)
                    if chtbl is None or np.all(np.isnan(chtbl)):
                        continue
                    if chtbl.shape != ptbl.shape:
                        continue
                    child_sum += np.nan_to_num(chtbl, nan=0.0)
                    n_children_with_data += 1

                if n_children_with_data == 0:
                    continue
                n_hier_checked += 1
                max_cell_diff = float(np.nanmax(
                    np.abs(child_sum - np.nan_to_num(ptbl, nan=0.0))
                ))
                ptotal = float(np.nansum(np.abs(ptbl)))
                pct_err = (100 * max_cell_diff / ptotal
                           if ptotal > EPSILON else 0.0)
                if pct_err > HIER_TOL_PCT:
                    _add(ptid, prec.name, yr, 'hierarchical_consistency',
                         'FAIL',
                         f'max_cell_diff={max_cell_diff:.2f} '
                         f'({pct_err:.3f}% of total)')
                    n_hier_fail += 1

        self._log(f"    [3] Hierarchical consistency: {n_hier_fail} failures "
                  f"/ {n_hier_checked} checked")

        # ── 4. Total population match ──
        #  Only meaningful for age_sex subjects where grand total = total pop.
        #  Education subjects measure pop 15+, hh_size measures households.
        is_age_sex = 'age_sex' in e_subject_id and 'educ' not in e_subject_id
        n_pop_fail = 0
        n_pop_checked = 0
        POP_TOL_PCT = 0.1  # 0.1% tolerance
        if not is_age_sex:
            self._log(f"    [4] Population match: skipped "
                      f"(not applicable for {e_subject_id})")
        for tid, rec in (e_records.items() if is_age_sex else []):
            if rec.level not in (LEVEL_GMINA,):
                continue
            if tid[-1] not in RODZ_AGGREGATION_SET:
                continue
            ct = rec.cross_tables[e_subject_id]
            for yr in year_range:
                tbl = ct.tables.get(yr)
                if tbl is None or np.all(np.isnan(tbl)):
                    continue
                ts = pd.Timestamp(yr, 1, 1)
                pop_val = rec.pop.get(ts, np.nan)
                if pd.isna(pop_val) or pop_val <= 0:
                    continue
                # Grand total from table
                if ndim == 1:
                    og0 = ogolem_idx.get(0)
                    if og0 is not None:
                        grand_total = float(tbl[og0])
                    else:
                        grand_total = float(np.nansum(tbl))
                elif ndim == 2:
                    og0 = ogolem_idx.get(0)
                    og1 = ogolem_idx.get(1)
                    if og0 is not None and og1 is not None:
                        grand_total = float(tbl[og0, og1])
                    elif og0 is not None:
                        grand_total = float(np.nansum(tbl[og0, :]))
                    elif og1 is not None:
                        grand_total = float(np.nansum(tbl[:, og1]))
                    else:
                        grand_total = float(np.nansum(tbl))
                else:
                    grand_total = float(np.nansum(tbl))

                n_pop_checked += 1
                pct_err = (100 * abs(grand_total - pop_val) / pop_val
                           if pop_val > EPSILON else 0.0)
                if pct_err > POP_TOL_PCT:
                    _add(tid, rec.name, yr, 'population_match', 'FAIL',
                         f'E_total={grand_total:.0f} pop={pop_val:.0f} '
                         f'err={pct_err:.3f}%')
                    n_pop_fail += 1

        self._log(f"    [4] Population match: {n_pop_fail} failures "
                  f"/ {n_pop_checked} checked (tol={POP_TOL_PCT}%)")

        # ── 5. Temporal smoothness ──
        n_smooth_flag = 0
        JUMP_THRESHOLD = 0.20  # 20% relative change
        gmina_tids = [tid for tid in e_records
                      if e_records[tid].level == LEVEL_GMINA
                      and tid[-1] in RODZ_AGGREGATION_SET]

        for tid in gmina_tids:
            ct = e_records[tid].cross_tables[e_subject_id]
            yrs_sorted = sorted(yr for yr in year_range
                                if yr in ct.tables
                                and ct.tables[yr] is not None
                                and not np.all(np.isnan(ct.tables[yr])))
            if len(yrs_sorted) < 2:
                continue
            # Mean table across years for relative reference
            all_tbls = [ct.tables[yr] for yr in yrs_sorted]
            stacked = np.stack(all_tbls, axis=0)
            mean_tbl = np.nanmean(stacked, axis=0)

            for i in range(1, len(yrs_sorted)):
                yr0, yr1 = yrs_sorted[i - 1], yrs_sorted[i]
                if yr1 - yr0 > 1:
                    continue  # Only check consecutive years
                tbl0 = ct.tables[yr0]
                tbl1 = ct.tables[yr1]
                # Mask out ogółem cells for checking
                if ndim == 1:
                    noi = non_ogolem_slices[0]
                    core0 = tbl0[noi]
                    core1 = tbl1[noi]
                    core_mean = mean_tbl[noi]
                elif ndim == 2:
                    noi0 = non_ogolem_slices[0]
                    noi1 = non_ogolem_slices[1]
                    core0 = tbl0[np.ix_(noi0, noi1)]
                    core1 = tbl1[np.ix_(noi0, noi1)]
                    core_mean = mean_tbl[np.ix_(noi0, noi1)]
                else:
                    continue

                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_change = np.abs(core1 - core0) / (core_mean + EPSILON)
                max_rel = float(np.nanmax(rel_change))
                if max_rel > JUMP_THRESHOLD:
                    _add(tid, e_records[tid].name, yr1,
                         'temporal_smoothness', 'WARN',
                         f'max_rel_change={max_rel:.3f} '
                         f'({yr0}→{yr1})')
                    n_smooth_flag += 1

        self._log(f"    [5] Temporal smoothness: {n_smooth_flag} warnings "
                  f"(threshold={JUMP_THRESHOLD*100:.0f}%)")

        # ── 6. Sub-division consistency ──
        # For rodz-3 gminas: table ≈ rodz-4 + rodz-5
        n_subdiv_fail = 0
        n_subdiv_checked = 0
        SUBDIV_TOL = 1.0
        for tid, rec in e_records.items():
            if rec.level != LEVEL_GMINA or rec.rodz != '3':
                continue
            # Find rodz-4 and rodz-5 siblings
            base = tid[:-1]
            r4_tid = base + '4'
            r5_tid = base + '5'
            r4_rec = self.db._records.get(r4_tid)
            r5_rec = self.db._records.get(r5_tid)
            if r4_rec is None and r5_rec is None:
                continue
            ct3 = rec.cross_tables.get(e_subject_id)
            if ct3 is None:
                continue
            for yr in year_range:
                tbl3 = ct3.tables.get(yr)
                if tbl3 is None or np.all(np.isnan(tbl3)):
                    continue
                # Sum rodz-4 + rodz-5
                sub_sum = np.zeros_like(tbl3)
                has_sub = False
                for sr_tid, sr_rec in [(r4_tid, r4_rec), (r5_tid, r5_rec)]:
                    if sr_rec is None:
                        continue
                    sr_ct = sr_rec.cross_tables.get(e_subject_id)
                    if sr_ct is None:
                        continue
                    sr_tbl = sr_ct.tables.get(yr)
                    if sr_tbl is None or np.all(np.isnan(sr_tbl)):
                        continue
                    if sr_tbl.shape != tbl3.shape:
                        continue
                    sub_sum += np.nan_to_num(sr_tbl, nan=0.0)
                    has_sub = True
                if not has_sub:
                    continue
                n_subdiv_checked += 1
                max_diff = float(np.nanmax(
                    np.abs(np.nan_to_num(tbl3, nan=0.0) - sub_sum)
                ))
                if max_diff > SUBDIV_TOL:
                    _add(tid, rec.name, yr, 'subdivision_consistency',
                         'FAIL', f'max_diff={max_diff:.2f}')
                    n_subdiv_fail += 1

        self._log(f"    [6] Sub-division consistency: {n_subdiv_fail} failures "
                  f"/ {n_subdiv_checked} checked")

        # ── 7. Education ↔ age×sex population coherence ──
        # Education data covers pop 13+ (1990) or 15+ (2000).
        # Check: E_educ total ≈ E_age_sex total − (0-4)+(5-9)+(10-14).
        is_educ = 'educ' in e_subject_id and 'sex' not in e_subject_id
        n_educ_pop_fail = 0
        n_educ_pop_checked = 0
        EDUC_POP_TOL_PCT = 5.0  # 5% tolerance (different age boundaries)
        YOUNG_AGE_LABELS = {'0-4', '5-9', '10-14'}

        if is_educ:
            # Determine which E_age_sex subject to compare against
            section = '2000' if '2000' in e_subject_id else '1990'
            age_sex_sid = f'E_age_sex_{section}'

            for tid, rec in e_records.items():
                if rec.level != 6:   # gmina only
                    continue
                e_ct = rec.cross_tables.get(e_subject_id)
                a_ct = rec.cross_tables.get(age_sex_sid)
                if e_ct is None or a_ct is None:
                    continue
                # Identify young-age indices in age×sex
                if hasattr(a_ct, 'dim_names') and a_ct.dim_names:
                    age_dim = a_ct.dim_names[0]
                    age_labels = a_ct.dim_labels.get(age_dim, [])
                    young_idx = [
                        i for i, lbl in enumerate(age_labels)
                        if lbl in YOUNG_AGE_LABELS
                    ]
                else:
                    young_idx = []

                for yr in year_range:
                    e_tbl = e_ct.tables.get(yr)
                    a_tbl = a_ct.tables.get(yr)
                    if e_tbl is None or a_tbl is None:
                        continue
                    if np.all(np.isnan(e_tbl)) or np.all(np.isnan(a_tbl)):
                        continue

                    educ_total = float(np.nansum(e_tbl))
                    age_sex_total = float(np.nansum(a_tbl))
                    # Subtract young age groups
                    young_pop = 0.0
                    for yi in young_idx:
                        young_pop += float(np.nansum(a_tbl[yi]))
                    eligible_pop = age_sex_total - young_pop

                    if eligible_pop <= 0:
                        continue
                    n_educ_pop_checked += 1
                    pct_err = 100 * abs(educ_total - eligible_pop) / eligible_pop
                    if pct_err > EDUC_POP_TOL_PCT:
                        _add(tid, rec.name, yr, 'educ_age_coherence',
                             'WARN',
                             f'educ_total={educ_total:.0f} '
                             f'eligible_pop={eligible_pop:.0f} '
                             f'err={pct_err:.1f}%')
                        n_educ_pop_fail += 1

            self._log(
                f"    [7] Educ↔age_sex coherence: {n_educ_pop_fail} warnings "
                f"/ {n_educ_pop_checked} checked (tol={EDUC_POP_TOL_PCT}%)"
            )
        else:
            self._log(f"    [7] Educ↔age_sex coherence: skipped "
                      f"(not applicable for {e_subject_id})")

        df = pd.DataFrame(issues)
        if df.empty:
            df = pd.DataFrame(columns=['teryt_id', 'name', 'year',
                                       'check', 'status', 'detail'])
        self._log(f"  Validation complete: {len(df)} issues found")
        return df

    # ------------------------------------------------------------------
    # Leave-one-out cross-validation (work chunk E, item 29)
    # ------------------------------------------------------------------

    def leave_one_out_cv(
        self,
        variable_type: str,
        prediction_section: str,
        holdout_year: int,
    ) -> pd.DataFrame:
        """Leave-one-out cross-validation for a specific pipeline.

        Re-runs the estimation pipeline for the given (variable_type,
        prediction_section) pair, but with the *holdout_year* census data
        masked out.  Then compares the predicted table at *holdout_year*
        with the actual observed data.

        Parameters
        ----------
        variable_type : str
            e.g. 'age_sex', 'educ', 'hh_size'
        prediction_section : str
            '2000' or '1990'
        holdout_year : int
            Census year to exclude (must be in CENSUS_YEARS and within
            the prediction range).

        Returns
        -------
        pd.DataFrame
            Per-gmina comparison with columns:
            ['teryt_id', 'name', 'cell_rmse', 'cell_rmse_pct',
             'chi_sq', 'marginal_err', 'total_pop_err_pct']
        """
        from geoTERYT_db import LEVEL_GMINA
        import copy

        key = (variable_type, prediction_section)
        e_sid = E_SUBJECT_NAMES.get(key)
        if e_sid is None:
            raise ValueError(f"Unknown pipeline key: {key}")

        year_range = (PREDICTION_2000_RANGE if prediction_section == '2000'
                      else PREDICTION_1990_RANGE)
        if holdout_year not in year_range:
            raise ValueError(
                f"holdout_year={holdout_year} not in {prediction_section} "
                f"range")

        # Identify the M_ source subject from the first anchor
        anchor_cfg = ANCHOR_SUBJECTS.get(variable_type, {}).get(
            prediction_section, {}
        )
        anchor_sids = anchor_cfg.get('anchor_subjects', [])
        if not anchor_sids:
            raise ValueError(f"No anchor subjects for {key}")
        source_sid = anchor_sids[0]

        # Get dimensions
        dim_names, dim_labels = self._get_subject_dimensions(source_sid)
        ndim = len(dim_names)
        ogolem_idx, non_ogolem_slices = self._identify_ogolem(
            dim_names, dim_labels
        )
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)

        self._log(f"\n  LOOCV: {e_sid}, holdout={holdout_year}")
        self._log(f"    source={source_sid}, shape={full_shape}")

        # ── Step 1: Collect observed tables at holdout year ──
        observed: Dict[str, np.ndarray] = {}
        for tid, rec in self.db._records.items():
            if rec.level != LEVEL_GMINA or tid[-1] not in RODZ_AGGREGATION_SET:
                continue
            ct = rec.cross_tables.get(source_sid)
            if ct is None:
                continue
            tbl = ct.tables.get(holdout_year)
            if tbl is None or np.all(np.isnan(tbl)):
                continue
            if tbl.shape != full_shape:
                continue
            observed[tid] = tbl.copy()

        self._log(f"    Observed gminas at {holdout_year}: {len(observed)}")
        if len(observed) == 0:
            self._log("    No observed data at holdout year — cannot evaluate")
            return pd.DataFrame()

        # ── Step 2: Temporarily mask holdout year in source ──
        backup: Dict[str, np.ndarray] = {}
        for tid in observed:
            rec = self.db._records[tid]
            ct = rec.cross_tables.get(source_sid)
            if ct is not None and holdout_year in ct.tables:
                backup[tid] = ct.tables[holdout_year].copy()
                ct.tables[holdout_year] = np.full(full_shape, np.nan)

        # Also clear any existing E_ results for this pipeline
        e_backup: Dict[str, Dict[int, np.ndarray]] = {}
        for tid, rec in self.db._records.items():
            ect = rec.cross_tables.get(e_sid)
            if ect is not None:
                e_backup[tid] = {yr: ect.tables[yr].copy()
                                 for yr in ect.tables if yr in year_range}

        # Clear _completed so pipeline can re-run
        self._completed.discard(key)

        # ── Step 3: Re-run pipeline ──
        prev_verbose = self.verbose
        self.verbose = False
        try:
            self.run_pipeline(variable_type, prediction_section)
        except Exception as exc:
            self._log(f"    ⚠ Pipeline failed during LOOCV: {exc}")
        finally:
            self.verbose = prev_verbose

        # ── Step 4: Collect predicted tables ──
        predicted: Dict[str, np.ndarray] = {}
        for tid in observed:
            rec = self.db._records.get(tid)
            if rec is None:
                continue
            ect = rec.cross_tables.get(e_sid)
            if ect is None:
                continue
            ptbl = ect.tables.get(holdout_year)
            if ptbl is not None and not np.all(np.isnan(ptbl)):
                predicted[tid] = ptbl.copy()

        self._log(f"    Predicted gminas at holdout: {len(predicted)}")

        # ── Step 5: Restore original data ──
        for tid, orig_tbl in backup.items():
            rec = self.db._records[tid]
            ct = rec.cross_tables.get(source_sid)
            if ct is not None:
                ct.tables[holdout_year] = orig_tbl

        for tid, yr_dict in e_backup.items():
            rec = self.db._records.get(tid)
            if rec is None:
                continue
            ect = rec.cross_tables.get(e_sid)
            if ect is not None:
                for yr, arr in yr_dict.items():
                    ect.tables[yr] = arr

        self._completed.add(key)  # Mark as completed again

        # ── Step 6: Compute metrics ──
        rows = []
        for tid in sorted(observed):
            if tid not in predicted:
                continue
            obs = observed[tid]
            pred = predicted[tid]
            rec = self.db._records[tid]

            # Extract core (non-ogółem) cells
            if ndim == 1:
                noi = non_ogolem_slices[0]
                obs_core = obs[noi]
                pred_core = pred[noi]
            elif ndim == 2:
                noi0 = non_ogolem_slices[0]
                noi1 = non_ogolem_slices[1]
                obs_core = obs[np.ix_(noi0, noi1)]
                pred_core = pred[np.ix_(noi0, noi1)]
            else:
                obs_core = obs
                pred_core = pred

            # RMSE
            diff = pred_core - obs_core
            valid = ~np.isnan(diff)
            if valid.sum() == 0:
                continue
            cell_rmse = float(np.sqrt(np.mean(diff[valid] ** 2)))
            obs_mean = float(np.mean(np.abs(obs_core[valid])))
            cell_rmse_pct = (100 * cell_rmse / obs_mean
                             if obs_mean > EPSILON else 0.0)

            # Chi-squared distance
            with np.errstate(divide='ignore', invalid='ignore'):
                chi_cells = np.where(
                    obs_core > EPSILON,
                    (pred_core - obs_core) ** 2 / obs_core,
                    0.0
                )
            chi_sq = float(np.nansum(chi_cells))

            # Marginal error (row/col sums)
            if ndim == 2:
                obs_row = np.nansum(obs_core, axis=1)
                pred_row = np.nansum(pred_core, axis=1)
                marg_err = float(np.max(np.abs(obs_row - pred_row)))
            elif ndim == 1:
                marg_err = float(np.abs(np.nansum(obs_core)
                                        - np.nansum(pred_core)))
            else:
                marg_err = 0.0

            # Total pop error
            obs_total = float(np.nansum(obs_core))
            pred_total = float(np.nansum(pred_core))
            pop_err_pct = (100 * abs(pred_total - obs_total) / obs_total
                           if obs_total > EPSILON else 0.0)

            rows.append({
                'teryt_id': tid,
                'name': rec.name,
                'cell_rmse': cell_rmse,
                'cell_rmse_pct': cell_rmse_pct,
                'chi_sq': chi_sq,
                'marginal_err': marg_err,
                'total_pop_err_pct': pop_err_pct,
            })

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            self._log(
                f"    Evaluated: {len(result_df)} gminas\n"
                f"    Mean RMSE: {result_df['cell_rmse'].mean():.2f}\n"
                f"    Mean RMSE%: {result_df['cell_rmse_pct'].mean():.2f}%\n"
                f"    Mean χ²: {result_df['chi_sq'].mean():.2f}\n"
                f"    Mean pop err%: "
                f"{result_df['total_pop_err_pct'].mean():.3f}%"
            )
        return result_df

    # ------------------------------------------------------------------
    # Estimation confidence scoring (work chunk E, item 31)
    # ------------------------------------------------------------------

    def compute_confidence_scores(
        self,
        e_subject_id: str,
    ) -> pd.DataFrame:
        """Compute per-territory estimation confidence scores.

        For each territorial unit with E_ data, computes a confidence
        score (0–100) based on:
          - Number of census anchors available (0–4) → weight 30%
          - Number of years with direct observed data → weight 25%
          - Distance to nearest anchor year → weight 20%
          - Population size (log-scaled) → weight 15%
          - Whether data came from current TERYT or historical code → weight 10%

        Returns
        -------
        pd.DataFrame
            Columns: ['teryt_id', 'name', 'level', 'year',
                       'confidence', 'n_anchors', 'n_observed_years',
                       'dist_nearest_anchor', 'log_pop', 'used_historical']
        """
        from geoTERYT_db import LEVEL_GMINA

        year_range = (PREDICTION_2000_RANGE if '2000' in e_subject_id
                      else PREDICTION_1990_RANGE)

        # Determine anchor subjects
        key = None
        for k, sid in E_SUBJECT_NAMES.items():
            if sid == e_subject_id:
                key = k
                break
        if key is None:
            raise ValueError(f"Unknown E_ subject: {e_subject_id}")

        anchor_cfg = ANCHOR_SUBJECTS.get(key[0], {}).get(key[1], {})
        anchor_sids = anchor_cfg.get('anchor_subjects', [])
        source_sid = anchor_sids[0] if anchor_sids else None

        rows = []
        for tid, rec in self.db._records.items():
            if rec.level != LEVEL_GMINA or tid[-1] not in RODZ_AGGREGATION_SET:
                continue
            ect = rec.cross_tables.get(e_subject_id)
            if ect is None or not ect.years_with_data:
                continue

            # Count anchor years with data in source
            n_anchors = 0
            anchor_years_present = []
            if source_sid:
                sct = rec.cross_tables.get(source_sid)
                if sct is not None:
                    for cy in CENSUS_YEARS:
                        if cy in year_range:
                            tbl = sct.tables.get(cy)
                            if tbl is not None and not np.all(np.isnan(tbl)):
                                n_anchors += 1
                                anchor_years_present.append(cy)

            # Count years with observed source data in year_range
            n_observed_years = 0
            if source_sid:
                sct = rec.cross_tables.get(source_sid)
                if sct is not None:
                    for yr in year_range:
                        tbl = sct.tables.get(yr)
                        if tbl is not None and not np.all(np.isnan(tbl)):
                            n_observed_years += 1

            # Historical code usage
            used_historical = len(rec.historical_codes) > 1

            # Mean population (log-scaled)
            pop_vals = []
            for yr in year_range:
                ts = pd.Timestamp(yr, 1, 1)
                pv = rec.pop.get(ts, np.nan)
                if not pd.isna(pv) and pv > 0:
                    pop_vals.append(pv)
            log_pop = np.log10(np.mean(pop_vals)) if pop_vals else 0.0

            for yr in year_range:
                tbl = ect.tables.get(yr)
                if tbl is None or np.all(np.isnan(tbl)):
                    continue

                # Distance to nearest anchor
                if anchor_years_present:
                    dist = min(abs(yr - ay) for ay in anchor_years_present)
                else:
                    dist = max(abs(yr - cy) for cy in CENSUS_YEARS
                               if cy in year_range) if any(
                        cy in year_range for cy in CENSUS_YEARS
                    ) else 20

                # ── Score components (each 0–1, weighted) ──
                # 1. Anchors: 0 → 0.0, 1 → 0.33, 2 → 0.67, 3+ → 1.0
                s_anchor = min(n_anchors / 3.0, 1.0)

                # 2. Observed years fraction
                total_years = len(list(year_range))
                s_observed = min(n_observed_years / total_years, 1.0)

                # 3. Distance to anchor (closer = better)
                # 0 → 1.0, 5 → 0.5, 10 → 0.25, 15+ → ~0
                s_dist = 1.0 / (1.0 + dist / 5.0)

                # 4. Population (log10, typical range 2.5-6)
                s_pop = min(max((log_pop - 2.0) / 4.0, 0.0), 1.0)

                # 5. Historical code (1.0 if no historical, 0.5 if has)
                s_hist = 0.5 if used_historical else 1.0

                confidence = 100.0 * (
                    0.30 * s_anchor
                    + 0.25 * s_observed
                    + 0.20 * s_dist
                    + 0.15 * s_pop
                    + 0.10 * s_hist
                )

                rows.append({
                    'teryt_id': tid,
                    'name': rec.name,
                    'level': rec.level,
                    'year': yr,
                    'confidence': round(confidence, 1),
                    'n_anchors': n_anchors,
                    'n_observed_years': n_observed_years,
                    'dist_nearest_anchor': dist,
                    'log_pop': round(log_pop, 2),
                    'used_historical': used_historical,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            self._log(
                f"  Confidence scores for {e_subject_id}: "
                f"{df['teryt_id'].nunique()} gminas, "
                f"{len(df)} gmina-years\n"
                f"    Mean confidence: {df['confidence'].mean():.1f}\n"
                f"    Median: {df['confidence'].median():.1f}\n"
                f"    Min: {df['confidence'].min():.1f}, "
                f"Max: {df['confidence'].max():.1f}"
            )
        return df

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_completed = len(self._completed)
        n_total = len(E_SUBJECT_NAMES)
        return (f"DemographicEstimator("
                f"completed={n_completed}/{n_total}, "
                f"Gurobi={'YES' if GUROBI_AVAILABLE else 'NO'})")
