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
from scipy.interpolate import CubicSpline

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
        Years before 1999 automatically fall back to 1999.

    Rules
    -----
    - Powiat  (teryt[-1] == '0'): children with rodz ∈ {1, 2, 3},
      with encompassing-child deduplication (Warsaw 1999–2001).
    - Voivodeship (teryt[2:] == '00000'): all powiats + all direct
      gminas with rodz ∈ {1, 2, 3}
    - Country (teryt == '0000000'): all voivodeships
    - NEVER include rodz 4, 5, 8, 9 in aggregation sums.
    """
    from geoTERYT_db import filter_aggregation_children
    tid = record.teryt_id

    # Country level
    if tid == '0000000':
        return record.get_children(year)

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
        raw = [
            child_tid for child_tid in record.get_children(year)
            if child_tid[-1] in RODZ_AGGREGATION_SET
        ]
        return filter_aggregation_children(raw, year, db._records)

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
    # Public entry points
    # ------------------------------------------------------------------

    def run_all(self):
        """Run every estimation pipeline in the correct dependency order.

        Order: age×sex → educ → educ_sex → hh_size → age_educ
        Within each: Prediction2000 first, then Prediction1990.
        """
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

            elif len(anchor_years) == 2:
                # Linear interpolation in log-space (geometric interpolation)
                y1, y2 = anchor_years
                lt1, lt2 = log_tables
                for yr in year_range:
                    if yr <= y1:
                        seed_tables[yr] = anchor_tables[0].copy()
                    elif yr >= y2:
                        seed_tables[yr] = anchor_tables[-1].copy()
                    else:
                        frac = (yr - y1) / (y2 - y1)
                        log_interp = lt1 * (1 - frac) + lt2 * frac
                        seed_tables[yr] = np.exp(log_interp)

            else:
                # ≥3 anchors: natural cubic spline per cell
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

    def _aggregate_to_parents(
        self,
        e_sid: str,
        year_range: range,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
        voiv_tids: List[str],
    ):
        """Aggregate gmina-level E_ tables up to powiat and voivodeship.

        For each (year, voivodeship):
          powiat_table  = Σ gmina_table  for gminas with rodz ∈ {1,2,3}
          voiv_table    = Σ powiat_table + Σ direct-gmina_table
        """
        full_shape = tuple(len(dim_labels[d]) for d in dim_names)
        n_pow = 0
        n_voiv = 0

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
                            self._store_estimated_cross_table(
                                child_tid, e_sid, year, powiat_total,
                                dim_names, dim_labels, is_observed=False,
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
                    self._store_estimated_cross_table(
                        voiv_tid, e_sid, year, voiv_total,
                        dim_names, dim_labels, is_observed=False,
                    )
                    n_voiv += 1

        self._log(
            f"    Aggregated: {n_pow} powiat-years, {n_voiv} voiv-years"
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

        elif len(anchor_years) == 2:
            y1, y2 = anchor_years
            lt1, lt2 = log_tables
            for yr in year_range:
                if yr <= y1:
                    result[yr] = anchor_tables[0].copy()
                elif yr >= y2:
                    result[yr] = anchor_tables[-1].copy()
                else:
                    f = (yr - y1) / (y2 - y1)
                    result[yr] = np.exp(lt1 * (1 - f) + lt2 * f)
        else:
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
                        observed[tid] = tbl
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
        self._log("  Phase C: old voivodeship marginal scaling (1986–1994)…")
        # Get old voivodeship teryt_ids
        country_rec = self.db._records.get('0000000')
        old_voi_tids = (
            country_rec.children_ids.get('old', [])
            if country_rec is not None else []
        )

        n_scaled = 0
        for year in range(1986, 1995):
            for ov_tid in old_voi_tids:
                ov_rec = self.db._records.get(ov_tid)
                if ov_rec is None:
                    continue
                ov_ct = ov_rec.cross_tables.get(source_sid)
                if ov_ct is None or ov_ct.shape != full_shape:
                    continue
                ov_tbl = ov_ct.tables.get(year)
                if ov_tbl is None or np.all(np.isnan(ov_tbl)):
                    continue

                # Collect gminas under this old voi
                children = ov_rec.get_children(year)
                gminas_in_ov = [
                    g for g in children
                    if g in seeds and year in seeds[g]
                ]
                if not gminas_in_ov:
                    continue

                # Scale gmina tables to match old voi total
                agg = np.zeros(full_shape, dtype=float)
                for g in gminas_in_ov:
                    agg += np.nan_to_num(seeds[g][year], nan=0.0)

                with np.errstate(divide='ignore', invalid='ignore'):
                    factors = np.where(
                        agg > EPSILON, ov_tbl / agg, 1.0,
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

        self._log(f"  Summary: {n_stored} cell-years for {len(seeds)} gminas")

    def _estimate_educ_2000(self, e_sid: str):
        """Education estimation for Prediction2000 (1999–2024).
        Will be implemented in work chunk D (v5.3), item 23."""
        raise NotImplementedError("See todo.md item 23.")

    def _estimate_educ_1990(self, e_sid: str):
        """Education estimation for Prediction1990 (1986–2002).
        Will be implemented in work chunk D (v5.3), item 24."""
        raise NotImplementedError("See todo.md item 24.")

    def _estimate_educ_sex_2000(self, e_sid: str):
        """Education × sex estimation for Prediction2000.
        Will be implemented in work chunk D (v5.3), item 25."""
        raise NotImplementedError("See todo.md item 25.")

    def _estimate_educ_sex_1990(self, e_sid: str):
        """Education × sex estimation for Prediction1990.
        Will be implemented in work chunk D (v5.3), item 25."""
        raise NotImplementedError("See todo.md item 25.")

    def _estimate_hh_size_2000(self, e_sid: str):
        """Household size estimation for Prediction2000.
        Will be implemented in work chunk D (v5.3), item 26."""
        raise NotImplementedError("See todo.md item 26.")

    def _estimate_hh_size_1990(self, e_sid: str):
        """Household size estimation for Prediction1990.
        Will be implemented in work chunk D (v5.3), item 27."""
        raise NotImplementedError("See todo.md item 27.")

    def _estimate_age_educ_2000(self, e_sid: str):
        """Age × education estimation for Prediction2000.
        Will be implemented in work chunk D (v5.3), item 28."""
        raise NotImplementedError("See todo.md item 28.")

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
    # Diagnostics — stub for work chunk E
    # ------------------------------------------------------------------

    def validate_results(self, e_subject_id: str) -> pd.DataFrame:
        """Run full consistency diagnostics for an E_ subject.
        Will be implemented in work chunk E (v5.4), item 30."""
        raise NotImplementedError("See todo.md item 30.")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_completed = len(self._completed)
        n_total = len(E_SUBJECT_NAMES)
        return (f"DemographicEstimator("
                f"completed={n_completed}/{n_total}, "
                f"Gurobi={'YES' if GUROBI_AVAILABLE else 'NO'})")
