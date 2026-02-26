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
            'anchor_subjects': ['M_age_sex', 'M_pop__age_sex'],
            'marginal_subjects': ['M_pop__age_sex'],
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
) -> List[str]:
    """Return the child teryt_ids whose data should be summed to produce
    the parent record's total.

    Rules
    -----
    - Powiat  (teryt[-1] == '0'): children with rodz ∈ {1, 2, 3}
    - Voivodeship (teryt[2:] == '00000'): all powiats + all direct
      gminas with rodz ∈ {1, 2, 3}
    - Country (teryt == '0000000'): all voivodeships
    - NEVER include rodz 4, 5, 8, 9 in aggregation sums.
    """
    tid = record.teryt_id

    # Country level
    if tid == '0000000':
        # All new voivodeships (02–32 even) — these cover the entire territory
        # Exclude old voivodeships (51+), the sub-Mazowieckie splits
        # (1300000, 1500000) to avoid double-counting with Mazowieckie.
        new_voiv_codes = {f'{c:02d}' for c in range(2, 33, 2)}
        return [r.teryt_id for r in db._records.values()
                if r.teryt_id[2:] == '00000'
                and r.teryt_id != '0000000'
                and len(r.teryt_id) == 7
                and r.teryt_id[:2] in new_voiv_codes
                and r.teryt_id not in ('1300000', '1500000')]

    # Voivodeship level
    if tid[2:] == '00000':
        children = []
        for child_tid in record.children_ids:
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
        return [
            child_tid for child_tid in record.children_ids
            if child_tid[-1] in RODZ_AGGREGATION_SET
        ]

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

        # Store DataSeries counterparts so that get_data_by_subject works
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
    # Layer 1 helpers — will be fully implemented in work chunk C
    # ------------------------------------------------------------------

    def _generate_seeds(
        self,
        teryt_ids: List[str],
        source_subject_id: str,
        year_range: range,
        dim_names: List[str],
        dim_labels: Dict[str, List[str]],
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Generate seed cross tables via log-linear interpolation.

        For each territorial unit, collects all years with observed data
        from *source_subject_id*, then interpolates in log-space to fill
        every year in *year_range*.

        Returns
        -------
        dict : teryt_id → {year → np.ndarray}
        """
        # Placeholder — full implementation in work chunk C (v5.2)
        raise NotImplementedError(
            "_generate_seeds() will be implemented in work chunk C (v5.2). "
            "See todo.md item 14."
        )

    # ------------------------------------------------------------------
    # Layer 2 helpers — will be fully implemented in work chunk C
    # ------------------------------------------------------------------

    def _fit_marginals_ipf(
        self,
        seed: np.ndarray,
        marginals: List[Tuple[np.ndarray, List[int]]],
    ) -> np.ndarray:
        """Fit a seed table to known marginals via N-dimensional IPF.

        Parameters
        ----------
        seed : np.ndarray
            Initial estimate (any shape).
        marginals : list of (target_array, dimensions)
            Each element specifies a marginal constraint.

        Returns
        -------
        np.ndarray : IPF-adjusted table.
        """
        # Placeholder — full implementation in work chunk C (v5.2)
        raise NotImplementedError(
            "_fit_marginals_ipf() will be implemented in work chunk C (v5.2). "
            "See todo.md item 16."
        )

    # ------------------------------------------------------------------
    # Layer 3 helpers — will be fully implemented in work chunk C
    # ------------------------------------------------------------------

    def _enforce_hierarchy_gurobi(
        self,
        variable_type: str,
        prediction_section: str,
        e_subject_id: str,
        year: int,
        voivodeship_tid: str,
    ):
        """Enforce hierarchical consistency via Gurobi QP.

        PRIMARY solver.  See todo.md item 18 for full formulation.
        """
        # Placeholder — full implementation in work chunk C (v5.2)
        raise NotImplementedError(
            "_enforce_hierarchy_gurobi() will be implemented in work chunk C "
            "(v5.2). See todo.md item 18."
        )

    def _enforce_hierarchy_ipf(
        self,
        variable_type: str,
        prediction_section: str,
        e_subject_id: str,
        year: int,
        voivodeship_tid: str,
    ):
        """Enforce hierarchical consistency via iterated multi-level IPF.

        FALLBACK solver when Gurobi is unavailable.
        See todo.md item 19 for algorithm.
        """
        # Placeholder — full implementation in work chunk C (v5.2)
        raise NotImplementedError(
            "_enforce_hierarchy_ipf() will be implemented in work chunk C "
            "(v5.2). See todo.md item 19."
        )

    # ------------------------------------------------------------------
    # Variable-specific pipelines — stubs for work chunks C & D
    # ------------------------------------------------------------------

    def _estimate_age_sex_2000(self, e_sid: str):
        """Age × sex estimation for Prediction2000 (1999–2024).
        Will be implemented in work chunk D (v5.3), item 20."""
        raise NotImplementedError("See todo.md item 20.")

    def _estimate_age_sex_1990(self, e_sid: str):
        """Age × sex estimation for Prediction1990 (1986–2002).
        Will be implemented in work chunk D (v5.3), item 21."""
        raise NotImplementedError("See todo.md item 21.")

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
