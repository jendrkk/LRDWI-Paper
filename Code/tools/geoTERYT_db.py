"""
GeoTERYT Database Module - Version 4.0
======================================

A comprehensive database system for Polish administrative divisions (TERYT) 
with full geometry support, historical tracking, and overlay operations.

NEW in v4.0:
- Added DataSeries class for storing numerical time series on records
- Added data attribute to TERYTRecord with add_data_point(), get_data_series(), etc.
- Added process_subject_data() static method on GeoTERYTDatabase (from GUS02)
- Added load_subject_data() for bulk loading of BDL data onto records
- Added aggregate_data() and get_distribution() for regional aggregation
- Updated save_complete() and load_complete_database() for data persistence

NEW in v3.1:
- Added historical_codes and code_by_year attributes to TERYTRecord
- Changed geometry loading: load_geometries() now ONLY stores in _geometries dict
- Added assign_geometries(year, level) - assign geometries for specific year only
- Added assign_missing_geometries(year, level) - find matching geometries for unchanged units
- Added impute_geometries_past_tid(year, level) - use code_by_year to find affiliated teryt_ids
- Added impute_from_best_candidates(condition) - fill geometry from geometry_best_candidate
- Added country_shape_check(year, level) - count holes vs missing geometries
- Updated save_complete() and load_complete_database() for new attributes

FIXES in v3.0:
- Added display() method to TERYTRecord for nice DataFrame-like display
- Added as_df() function to convert List[TERYTRecord] to DataFrame
- Added old_woj and old_woj_id attributes for pre-1999 voivodeship assignment
- Added save_complete() and load_complete_database() for full persistence with geometries
- Fixed geometry loading for pre-2012 files (handles 'obszar' column with NUTS-like codes)
- Historical TERYT IDs now properly displayed in get_unit_info()
- All past_* attributes now properly shown

FIXES in v2.0:
- Properly parses 'notes' column from CSV (string to dict conversion using ast.literal_eval)
- Correctly tracks changes by comparing year-over-year states
- Adds past_levels tracking (e.g., rural -> urban-rural transitions)
- Adds past_kinds tracking
- Fixed get_changed_units() method to actually return changed units
- Adds geometry clipping to Poland boundary (fixes Hel Peninsula water issue)
- Improved change detection by comparing consecutive years

Author: Jedrzej Slowinski and Claude Opus 4.5
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely import wkb
import warnings
import re
import os
import ast
import pickle
from IPython.display import display, HTML


# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# Pre-1999 Polish voivodeships (49 voivodeships system, 1975-1998)
PRE_1999_VOIVODESHIPS = {
    'biała podlaska': {'name': 'bialskopodlaskie', 'name_pl': 'Bialskopodlaskie'},
    'białystok': {'name': 'białostockie', 'name_pl': 'Białostockie'},
    'bielsko-biała': {'name': 'bielskie', 'name_pl': 'Bielskie'},
    'bydgoszcz': {'name': 'bydgoskie', 'name_pl': 'Bydgoskie'},
    'chełm': {'name': 'chełmskie', 'name_pl': 'Chełmskie'},
    'ciechanów': {'name': 'ciechanowskie', 'name_pl': 'Ciechanowskie'},
    'częstochowa': {'name': 'częstochowskie', 'name_pl': 'Częstochowskie'},
    'elbląg': {'name': 'elbląskie', 'name_pl': 'Elbląskie'},
    'gdańsk': {'name': 'gdańskie', 'name_pl': 'Gdańskie'},
    'gorzów wielkopolski': {'name': 'gorzowskie', 'name_pl': 'Gorzowskie'},
    'jelenia góra': {'name': 'jeleniogórskie', 'name_pl': 'Jeleniogórskie'},
    'kalisz': {'name': 'kaliskie', 'name_pl': 'Kaliskie'},
    'katowice': {'name': 'katowickie', 'name_pl': 'Katowickie'},
    'kielce': {'name': 'kieleckie', 'name_pl': 'Kieleckie'},
    'konin': {'name': 'konińskie', 'name_pl': 'Konińskie'},
    'koszalin': {'name': 'koszalińskie', 'name_pl': 'Koszalińskie'},
    'kraków': {'name': 'krakowskie', 'name_pl': 'Krakowskie'},
    'krosno': {'name': 'krośnieńskie', 'name_pl': 'Krośnieńskie'},
    'legnica': {'name': 'legnickie', 'name_pl': 'Legnickie'},
    'leszno': {'name': 'leszczyńskie', 'name_pl': 'Leszczyńskie'},
    'lublin': {'name': 'lubelskie', 'name_pl': 'Lubelskie'},
    'łomża': {'name': 'łomżyńskie', 'name_pl': 'Łomżyńskie'},
    'łódź': {'name': 'łódzkie', 'name_pl': 'Łódzkie'},
    'nowy sącz': {'name': 'nowosądeckie', 'name_pl': 'Nowosądeckie'},
    'olsztyn': {'name': 'olsztyńskie', 'name_pl': 'Olsztyńskie'},
    'opole': {'name': 'opolskie', 'name_pl': 'Opolskie'},
    'ostrołęka': {'name': 'ostrołęckie', 'name_pl': 'Ostrołęckie'},
    'piła': {'name': 'pilskie', 'name_pl': 'Pilskie'},
    'piotrków trybunalski': {'name': 'piotrkowskie', 'name_pl': 'Piotrkowskie'},
    'płock': {'name': 'płockie', 'name_pl': 'Płockie'},
    'poznań': {'name': 'poznańskie', 'name_pl': 'Poznańskie'},
    'przemyśl': {'name': 'przemyskie', 'name_pl': 'Przemyskie'},
    'radom': {'name': 'radomskie', 'name_pl': 'Radomskie'},
    'rzeszów': {'name': 'rzeszowskie', 'name_pl': 'Rzeszowskie'},
    'siedlce': {'name': 'siedleckie', 'name_pl': 'Siedleckie'},
    'sieradz': {'name': 'sieradzkie', 'name_pl': 'Sieradzkie'},
    'skierniewice': {'name': 'skierniewickie', 'name_pl': 'Skierniewickie'},
    'słupsk': {'name': 'słupskie', 'name_pl': 'Słupskie'},
    'suwałki': {'name': 'suwalskie', 'name_pl': 'Suwalskie'},
    'szczecin': {'name': 'szczecińskie', 'name_pl': 'Szczecińskie'},
    'tarnobrzeg': {'name': 'tarnobrzeskie', 'name_pl': 'Tarnobrzeskie'},
    'tarnów': {'name': 'tarnowskie', 'name_pl': 'Tarnowskie'},
    'toruń': {'name': 'toruńskie', 'name_pl': 'Toruńskie'},
    'wałbrzych': {'name': 'wałbrzyskie', 'name_pl': 'Wałbrzyskie'},
    'warszawa': {'name': 'warszawskie', 'name_pl': 'Warszawskie'},
    'włocławek': {'name': 'włocławskie', 'name_pl': 'Włocławskie'},
    'wrocław': {'name': 'wrocławskie', 'name_pl': 'Wrocławskie'},
    'zamość': {'name': 'zamojskie', 'name_pl': 'Zamojskie'},
    'zielona góra': {'name': 'zielonogórskie', 'name_pl': 'Zielonogórskie'},
}

# Administrative level codes
LEVEL_VOIVODESHIP = 2
LEVEL_POWIAT = 5
LEVEL_GMINA = 6

# RODZ (kind) codes to exclude when working with gmina-level data
# Codes 4 (town in urban-rural) and 5 (rural area in urban-rural) are sub-divisions
RODZ_SUB_DIVISIONS = ['4', '5']
RODZ_SUB_DIVISIONS_AND_DISTRICTS = ['4', '5', '8', '9']
RODZ_DISTRICTS = ['8', '9']

# Kind mapping
KIND_MAPPING = {
    '1': 'urban',
    '2': 'rural', 
    '3': 'urban-rural',
    '4': 'town',
    '5': 'village',
    '8': 'Warsaw district',
    '9': 'delegatura'
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def nuts_code_to_teryt(nuts_code: str) -> str:
    """
    Converts a NUTS code to a TERYT code.
    
    Used for pre-2012 geometry files that use NUTS-like codes in 'obszar' column.
    Example: 'PL214102011' -> '1214010' (14 = woj, 21 = pow, 01 = gmi, 1 = rodz)
    """
    if pd.isna(nuts_code) or nuts_code is None:
        return '0000000'
    
    nuts_code = str(nuts_code).strip()
    if len(nuts_code) < 7:
        return '0000000'
    
    k = 3
    gmina_id = nuts_code[-k:]
    powiat_id = nuts_code[-(k+2):-k]
    if len(nuts_code) == 11:
        voivodeship_id = nuts_code[1:3]
    elif len(nuts_code) == 10:
        voivodeship_id = nuts_code[2:4]
        powiat_id = nuts_code[4:6]
        gmina_id = nuts_code[6:9]
    else:
        voivodeship_id = nuts_code[2:4]
    return voivodeship_id + powiat_id + gmina_id


def teryt_to_short(teryt: str) -> str:
    """Converts a full 7-digit TERYT code to a 6-digit short version."""
    return str(teryt)[:6]


def as_df(records: List['TERYTRecord'], include_geometry: bool = False) -> pd.DataFrame:
    """
    Convert a list of TERYTRecords to a pandas DataFrame.
    
    Parameters:
    - records: List of TERYTRecord objects
    - include_geometry: If True, include geometry as WKT
    
    Returns:
    - pd.DataFrame with all record attributes
    """
    if not records:
        return pd.DataFrame()
    
    data = []
    for record in records:
        row = record.to_dict()
        if include_geometry and record.geometry is not None:
            row['geometry_wkt'] = record.geometry.wkt
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def parse_notes_column(notes_value) -> dict:
    """
    Parse the notes column which may be a dict, string representation of dict, or NaN.
    
    FIX: The notes column from CSV is saved as a string representation of a dict.
    This function properly parses it back into a dict.
    
    Returns a dict with 'number_of_changes' and 'changes' keys.
    """
    if pd.isna(notes_value):
        return {'number_of_changes': 0, 'changes': []}
    
    if isinstance(notes_value, dict):
        return notes_value
    
    if isinstance(notes_value, str):
        # Skip empty strings
        if not notes_value.strip():
            return {'number_of_changes': 0, 'changes': []}
        
        try:
            # Try to parse as Python literal (handles dicts saved by repr())
            parsed = ast.literal_eval(notes_value)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, RecursionError):
            pass
        
        # Try JSON parsing as fallback
        try:
            import json
            notes_value_json = notes_value.replace("'", '"')
            return json.loads(notes_value_json)
        except:
            pass
    
    return {'number_of_changes': 0, 'changes': []}


def safe_clip_geometry(geometry, clip_geometry):
    """
    Safely clip a geometry to another geometry, handling invalid geometries.
    
    Used to clip gmina geometries to Poland's land boundary (removes water areas).
    """
    if geometry is None or geometry.is_empty:
        return geometry
    
    try:
        # Make geometries valid
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        if not clip_geometry.is_valid:
            clip_geometry = make_valid(clip_geometry)
        
        # Perform intersection
        clipped = geometry.intersection(clip_geometry)
        
        if clipped.is_empty:
            return geometry  # Return original if clipping results in empty
        
        # Ensure we return a Polygon or MultiPolygon
        if clipped.geom_type == 'GeometryCollection':
            # Extract polygons from collection
            polys = []
            for g in clipped.geoms:
                if g.geom_type == 'Polygon':
                    polys.append(g)
                elif g.geom_type == 'MultiPolygon':
                    polys.extend(g.geoms)
            if polys:
                if len(polys) == 1:
                    clipped = polys[0]
                else:
                    clipped = MultiPolygon(polys)
        
        return clipped
    except Exception as e:
        # If clipping fails, return original
        warnings.warn(f"Geometry clipping failed: {e}")
        return geometry

# ==============================================================================
# DATA SERIES CLASS (NEW in v4.0)
# ==============================================================================

class DataSeries:
    """
    A time series of numerical data with metadata.
    
    Stores values as a pd.Series with DatetimeIndex (01.01.YYYY format)
    spanning YEAR_RANGE_FULL, along with descriptive metadata about
    the data source, subject, variable, and categorical dimensions.
    
    Attributes:
    - source_type: Data source ('BDL', 'Census', etc.)
    - subject_id: Subject identifier (e.g. 'P2137')
    - subject_name: Human-readable subject name (reserved for future use)
    - variable_id: Variable identifier within the subject
    - variable_name: Human-readable variable name (reserved for future use)
    - categories: Dict of categorical dimensions (e.g. {'n1': 'ogółem', 'n2': 'miasto'})
    - values: pd.Series with DatetimeIndex, NaN for missing years
    - cat_code: Dict mapping dimension name -> numerical category code (e.g. {'n1': 1, 'n2': 2})
    - cat_bounds: Dict mapping dimension name -> {'lower_bound': ..., 'upper_bound': ...}
    """
    
    def __init__(self, source_type: str, subject_id: str, variable_id,
                 subject_name: str = '', variable_name: str = '',
                 categories: dict = None):
        self.source_type = source_type
        self.subject_id = str(subject_id)
        self.subject_name = subject_name
        self.variable_id = str(variable_id)
        self.variable_name = variable_name
        self.categories = categories or {}
        # pd.Series indexed by DatetimeIndex spanning YEAR_RANGE_FULL (NEW in v4.2)
        self.values: pd.Series = pd.Series(
            data=np.nan,
            index=DATETIME_INDEX_FULL,
            dtype=float
        )
        # Category coding (NEW in v4.2)
        self.cat_code: Dict[str, int] = {}      # dim_name -> numerical code
        self.cat_bounds: Dict[str, dict] = {}    # dim_name -> {'lower_bound': ..., 'upper_bound': ...}
    
    def add_value(self, year, value):
        """Add a data point for a specific year."""
        try:
            yr = int(year)
            ts = pd.Timestamp(year=yr, month=1, day=1)
            if ts in self.values.index:
                self.values[ts] = float(value)
        except (ValueError, TypeError):
            pass  # Skip non-numeric values
    
    def get_value(self, year: int) -> Optional[float]:
        """Get value for a specific year. Returns None if not available."""
        try:
            ts = pd.Timestamp(year=int(year), month=1, day=1)
            val = self.values.get(ts, np.nan)
            return None if pd.isna(val) else float(val)
        except (ValueError, KeyError):
            return None
    
    @property
    def years(self) -> List[int]:
        """Sorted list of years with non-NaN data."""
        return sorted([ts.year for ts in self.values.dropna().index])
    
    @property
    def n_years(self) -> int:
        """Number of years with non-NaN data."""
        return int(self.values.notna().sum())
    
    @property
    def key(self) -> Tuple[str, str, str]:
        """The tuple key identifying this series: (source_type, subject_id, variable_id)."""
        return (self.source_type, self.subject_id, self.variable_id)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        # Store only non-NaN values as {year_int: value}
        vals = {ts.year: v for ts, v in self.values.items() if not pd.isna(v)}
        return {
            'source_type': self.source_type,
            'subject_id': self.subject_id,
            'subject_name': self.subject_name,
            'variable_id': self.variable_id,
            'variable_name': self.variable_name,
            'categories': self.categories,
            'values': vals,
            'cat_code': self.cat_code if self.cat_code else None,
            'cat_bounds': self.cat_bounds if self.cat_bounds else None
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'DataSeries':
        """Reconstruct from dictionary."""
        ds = cls(
            source_type=d['source_type'],
            subject_id=d['subject_id'],
            variable_id=d['variable_id'],
            subject_name=d.get('subject_name', ''),
            variable_name=d.get('variable_name', ''),
            categories=d.get('categories', {})
        )
        # Restore values from {year: value} dict
        for k, v in d.get('values', {}).items():
            ds.add_value(int(k), v)
        # Restore cat_code and cat_bounds (NEW in v4.2)
        ds.cat_code = d.get('cat_code') or {}
        ds.cat_bounds = d.get('cat_bounds') or {}
        return ds
    
    def __repr__(self):
        cats = ', '.join(f'{k}={v}' for k, v in self.categories.items()) if self.categories else 'none'
        yr_range = f"{min(self.years)}-{max(self.years)}" if self.years else 'no data'
        return f"DataSeries({self.source_type}/{self.subject_id}/{self.variable_id}, cats=[{cats}], {self.n_years} years [{yr_range}])"


# ==============================================================================
# CROSS TABLE CLASS (NEW in v4.1)
# ==============================================================================

YEAR_RANGE_FULL = list(range(1988, 2026))  # 1988–2025

# DatetimeIndex for the full time span (NEW in v4.2)
DATETIME_INDEX_FULL = pd.DatetimeIndex(
    [pd.Timestamp(year=y, month=1, day=1) for y in YEAR_RANGE_FULL]
)

class CrossTable:
    """
    Multi-dimensional cross table for a subject on a TERYTRecord.
    
    Stores an M-dimensional numpy array per year, where M is the number of
    non-constant categorical dimensions (n1, n2, ...) in the subject.
    
    For a subject with 2 categories (e.g., n1=age, n2=gender), the cross table
    for each year is a 2D matrix. For 3 categories, it is a 3D tensor, etc.
    For a single category, it is a 1D vector (one row).
    
    Years without data are stored as NaN-filled arrays of the same shape.
    
    Attributes:
    - subject_id: Subject identifier
    - subject_name: Human-readable subject name
    - dim_names: Ordered list of dimension names (e.g., ['n1', 'n2'])
    - dim_labels: Dict mapping dim name -> ordered list of labels (e.g., {'n1': ['ogółem', '0-4', ...], 'n2': ['ogółem', 'mężczyźni', 'kobiety']})
    - tables: Dict mapping year (int) -> np.ndarray (shape determined by dim_labels)
    - year_range: List of years the cross table spans (default: 1988–2024)
    """
    
    def __init__(self, subject_id: str, dim_names: List[str],
                 dim_labels: Dict[str, List[str]], subject_name: str = '',
                 year_range: List[int] = None):
        self.subject_id = str(subject_id)
        self.subject_name = subject_name
        self.dim_names = list(dim_names)
        self.dim_labels = {k: list(v) for k, v in dim_labels.items()}
        self.year_range = year_range or YEAR_RANGE_FULL
        # Compute shape from dim_labels
        self._shape = tuple(len(self.dim_labels[d]) for d in self.dim_names)
        # Initialize tables: year -> ndarray (NaN for missing)
        self.tables: Dict[int, np.ndarray] = {}
        for year in self.year_range:
            self.tables[year] = np.full(self._shape, np.nan)
    
    @property
    def shape(self) -> tuple:
        """Shape of each year's cross table."""
        return self._shape
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.dim_names)
    
    @property
    def years_with_data(self) -> List[int]:
        """Years that have at least one non-NaN value."""
        return sorted(y for y, t in self.tables.items() if not np.all(np.isnan(t)))
    
    @property
    def years_missing(self) -> List[int]:
        """Years that are entirely NaN."""
        return sorted(y for y, t in self.tables.items() if np.all(np.isnan(t)))
    
    def get_table(self, year: int) -> Optional[np.ndarray]:
        """Get the cross table array for a given year."""
        return self.tables.get(int(year))
    
    def set_table(self, year: int, table: np.ndarray):
        """
        Manually set the cross table for a given year.
        
        The table must have the same shape as existing tables.
        """
        year = int(year)
        table = np.asarray(table, dtype=float)
        if table.shape != self._shape:
            raise ValueError(
                f"Table shape {table.shape} does not match expected {self._shape}"
            )
        self.tables[year] = table
    
    def get_as_dataframe(self, year: int) -> pd.DataFrame:
        """
        Get the cross table for a year as a pandas DataFrame.
        
        For 1D: single-column DataFrame with dim_labels as index.
        For 2D: DataFrame with dim_labels[dim0] as index, dim_labels[dim1] as columns.
        For 3D+: MultiIndex DataFrame (first N-1 dims as row MultiIndex, last dim as columns).
        """
        table = self.get_table(year)
        if table is None:
            return pd.DataFrame()
        
        if self.ndim == 1:
            return pd.DataFrame(
                table, index=self.dim_labels[self.dim_names[0]],
                columns=['value']
            )
        elif self.ndim == 2:
            return pd.DataFrame(
                table,
                index=self.dim_labels[self.dim_names[0]],
                columns=self.dim_labels[self.dim_names[1]]
            )
        else:
            # M-dimensional: flatten to 2D with MultiIndex rows
            # Last dimension becomes columns, rest become row MultiIndex
            import itertools
            row_dims = self.dim_names[:-1]
            col_dim = self.dim_names[-1]
            row_labels_product = list(itertools.product(
                *[self.dim_labels[d] for d in row_dims]
            ))
            row_index = pd.MultiIndex.from_tuples(row_labels_product, names=row_dims)
            flat = table.reshape(-1, table.shape[-1])
            return pd.DataFrame(flat, index=row_index,
                                columns=self.dim_labels[col_dim])
    
    def deconstruct_to_data_points(self, year: int) -> List[dict]:
        """
        Deconstruct the cross table for a year into raw data points.
        
        Returns a list of dicts with category values and numeric value,
        suitable for feeding back into add_data_point().
        
        Returns:
        - List of dicts: [{'n1': ..., 'n2': ..., 'value': ...}, ...]
        """
        import itertools
        table = self.get_table(year)
        if table is None:
            return []
        
        points = []
        indices = list(itertools.product(
            *[range(len(self.dim_labels[d])) for d in self.dim_names]
        ))
        for idx in indices:
            val = table[idx]
            if np.isnan(val):
                continue
            categories = {}
            for i, dim in enumerate(self.dim_names):
                categories[dim] = self.dim_labels[dim][idx[i]]
            points.append({**categories, 'value': float(val)})
        
        return points
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        tables_serialized = {}
        for year, table in self.tables.items():
            tables_serialized[year] = table.tolist()
        return {
            'subject_id': self.subject_id,
            'subject_name': self.subject_name,
            'dim_names': self.dim_names,
            'dim_labels': self.dim_labels,
            'year_range': self.year_range,
            'tables': tables_serialized
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'CrossTable':
        """Reconstruct from dictionary."""
        ct = cls(
            subject_id=d['subject_id'],
            dim_names=d['dim_names'],
            dim_labels=d['dim_labels'],
            subject_name=d.get('subject_name', ''),
            year_range=d.get('year_range', YEAR_RANGE_FULL)
        )
        for year_str, table_list in d.get('tables', {}).items():
            ct.tables[int(year_str)] = np.array(table_list, dtype=float)
        return ct
    
    def __repr__(self):
        data_years = self.years_with_data
        yr_str = f"{data_years[0]}-{data_years[-1]}" if data_years else 'no data'
        return (f"CrossTable({self.subject_id}, dims={self.dim_names}, "
                f"shape={self._shape}, {len(data_years)} years with data [{yr_str}])")
    
    def __add__(self, other: 'CrossTable') -> 'CrossTable':
        """
        Add two cross tables element-wise (for aggregation across TERYTs).
        
        NaN + x = x (NaN is treated as 0 for aggregation purposes).
        """
        if self.subject_id != other.subject_id:
            raise ValueError(f"Cannot add cross tables from different subjects: "
                             f"{self.subject_id} vs {other.subject_id}")
        if self._shape != other._shape:
            raise ValueError(f"Cannot add cross tables with different shapes: "
                             f"{self._shape} vs {other._shape}")
        
        result = CrossTable(
            subject_id=self.subject_id,
            dim_names=self.dim_names,
            dim_labels=self.dim_labels,
            subject_name=self.subject_name,
            year_range=self.year_range
        )
        for year in result.year_range:
            a = self.tables.get(year, np.full(self._shape, np.nan))
            b = other.tables.get(year, np.full(self._shape, np.nan))
            # np.nansum: NaN + NaN = 0, NaN + x = x
            # We want NaN + NaN = NaN, so handle that
            both_nan = np.isnan(a) & np.isnan(b)
            summed = np.where(np.isnan(a), 0, a) + np.where(np.isnan(b), 0, b)
            summed[both_nan] = np.nan
            result.tables[year] = summed
        
        return result


# ==============================================================================
# DATABASE RECORD CLASS
# ==============================================================================

class TERYTRecord:
    """
    Represents a single administrative division record with full metadata.
    
    Tracks historical changes including:
    - past_teryt_ids: All previous TERYT IDs this unit had
    - past_names: All previous names
    - past_levels: All previous administrative levels (NEW in v2.0)
    - past_kinds: All previous kinds (urban, rural, etc.)
    - changes: Detailed list of all changes with years
    - old_woj: Pre-1999 voivodeship name (NEW in v3.0)
    - old_woj_id: Pre-1999 voivodeship index (NEW in v3.0)
    - historical_codes: List of all TERYT codes this unit ever had (NEW in v3.1)
    - code_by_year: Dict mapping year -> teryt_id for this unit (NEW in v3.1)
    """
    
    def __init__(self, teryt_id: str, name: str, name_dod: str = None,
                 level: int = None, kind: str = None):
        self.teryt_id = str(teryt_id).zfill(7)
        self.nts_id = teryt_id[:-1] # NTS ID is TERYT without the last digit
        self.name = name
        self.name_dod = name_dod
        self.level = level
        self.kind = kind
        self.years_valid: Set[int] = set()
        self.past_names: List[Tuple[str, int]] = []  # (name, year_until)
        self.past_teryt_ids: List[Tuple[str, int]] = []  # (teryt_id, year_until)
        self.past_levels: List[Tuple[int, int]] = []  # (level, year_until) - NEW
        self.past_kinds: List[Tuple[str, int]] = []  # (kind, year_until)
        self.changes: List[dict] = []
        
        # Geometry attributes
        self.geometry = None
        self.geometry_best_candidate = None
        self.geometry_year: Optional[int] = None
        self.geometry_notes: Optional[str] = None
        
        # Additional metadata
        self.woj = self.teryt_id[:2]
        self.pow = self.teryt_id[2:4]
        self.gmi = self.teryt_id[4:6]
        self.rodz = self.teryt_id[6]
        
        # Pre-1999 voivodeship assignment (NEW in v3.0)
        self.old_woj: Optional[str] = None  # Name of pre-1999 voivodeship
        self.old_woj_id: Optional[int] = None  # Index of pre-1999 voivodeship
        
        # Historical codes tracking (NEW in v3.1)
        self.historical_codes: List[str] = []  # All TERYT codes this unit ever had
        self.code_by_year: Dict[int, str] = {}  # year -> teryt_id mapping
        
        # Track if this unit underwent any changes
        self.has_changes = False
        
        # Childern and parent relationships (for hierarchy)
        self.parent_id: Optional[str] = None
        self.children_ids: List[str] = []
        
        # Data storage (NEW in v4.0)
        # Key: (source_type, subject_id, variable_id) -> DataSeries
        self.data: Dict[tuple, 'DataSeries'] = {}
        
        # Cross table storage (NEW in v4.1)
        # Key: subject_id -> CrossTable
        self.cross_tables: Dict[str, 'CrossTable'] = {}
        
        # Population and classification (NEW in v4.2)
        # Total population time series indexed by DatetimeIndex
        self.pop: pd.Series = pd.Series(
            data=np.nan,
            index=DATETIME_INDEX_FULL,
            dtype=float
        )
        # Urban/rural classification DataFrame indexed by DatetimeIndex
        self.pop_class: pd.DataFrame = pd.DataFrame(
            {'pop_class_code': pd.Series(dtype=int),
             'pop_class_label': pd.Series(dtype=str)},
            index=pd.DatetimeIndex([])
        )
    
    def add_year(self, year: int):
        """Add a year when this division was valid."""
        self.years_valid.add(year)
    
    def add_past_name(self, name: str, year_until: int):
        """Add a previous name with the year it was valid until."""
        if name and name != self.name:
            entry = (name, year_until)
            if entry not in self.past_names:
                self.past_names.append(entry)
                self.has_changes = True
    
    def add_past_teryt_id(self, teryt_id: str, year_until: int):
        """Add a previous TERYT ID with the year it was valid until."""
        teryt_id = str(teryt_id).zfill(7)
        if teryt_id and teryt_id != self.teryt_id and teryt_id != '0000000':
            entry = (teryt_id, year_until)
            if entry not in self.past_teryt_ids:
                self.past_teryt_ids.append(entry)
                self.has_changes = True
    
    def add_past_level(self, level: int, year_until: int):
        """Add a previous level with the year it was valid until."""
        if level is not None and level != self.level:
            entry = (int(level), year_until)
            if entry not in self.past_levels:
                self.past_levels.append(entry)
                self.has_changes = True
    
    def add_past_kind(self, kind: str, year_until: int):
        """Add a previous kind with the year it was valid until."""
        if kind and kind != self.kind:
            entry = (kind, year_until)
            if entry not in self.past_kinds:
                self.past_kinds.append(entry)
                self.has_changes = True
    
    def add_change(self, change: dict):
        """Add a change record."""
        if change not in self.changes:
            self.changes.append(change)
            self.has_changes = True
    
    def set_geometry(self, geometry, year: int):
        """Set the geometry and its source year."""
        self.geometry = geometry
        self.geometry_year = year
    
    def set_parent(self, parent_id: str):
        """Set the parent TERYT ID."""
        self.parent_id = parent_id
    
    def set_children(self, children_ids: List[str]):
        """Set the list of children TERYT IDs."""
        self.children_ids = children_ids
    
    # ------------------------------------------------------------------
    # Data management methods (NEW in v4.0)
    # ------------------------------------------------------------------
    
    def add_data_point(self, source_type: str, subject_id: str, variable_id,
                       year, value, categories: dict = None,
                       subject_name: str = '', variable_name: str = ''):
        """
        Add a single data point to this record.
        Creates a new DataSeries if one doesn't exist for the given key.
        
        Parameters:
        - source_type: Data source ('BDL', 'Census', etc.)
        - subject_id: Subject ID (e.g. 'P2137')
        - variable_id: Variable ID within the subject
        - year: Year of the data point
        - value: Numerical value
        - categories: Dict of categorical dimensions (e.g. {'n1': 'ogółem'})
        - subject_name: Human-readable subject name (reserved for future)
        - variable_name: Human-readable variable name (reserved for future)
        """
        key = (source_type, str(subject_id), str(variable_id))
        if key not in self.data:
            self.data[key] = DataSeries(
                source_type=source_type,
                subject_id=str(subject_id),
                variable_id=str(variable_id),
                subject_name=subject_name,
                variable_name=variable_name,
                categories=categories
            )
        self.data[key].add_value(year, value)
    
    def get_data_series(self, source_type: str, subject_id: str, variable_id) -> Optional['DataSeries']:
        """Get a specific DataSeries by key, or None if not found."""
        key = (source_type, str(subject_id), str(variable_id))
        return self.data.get(key)
    
    def get_data_by_subject(self, subject_id: str) -> Dict[tuple, 'DataSeries']:
        """Get all DataSeries for a given subject."""
        return {k: v for k, v in self.data.items() if k[1] == str(subject_id)}
    
    def list_data_keys(self) -> List[tuple]:
        """List all (source_type, subject_id, variable_id) keys stored."""
        return list(self.data.keys())
    
    def list_subjects(self) -> List[str]:
        """List all unique subject IDs stored on this record."""
        return list(set(k[1] for k in self.data.keys()))
    
    @property
    def has_data(self) -> bool:
        """Whether this record has any data stored."""
        return len(self.data) > 0
    
    @property
    def n_data_series(self) -> int:
        """Number of data series stored."""
        return len(self.data)
    
    # ------------------------------------------------------------------
    # Cross table methods (NEW in v4.1)
    # ------------------------------------------------------------------
    
    def build_cross_table(self, subject_id: str, subject_name: str = '') -> Optional['CrossTable']:
        """
        Build a CrossTable for a subject from existing DataSeries on this record.
        
        Automatically detects dimensions from the categories of stored variables.
        If the subject has 0 or 1 variables, returns None (cross table not meaningful).
        
        Parameters:
        - subject_id: Subject ID
        - subject_name: Human-readable name for the cross table
        
        Returns:
        - CrossTable if built, None if not enough variables
        """
        subject_data = self.get_data_by_subject(str(subject_id))
        if not subject_data:
            return None
        
        # Collect all unique dimension names and their labels
        dim_values = {}  # dim_name -> set of labels
        for key, series in subject_data.items():
            if series.categories:
                for dim_name, label in series.categories.items():
                    if dim_name not in dim_values:
                        dim_values[dim_name] = set()
                    dim_values[dim_name].add(str(label))
        
        if not dim_values:
            return None
        
        # Sort dimension names and labels for deterministic ordering
        dim_names = sorted(dim_values.keys())
        dim_labels = {d: sorted(dim_values[d]) for d in dim_names}
        
        # Create cross table
        ct = CrossTable(
            subject_id=str(subject_id),
            dim_names=dim_names,
            dim_labels=dim_labels,
            subject_name=subject_name
        )
        
        # Fill in values from DataSeries
        for key, series in subject_data.items():
            if not series.categories:
                continue
            # Find the index for each dimension
            idx = []
            valid = True
            for dim in dim_names:
                label = str(series.categories.get(dim, ''))
                if label in dim_labels[dim]:
                    idx.append(dim_labels[dim].index(label))
                else:
                    valid = False
                    break
            if not valid:
                continue
            
            idx_tuple = tuple(idx)
            for ts, val in series.values.dropna().items():
                yr = ts.year
                if yr in ct.tables:
                    ct.tables[yr][idx_tuple] = float(val)
        
        self.cross_tables[str(subject_id)] = ct
        return ct
    
    def get_cross_table(self, subject_id: str) -> Optional['CrossTable']:
        """Get the CrossTable for a subject, or None if not built."""
        return self.cross_tables.get(str(subject_id))
    
    def set_cross_table(self, subject_id: str, cross_table: 'CrossTable'):
        """
        Manually set/replace a CrossTable for a subject.
        
        Parameters:
        - subject_id: Subject ID
        - cross_table: CrossTable instance to store
        """
        self.cross_tables[str(subject_id)] = cross_table
    
    def insert_cross_table_year(self, subject_id: str, year: int,
                                table: np.ndarray, subject_name: str = '',
                                dim_names: List[str] = None,
                                dim_labels: Dict[str, List[str]] = None):
        """
        Insert a cross table for a single year, creating the CrossTable if needed.
        
        Parameters:
        - subject_id: Subject ID
        - year: Year for this table
        - table: numpy array with the values
        - subject_name: Human-readable name (used only if creating new)
        - dim_names: Dimension names (required if creating new)
        - dim_labels: Dimension labels (required if creating new)
        """
        subject_id = str(subject_id)
        ct = self.cross_tables.get(subject_id)
        if ct is None:
            if dim_names is None or dim_labels is None:
                raise ValueError("dim_names and dim_labels required when creating new CrossTable")
            ct = CrossTable(
                subject_id=subject_id,
                dim_names=dim_names,
                dim_labels=dim_labels,
                subject_name=subject_name
            )
            self.cross_tables[subject_id] = ct
        ct.set_table(year, table)
    
    def deconstruct_cross_table(self, subject_id: str, year: int,
                                source_type: str = 'CrossTable') -> int:
        """
        Deconstruct a cross table for a year back into raw DataSeries data points.
        
        The deconstructed data is stored in self.data as new DataSeries.
        
        Parameters:
        - subject_id: Subject ID
        - year: Year to deconstruct
        - source_type: Source type label for the created data points
        
        Returns:
        - Number of data points created
        """
        ct = self.cross_tables.get(str(subject_id))
        if ct is None:
            return 0
        
        points = ct.deconstruct_to_data_points(year)
        count = 0
        for pt in points:
            # Build categories and variable_id from dimension values
            categories = {k: v for k, v in pt.items() if k != 'value'}
            var_id = '|'.join(str(categories.get(d, '')) for d in ct.dim_names)
            self.add_data_point(
                source_type=source_type,
                subject_id=str(subject_id),
                variable_id=var_id,
                year=year,
                value=pt['value'],
                categories=categories,
                subject_name=ct.subject_name
            )
            count += 1
        return count
    
    def list_cross_tables(self) -> List[str]:
        """List subject IDs that have cross tables."""
        return list(self.cross_tables.keys())
    
    @property
    def has_cross_tables(self) -> bool:
        """Whether this record has any cross tables."""
        return len(self.cross_tables) > 0
    
    @property
    def n_cross_tables(self) -> int:
        """Number of cross tables stored."""
        return len(self.cross_tables)
    
    @property
    def first_year(self) -> Optional[int]:
        return min(self.years_valid) if self.years_valid else None
    
    @property
    def last_year(self) -> Optional[int]:
        return max(self.years_valid) if self.years_valid else None
    
    @property
    def has_geometry(self) -> bool:
        return self.geometry is not None
    
    @property
    def short_teryt(self) -> str:
        return self.teryt_id[:6]
    
    @property
    def all_teryt_ids(self) -> List[str]:
        """Get all TERYT IDs this unit ever had (current + past)."""
        ids = [self.teryt_id]
        ids.extend([tid for tid, _ in self.past_teryt_ids])
        return ids
    
    @property
    def all_names(self) -> List[str]:
        """Get all names this unit ever had (current + past)."""
        names = [self.name]
        names.extend([name for name, _ in self.past_names])
        return list(set(names))
    
    @property
    def n_changes(self) -> int:
        """Total number of changes for this unit."""
        return len(self.changes)
    
    def to_dict(self) -> dict:
        """Convert record to dictionary."""
        return {
            'teryt_id': self.teryt_id,
            'name': self.name,
            'name_dod': self.name_dod,
            'level': self.level,
            'kind': self.kind,
            'woj': self.woj,
            'pow': self.pow,
            'gmi': self.gmi,
            'rodz': self.rodz,
            'years_valid': sorted(self.years_valid),
            'first_year': self.first_year,
            'last_year': self.last_year,
            'past_names': self.past_names,
            'past_teryt_ids': self.past_teryt_ids,
            'past_levels': self.past_levels,
            'past_kinds': self.past_kinds,
            'all_teryt_ids': self.all_teryt_ids,
            'all_names': self.all_names,
            'changes': self.changes,
            'has_changes': self.has_changes,
            'n_changes': self.n_changes,
            'has_geometry': self.has_geometry,
            'geometry_year': self.geometry_year,
            'geometry_notes': self.geometry_notes,
            'old_woj': self.old_woj,
            'old_woj_id': self.old_woj_id,
            'historical_codes': self.historical_codes,
            'code_by_year': self.code_by_year,
            'has_data': self.has_data,
            'n_data_series': self.n_data_series,
            'data_subjects': self.list_subjects(),
            'has_cross_tables': self.has_cross_tables,
            'n_cross_tables': self.n_cross_tables,
            'cross_table_subjects': self.list_cross_tables(),
            'has_pop': bool(self.pop.notna().any()),
            'pop_years': sorted([ts.year for ts in self.pop.dropna().index]) if self.pop.notna().any() else [],
            'has_pop_class': len(self.pop_class) > 0,
        }
    
    def display(self):
        """
        Display this record in a nice DataFrame-like format (for Jupyter notebooks).
        """
        data = self.to_dict()
        # Format some fields for better readability
        formatted = {
            'Field': [],
            'Value': []
        }
        
        for key, value in data.items():
            formatted['Field'].append(key)
            if isinstance(value, (list, set)):
                if len(value) == 0:
                    formatted['Value'].append('[]')
                elif len(value) > 5:
                    formatted['Value'].append(f'{str(value[:5])}... ({len(value)} items)')
                else:
                    formatted['Value'].append(str(value))
            else:
                formatted['Value'].append(str(value))
        
        df = pd.DataFrame(formatted)
        try:
            display(df)
        except:
            print(df.to_string(index=False))
    
    def __repr__(self):
        status = []
        if self.has_changes:
            status.append(f"changes={self.n_changes}")
        if self.past_levels:
            status.append(f"level_changes={len(self.past_levels)}")
        status_str = ", ".join(status) if status else "no changes"
        return f"TERYTRecord({self.teryt_id}, {self.name}, years={self.first_year}-{self.last_year}, {status_str})"


# ==============================================================================
# MAIN DATABASE CLASS
# ==============================================================================

class GeoTERYTDatabase:
    """
    Comprehensive database for Polish administrative divisions with geometry support.
    
    Version 2.0 improvements:
    - Properly parses notes column from CSV (string to dict conversion)
    - Correctly tracks changes by comparing year-over-year states
    - Tracks past_levels and past_kinds  
    - Clips geometries to Poland boundary (fixes water area issues)
    - Fixed get_changed_units() method
    """
    
    def __init__(self):
        """Initialize an empty database."""
        # Main storage: teryt_id -> TERYTRecord
        self._records: Dict[str, TERYTRecord] = {}
        
        # Indices for fast lookup
        self._by_year: Dict[int, Set[str]] = {}
        self._by_name: Dict[str, Set[str]] = {}
        self._by_level: Dict[int, Set[str]] = {}
        self._by_kind: Dict[str, Set[str]] = {}
        self._by_voivodeship: Dict[str, Set[str]] = {}
        
        # Track ID changes: old_id -> new_id
        self._id_transitions: Dict[str, str] = {}
        
        # Geometry storage
        self._geometries: Dict[int, gpd.GeoDataFrame] = {}
        self._poland_boundary = None
        self._poland_gdf = None
        self._old_voivodships = None
        
        # Metadata
        self._year_range: Tuple[int, int] = (1999, 2024)
        self._crs = "EPSG:2180"
        self._built = False
    
    def set_poland_boundary(self, boundary_gdf: gpd.GeoDataFrame):
        """
        Set the Poland boundary for clipping geometries.
        
        This removes water areas (like the Baltic Sea around Hel Peninsula).
        
        Parameters:
        - boundary_gdf: GeoDataFrame with Poland's boundary (will use unary_union)
        """
        if boundary_gdf.crs != self._crs:
            boundary_gdf = boundary_gdf.to_crs(self._crs)
        self._poland_boundary = unary_union(boundary_gdf.geometry)
        if not self._poland_boundary.is_valid:
            self._poland_boundary = make_valid(self._poland_boundary)

    def set_poland_gdf(self, boundary_gdf: gpd.GeoDataFrame):
        """
        Set the Poland GeoDataFrame for plotting and other purposes.
        
        Parameters:
        - boundary_gdf: GeoDataFrame with Poland's boundary
        """
        if boundary_gdf.crs != self._crs:
            boundary_gdf = boundary_gdf.to_crs(self._crs)
        self._poland_gdf = boundary_gdf

    def set_old_voivodship_gdf(self, old_voiv_gdf: gpd.GeoDataFrame):
        """
        Set the old voivodeships GeoDataFrame for pre-1999 assignment.
        
        Parameters:
        - old_voiv_gdf: GeoDataFrame with pre-1999 voivodeships
        """
        if old_voiv_gdf.crs != self._crs:
            old_voiv_gdf = old_voiv_gdf.to_crs(self._crs)
        
        def build_old_woj_mapping(row):
            mapping = {}
            for idx, row in old_voiv_gdf.iterrows():
                name = row.get('name', '').lower()
                mapping[name] = {
                    'name': row.get('name'),
                    'name_pl': row.get('name_pl'),
                    'geometry': row.geometry
                }
        
        self._old_voivodships = old_voiv_gdf

    def get_poland_boundary(self) -> gpd.GeoDataFrame:
        
        if self._poland_gdf is not None:
            return self._poland_gdf

    def get_poland_gdf(self) -> gpd.GeoDataFrame:
        
        if self._poland_boundary is not None:
            return self._poland_boundary

    def get_old_voivodship_gdf(self) -> gpd.GeoDataFrame:
        
        if self._old_voivodships is not None:
            return self._old_voivodships

    def build_from_harmonized(self, mega_df: pd.DataFrame, verbose: bool = True):
        """
        Build the database from a harmonized TERYT mega DataFrame.
        
        FIXED: Now properly:
        - Parses notes column from CSV string format
        - Tracks changes by comparing consecutive years
        - Records past_levels when a unit changes level (e.g., rural -> urban-rural)
        
        Parameters:
        - mega_df: DataFrame from harmonize_teryt() or loaded from CSV
        - verbose: Print progress messages
        """
        if verbose:
            print("Building GeoTERYT database v2.0 from harmonized data...")
        
        # Clear existing data
        self._records.clear()
        self._by_year.clear()
        self._by_name.clear()
        self._by_level.clear()
        self._by_kind.clear()
        self._by_voivodeship.clear()
        self._id_transitions.clear()
        
        years = sorted(mega_df['year'].unique())
        self._year_range = (min(years), max(years))
        
        # ======================================================================
        # PASS 1: Collect all unit states by year
        # ======================================================================
        if verbose:
            print("  Pass 1: Collecting unit states by year...")
        
        # Structure: teryt_id -> {year: {state data}}
        unit_states = {}
        
        for year in years:
            year_data = mega_df[mega_df['year'] == year]
            
            for _, row in year_data.iterrows():
                teryt_id = str(row.get('id', '')).zfill(7)
                if teryt_id == '0000000' or len(teryt_id) != 7:
                    continue
                
                if teryt_id not in unit_states:
                    unit_states[teryt_id] = {}
                
                # Parse the notes column properly!
                notes = parse_notes_column(row.get('notes'))
                
                unit_states[teryt_id][year] = {
                    'name': row.get('NAZWA', row.get('name', '')),
                    'name_dod': row.get('NAZWA_DOD', row.get('name_dod', '')),
                    'level': row.get('level'),
                    'kind': row.get('kind'),
                    'if_changed': row.get('if_changed', False),
                    'when_changed': row.get('when_changed'),
                    'notes': notes
                }
        
        if verbose:
            print(f"    Found {len(unit_states)} unique TERYT IDs across all years")
        
        # ======================================================================
        # PASS 2: Build records with proper change tracking
        # ======================================================================
        if verbose:
            print("  Pass 2: Building records with change detection...")
        
        change_stats = {
            'name_changes': 0,
            'kind_changes': 0,
            'level_changes': 0,
            'notes_changes': 0
        }
        
        for teryt_id, year_states in unit_states.items():
            sorted_years = sorted(year_states.keys())
            
            # Use the latest state for the main record
            latest_year = max(sorted_years)
            latest_state = year_states[latest_year]
            
            record = TERYTRecord(
                teryt_id=teryt_id,
                name=latest_state['name'],
                name_dod=latest_state['name_dod'],
                level=latest_state['level'],
                kind=latest_state['kind']
            )
            
            # Track all years this unit existed
            for year in sorted_years:
                record.add_year(year)
            
            # Track changes by comparing consecutive years
            prev_name = None
            prev_level = None
            prev_kind = None
            is_first_year = True
            
            def values_differ(val1, val2):
                """Compare two values, treating NaN/None as equal."""
                if pd.isna(val1) and pd.isna(val2):
                    return False
                if pd.isna(val1) or pd.isna(val2):
                    return True
                return val1 != val2
            
            for year in sorted_years:
                state = year_states[year]
                current_name = state['name']
                current_level = state['level']
                current_kind = state['kind']
                
                # Skip NaN comparisons on first year
                if not is_first_year:
                    # Detect NAME changes
                    if prev_name is not None and values_differ(current_name, prev_name):
                        record.add_past_name(prev_name, year - 1)
                        record.add_change({
                            'year': year,
                            'type': 'name_change',
                            'from': prev_name,
                            'to': current_name
                        })
                        change_stats['name_changes'] += 1
                    
                    # Detect LEVEL changes (e.g., from level 6 to level 5)
                    # Only if both values are not NaN
                    if prev_level is not None and not pd.isna(prev_level) and not pd.isna(current_level):
                        if values_differ(current_level, prev_level):
                            record.add_past_level(prev_level, year - 1)
                            record.add_change({
                                'year': year,
                                'type': 'level_change',
                                'from': prev_level,
                                'to': current_level
                            })
                            change_stats['level_changes'] += 1
                
                # Detect KIND changes (e.g., rural -> urban-rural)
                # Only if both values are not NaN
                if not is_first_year and prev_kind is not None and not pd.isna(prev_kind) and not pd.isna(current_kind):
                    if values_differ(current_kind, prev_kind):
                        record.add_past_kind(prev_kind, year - 1)
                        record.add_change({
                            'year': year,
                            'type': 'kind_change',
                            'from': prev_kind,
                            'to': current_kind
                        })
                        change_stats['kind_changes'] += 1
                
                # Also add changes from the notes column (from harmonization)
                # Only add on the year the change actually happened to avoid duplicates
                notes = state['notes']
                when_changed = state.get('when_changed')
                if notes and notes.get('number_of_changes', 0) > 0:
                    # Only add notes changes if when_changed matches this year,
                    # OR if when_changed is NaN and this is the first year
                    should_add = False
                    if when_changed is not None and not pd.isna(when_changed):
                        if int(when_changed) == year:
                            should_add = True
                    elif is_first_year:
                        # If no when_changed, add to first year only
                        should_add = True
                    
                    if should_add:
                        for change_desc in notes.get('changes', []):
                            record.add_change({
                                'year': year,
                                'type': 'reform',
                                'description': change_desc,
                                'when_changed': when_changed
                            })
                            change_stats['notes_changes'] += 1
                
                # Also mark if_changed flag
                if state.get('if_changed'):
                    record.has_changes = True
                
                # Update previous values for next iteration
                is_first_year = False
                prev_name = current_name
                prev_level = current_level
                prev_kind = current_kind
            
            self._records[teryt_id] = record
        
        # ======================================================================
        # PASS 2.5: Populate historical_codes and code_by_year from mega_df
        # ======================================================================
        if verbose:
            print("  Pass 2.5: Populating historical codes and code_by_year...")
        
        # Check if the columns exist in mega_df
        has_historical_codes = 'historical_codes' in mega_df.columns
        has_code_by_year = 'code_by_year' in mega_df.columns
        
        if has_historical_codes or has_code_by_year:
            # FIXED: Iterate over ALL teryt_ids in _records and find them in mega_df
            # (not just latest year, since historical teryt_ids may not exist in latest year)
            for teryt_id, record in self._records.items():
                # Find any row with this teryt_id in mega_df
                matching_rows = mega_df[mega_df['id'] == teryt_id]
                
                if len(matching_rows) == 0:
                    continue
                
                # Use the first matching row (historical_codes and code_by_year are same across all years)
                row = matching_rows.iloc[0]
                
                # Populate historical_codes
                if has_historical_codes:
                    hist_codes = row.get('historical_codes')
                    if pd.notna(hist_codes):
                        if isinstance(hist_codes, str):
                            try:
                                hist_codes = ast.literal_eval(hist_codes)
                            except:
                                hist_codes = []
                        if isinstance(hist_codes, list):
                            record.historical_codes = [str(c).zfill(7) for c in hist_codes]
                
                # Populate code_by_year
                if has_code_by_year:
                    cby = row.get('code_by_year')
                    if pd.notna(cby):
                        if isinstance(cby, str):
                            try:
                                cby = ast.literal_eval(cby)
                            except:
                                cby = {}
                        if isinstance(cby, dict):
                            # Convert keys to int and values to zfilled strings
                            record.code_by_year = {
                                int(year): str(code).zfill(7) 
                                for year, code in cby.items()
                            }
            
            if verbose:
                records_with_hist = sum(1 for r in self._records.values() if r.historical_codes)
                records_with_cby = sum(1 for r in self._records.values() if r.code_by_year)
                print(f"    Records with historical_codes: {records_with_hist}")
                print(f"    Records with code_by_year: {records_with_cby}")
        else:
            if verbose:
                print("    Note: historical_codes and code_by_year columns not found in mega_df")
        
        if verbose:
            print(f"    Name changes detected: {change_stats['name_changes']}")
            print(f"    Kind changes detected: {change_stats['kind_changes']}")
            print(f"    Level changes detected: {change_stats['level_changes']}")
            print(f"    Changes from notes: {change_stats['notes_changes']}")
        
        # ======================================================================
        # PASS 3: Build indices
        # ======================================================================
        if verbose:
            print("  Pass 3: Building indices...")
        
        for teryt_id, record in self._records.items():
            # Year index
            for year in record.years_valid:
                if year not in self._by_year:
                    self._by_year[year] = set()
                self._by_year[year].add(teryt_id)
            
            # Name index
            name_lower = record.name.lower() if record.name else ''
            if name_lower:
                if name_lower not in self._by_name:
                    self._by_name[name_lower] = set()
                self._by_name[name_lower].add(teryt_id)
            
            # Level index
            if record.level is not None:
                level = int(record.level)
                if level not in self._by_level:
                    self._by_level[level] = set()
                self._by_level[level].add(teryt_id)
            
            # Kind index
            if record.kind:
                if record.kind not in self._by_kind:
                    self._by_kind[record.kind] = set()
                self._by_kind[record.kind].add(teryt_id)
            
            # Voivodeship index
            if record.woj not in self._by_voivodeship:
                self._by_voivodeship[record.woj] = set()
            self._by_voivodeship[record.woj].add(teryt_id)
        
        self._built = True
        
        # Summary statistics
        units_with_changes = sum(1 for r in self._records.values() if r.has_changes)
        units_with_level_changes = sum(1 for r in self._records.values() if r.past_levels)
        units_with_kind_changes = sum(1 for r in self._records.values() if r.past_kinds)
        
        if verbose:
            print(f"\n✓ Database built successfully!")
            print(f"  Total records: {len(self._records):,}")
            print(f"  Year range: {self._year_range[0]} - {self._year_range[1]}")
            print(f"  Voivodeships: {len(self._by_level.get(2, set()))}")
            print(f"  Powiats: {len(self._by_level.get(5, set()))}")
            print(f"  Gminas: {len(self._by_level.get(6, set()))}")
            print(f"  Units with changes: {units_with_changes}")
            print(f"  Units with level changes: {units_with_level_changes}")
            print(f"  Units with kind changes: {units_with_kind_changes}")
    
    def load_geometries(self, gdf_dict: Dict[str, gpd.GeoDataFrame], 
                        teryt_column: str = 'teryt',
                        clip_to_poland: bool = True,
                        verbose: bool = True):
        """
        Load geometries from a dictionary of GeoDataFrames.
        
        FIXED in v3.0: Now handles pre-2012 files with 'obszar' column containing
        NUTS-like codes that need to be converted to TERYT codes.
        
        Parameters:
        - gdf_dict: Dictionary mapping year/key strings to GeoDataFrames
        - teryt_column: Column name containing TERYT codes (tries 'teryt' then 'jpt_kod_je')
        - clip_to_poland: If True and Poland boundary is set, clip geometries to land
        - verbose: Print progress
        
        NEW in v3.1: This method now ONLY stores geometries in _geometries dict.
        It does NOT assign geometries to records. Use assign_geometries() methods instead.
        """
        if verbose:
            print("Loading geometries into database (v3.1 - storage only)...")
            if clip_to_poland and self._poland_boundary is not None:
                print("  (Clipping to Poland boundary enabled)")
        
        for key, gdf in gdf_dict.items():
            # Extract year from key
            match = re.search(r'\d{4}', str(key))
            if not match:
                if verbose:
                    print(f"  Skipping {key}: no year found")
                continue
            
            year = int(match.group(0))
            
            if verbose:
                print(f"  Processing {key} (year {year})...")
            
            # Ensure CRS
            if gdf.crs is None:
                gdf = gdf.set_crs(self._crs)
            elif str(gdf.crs) != self._crs:
                gdf = gdf.to_crs(self._crs)
            
            # Standardize column names to lowercase for consistent matching
            gdf = gdf.copy()
            gdf.columns = [col.lower() for col in gdf.columns]
            
            # Find TERYT column or create it from 'obszar' (pre-2012 files)
            actual_teryt_col = None
            needs_conversion = False
            
            # Check for standard TERYT columns first
            for col in [teryt_column.lower(), 'teryt', 'jpt_kod_je']:
                if col in gdf.columns:
                    actual_teryt_col = col
                    break
            
            # If no TERYT column found, check for 'obszar' (NUTS-like codes in pre-2012 files)
            if actual_teryt_col is None and 'obszar' in gdf.columns:
                actual_teryt_col = 'obszar'
                needs_conversion = True
                if verbose:
                    print(f"    Using 'obszar' column (will convert NUTS codes to TERYT)")
            
            if actual_teryt_col is None:
                if verbose:
                    print(f"    Warning: No TERYT or 'obszar' column found")
                    print(f"    Available columns: {list(gdf.columns)}")
                # Store anyway but without teryt column
                self._geometries[year] = gdf
                continue
            
            # Create standardized 'teryt_id' column
            if needs_conversion:
                gdf['teryt_id'] = gdf[actual_teryt_col].apply(nuts_code_to_teryt)
            else:
                gdf['teryt_id'] = gdf[actual_teryt_col].apply(lambda x: str(x).zfill(7) if pd.notna(x) else '0000000')
            
            # Handle 6-digit codes (append 0)
            gdf['teryt_id'] = gdf['teryt_id'].apply(lambda x: x + '0' if len(x) == 6 else x)
            
            # Handle 8-digit codes (truncate last digit)
            gdf['teryt_id'] = gdf['teryt_id'].apply(lambda x: x[:-1] if len(x) == 8 else x)
            
            # Optionally clip geometries to Poland boundary
            if clip_to_poland and self._poland_boundary is not None:
                gdf['geometry'] = gdf['geometry'].apply(
                    lambda g: safe_clip_geometry(g, self._poland_boundary)
                )
            
            # Store the prepared GeoDataFrame
            self._geometries[year] = gdf
            
            if verbose:
                print(f"    Stored {len(gdf)} geometries for year {year}")
        
        if verbose:
            print(f"\n  ✓ Geometry data stored for years: {sorted(self._geometries.keys())}")
            print(f"    NOTE: Use assign_geometries() methods to assign geometries to records")
    
    def load_poland_shape(self, gadm_path: Union[str, Path] = None, verbose: bool = True):
        """
        Load Poland shape from GADM shapefile for geometry clipping.
        
        This should be called BEFORE load_geometries() to enable clipping.
        
        Parameters:
        - gadm_path: Path to GADM level 0 or 1 shapefile
        - verbose: Print progress
        """
        if verbose:
            print(f"Loading Poland shape from {gadm_path}...")
        
        if gadm_path is None:
            gadm_path = Path('/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/Geospatial/gadm41_POL_shp/gadm41_POL_0.shp')
        
        gdf = gpd.read_file(gadm_path)
        gdf = gdf.to_crs("EPSG:2180")  # Polish projected CRS
        self.set_poland_boundary(gdf)
        self.set_poland_gdf(gdf)
        
        if verbose:
            print("  ✓ Poland shape loaded and set for clipping")
    
    def link_children_to_parents(self, verbose: bool = True):
        """
        Link child units to their parents based on TERYT hierarchy.
        
        This populates the parent_id and children_ids fields in each record.
        
        Parameters:
        - verbose: Print progress
        """
        if verbose:
            print("Linking child units to their parents...")
        
        # Create a fake root parent for top-level units (województwa) with ID '0000000'
        root_parent = TERYTRecord(
            teryt_id='0000000',
            name='Poland',
            name_dod='',
            level=0,
            kind='country'
        )
        self._records['0000000'] = root_parent
        all_teryt_ids = pd.DataFrame(list(self._records.keys()))
        
        i = 0
        for teryt_id, record in self._records.items():
            if len(teryt_id) != 7 or teryt_id == '0000000':
                continue
            
            # Determine parent ID based on TERYT structure
            woj = teryt_id[:2]
            pow = teryt_id[2:4]
            gmi = teryt_id[4:6]
            rodz = teryt_id[6]
            level = record.level
            
            parent_id = None
            children_ids = pd.DataFrame()
            
            if level == 2:  # Voivodeship
                parent_id = '0000000'  # Root parent
                # Children = powiats: same woj, gmi='00', but NOT the voivodeship itself
                mask = (all_teryt_ids[0].str[:2] == woj) & \
                       (all_teryt_ids[0].str[4:6] == '00') & \
                       (all_teryt_ids[0].str[2:4] != '00')
                children_ids = all_teryt_ids[mask]
            elif level == 5:  # Powiat
                parent_id = woj + '00000'
                # Children = gminas (rodz 1,2,3) under this powiat
                mask = (all_teryt_ids[0].str[:4] == woj + pow) & \
                       (all_teryt_ids[0].str[6].isin(['1', '2', '3']))
                children_ids = all_teryt_ids[mask]
            elif level == 6:  # Gmina
                if rodz in ['1', '2', '3']:  # Regular gmina
                    parent_id = woj + pow + '000'
                    # Children = sub-parts (rodz 4,5,8,9) with same woj+pow+gmi
                    mask = (all_teryt_ids[0].str[:6] == woj + pow + gmi) & \
                           (all_teryt_ids[0].str[6].isin(['4', '5', '8', '9']))
                    children_ids = all_teryt_ids[mask]
                elif rodz in ['4', '5']:  # City/rural part of gmina miejsko-wiejska
                    parent_id = woj + pow + gmi + '3'
                else:  # Parts of cities with powiat rights (dzielnice, delegatury)
                    parent_id = woj + pow + gmi + '1'
            
            record.set_parent(parent_id)
            child_list = list(children_ids[0].values) if not children_ids.empty else []
            record.set_children(child_list)
            i += 1 
        
        if verbose:
            print(f"  ✓ Linked children to parents for {i} records")
    
    # ==========================================================================
    # SEARCH METHODS
    # ==========================================================================
    
    def get_by_teryt_id(self, teryt_id: str) -> Optional[TERYTRecord]:
        """Get a record by its TERYT ID."""
        teryt_id = str(teryt_id).zfill(7)
        return self._records.get(teryt_id)
    
    def get_unit_info(self, teryt_id: str) -> Optional[dict]:
        """Get detailed information about a unit by its TERYT ID."""
        record = self.get_by_teryt_id(teryt_id)
        return record.to_dict() if record else "Unit not found"
    
    def search_by_name(self, name: str, exact: bool = False) -> List[TERYTRecord]:
        """Search for divisions by name."""
        name_lower = name.lower()
        results = []
        
        if exact:
            if name_lower in self._by_name:
                results = [self._records[tid] for tid in self._by_name[name_lower]]
        else:
            for stored_name, teryt_ids in self._by_name.items():
                if name_lower in stored_name:
                    results.extend(self._records[tid] for tid in teryt_ids)
        
        return results
    
    def get_divisions_by_year(self, year: int, level: Optional[int] = None,
                               kind: Optional[str] = None,
                               exclude_subdivisions_and_districts: bool = True,
                               exclude_subdivisions: bool = True,
                               exclude_districts: bool = True) -> List[TERYTRecord]:
        """Get all divisions valid in a specific year."""
        if year not in self._by_year:
            return []
        
        teryt_ids = self._by_year[year].copy()
        
        if level is not None:
            level_ids = self._by_level.get(level, set())
            teryt_ids = teryt_ids & level_ids
        
        if kind is not None:
            kind_ids = self._by_kind.get(kind, set())
            teryt_ids = teryt_ids & kind_ids
        
        records = [self._records[tid] for tid in teryt_ids]
        
        if exclude_subdivisions_and_districts:
            records = [r for r in records if r.rodz not in RODZ_SUB_DIVISIONS_AND_DISTRICTS]
            return records
        
        if exclude_subdivisions:
            records = [r for r in records if r.rodz not in RODZ_SUB_DIVISIONS]
            return records

        if exclude_districts:
            records = [r for r in records if r.rodz not in RODZ_DISTRICTS]
            return records
        
        return records
    
    def get_snapshot(self, year: int) -> List[TERYTRecord]:
        """
        Get all administrative unit records that were valid in a specific year.
        
        Unlike get_divisions_by_year(), this method returns ALL records without
        filtering by level, kind, or excluding subdivisions. Use this for
        geometry assignment operations that need to process all units.
        
        Parameters:
        - year: The year to get divisions for
        
        Returns:
        - List of TERYTRecord objects valid in that year
        """
        if year not in self._by_year:
            return []
        
        teryt_ids = self._by_year[year]
        return [self._records[tid] for tid in teryt_ids]
    
    def get_divisions_by_level(self, level: int) -> List[TERYTRecord]:
        """Get all divisions at a specific administrative level."""
        teryt_ids = self._by_level.get(level, set())
        return [self._records[tid] for tid in teryt_ids]
    
    def get_gminas_in_voivodeship(self, woj_code: str, year: Optional[int] = None) -> List[TERYTRecord]:
        """Get all gminas in a voivodeship."""
        woj_code = str(woj_code).zfill(2)
        
        if woj_code not in self._by_voivodeship:
            return []
        
        teryt_ids = self._by_voivodeship[woj_code].copy()
        gmina_ids = self._by_level.get(LEVEL_GMINA, set())
        teryt_ids = teryt_ids & gmina_ids
        
        if year is not None:
            year_ids = self._by_year.get(year, set())
            teryt_ids = teryt_ids & year_ids
        
        records = [self._records[tid] for tid in teryt_ids]
        records = [r for r in records if r.rodz not in RODZ_SUB_DIVISIONS_AND_DISTRICTS]
        
        return records
    
    def get_changed_units(self, year_from: Optional[int] = None, 
                          year_to: Optional[int] = None,
                          change_type: Optional[str] = None) -> List[TERYTRecord]:
        """
        Get units that experienced changes in a time period.
        
        FIXED in v2.0: Now properly checks both:
        - has_changes flag
        - actual change records with year filtering
        
        Parameters:
        - year_from: Start year (inclusive), None for no lower bound
        - year_to: End year (inclusive), None for no upper bound
        - change_type: Filter by change type ('name_change', 'kind_change', 'level_change', 'reform')
        """
        results = []
        
        for record in self._records.values():
            # Skip if no changes at all
            if not record.has_changes and not record.changes:
                continue
            
            # If no time filter, return all changed units
            if year_from is None and year_to is None and change_type is None:
                results.append(record)
                continue
            
            # Check if any change falls within the time range
            has_relevant_change = False
            
            for change in record.changes:
                change_year = change.get('year')
                if change_year is None:
                    change_year = change.get('when_changed')
                if change_year is None:
                    continue
                
                # Handle numpy types
                if hasattr(change_year, 'item'):
                    change_year = change_year.item()
                try:
                    change_year = int(change_year)
                except (ValueError, TypeError):
                    continue
                
                # Check year range
                if year_from is not None and change_year < year_from:
                    continue
                if year_to is not None and change_year > year_to:
                    continue
                
                # Check change type
                if change_type is not None and change.get('type') != change_type:
                    continue
                
                has_relevant_change = True
                break
            
            # Also check past_* lists for changes in time range (if no type filter)
            if not has_relevant_change and change_type is None:
                # Check past_names
                for _, year_until in record.past_names:
                    check_year = year_until + 1  # Change happened in the next year
                    if (year_from is None or check_year >= year_from) and \
                       (year_to is None or check_year <= year_to):
                        has_relevant_change = True
                        break
                
                # Check past_levels
                if not has_relevant_change:
                    for _, year_until in record.past_levels:
                        check_year = year_until + 1
                        if (year_from is None or check_year >= year_from) and \
                           (year_to is None or check_year <= year_to):
                            has_relevant_change = True
                            break
                
                # Check past_kinds
                if not has_relevant_change:
                    for _, year_until in record.past_kinds:
                        check_year = year_until + 1
                        if (year_from is None or check_year >= year_from) and \
                           (year_to is None or check_year <= year_to):
                            has_relevant_change = True
                            break
            
            if has_relevant_change:
                results.append(record)
        
        return results
    
    def get_units_with_level_changes(self) -> List[TERYTRecord]:
        """Get all units that changed their administrative level."""
        return [r for r in self._records.values() if r.past_levels]
    
    def get_units_with_kind_changes(self) -> List[TERYTRecord]:
        """Get all units that changed their kind (urban/rural/etc)."""
        return [r for r in self._records.values() if r.past_kinds]
    
    # ==========================================================================
    # GEOMETRY METHODS
    # ==========================================================================
    
    def get_geometry(self, teryt_id: str, year: Optional[int] = None):
        """Get geometry for a specific TERYT ID."""
        record = self.get_by_teryt_id(teryt_id)
        
        if record is None:
            return None
        
        if record.has_geometry:
            return record.geometry
        
        return self._find_best_geometry(teryt_id, year)
    
    def _find_best_geometry(self, teryt_id: str, target_year: Optional[int] = None):
        """Find the best available geometry for a TERYT ID."""
        teryt_id = str(teryt_id).zfill(7)
        short_teryt = teryt_id[:6]
        
        available = []
        
        for year, gdf in self._geometries.items():
            # Find TERYT column
            teryt_col = None
            for col in ['teryt', 'jpt_kod_je', 'TERYT', 'JPT_KOD_JE']:
                if col in gdf.columns:
                    teryt_col = col
                    break
            if teryt_col is None:
                continue
            
            mask = (gdf[teryt_col].astype(str).str.zfill(7) == teryt_id) | \
                   (gdf[teryt_col].astype(str).str[:6] == short_teryt)
            
            matches = gdf[mask]
            if len(matches) > 0:
                geom = matches.iloc[0].geometry
                # Optionally clip
                if self._poland_boundary is not None:
                    geom = safe_clip_geometry(geom, self._poland_boundary)
                available.append((year, geom))
        
        if not available:
            return None
        
        if target_year is not None:
            available.sort(key=lambda x: abs(x[0] - target_year))
        else:
            available.sort(key=lambda x: -x[0])  # Most recent first
        
        geom_year = available[0][0]

        # Update the record geometry year 
        record = self.get_by_teryt_id(teryt_id)
        if record is not None:
            record.geometry_year = geom_year
        
        return available[0][1]
    
    # ==========================================================================
    # GEOMETRY ASSIGNMENT METHODS (NEW in v3.1)
    # ==========================================================================
    
    def assign_geometries(self, year: int, level: Optional[int] = None,
                          verbose: bool = True) -> dict:
        """
        Assign geometries to records for a specific year.
        
        Only assigns geometries where the teryt_id matches exactly in the 
        geometry file for that year. Does NOT try to find geometries from
        other years for unchanged units.
        
        Parameters:
        - year: The year for which to assign geometries
        - level: Optional level filter (2=voivodeship, 5=powiat, 6=gmina)
        - verbose: Print progress
        
        Returns:
        - dict with assignment statistics
        """
        if year not in self._geometries:
            if verbose:
                print(f"  No geometry data available for year {year}")
            return {'assigned': 0, 'not_found': 0, 'error': f'No geometry for year {year}'}
        
        if verbose:
            print(f"Assigning geometries for year {year}...")
        
        gdf = self._geometries[year]
        
        # Ensure teryt_id column exists
        if 'teryt_id' not in gdf.columns:
            if verbose:
                print(f"  Warning: No teryt_id column in geometry for year {year}")
            return {'assigned': 0, 'not_found': 0, 'error': 'No teryt_id column'}
        
        # Get snapshot of units valid in this year
        snapshot = self.get_snapshot(year)
        if level is not None:
            snapshot = [r for r in snapshot if r.level == level]
        
        # Build lookup from teryt_id to geometry
        geom_lookup = {}
        for _, row in gdf.iterrows():
            tid = str(row['teryt_id']).zfill(7)
            geom_lookup[tid] = row.geometry
            # Also store 6-digit version for flexible matching
            short_tid = tid[:6]
            if short_tid not in geom_lookup:
                geom_lookup[short_tid] = row.geometry
        
        assigned = 0
        not_found = 0
        already_has = 0
        
        for record in snapshot:
            if record.has_geometry:
                already_has += 1
                continue
            
            teryt_id = record.teryt_id
            geom = geom_lookup.get(teryt_id)
            
            # Try 6-digit match if exact match fails
            if geom is None:
                geom = geom_lookup.get(teryt_id[:6])
            
            if geom is not None:
                record.set_geometry(geom, year)
                assigned += 1
            else:
                not_found += 1
        
        if verbose:
            print(f"  ✓ Assigned: {assigned}")
            print(f"  ✓ Already had geometry: {already_has}")
            print(f"  ✗ Not found: {not_found}")
        
        return {'assigned': assigned, 'not_found': not_found, 'already_has': already_has}
    
    def assign_missing_geometries(self, year: int, level: Optional[int] = None,
                                   verbose: bool = True) -> dict:
        """
        For units without geometry that haven't changed, find matching geometries
        from other years by comparing across years.
        
        For unchanged units (same teryt_id across years), if a geometry exists
        in another year, assign it. Prefers geometry from year closest to target.
        
        Parameters:
        - year: The year for which to find missing geometries
        - level: Optional level filter
        - verbose: Print progress
        
        Returns:
        - dict with assignment statistics
        """
        if verbose:
            print(f"Finding missing geometries for year {year}...")
        
        # Get units valid in this year that don't have geometry
        snapshot = self.get_snapshot(year)
        if level is not None:
            snapshot = [r for r in snapshot if r.level == level]
        
        missing = [r for r in snapshot if not r.has_geometry]
        
        if verbose:
            print(f"  Units without geometry: {len(missing)}")
        
        if len(missing) == 0:
            return {'assigned': 0, 'total_missing': 0}
        
        # For each missing unit, check if it's unchanged (same teryt_id in other years)
        assigned = 0
        candidates_found = 0
        
        for record in missing:
            teryt_id = str(record.teryt_id).zfill(7)
            short_tid = teryt_id[:6]
            
            # Check if this unit has any changes
            if record.has_changes:
                # For units with changes, we need impute_geometries_past_tid instead
                continue
            
            # Collect all matching geometries from all years
            all_candidates = []
            
            for geom_year, gdf in self._geometries.items():
                if 'teryt_id' not in gdf.columns:
                    continue
                
                # Match using both 7-digit (full) and 6-digit (short) teryt_id format
                gdf_teryt = gdf['teryt_id'].astype(str).str.zfill(7)
                mask = (gdf_teryt == teryt_id) | (gdf_teryt.str[:6] == short_tid)
                matches = gdf[mask]
                
                if len(matches) > 0:
                    geom = matches.iloc[0].geometry
                    if geom is not None and not geom.is_empty:
                        all_candidates.append((geom_year, geom))
            
            if all_candidates:
                # Sort by year proximity to target (prefer nearest year)
                all_candidates.sort(key=lambda x: abs(x[0] - year))
                best_year, best_geom = all_candidates[0]
                
                if verbose and len(all_candidates) > 1:
                    print(f"  Found {len(all_candidates)} candidates for {record.teryt_id}, using year {best_year}")
                record.set_geometry(best_geom, best_year)
                record.geometry_notes = f"assigned_from_year_{best_year}"
                assigned += 1
                candidates_found += 1
        
        if verbose:
            print(f"  ✓ Assigned from other years: {assigned}")
            still_missing = sum(1 for r in snapshot if not r.has_geometry)
            print(f"  Remaining without geometry: {still_missing}")
        
        return {'assigned': assigned, 'total_missing': len(missing), 'still_missing': len(missing) - assigned}
    
    def impute_geometries_past_tid(self, year: int, level: Optional[int] = None,
                                    verbose: bool = True) -> dict:
        """
        For units without geometry, use code_by_year to find affiliated teryt_ids
        and search for geometry candidates.
        
        Uses the code_by_year attribute to find which teryt_id this unit had in
        different years, then searches for matching geometries.
        
        Parameters:
        - year: The year for which to find missing geometries
        - level: Optional level filter
        - verbose: Print progress
        
        Returns:
        - dict with imputation statistics
        """
        if verbose:
            print(f"Imputing geometries using past teryt_ids for year {year}...")
        
        # Get units valid in this year that don't have geometry
        snapshot = self.get_snapshot(year)
        if level is not None:
            snapshot = [r for r in snapshot if r.level == level]
        
        missing = [r for r in snapshot if not r.has_geometry]
        
        if verbose:
            print(f"  Units without geometry: {len(missing)}")
        
        if len(missing) == 0:
            return {'candidates_found': 0, 'imputed': 0, 'total_missing': 0}
        
        candidates_found = 0
        imputed = 0
        
        for record in missing:
            # Collect all teryt_ids to search for - include current, historical, and code_by_year
            search_ids = set()
            
            # Add current teryt_id
            search_ids.add(str(record.teryt_id).zfill(7))
            
            # Add historical codes
            if record.historical_codes:
                for code in record.historical_codes:
                    search_ids.add(str(code).zfill(7))
            
            # Add codes from code_by_year
            if record.code_by_year:
                for code in record.code_by_year.values():
                    search_ids.add(str(code).zfill(7))
            
            # Search for geometries of all affiliated teryt_ids
            candidate_geoms = []
            
            for geom_year, gdf in self._geometries.items():
                if 'teryt_id' not in gdf.columns:
                    continue
                
                # Normalize teryt_id column
                gdf_teryt = gdf['teryt_id'].astype(str).str.zfill(7)
                
                for search_id in search_ids:
                    short_id = search_id[:6]
                    
                    # Match using both 7-digit (full) and 6-digit (short) teryt_id format
                    mask = (gdf_teryt == search_id) | (gdf_teryt.str[:6] == short_id)
                    matches = gdf[mask]
                    
                    if len(matches) > 0:
                        geom = matches.iloc[0].geometry
                        if geom is not None and not geom.is_empty:
                            candidate_geoms.append({
                                'geometry': geom,
                                'source_year': geom_year,
                                'source_teryt_id': search_id,
                            })

            if candidate_geoms:
                candidates_found += 1
                
                # Remove duplicates (same geometry from multiple searches)
                # by using geometry's wkt representation as key
                seen_geoms = {}
                for cand in candidate_geoms:
                    geom_key = cand['geometry'].wkt[:100]  # Use first 100 chars of wkt as key
                    if geom_key not in seen_geoms:
                        seen_geoms[geom_key] = cand
                    else:
                        # Keep the one from year closer to target
                        existing = seen_geoms[geom_key]
                        if abs(cand['source_year'] - year) < abs(existing['source_year'] - year):
                            seen_geoms[geom_key] = cand
                
                unique_candidates = list(seen_geoms.values())
                
                # Sort by year proximity to target (prefer nearest year)
                unique_candidates.sort(key=lambda x: abs(x['source_year'] - year))
                
                best = unique_candidates[0]
                record.geometry_best_candidate = best['geometry']
                record.geometry_notes = f"candidate_from_{best['source_teryt_id']}_year_{best['source_year']}"
                
                # Assign directly if we found a good match
                record.set_geometry(best['geometry'], best['source_year'])
                record.geometry_notes = f"imputed_from_{best['source_teryt_id']}_year_{best['source_year']}"
                imputed += 1
        
        if verbose:
            print(f"  ✓ Candidates found: {candidates_found}")
            print(f"  ✓ Directly imputed: {imputed}")
            still_missing = sum(1 for r in snapshot if not r.has_geometry)
            print(f"  Remaining without geometry: {still_missing}")
        
        return {
            'candidates_found': candidates_found, 
            'imputed': imputed, 
            'total_missing': len(missing),
            'still_missing': len(missing) - imputed
        }
    
    def impute_from_best_candidates(self, condition: Optional[str] = None,
                                     verbose: bool = True) -> dict:
        """
        Fill geometry from geometry_best_candidate based on geometry_notes.
        
        Parameters:
        - condition: Optional string to filter by geometry_notes (e.g., 'candidate_from')
        - verbose: Print progress
        
        Returns:
        - dict with imputation statistics
        """
        if verbose:
            print("Imputing geometries from best candidates...")
        
        imputed = 0
        skipped = 0
        
        for record in self._records.values():
            if record.has_geometry:
                continue
            
            if record.geometry_best_candidate is None:
                continue
            
            # Check condition if specified
            if condition is not None:
                if record.geometry_notes is None or condition not in record.geometry_notes:
                    skipped += 1
                    continue
            
            # Assign the best candidate
            record.geometry = record.geometry_best_candidate
            record.geometry_notes = record.geometry_notes.replace('candidate_from', 'imputed_from') if record.geometry_notes else 'imputed'
            imputed += 1
        
        if verbose:
            print(f"  ✓ Imputed: {imputed}")
            if skipped > 0:
                print(f"  Skipped (condition not met): {skipped}")
        
        return {'imputed': imputed, 'skipped': skipped}
    
    def country_shape_check(self, year: int, level: int = 6,
                            verbose: bool = True) -> dict:
        """
        Check geometry coverage by overlaying with Poland boundary.
        
        Counts holes (areas inside Poland not covered by any unit geometry)
        and compares with count of units missing geometries.
        
        Parameters:
        - year: Year to check
        - level: Administrative level (default 6 = gmina)
        - verbose: Print progress
        
        Returns:
        - dict with coverage statistics
        """
        if self._poland_boundary is None:
            if verbose:
                print("  No Poland boundary loaded. Use load_poland_shape() first.")
            return {'error': 'No Poland boundary loaded'}
        
        if verbose:
            print(f"Checking geometry coverage for year {year}, level {level}...")
        
        # Get all units for this year and level
        snapshot = self.get_snapshot(year)
        snapshot = [r for r in snapshot if r.level == level]
        
        # Filter out subdivisions
        snapshot = [r for r in snapshot if r.rodz not in RODZ_SUB_DIVISIONS_AND_DISTRICTS]
        
        total_units = len(snapshot)
        units_with_geom = sum(1 for r in snapshot if r.has_geometry)
        units_missing_geom = total_units - units_with_geom
        
        if verbose:
            print(f"  Total units: {total_units}")
            print(f"  With geometry: {units_with_geom}")
            print(f"  Missing geometry: {units_missing_geom}")
        
        # Create union of all geometries
        geometries = [r.geometry for r in snapshot if r.has_geometry and r.geometry is not None]
        
        if not geometries:
            if verbose:
                print("  No geometries to check")
            return {
                'total_units': total_units,
                'units_with_geom': 0,
                'units_missing_geom': units_missing_geom,
                'coverage_area': 0,
                'uncovered_area': self._poland_boundary.area if self._poland_boundary else 0,
                'coverage_percent': 0
            }
        
        try:
            from shapely.ops import unary_union
            
            # Union all geometries
            all_geoms_union = unary_union(geometries)
            
            # Calculate coverage
            poland_area = self._poland_boundary.area
            coverage_area = all_geoms_union.area
            
            # Find uncovered area (holes)
            uncovered = self._poland_boundary.difference(all_geoms_union)
            uncovered_area = uncovered.area if uncovered else 0
            
            coverage_percent = (coverage_area / poland_area) * 100 if poland_area > 0 else 0
            
            # Count distinct uncovered regions (holes)
            n_holes = 0
            if uncovered and not uncovered.is_empty:
                if uncovered.geom_type == 'MultiPolygon':
                    n_holes = len(uncovered.geoms)
                elif uncovered.geom_type == 'Polygon':
                    n_holes = 1
                elif uncovered.geom_type == 'GeometryCollection':
                    n_holes = sum(1 for g in uncovered.geoms if g.geom_type in ['Polygon', 'MultiPolygon'])
            
            if verbose:
                print(f"  Coverage: {coverage_percent:.2f}%")
                print(f"  Uncovered area: {uncovered_area/1e6:.2f} km²")
                print(f"  Number of holes/gaps: {n_holes}")
                if n_holes != units_missing_geom:
                    print(f"  ⚠ Mismatch: {n_holes} holes vs {units_missing_geom} missing units")
            
            return {
                'total_units': total_units,
                'units_with_geom': units_with_geom,
                'units_missing_geom': units_missing_geom,
                'coverage_area_km2': coverage_area / 1e6,
                'uncovered_area_km2': uncovered_area / 1e6,
                'coverage_percent': coverage_percent,
                'n_holes': n_holes,
                'holes_match_missing': n_holes == units_missing_geom,
                'uncovered_geometry': uncovered
            }
            
        except Exception as e:
            if verbose:
                print(f"  Error during coverage check: {e}")
            return {
                'error': str(e),
                'total_units': total_units,
                'units_with_geom': units_with_geom,
                'units_missing_geom': units_missing_geom
            }
    
    def to_geodataframe(self, year: Optional[int] = None, 
                        level: Optional[int] = None,
                        kind: Optional[str] = None,
                        exclude_subdivisions_and_districts: bool = True,
                        exclude_subdivisions: bool = True,
                        exclude_districts: bool = True,
                        include_all_attributes: bool = True,
                        only_with_geometry: bool = False) -> gpd.GeoDataFrame:
        """
        Convert database records to a GeoDataFrame.
        
        Parameters:
        - year: Filter by specific year
        - level: Filter by administrative level
        - kind: Filter by kind
        - exclude_subdivisions: Exclude RODZ 4,5
        - include_all_attributes: Include all record attributes
        - only_with_geometry: Only include records with geometry
        """
        if year is not None:
            records = self.get_divisions_by_year(year, level=level, kind=kind,
                                                  exclude_subdivisions=exclude_subdivisions)
        else:
            records = list(self._records.values())
            if level is not None:
                records = [r for r in records if r.level == level]
            if kind is not None:
                records = [r for r in records if r.kind == kind]
            if exclude_subdivisions_and_districts:
                records = [r for r in records if r.rodz not in RODZ_SUB_DIVISIONS_AND_DISTRICTS]
            elif exclude_subdivisions:
                records = [r for r in records if r.rodz not in RODZ_SUB_DIVISIONS]
            elif exclude_districts:
                records = [r for r in records if r.rodz not in RODZ_DISTRICTS]
        
        if only_with_geometry:
            records = [r for r in records if r.has_geometry]
        
        data = []
        geometries = []
        
        for record in records:
            row_data = {
                'teryt_id': record.teryt_id,
                'name': record.name,
                'level': record.level,
                'kind': record.kind,
                'woj': record.woj,
                'pow': record.pow,
                'gmi': record.gmi,
                'rodz': record.rodz,
            }
            
            if include_all_attributes:
                row_data.update({
                    'name_dod': record.name_dod,
                    'first_year': record.first_year,
                    'last_year': record.last_year,
                    'has_geometry': record.has_geometry,
                    'geometry_year': record.geometry_year,
                    'n_changes': record.n_changes,
                    'has_changes': record.has_changes,
                    'past_names': str(record.past_names) if record.past_names else None,
                    'past_teryt_ids': str(record.past_teryt_ids) if record.past_teryt_ids else None,
                    'past_levels': str(record.past_levels) if record.past_levels else None,
                    'past_kinds': str(record.past_kinds) if record.past_kinds else None,
                })
            
            data.append(row_data)
            geom = self.get_geometry(record.teryt_id, year)
            geometries.append(geom)
        
        return gpd.GeoDataFrame(data, geometry=geometries, crs=self._crs)
    
    def merge_to_level(self, year: int, target_level: int) -> gpd.GeoDataFrame:
        """
        Merge gmina geometries up to a higher administrative level.
        
        Parameters:
        - year: Year for which to get gminas
        - target_level: Target level (2 for voivodeship, 5 for powiat)
        """
        gdf = self.to_geodataframe(year=year, level=LEVEL_GMINA, exclude_subdivisions=True)
        
        if len(gdf) == 0:
            return gpd.GeoDataFrame(columns=['teryt_id', 'name', 'geometry'], crs=self._crs)
        
        if target_level == LEVEL_VOIVODESHIP:
            gdf['group_id'] = gdf['woj']
        elif target_level == LEVEL_POWIAT:
            gdf['group_id'] = gdf['woj'] + gdf['pow']
        else:
            raise ValueError(f"Invalid target level: {target_level}. Use 2 or 5.")
        
        gdf_valid = gdf[gdf.geometry.notna()].copy()
        
        if len(gdf_valid) == 0:
            return gpd.GeoDataFrame(columns=['teryt_id', 'name', 'geometry'], crs=self._crs)
        
        merged = gdf_valid.dissolve(by='group_id', as_index=False)
        
        merged_data = []
        for _, row in merged.iterrows():
            group_id = row['group_id']
            
            if target_level == LEVEL_VOIVODESHIP:
                parent_teryt = group_id + "00000"
            else:
                parent_teryt = group_id + "000"
            
            parent_record = self.get_by_teryt_id(parent_teryt)
            name = parent_record.name if parent_record else f"Unknown ({group_id})"
            
            merged_data.append({
                'teryt_id': parent_teryt,
                'name': name,
                'level': target_level,
                'geometry': row.geometry
            })
        
        return gpd.GeoDataFrame(merged_data, crs=self._crs)
    
    # ==========================================================================
    # OVERLAY OPERATIONS
    # ==========================================================================
    
    def overlay_gminas_to_regions(self, gminas_gdf: gpd.GeoDataFrame,
                                   regions_gdf: gpd.GeoDataFrame,
                                   region_id_column: str = 'region_id',
                                   method: str = 'centroid') -> gpd.GeoDataFrame:
        """
        Overlay gminas onto regions and assign each gmina to a region.
        
        Parameters:
        - gminas_gdf: GeoDataFrame with gmina geometries
        - regions_gdf: GeoDataFrame with region boundaries
        - region_id_column: Column in regions_gdf with region identifier
        - method: 'centroid' (point in polygon) or 'area' (max overlap)
        """
        result = gminas_gdf.copy()
        result['assigned_region'] = None
        
        if regions_gdf.crs != gminas_gdf.crs:
            regions_gdf = regions_gdf.to_crs(gminas_gdf.crs)
        
        for idx, gmina_row in result.iterrows():
            geom = gmina_row.geometry
            
            if geom is None or geom.is_empty:
                continue
            
            if method == 'centroid':
                centroid = geom.centroid
                for _, region_row in regions_gdf.iterrows():
                    if region_row.geometry is not None and region_row.geometry.contains(centroid):
                        result.at[idx, 'assigned_region'] = region_row[region_id_column]
                        break
                        
            elif method == 'area':
                max_area = 0
                best_region = None
                for _, region_row in regions_gdf.iterrows():
                    try:
                        if region_row.geometry is None:
                            continue
                        intersection = geom.intersection(region_row.geometry)
                        area = intersection.area
                        if area > max_area:
                            max_area = area
                            best_region = region_row[region_id_column]
                    except:
                        continue
                result.at[idx, 'assigned_region'] = best_region
        
        return result
    
    def assign_gminas_to_pre1999_voivodeships(self, year: int,
                                               pre1999_boundaries_gdf: gpd.GeoDataFrame,
                                               voivodeship_name_col: str = 'name',
                                               voivodeship_id_col: str = 'old_woj_id',
                                               method: str = 'centroid',
                                               update_records: bool = True,
                                               verbose: bool = True) -> gpd.GeoDataFrame:
        """
        Assign gminas from a given year to pre-1999 voivodeship boundaries.
        
        NEW in v3.0: Also updates the old_woj and old_woj_id attributes in 
        the database records if update_records=True.
        
        Parameters:
        - year: Year for gmina data
        - pre1999_boundaries_gdf: GeoDataFrame with 49 voivodeship boundaries
        - voivodeship_name_col: Column name for voivodeship names
        - voivodeship_id_col: Column name for voivodeship IDs (created if missing)
        - method: 'centroid' or 'area'
        - update_records: If True, update old_woj/old_woj_id in database records
        - verbose: Print progress
        """
        gminas_gdf = self.to_geodataframe(year=year, level=LEVEL_GMINA, 
                                           exclude_subdivisions=True,
                                           only_with_geometry=True)
        
        if verbose:
            print(f"Assigning {len(gminas_gdf)} gminas to pre-1999 voivodeships...")
        
        # Ensure the voivodeship ID column exists
        pre1999_copy = pre1999_boundaries_gdf.copy()
        if voivodeship_id_col not in pre1999_copy.columns:
            pre1999_copy[voivodeship_id_col] = range(1, len(pre1999_copy) + 1)
        
        # Create a mapping from name to ID
        name_to_id = dict(zip(pre1999_copy[voivodeship_name_col], pre1999_copy[voivodeship_id_col]))
        
        result = self.overlay_gminas_to_regions(
            gminas_gdf, 
            pre1999_copy,
            region_id_column=voivodeship_name_col,
            method=method
        )
        
        result = result.rename(columns={'assigned_region': 'pre1999_voivodeship'})
        
        # Add the voivodeship ID column
        result['pre1999_voivodeship_id'] = result['pre1999_voivodeship'].map(name_to_id)
        
        # Update database records if requested
        if update_records:
            updated = 0
            for _, row in result.iterrows():
                teryt_id = row['teryt_id']
                old_woj = row.get('pre1999_voivodeship')
                old_woj_id = row.get('pre1999_voivodeship_id')
                
                if teryt_id in self._records:
                    self._records[teryt_id].old_woj = old_woj
                    self._records[teryt_id].old_woj_id = old_woj_id
                    updated += 1
            
            if verbose:
                assigned = result['pre1999_voivodeship'].notna().sum()
                print(f"  ✓ Assigned {assigned} gminas to voivodeships")
                print(f"  ✓ Updated {updated} database records with old_woj/old_woj_id")
        
        return result
    
    def update_affilated_gminas_to_pre1999_voivodeships(self):
        """
        For gminas that have no old_woj_id set, but they have historical codes
        that match gminas with old_woj_id, assign them the same old_woj_id.
        """
        # Build mapping from teryt_id to old_woj_id for gminas that have it set
        teryt_to_old_woj = {}
        for record in self._records.values():
            if record.level == LEVEL_GMINA and record.rodz not in RODZ_SUB_DIVISIONS_AND_DISTRICTS and record.old_woj_id is not None:
                teryt_to_old_woj[str(record.teryt_id).zfill(7)] = [record.old_woj, record.old_woj_id]
        
        updated = 0
        for record in self._records.values():
            if record.level == LEVEL_GMINA and record.rodz not in RODZ_SUB_DIVISIONS_AND_DISTRICTS and record.old_woj_id is None:
                # Check historical codes for matches
                if record.historical_codes:
                    for code in record.historical_codes:
                        code_str = str(code).zfill(7)
                        if code_str in teryt_to_old_woj:
                            record.old_woj = teryt_to_old_woj[code_str][0]
                            record.old_woj_id = teryt_to_old_woj[code_str][1]
                            updated += 1
                            break
        
        print(f"Updated {updated} gminas with old_woj_id based on historical codes.")
    
    # ==========================================================================
    # EXPORT METHODS
    # ==========================================================================
    
    def to_dataframe(self, include_geometry: bool = False) -> pd.DataFrame:
        """Export all records to a pandas DataFrame."""
        data = []
        for record in self._records.values():
            row = record.to_dict()
            if include_geometry and record.geometry is not None:
                row['geometry_wkt'] = record.geometry.wkt
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_to_geopackage(self, filepath: Union[str, Path], 
                              year: Optional[int] = None,
                              level: Optional[int] = None):
        """Export database to a GeoPackage file."""
        gdf = self.to_geodataframe(year=year, level=level, only_with_geometry=True)
        gdf.to_file(filepath, driver='GPKG')
    
    def export_to_shapefile(self, filepath: Union[str, Path],
                             year: Optional[int] = None,
                             level: Optional[int] = None):
        """Export database to a Shapefile."""
        gdf = self.to_geodataframe(year=year, level=level, only_with_geometry=True)
        gdf.to_file(filepath, driver='ESRI Shapefile')
    
    # ==========================================================================
    # STATISTICS AND SUMMARY
    # ==========================================================================
    
    def summary(self) -> dict:
        """Get a summary of the database contents."""
        return {
            'total_records': len(self._records),
            'year_range': self._year_range,
            'voivodeships': len(self._by_level.get(2, set())),
            'powiats': len(self._by_level.get(5, set())),
            'gminas': len(self._by_level.get(6, set())),
            'records_with_geometry': sum(1 for r in self._records.values() if r.has_geometry),
            'records_with_changes': sum(1 for r in self._records.values() if r.has_changes),
            'records_with_level_changes': sum(1 for r in self._records.values() if r.past_levels),
            'records_with_kind_changes': sum(1 for r in self._records.values() if r.past_kinds),
            'geometry_years_available': sorted(self._geometries.keys()),
            'unique_kinds': list(self._by_kind.keys()),
            'unique_voivodeships': list(self._by_voivodeship.keys()),
            'has_poland_boundary': self._poland_boundary is not None,
        }
    
    def print_summary(self):
        """Print a formatted summary of the database."""
        s = self.summary()
        print("=" * 60)
        print("GeoTERYT Database Summary (v3.0)")
        print("=" * 60)
        print(f"Total records:           {s['total_records']:,}")
        print(f"Year range:              {s['year_range'][0]} - {s['year_range'][1]}")
        print("-" * 60)
        print("Administrative levels:")
        print(f"  Voivodeships (2):      {s['voivodeships']}")
        print(f"  Powiats (5):           {s['powiats']}")
        print(f"  Gminas (6):            {s['gminas']}")
        print("-" * 60)
        print("Change tracking:")
        print(f"  Records with changes:      {s['records_with_changes']}")
        print(f"  Records with level changes: {s['records_with_level_changes']}")
        print(f"  Records with kind changes:  {s['records_with_kind_changes']}")
        print("-" * 60)
        print("Geometry:")
        print(f"  Records with geometry: {s['records_with_geometry']}")
        print(f"  Geometry years:        {s['geometry_years_available']}")
        print(f"  Poland boundary set:   {s['has_poland_boundary']}")
        # Pre-1999 voivodeship info
        records_with_old_woj = sum(1 for r in self._records.values() if r.old_woj is not None)
        if records_with_old_woj > 0:
            print("-" * 60)
            print("Pre-1999 Voivodeship Overlay:")
            print(f"  Records with old_woj:  {records_with_old_woj}")
        # Data summary (NEW in v4.0)
        data_summary = self.get_data_summary()
        if data_summary['records_with_data'] > 0:
            print("-" * 60)
            print("Data:")
            print(f"  Records with data:     {data_summary['records_with_data']}")
            print(f"  Subjects loaded:       {data_summary['subjects']}")
            print(f"  Total data series:     {data_summary['total_data_series']:,}")
            print(f"  Total data points:     {data_summary['total_data_points']:,}")
        print("=" * 60)
    
    # ==========================================================================
    # DATA LOADING AND AGGREGATION METHODS (NEW in v4.0)
    # ==========================================================================
    
    @staticmethod
    def process_subject_data(df_demographic: pd.DataFrame, df_variables: pd.DataFrame,
                             subject_id: str) -> pd.DataFrame:
        """
        Process BDL or Census demographic data for a given subject ID.
        
        Filters, merges, expands and normalizes raw data into a flat
        DataFrame with one row per (unit, variable, year) observation.
        
        Works with both BDL time-series data (multiple years per row in
        values column) and Census cross-sectional data (single year per row).
        
        Parameters:
        - df_demographic: Raw data (bdl_demographic_data.csv or census data CSV)
        - df_variables: Variable metadata (bdl_variables_level6.csv or census_meta.csv)
        - subject_id: Subject ID to process (e.g. 'P2137')
        
        Returns:
        - DataFrame with columns including: nuts_id, name, variableId, subjectId,
          var_id, n1..n5 (non-constant only), year, val, attrId, teryt_id
        """
        # Filter by subjectId
        df_subject = df_demographic[df_demographic['subjectId'] == subject_id].copy()
        
        if df_subject.empty:
            warnings.warn(f"No data found for subject {subject_id}")
            return pd.DataFrame()
        
        # Get variable metadata for this subject
        # Use only columns that exist in the metadata
        variable_ids = df_subject['variableId'].unique()
        n_cols = [c for c in ['n1', 'n2', 'n3', 'n4', 'n5'] if c in df_variables.columns]
        subset_cols = ['id'] + n_cols
        df_variables_subset = df_variables[df_variables['id'].isin(variable_ids)][subset_cols]
        
        # Merge subject data with variables
        df_merged = pd.merge(
            df_subject, df_variables_subset,
            left_on='variableId', right_on='id', how='left'
        )
        
        # Remove constant columns (n-columns that have only one unique value)
        for col in n_cols:
            if col in df_merged.columns and df_merged[col].nunique() <= 1:
                df_merged = df_merged.drop(columns=[col])
        
        # Parse values column (stored as string repr of list of dicts)
        # IMPORTANT: always return a list so that .explode() works correctly
        # for both BDL (multi-year) and Census (single-year) data
        def parse_values_column(value):
            try:
                value_list = ast.literal_eval(value)
                if isinstance(value_list, list):
                    return value_list  # always return the list, never unwrap
                return [value_list]  # wrap single dicts in a list
            except (ValueError, SyntaxError):
                return None
        
        df_merged['values'] = df_merged['values'].apply(parse_values_column)
        
        # Expand and normalize
        df_expanded = df_merged.explode('values').reset_index(drop=True)
        values_normalized = pd.json_normalize(df_expanded['values'])
        df_expanded = pd.concat(
            [df_expanded.drop(columns=['values']), values_normalized], axis=1
        )
        
        # Rename and format columns
        df_expanded = df_expanded.rename(columns={"id_x": "nuts_id", "id_y": "var_id"})
        df_expanded['nuts_id'] = df_expanded['nuts_id'].apply(
            lambda x: str(int(x)).zfill(12)
        )
        df_expanded['teryt_id'] = df_expanded['nuts_id'].apply(nuts_code_to_teryt)
        
        return df_expanded
    
    def load_subject_data(self, df_expanded: pd.DataFrame, source_type: str = 'BDL',
                          subject_id: str = None, verbose: bool = True,
                          subject_name: str = "") -> dict:
        """
        Load processed subject data onto TERYTRecord objects.
        
        Each unique (variableId, categories) combination becomes a separate
        DataSeries on the matching TERYTRecord.
        
        Parameters:
        - df_expanded: Output of process_subject_data()
        - source_type: Data source type ('BDL', 'Census', etc.)
        - subject_id: Override subject_id (if None, inferred from data)
        - verbose: Print progress
        
        Returns:
        - dict with loading statistics
        """
        if df_expanded.empty:
            if verbose:
                print("  ⚠ Empty DataFrame, nothing to load")
            return {'matched_teryts': 0, 'unmatched_teryts': 0, 'total_data_points': 0}
        
        # Detect subject_id from data if not provided
        if subject_id is None:
            subject_id = str(df_expanded['subjectId'].iloc[0])
        
        # Detect category columns (n1, n2, etc.)
        category_cols = [c for c in df_expanded.columns if re.match(r'^n\d+$', c)]
        
        # Detect variable ID column
        var_col = 'var_id' if 'var_id' in df_expanded.columns else 'variableId'
        
        matched = 0
        unmatched = 0
        unmatched_teryts = set()
        total_points = 0
        
        # Group by teryt_id for efficient loading
        for teryt_id, group in df_expanded.groupby('teryt_id'):
            teryt_id = str(teryt_id).zfill(7)
            
            # Try exact match
            record = self._records.get(teryt_id)
            
            # Try 6-digit match if exact match fails
            if record is None:
                short_id = teryt_id[:6]
                candidates = [tid for tid in self._records if tid[:6] == short_id and tid[-1] in ['1','2','3']]
                if len(candidates) == 1:
                    record = self._records[candidates[0]]
            
            if record is None:
                unmatched += 1
                unmatched_teryts.add(teryt_id)
                continue
            
            matched += 1
            
            # Load each variable's time series
            for var_id, var_group in group.groupby(var_col):
                # Get categories for this variable
                categories = {}
                for col in category_cols:
                    vals = var_group[col].unique()
                    if len(vals) == 1:
                        categories[col] = str(vals[0])
                
                var_name = ""
                for cat in list(categories.values()):
                    var_name += f"{cat}/"
                
                # Add data points for each year
                for _, row in var_group.iterrows():
                    year = row.get('year')
                    val = row.get('val')
                    if year is not None and val is not None:
                        record.add_data_point(
                            source_type=source_type,
                            subject_id=subject_id,
                            variable_id=str(var_id),
                            year=year,
                            value=val,
                            categories=categories,
                            subject_name=subject_name,
                            variable_name=var_name
                        )
                        total_points += 1
        
        stats = {
            'matched_teryts': matched,
            'unmatched_teryts': unmatched,
            'total_data_points': total_points,
            'unmatched_teryt_ids': sorted(unmatched_teryts)
        }
        
        if verbose:
            print(f"  ✓ Loaded {total_points:,} data points for subject {subject_id}")
            print(f"  ✓ Matched {matched} TERYT records, {unmatched} unmatched")
            if unmatched > 0:
                print(f"  ⚠ Unmatched TERYT IDs (first 10): {sorted(unmatched_teryts)[:10]}")
        
        return stats
    
    def aggregate_data(self, records: List[TERYTRecord], subject_id: str, year,
                       agg_func: str = 'sum') -> pd.DataFrame:
        """
        Aggregate data across multiple TERYTRecords for a given subject and year.
        
        Useful for computing regional totals (e.g., voivodeship population from gminas).
        
        Parameters:
        - records: List of TERYTRecord objects to aggregate
        - subject_id: Subject ID to aggregate
        - year: Year to get values for (int or str)
        - agg_func: 'sum' or 'mean'
        
        Returns:
        - DataFrame with variable_id, categories, and aggregated values
        """
        year = int(year)
        rows = []
        for record in records:
            subject_data = record.get_data_by_subject(subject_id)
            for key, series in subject_data.items():
                val = series.get_value(year)
                if val is not None:
                    row = {
                        'variable_id': series.variable_id,
                        **series.categories,
                        'value': val
                    }
                    rows.append(row)
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Determine group columns (everything except 'value')
        group_cols = [c for c in df.columns if c != 'value']
        
        if agg_func == 'sum':
            result = df.groupby(group_cols, as_index=False)['value'].sum()
        elif agg_func == 'mean':
            result = df.groupby(group_cols, as_index=False)['value'].mean()
        else:
            raise ValueError(f"Unknown agg_func: {agg_func}")
        
        return result
    
    def get_distribution(self, records: List[TERYTRecord], subject_id: str, year,
                         row_category: str = None, col_category: str = None,
                         agg_func: str = 'sum') -> pd.DataFrame:
        """
        Get joint or marginal distributions from demographic data.
        
        Aggregates data across the given records and pivots by the specified
        categorical dimensions.
        
        Parameters:
        - records: List of TERYTRecord objects
        - subject_id: Subject ID
        - year: Year to get data for
        - row_category: Category column for rows (e.g., 'n1' for gender)
        - col_category: Category column for columns (e.g., 'n2' for urban/rural)
        - agg_func: 'sum' or 'mean'
        
        Returns:
        - Pivot table DataFrame (joint) or Series (marginal)
        """
        agg_df = self.aggregate_data(records, subject_id, year, agg_func)
        
        if agg_df.empty:
            return pd.DataFrame()
        
        if row_category and col_category:
            # Joint distribution (pivot table)
            if row_category in agg_df.columns and col_category in agg_df.columns:
                return agg_df.pivot_table(
                    index=row_category, columns=col_category,
                    values='value', aggfunc=agg_func
                )
        elif row_category:
            # Marginal distribution by row_category
            if row_category in agg_df.columns:
                return agg_df.groupby(row_category)['value'].agg(agg_func)
        elif col_category:
            # Marginal distribution by col_category
            if col_category in agg_df.columns:
                return agg_df.groupby(col_category)['value'].agg(agg_func)
        
        return agg_df
    
    def get_data_summary(self) -> dict:
        """
        Get a summary of all data stored across records.
        
        Returns:
        - dict with counts of records with data, subjects, total series, etc.
        """
        records_with_data = 0
        all_subjects = set()
        total_series = 0
        total_points = 0
        
        for record in self._records.values():
            if record.has_data:
                records_with_data += 1
                total_series += record.n_data_series
                for key, series in record.data.items():
                    all_subjects.add(key[1])
                    total_points += series.n_years
        
        return {
            'records_with_data': records_with_data,
            'total_records': len(self._records),
            'subjects': sorted(all_subjects),
            'n_subjects': len(all_subjects),
            'total_data_series': total_series,
            'total_data_points': total_points
        }
    
    # ==========================================================================
    # QUALITY OF LIFE / INSPECTION METHODS (NEW in v4.1)
    # ==========================================================================
    
    def subject_availability(self, subject_ids: List[str] = None,
                             level: int = None,
                             mode: str = 'bool') -> pd.DataFrame:
        """
        Get a DataFrame showing subject data availability across TERYTRecords.
        
        Parameters:
        - subject_ids: List of subject IDs to check (None = all found)
        - level: Filter records by admin level (2=woj, 4=pow, 6=gmi, None=all)
        - mode: 
            'bool' - True/False whether record has any data for subject
            'years' - comma-separated string of years with data
            'count' - number of data series for the subject
        
        Returns:
        - DataFrame with teryt_ids as index and subject_ids as columns
        """
        # Collect records
        records = list(self._records.values())
        if level is not None:
            records = [r for r in records if r.level == level]
        
        # Discover all subjects if not specified
        if subject_ids is None:
            subject_ids = sorted(set(
                k[1] for r in records for k in r.data.keys()
            ))
        
        rows = []
        for record in records:
            row = {'teryt_id': record.teryt_id, 'name': record.name}
            for sid in subject_ids:
                subj_data = record.get_data_by_subject(sid)
                if mode == 'bool':
                    row[sid] = len(subj_data) > 0
                elif mode == 'years':
                    all_years = set()
                    for ds in subj_data.values():
                        all_years.update(ds.years)
                    row[sid] = ','.join(str(y) for y in sorted(all_years)) if all_years else ''
                elif mode == 'count':
                    row[sid] = len(subj_data)
                else:
                    row[sid] = len(subj_data) > 0
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index('teryt_id')
        return df
    
    def get_subject_dataframe(self, subject_id: str, teryt_id: str = None,
                              year: int = None) -> pd.DataFrame:
        """
        Reconstruct a DataFrame from stored DataSeries for a given subject.
        
        This effectively reverses load_subject_data(): it collects all data
        points from all matching records and returns one flat DataFrame.
        
        Parameters:
        - subject_id: Subject ID to retrieve
        - teryt_id: Restrict to a single TERYT record (None = all records)
        - year: Restrict to a single year (None = all years)
        
        Returns:
        - DataFrame with columns: teryt_id, name, variable_id, source_type,
          year, value, + category columns (n1, n2, ...)
        """
        rows = []
        if teryt_id:
            records = [self._records.get(str(teryt_id).zfill(7))]
            records = [r for r in records if r is not None]
        else:
            records = list(self._records.values())
        
        for record in records:
            subj_data = record.get_data_by_subject(str(subject_id))
            for key, series in subj_data.items():
                src, sid, vid = key
                for ts, val in series.values.dropna().items():
                    yr = ts.year
                    if year is not None and yr != int(year):
                        continue
                    row = {
                        'teryt_id': record.teryt_id,
                        'name': record.name,
                        'source_type': src,
                        'subject_id': sid,
                        'variable_id': vid,
                        'year': int(yr),
                        'value': val,
                    }
                    if series.categories:
                        row.update(series.categories)
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_variable_values(self, subject_id: str, year: int,
                            level: int = None) -> pd.DataFrame:
        """
        Get all values of a subject for a given year across all teryts.
        
        Useful to quickly see e.g. population by gmina for a given year.
        
        Parameters:
        - subject_id: Subject ID
        - year: Year
        - level: Filter by admin level (None = all)
        
        Returns:
        - DataFrame with teryt_id, name, + one column per variable
        """
        year = int(year)
        rows = []
        for record in self._records.values():
            if level is not None and record.level != level:
                continue
            subj_data = record.get_data_by_subject(str(subject_id))
            if not subj_data:
                continue
            row = {'teryt_id': record.teryt_id, 'name': record.name}
            for key, series in subj_data.items():
                val = series.get_value(year)
                if val is not None:
                    # Use variable_name if available, else variable_id
                    col_name = series.variable_name.rstrip('/') if series.variable_name else key[2]
                    row[col_name] = val
            if len(row) > 2:  # has at least one value
                rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index('teryt_id')
        return df
    
    # ==========================================================================
    # CROSS TABLE DATABASE METHODS (NEW in v4.1)
    # ==========================================================================
    
    def build_cross_tables(self, subject_id: str, subject_name: str = '',
                           level: int = None, verbose: bool = True) -> int:
        """
        Build cross tables for a subject across all matching records.
        
        Should be called after load_subject_data(). Only builds for records
        that have 2+ variables for the subject (otherwise cross table is trivial).
        
        Parameters:
        - subject_id: Subject ID
        - subject_name: Human-readable name
        - level: Filter by admin level (None = all)
        - verbose: Print progress
        
        Returns:
        - Number of cross tables built
        """
        count = 0
        for record in self._records.values():
            if level is not None and record.level != level:
                continue
            subj_data = record.get_data_by_subject(str(subject_id))
            if len(subj_data) < 2:
                continue
            ct = record.build_cross_table(str(subject_id), subject_name=subject_name)
            if ct is not None:
                count += 1
        
        if verbose:
            print(f"  ✓ Built {count} cross tables for subject {subject_id}"
                  f"{f' ({subject_name})' if subject_name else ''}")
        
        return count
    
    def aggregate_cross_tables(self, teryt_ids: List[str],
                               subject_id: str) -> Optional['CrossTable']:
        """
        Aggregate cross tables across multiple TERYTs (element-wise sum).
        
        Handles cross tables with different label sets by building a union
        of all dimension labels and aligning data before summing.
        
        Parameters:
        - teryt_ids: List of TERYT IDs to aggregate
        - subject_id: Subject ID
        
        Returns:
        - Aggregated CrossTable, or None if no data found
        """
        import itertools
        
        # Collect all cross tables
        all_cts = []
        for tid in teryt_ids:
            record = self._records.get(str(tid).zfill(7))
            if record is None:
                continue
            ct = record.get_cross_table(str(subject_id))
            if ct is not None:
                all_cts.append(ct)
        
        if not all_cts:
            return None
        
        # Build union of labels for each dimension (preserving order)
        dim_names = all_cts[0].dim_names
        union_labels = {d: [] for d in dim_names}
        for ct in all_cts:
            for d in dim_names:
                for label in ct.dim_labels[d]:
                    if label not in union_labels[d]:
                        union_labels[d].append(label)
        
        union_shape = tuple(len(union_labels[d]) for d in dim_names)
        
        result = CrossTable(
            subject_id=all_cts[0].subject_id,
            dim_names=dim_names,
            dim_labels=union_labels,
            subject_name=all_cts[0].subject_name,
            year_range=all_cts[0].year_range
        )
        
        for ct in all_cts:
            # Build index mapping: ct label index -> result label index
            idx_maps = []
            for d in dim_names:
                mapping = [union_labels[d].index(label) for label in ct.dim_labels[d]]
                idx_maps.append(mapping)
            
            for year in ct.year_range:
                src = ct.tables.get(year)
                if src is None or np.all(np.isnan(src)):
                    continue
                
                dst = result.tables[year]
                # Iterate over all index combinations in the source
                for src_idx in itertools.product(*[range(len(m)) for m in idx_maps]):
                    dst_idx = tuple(idx_maps[i][src_idx[i]] for i in range(len(dim_names)))
                    val = src[src_idx]
                    if np.isnan(val):
                        continue
                    if np.isnan(dst[dst_idx]):
                        dst[dst_idx] = val
                    else:
                        dst[dst_idx] += val
        
        return result
    
    def get_cross_table_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all cross tables in the database.
        
        Returns:
        - DataFrame with subject_id, n_records, dims, shape, years_with_data
        """
        subject_info = {}
        for record in self._records.values():
            for sid, ct in record.cross_tables.items():
                if sid not in subject_info:
                    subject_info[sid] = {
                        'subject_id': sid,
                        'subject_name': ct.subject_name,
                        'n_records': 0,
                        'dim_names': ct.dim_names,
                        'shape': ct.shape,
                    }
                subject_info[sid]['n_records'] += 1
        
        if not subject_info:
            return pd.DataFrame()
        
        rows = []
        for sid, info in subject_info.items():
            rows.append({
                'subject_id': info['subject_id'],
                'subject_name': info['subject_name'],
                'n_records': info['n_records'],
                'dimensions': ' × '.join(info['dim_names']),
                'shape': info['shape'],
            })
        
        return pd.DataFrame(rows).set_index('subject_id')
    
    # ==========================================================================
    # DATA UNIFICATION METHODS (NEW in v4.2)
    # ==========================================================================
    
    @staticmethod
    def filter_subject_data(df_processed: pd.DataFrame, subject_id: str,
                            filters: Dict[str, str] = None) -> pd.DataFrame:
        """
        Filter processed subject data by specific dimension values.
        
        Used to reduce dimensions before loading (e.g., keep only
        'miejsce zamieszkania' in n2 for subject P1336).
        
        Parameters:
        - df_processed: Output of process_subject_data()
        - subject_id: Subject ID (for logging)
        - filters: Dict mapping column name -> value to keep
                   e.g. {'n2': 'miejsce zamieszkania', 'n3': 'stan na 30 czerwca'}
        
        Returns:
        - Filtered DataFrame
        """
        if filters is None or df_processed.empty:
            return df_processed
        
        df = df_processed.copy()
        for col, val in filters.items():
            if col in df.columns:
                before = len(df)
                df = df[df[col] == val].copy()
                # Drop the now-constant filter column
                if df[col].nunique() <= 1:
                    df = df.drop(columns=[col])
        
        return df
    
    # ------------------------------------------------------------------
    # Merged subject creation (NEW in v4.3 - replaces unify_census_subjects)
    # ------------------------------------------------------------------
    
    @staticmethod
    def _parse_numeric_bounds(label: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse a category label to extract (lower_bound, upper_bound).
        
        Handles age groups, household sizes, and other numeric range categories.
        Returns (None, None) for non-parseable labels (e.g. 'ogółem').
        
        Examples:
            '0-4' → (0, 4)
            '15-19' → (15, 19)
            '85 i więcej' → (85, None)
            '14 lat i mniej' → (None, 14)
            '3-osobowe' → (3, 3)
            '5-osobowe i większe' → (5, None)
            'ogółem' → (None, None)
        """
        label_lower = label.lower().strip()
        
        # Skip totals
        if label_lower in ['ogółem', 'total', 'razem']:
            return None, None
        
        # Pattern: "X-Y" or "X–Y" (range)
        m = re.search(r'(\d+)\s*[-–]\s*(\d+)', label)
        if m:
            return int(m.group(1)), int(m.group(2))
        
        # Pattern: number followed eventually by "i więcej" / "i większe"
        m = re.search(r'(\d+).*?(?:i\s+więcej|i\s+większe|i\s+więc|i\s+wiecej)', label_lower)
        if m:
            return int(m.group(1)), None
        
        # Pattern: "X i mniej"
        m = re.search(r'(\d+).*?i\s+mniej', label_lower)
        if m:
            return None, int(m.group(1))
        
        # Pattern: "poniżej X"
        m = re.search(r'poniżej\s*(\d+)', label_lower)
        if m:
            return None, int(m.group(1)) - 1
        
        # Pattern: "X-osobowe" or "X osob" (single size, household)
        m = re.search(r'(\d+)\s*[-–]?\s*osob', label_lower)
        if m:
            n = int(m.group(1))
            return n, n
        
        return None, None
    
    @staticmethod
    def _compute_unified_bins(sources_labels: Dict[str, List[str]]
                              ) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        """
        Compute unified bins from multiple sources with possibly different binning.
        
        Uses the "common break points" approach:
        1. Parse each source's labels to extract (lower, upper) bounds
        2. Compute break points for each source (lower bounds + upper+1)
        3. Unified break points = intersection of all sources' break points
        4. Unified bins = consecutive pairs of common break points
        5. Each source label maps to the unified bin containing its lower bound
        
        Multiple source labels can map to the same unified label when the source
        has finer bins — values should be SUMMED by the caller.
        
        Parameters:
        - sources_labels: {source_id: [list of labels including 'ogółem']}
        
        Returns:
        - unified_labels: List of unified bin labels (including 'ogółem')
        - mapping: {source_id: {source_label: unified_label}}
        """
        total_labels_set = {'ogółem', 'total', 'razem'}
        
        # Step 1: Parse bounds for each source
        source_bins = {}       # sid -> [(lower, upper, label)]
        source_non_parseable = {}  # sid -> [labels that aren't numeric bins]
        source_break_points = {}
        
        for sid, labels in sources_labels.items():
            bins = []
            non_parseable = []
            for label in labels:
                if label.lower() in total_labels_set:
                    continue
                lb, ub = GeoTERYTDatabase._parse_numeric_bounds(label)
                if lb is not None or ub is not None:
                    if lb is None:
                        lb = 0  # default open lower bound to 0
                    bins.append((lb, ub, label))
                else:
                    non_parseable.append(label)
            bins.sort(key=lambda x: (x[0], x[1] if x[1] is not None else float('inf')))
            source_bins[sid] = bins
            source_non_parseable[sid] = non_parseable
            
            # Extract break points
            bps = set()
            for lb, ub, _ in bins:
                bps.add(lb)
                if ub is not None:
                    bps.add(ub + 1)
            bps.add(float('inf'))
            source_break_points[sid] = bps
        
        # Step 2: Common break points
        if source_break_points:
            common_bps = sorted(set.intersection(*source_break_points.values()))
        else:
            common_bps = [0, float('inf')]
        
        # Step 3: Build unified bin labels
        unified_bin_labels = []
        for i in range(len(common_bps) - 1):
            lb = int(common_bps[i])
            next_bp = common_bps[i + 1]
            if next_bp == float('inf'):
                label = f"{lb} i więcej"
            else:
                ub = int(next_bp - 1)
                label = f"{lb}-{ub}" if lb != ub else str(lb)
            unified_bin_labels.append(label)
        
        # Non-parseable labels: union across sources
        all_non_parseable = []
        for labels in source_non_parseable.values():
            for l in labels:
                if l not in all_non_parseable:
                    all_non_parseable.append(l)
        
        # Final: ogółem + bins + non-parseable
        unified_labels = ['ogółem'] + unified_bin_labels + sorted(all_non_parseable)
        
        # Step 4: Build mapping for each source
        mapping = {}
        for sid in sources_labels:
            m = {}
            # Map totals
            for label in sources_labels[sid]:
                if label.lower() in total_labels_set:
                    m[label] = 'ogółem'
            # Map bin labels by lower bound
            for lb, ub, label in source_bins[sid]:
                for i in range(len(common_bps) - 1):
                    u_lb = common_bps[i]
                    u_next = common_bps[i + 1]
                    if lb >= u_lb and (u_next == float('inf') or lb < u_next):
                        m[label] = unified_bin_labels[i]
                        break
            # Map non-parseable (exact match)
            for label in source_non_parseable[sid]:
                m[label] = label
            mapping[sid] = m
        
        return unified_labels, mapping
    
    def create_merged_subjects(self, subject_names_dict: Dict[str, str],
                               verbose: bool = True) -> Dict[str, List[str]]:
        """
        Create new merged subjects from groups of subjects sharing the same topic.
        
        DOES NOT modify original subjects — keeps all raw data intact.
        Creates new DataSeries under merged subject IDs (prefixed with 'M_').
        
        For each merge group:
        1. Detects semantic dimension types (age, sex, education, etc.)
        2. Aligns dimensions across sources (e.g., BDL n1=age/n2=sex vs Census n1=sex/n2=age)
        3. For range-based dimensions (age, hh_size): computes unified bins
           via common break points and sums finer bins
        4. For exact-match dimensions (sex): matches labels directly
        5. BDL data takes priority; census fills only NaN years
        
        Parameters:
        - subject_names_dict: Dict mapping subject_id -> subject_name
        
        Returns:
        - Dict mapping merged_subject_id -> list of source subject_ids
        """
        import itertools
        
        # Group subjects by name → find merge groups
        name_to_ids = {}
        for sid, name in subject_names_dict.items():
            name_to_ids.setdefault(name, []).append(sid)
        merge_groups = {name: sids for name, sids in name_to_ids.items() if len(sids) > 1}
        
        if verbose:
            print(f"Found {len(merge_groups)} subject groups to merge:")
            for name, sids in merge_groups.items():
                print(f"  {name}: {sids}")
        
        result = {}
        
        for group_name, sids in merge_groups.items():
            merged_sid = f"M_{group_name}"
            
            # ---- Step 1: Collect dimension labels per source per n-dim ----
            source_dim_labels = {}  # sid -> {n_dim -> set(labels)}
            for sid in sids:
                source_dim_labels[sid] = {}
                for record in self._records.values():
                    subj_data = record.get_data_by_subject(sid)
                    for key, series in subj_data.items():
                        for dim, label in series.categories.items():
                            source_dim_labels[sid].setdefault(dim, set()).add(str(label))
            
            # Keep only sources that actually have data
            sids_with_data = [s for s in sids if source_dim_labels.get(s)]
            if len(sids_with_data) < 1:
                if verbose:
                    print(f"  ⚠ Skipping {group_name}: no sources with data")
                continue
            
            # ---- Step 2: Detect semantic type of each n-dim ----
            source_dim_types = {}  # sid -> {n_dim -> semantic_type}
            for sid in sids_with_data:
                source_dim_types[sid] = {}
                for dim, labels in source_dim_labels[sid].items():
                    if self._detect_gender_dim(labels):
                        source_dim_types[sid][dim] = 'sex'
                    elif self._detect_age_dim(labels):
                        source_dim_types[sid][dim] = 'age'
                    elif self._detect_education_dim(labels):
                        source_dim_types[sid][dim] = 'education'
                    else:
                        # Check for hh_size patterns
                        hh_pattern = re.compile(r'\d+\s*[-–]?\s*osob', re.IGNORECASE)
                        if sum(1 for l in labels if hh_pattern.search(l)) >= 2:
                            source_dim_types[sid][dim] = 'hh_size'
                        else:
                            source_dim_types[sid][dim] = 'other'
            
            # ---- Step 3: Canonical dimension mapping ----
            # Use BDL subject as canonical (it has the most years of data)
            bdl_sid = None
            for s in sids_with_data:
                for r in self._records.values():
                    if any(k[0] == 'BDL' and k[1] == s for k in r.data.keys()):
                        bdl_sid = s
                        break
                if bdl_sid:
                    break
            
            canonical_sid = bdl_sid or sids_with_data[0]
            canonical_types = source_dim_types.get(canonical_sid, {})
            
            # Map: semantic_type -> canonical n_dim name
            type_to_canon_dim = {}
            for n_dim, sem_type in canonical_types.items():
                type_to_canon_dim[sem_type] = n_dim
            
            # Map: (source_sid, source_n_dim) -> canonical n_dim
            dim_alignment = {}
            for sid in sids_with_data:
                for n_dim, sem_type in source_dim_types[sid].items():
                    if sem_type in type_to_canon_dim:
                        dim_alignment[(sid, n_dim)] = type_to_canon_dim[sem_type]
                    else:
                        dim_alignment[(sid, n_dim)] = n_dim
            
            # ---- Step 4: Compute unified labels per canonical dimension ----
            canonical_dim_names = sorted(type_to_canon_dim.values())
            unified_labels = {}
            label_mappings = {}  # canon_dim -> {sid -> {src_label -> unified_label}}
            
            for sem_type, canon_dim in type_to_canon_dim.items():
                if sem_type in ('age', 'hh_size'):
                    # Range-based: compute unified bins
                    range_labels_per_source = {}
                    for sid in sids_with_data:
                        for n_dim, st in source_dim_types[sid].items():
                            if st == sem_type:
                                range_labels_per_source[sid] = sorted(source_dim_labels[sid][n_dim])
                    
                    unified, mapping = self._compute_unified_bins(range_labels_per_source)
                    unified_labels[canon_dim] = unified
                    label_mappings[canon_dim] = mapping
                
                elif sem_type == 'sex':
                    # Standard sex labels
                    std_labels = ['ogółem', 'mężczyźni', 'kobiety']
                    unified_labels[canon_dim] = std_labels
                    label_mappings[canon_dim] = {}
                    for sid in sids_with_data:
                        for n_dim, st in source_dim_types[sid].items():
                            if st == 'sex':
                                label_mappings[canon_dim][sid] = {
                                    l: l for l in source_dim_labels[sid][n_dim]
                                    if l in std_labels
                                }
                
                else:
                    # Exact match: union of labels
                    all_labels = set()
                    for sid in sids_with_data:
                        for n_dim, st in source_dim_types[sid].items():
                            if st == sem_type:
                                all_labels.update(source_dim_labels[sid][n_dim])
                    unified_labels[canon_dim] = sorted(all_labels)
                    label_mappings[canon_dim] = {}
                    for sid in sids_with_data:
                        for n_dim, st in source_dim_types[sid].items():
                            if st == sem_type:
                                label_mappings[canon_dim][sid] = {
                                    l: l for l in source_dim_labels[sid][n_dim]
                                }
            
            if verbose:
                print(f"\n  Merged subject: {merged_sid}")
                for cdim in canonical_dim_names:
                    labels = unified_labels.get(cdim, [])
                    print(f"    {cdim} ({type_to_canon_dim and [t for t,d in type_to_canon_dim.items() if d==cdim]}): "
                          f"{len(labels)} labels")
            
            # ---- Step 5: Generate variable IDs for unified categories ----
            cat_combos = list(itertools.product(
                *[unified_labels[d] for d in canonical_dim_names]
            ))
            var_id_map = {}  # frozen category tuple -> var_id string
            for i, combo in enumerate(cat_combos):
                cats = {canonical_dim_names[j]: combo[j] for j in range(len(canonical_dim_names))}
                cats_key = tuple(sorted(cats.items()))
                var_id_map[cats_key] = f"M{i+1:04d}"
            
            # ---- Step 6: Create merged DataSeries for each record ----
            # Process BDL source first, then census (BDL takes priority)
            bdl_sources = [s for s in sids_with_data 
                          if any(k[0] == 'BDL' for r in self._records.values() 
                                for k in r.data.keys() if k[1] == s)]
            census_sources = [s for s in sids_with_data if s not in bdl_sources]
            ordered_sources = bdl_sources + census_sources
            
            records_merged = 0
            series_created = 0
            
            for record in self._records.values():
                record_has_merged = False
                
                for sid in ordered_sources:
                    subj_data = record.get_data_by_subject(sid)
                    if not subj_data:
                        continue
                    
                    # Accumulate values per unified category within this source
                    temp_values = {}  # cats_key -> pd.Series
                    
                    for key, series in subj_data.items():
                        if not series.categories:
                            continue
                        
                        # Map source dims to canonical dims + map labels
                        mapped_cats = {}
                        skip = False
                        for src_dim, src_label in series.categories.items():
                            canon_dim = dim_alignment.get((sid, src_dim))
                            if canon_dim is None or canon_dim not in label_mappings:
                                skip = True
                                break
                            
                            src_mapping = label_mappings.get(canon_dim, {}).get(sid, {})
                            unified_label = src_mapping.get(src_label)
                            if unified_label is None:
                                skip = True
                                break
                            
                            mapped_cats[canon_dim] = unified_label
                        
                        if skip or len(mapped_cats) != len(canonical_dim_names):
                            continue
                        
                        cats_key = tuple(sorted(mapped_cats.items()))
                        var_id = var_id_map.get(cats_key)
                        if var_id is None:
                            continue
                        
                        # Sum within this source (aggregates finer bins)
                        if cats_key not in temp_values:
                            temp_values[cats_key] = pd.Series(
                                data=np.nan, index=DATETIME_INDEX_FULL, dtype=float
                            )
                        
                        for ts, val in series.values.dropna().items():
                            existing = temp_values[cats_key].get(ts, np.nan)
                            if pd.isna(existing):
                                temp_values[cats_key][ts] = val
                            else:
                                temp_values[cats_key][ts] = existing + val
                    
                    # Merge temp into target (only fill NaN positions)
                    for cats_key, temp_series in temp_values.items():
                        var_id = var_id_map[cats_key]
                        merged_key = ('Merged', merged_sid, var_id)
                        
                        if merged_key not in record.data:
                            cats_dict = dict(cats_key)
                            record.data[merged_key] = DataSeries(
                                source_type='Merged',
                                subject_id=merged_sid,
                                variable_id=var_id,
                                subject_name=group_name,
                                categories=cats_dict
                            )
                            series_created += 1
                        
                        target = record.data[merged_key]
                        for ts, val in temp_series.dropna().items():
                            if pd.isna(target.values.get(ts, np.nan)):
                                target.values[ts] = val
                    
                    if temp_values:
                        record_has_merged = True
                
                if record_has_merged:
                    records_merged += 1
            
            result[merged_sid] = sids_with_data
            
            if verbose:
                print(f"    ✓ {records_merged} records, {series_created} merged series created")
        
        return result
    
    def extract_population(self, subject_names_dict: Dict[str, str],
                           verbose: bool = True) -> int:
        """
        Extract total population for each TERYTRecord from pop__ subjects.
        
        For each record with BDL data in subjects with names starting with 'pop__',
        finds variables where ALL dimension labels are 'ogółem' or 'total',
        and stores the total population in TERYTRecord.pop as pd.Series.
        Census data is also added where available.
        
        Parameters:
        - subject_names_dict: Dict mapping subject_id -> human-readable name
        - verbose: Print progress
        
        Returns:
        - Number of records with population data extracted
        """
        pop_subjects = [sid for sid, name in subject_names_dict.items()
                        if name.startswith('pop__')]
        
        total_labels = {'ogółem', 'total', 'Ogółem', 'Total'}
        count = 0
        
        for record in self._records.values():
            pop_found = False
            
            for sid in pop_subjects:
                subj_data = record.get_data_by_subject(sid)
                if not subj_data:
                    continue
                
                # Find the "total" variable (all categories are 'ogółem'/'total')
                for key, series in subj_data.items():
                    if not series.categories:
                        continue
                    
                    all_total = all(
                        v in total_labels
                        for v in series.categories.values()
                    )
                    
                    if all_total:
                        # Merge values into record.pop
                        for ts, val in series.values.dropna().items():
                            if pd.isna(record.pop.get(ts, np.nan)):
                                record.pop[ts] = val
                        pop_found = True
            
            if pop_found:
                count += 1
        
        if verbose:
            print(f"  ✓ Extracted population for {count} records")
        
        return count
    
    def classify_population(self, verbose: bool = True) -> int:
        """
        Classify each TERYTRecord by urban/rural based on pop data and rodz.
        
        Applied only to level=6 records with at least one year of pop data.
        Stores result in TERYTRecord.pop_class as pd.DataFrame with columns
        pop_class_code (int) and pop_class_label (str).
        
        Classification:
        - rodz in ['1','4','8','9'] (urban): classified by population size
        - rodz in ['2','5'] (rural): always class 1 ('wieś')
        
        Parameters:
        - verbose: Print progress
        
        Returns:
        - Number of records classified
        """
        count = 0
        
        for record in self._records.values():
            if record.level != 6:
                continue
            if not record.pop.notna().any():
                continue
            
            rows = []
            for ts in DATETIME_INDEX_FULL:
                pop_val = record.pop.get(ts, np.nan)
                
                if record.rodz in ['2', '5']:
                    rows.append({
                        'date': ts,
                        'pop_class_code': 1,
                        'pop_class_label': 'wieś'
                    })
                elif record.rodz in ['1', '4', '8', '9']:
                    if pd.isna(pop_val):
                        rows.append({
                            'date': ts,
                            'pop_class_code': np.nan,
                            'pop_class_label': ''
                        })
                    elif pop_val <= 20000:
                        rows.append({
                            'date': ts,
                            'pop_class_code': 2,
                            'pop_class_label': 'miasto do 20 000'
                        })
                    elif pop_val <= 50000:
                        rows.append({
                            'date': ts,
                            'pop_class_code': 3,
                            'pop_class_label': 'miasto od 20 001 do 50 000'
                        })
                    elif pop_val <= 100000:
                        rows.append({
                            'date': ts,
                            'pop_class_code': 4,
                            'pop_class_label': 'miasto od 50 001 do 100 000'
                        })
                    elif pop_val <= 500000:
                        rows.append({
                            'date': ts,
                            'pop_class_code': 5,
                            'pop_class_label': 'miasto od 100 001 do 500 000'
                        })
                    else:
                        rows.append({
                            'date': ts,
                            'pop_class_code': 6,
                            'pop_class_label': 'miasto 500 001 i więcej'
                        })
                else:
                    continue
            
            if rows:
                record.pop_class = pd.DataFrame(rows).set_index('date')
                count += 1
        
        if verbose:
            print(f"  ✓ Classified {count} records by urban/rural")
        
        return count
    
    def code_dimension_labels(self, subject_names_dict: Dict[str, str],
                              verbose: bool = True) -> int:
        """
        Translate categorical string labels to numerical codes for all DataSeries.
        
        Rules:
        - Gender: 'mężczyźni'=1, 'kobiety'=2, 'ogółem'=0
        - Education: lowest=1 ... highest=max, 'ogółem'=0
        - Age: sequential codes + extract lower/upper bounds
        - Household size: labels with numbers get sequential codes,
          labels without numbers start from 101
        
        Stores codes in DataSeries.cat_code and bounds in DataSeries.cat_bounds.
        
        Parameters:
        - subject_names_dict: Dict mapping subject_id -> human-readable name
        - verbose: Print progress
        
        Returns:
        - Number of DataSeries coded
        """
        # First, collect all unique category labels per subject per dimension
        subject_dims = {}  # subject_id -> {dim_name: set(labels)}
        for record in self._records.values():
            for key, series in record.data.items():
                sid = key[1]
                if sid not in subject_dims:
                    subject_dims[sid] = {}
                for dim, label in series.categories.items():
                    if dim not in subject_dims[sid]:
                        subject_dims[sid][dim] = set()
                    subject_dims[sid][dim].add(str(label))
        
        # Build coding maps per subject per dimension
        coding_maps = {}  # subject_id -> {dim_name: {label: code}}
        bounds_maps = {}  # subject_id -> {dim_name: {label: {'lower_bound': ..., 'upper_bound': ...}}}
        
        for sid, dims in subject_dims.items():
            subject_name = subject_names_dict.get(sid, '')
            coding_maps[sid] = {}
            bounds_maps[sid] = {}
            
            for dim, labels in dims.items():
                labels_sorted = sorted(labels)
                code_map = {}
                bound_map = {}
                
                # Detect dimension type from labels
                is_gender = any(l.lower() in ['mężczyźni', 'kobiety'] for l in labels)
                is_education = 'educ' in subject_name
                is_age = any(l.lower() in ['pop__age_sex', 'pop__age', 'pop__age_educ'] 
                            for l in [subject_name])
                is_hh_size = 'hh_size' in subject_name
                
                if is_gender and self._detect_gender_dim(labels):
                    code_map = self._code_gender(labels_sorted)
                elif is_age and self._detect_age_dim(labels):
                    code_map, bound_map = self._code_age(labels_sorted)
                elif is_education and self._detect_education_dim(labels):
                    code_map = self._code_education(labels_sorted)
                elif is_hh_size:
                    code_map = self._code_hh_size(labels_sorted)
                else:
                    # Default: sequential coding
                    for i, label in enumerate(labels_sorted):
                        code_map[label] = 0 if label.lower() in ['ogółem', 'total'] else i + 1
                
                coding_maps[sid][dim] = code_map
                if bound_map:
                    bounds_maps[sid][dim] = bound_map
        
        # Apply codes to all DataSeries
        total_coded = 0
        for record in self._records.values():
            for key, series in record.data.items():
                sid = key[1]
                if sid not in coding_maps:
                    continue
                
                for dim, label in series.categories.items():
                    if sid in coding_maps and dim in coding_maps[sid]:
                        code = coding_maps[sid][dim].get(str(label))
                        if code is not None:
                            series.cat_code[dim] = code
                    
                    if sid in bounds_maps and dim in bounds_maps[sid]:
                        bounds = bounds_maps[sid][dim].get(str(label))
                        if bounds is not None:
                            series.cat_bounds[dim] = bounds
                
                if series.cat_code:
                    total_coded += 1
        
        if verbose:
            print(f"  ✓ Coded dimension labels for {total_coded} DataSeries "
                  f"across {len(coding_maps)} subjects")
        
        return total_coded
    
    @staticmethod
    def _detect_gender_dim(labels: set) -> bool:
        """Check if a set of labels represents gender categories."""
        lower = {l.lower() for l in labels}
        return bool(lower & {'mężczyźni', 'kobiety', 'mężczyzni'})
    
    @staticmethod
    def _detect_age_dim(labels: set) -> bool:
        """Check if a set of labels represents age categories."""
        import re
        age_pattern = re.compile(r'\d+\s*[-–]\s*\d+|i więcej|i mniej|lat i|roku życia')
        return sum(1 for l in labels if age_pattern.search(l)) >= 2
    
    @staticmethod
    def _detect_education_dim(labels: set) -> bool:
        """Check if a set of labels represents education levels."""
        edu_keywords = {'podstawowe', 'średnie', 'wyższe', 'zasadnicze', 'gimnazjalne',
                        'policealne', 'zawodowe', 'niepełne'}
        lower = {l.lower() for l in labels}
        return bool(lower & edu_keywords)
    
    @staticmethod
    def _code_gender(labels: List[str]) -> Dict[str, int]:
        """Code gender labels: ogółem=0, mężczyźni=1, kobiety=2."""
        code_map = {}
        for label in labels:
            l = label.lower()
            if l in ['ogółem', 'total', 'razem']:
                code_map[label] = 0
            elif l in ['mężczyźni', 'mężczyzni']:
                code_map[label] = 1
            elif l == 'kobiety':
                code_map[label] = 2
            else:
                code_map[label] = 0
        return code_map
    
    @staticmethod
    def _code_age(labels: List[str]) -> Tuple[Dict[str, int], Dict[str, dict]]:
        """
        Code age categories sequentially and extract numeric bounds.
        ogółem/total → code 0.
        """
        import re
        code_map = {}
        bound_map = {}
        
        # Separate 'ogółem'/'total' from actual age groups
        age_labels = []
        total_labels = []
        for label in labels:
            if label.lower() in ['ogółem', 'total', 'razem']:
                total_labels.append(label)
            else:
                age_labels.append(label)
        
        # Extract numeric info for sorting
        def extract_lower(label):
            m = re.search(r'(\d+)', label)
            return int(m.group(1)) if m else 999
        
        age_labels.sort(key=extract_lower)
        
        for label in total_labels:
            code_map[label] = 0
            bound_map[label] = {'lower_bound': np.nan, 'upper_bound': np.nan}
        
        for i, label in enumerate(age_labels, start=1):
            code_map[label] = i
            
            # Extract bounds
            lower = np.nan
            upper = np.nan
            
            # Pattern: "X-Y" or "X–Y"
            m = re.search(r'(\d+)\s*[-–]\s*(\d+)', label)
            if m:
                lower = int(m.group(1))
                upper = int(m.group(2))
            else:
                # Pattern: "X i więcej" or "X lat i więcej"
                m = re.search(r'(\d+)\s*(i więcej|lat i więcej|roku życia i więcej|i więc)', label)
                if m:
                    lower = int(m.group(1))
                    upper = np.nan
                else:
                    # Pattern: "X i mniej" or "poniżej X"
                    m = re.search(r'(\d+)\s*(i mniej|lat i mniej)', label)
                    if m:
                        lower = np.nan
                        upper = int(m.group(1))
                    else:
                        # Try to extract just a number
                        m = re.search(r'(\d+)', label)
                        if m:
                            lower = int(m.group(1))
            
            bound_map[label] = {'lower_bound': lower, 'upper_bound': upper}
        
        return code_map, bound_map
    
    @staticmethod
    def _code_education(labels: List[str]) -> Dict[str, int]:
        """
        Code education levels from lowest (1) to highest.
        ogółem/total → 0.
        """
        # Education level hierarchy (Polish system, approximate ordering)
        edu_order = [
            'niepełne podstawowe', 'bez wykształcenia', 'nieustalone',
            'podstawowe', 'podstawowe ukończone',
            'gimnazjalne',
            'zasadnicze zawodowe', 'zasadnicze',
            'średnie', 'średnie zawodowe', 'średnie ogólnokształcące',
            'policealne', 'policealne i średnie zawodowe',
            'wyższe', 'wyższe ze stopniem',
        ]
        
        code_map = {}
        other_code = 100
        
        for label in labels:
            l = label.lower().strip()
            if l in ['ogółem', 'total', 'razem']:
                code_map[label] = 0
                continue
            
            # Find best match in hierarchy
            matched = False
            for rank, edu_level in enumerate(edu_order, start=1):
                if edu_level in l or l in edu_level:
                    code_map[label] = rank
                    matched = True
                    break
            
            if not matched:
                code_map[label] = other_code
                other_code += 1
        
        return code_map
    
    @staticmethod
    def _code_hh_size(labels: List[str]) -> Dict[str, int]:
        """
        Code household size labels.
        Labels with numbers get sequential codes starting from 1.
        Labels without numbers get codes starting from 101.
        """
        import re
        
        numbered = []
        unnumbered = []
        
        for label in labels:
            m = re.search(r'(\d+)', label)
            if m and label.lower() not in ['ogółem', 'total', 'razem']:
                numbered.append((int(m.group(1)), label))
            elif label.lower() in ['ogółem', 'total', 'razem']:
                unnumbered.append((0, label))  # ogółem gets special treatment
            else:
                unnumbered.append((1, label))
        
        numbered.sort(key=lambda x: x[0])
        
        code_map = {}
        for i, (num, label) in enumerate(numbered, start=1):
            code_map[label] = i
        
        next_code = 101
        for _, label in unnumbered:
            if label.lower() in ['ogółem', 'total', 'razem']:
                code_map[label] = 0
            else:
                code_map[label] = next_code
                next_code += 1
        
        return code_map
    
    # ==========================================================================
    # PERSISTENCE METHODS (NEW in v3.0, updated in v4.0)
    # ==========================================================================
    
    def save_complete(self, filepath: Union[str, Path], verbose: bool = True):
        """
        Save the complete database to a single file, including all geometries.
        
        NEW in v3.0: Uses pickle to save the entire database state so it can be
        restored later without re-building from raw data.
        
        Parameters:
        - filepath: Path for the output file (will add .pkl extension if not present)
        - verbose: Print progress
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.pkl')
        
        if verbose:
            print(f"Saving complete database to {filepath}...")
        
        # Prepare data for serialization
        # Convert geometries to WKB (Well-Known Binary) for efficient storage
        records_data = {}
        for teryt_id, record in self._records.items():
            rec_dict = {
                'teryt_id': record.teryt_id,
                'name': record.name,
                'name_dod': record.name_dod,
                'level': record.level,
                'kind': record.kind,
                'years_valid': record.years_valid,
                'past_names': record.past_names,
                'past_teryt_ids': record.past_teryt_ids,
                'past_levels': record.past_levels,
                'past_kinds': record.past_kinds,
                'changes': record.changes,
                'has_changes': record.has_changes,
                'geometry_year': record.geometry_year,
                'geometry_notes': record.geometry_notes,  # NEW in v3.1
                'old_woj': record.old_woj,
                'old_woj_id': record.old_woj_id,
                'historical_codes': record.historical_codes,  # NEW in v3.1
                'code_by_year': record.code_by_year,  # NEW in v3.1
                'geometry_wkb': record.geometry.wkb if record.geometry else None,
                'geometry_best_candidate_wkb': record.geometry_best_candidate.wkb if record.geometry_best_candidate else None,  # NEW in v3.1
                'has_geometry': record.has_geometry,
                'parent_teryt_id': record.parent_id,  # NEW in v3.1
                'child_teryt_ids': record.children_ids,  # NEW in v3.1
                # Data storage (NEW in v4.0)
                'data': {f"{k[0]}|{k[1]}|{k[2]}": v.to_dict() for k, v in record.data.items()} if record.data else None,
                # Cross table storage (NEW in v4.1)
                'cross_tables': {k: v.to_dict() for k, v in record.cross_tables.items()} if record.cross_tables else None,
                # Population and classification (NEW in v4.2)
                'pop': {ts.year: v for ts, v in record.pop.items() if not pd.isna(v)} if record.pop.notna().any() else None,
                'pop_class': record.pop_class.to_dict('index') if len(record.pop_class) > 0 else None
            }
            records_data[teryt_id] = rec_dict
        
        # Prepare geometry GeoDataFrames for storage (as WKB)
        geometries_data = {}
        for year, gdf in self._geometries.items():
            # Store just essential columns with geometry as WKB
            gdf_copy = gdf.copy()
            gdf_copy['geometry_wkb'] = gdf_copy.geometry.apply(lambda g: g.wkb if g else None)
            geometries_data[year] = gdf_copy.drop(columns=['geometry']).to_dict('records')
        
        # Poland boundary
        poland_wkb = self._poland_boundary.wkb if self._poland_boundary else None
        
        save_data = {
            'version': '4.2',
            'records': records_data,
            'by_year': {k: list(v) for k, v in self._by_year.items()},
            'by_name': {k: list(v) for k, v in self._by_name.items()},
            'by_level': {k: list(v) for k, v in self._by_level.items()},
            'by_kind': {k: list(v) for k, v in self._by_kind.items()},
            'by_voivodeship': {k: list(v) for k, v in self._by_voivodeship.items()},
            'id_transitions': self._id_transitions,
            'geometries': geometries_data,
            'poland_boundary_wkb': poland_wkb,
            'year_range': self._year_range,
            'crs': self._crs,
            'built': self._built
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if verbose:
            file_size = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved {len(records_data)} records")
            n_with_data = sum(1 for r in self._records.values() if r.has_data)
            if n_with_data > 0:
                print(f"  ✓ Records with data: {n_with_data}")
            n_with_ct = sum(1 for r in self._records.values() if r.has_cross_tables)
            if n_with_ct > 0:
                print(f"  ✓ Records with cross tables: {n_with_ct}")
            print(f"  ✓ File size: {file_size:.1f} MB")
            print(f"  ✓ Path: {filepath}")
    
    def get_year_statistics(self, year: int) -> dict:
        """Get statistics for a specific year."""
        if year not in self._by_year:
            return {'error': f'Year {year} not in database'}
        
        teryt_ids = self._by_year[year]
        records = [self._records[tid] for tid in teryt_ids]
        
        level_counts = {}
        kind_counts = {}
        
        for r in records:
            level = r.level
            if level is not None:
                level = int(level)
                if level not in level_counts:
                    level_counts[level] = 0
                level_counts[level] += 1
            
            kind = r.kind
            if kind:
                if kind not in kind_counts:
                    kind_counts[kind] = 0
                kind_counts[kind] += 1
        
        return {
            'year': year,
            'total_units': len(records),
            'level_counts': level_counts,
            'kind_counts': kind_counts,
            'units_with_geometry': sum(1 for r in records if r.has_geometry),
            'units_with_changes': sum(1 for r in records if r.has_changes),
        }
    
    # ==========================================================================
    # CONVENIENCE PROPERTIES
    # ==========================================================================
    
    @property
    def years(self) -> List[int]:
        """List of all years in the database."""
        return sorted(self._by_year.keys())
    
    @property
    def n_records(self) -> int:
        """Total number of records."""
        return len(self._records)
    
    @property
    def is_built(self) -> bool:
        """Whether the database has been built."""
        return self._built
    
    def __len__(self):
        return len(self._records)
    
    def __repr__(self):
        if self._built:
            return f"GeoTERYTDatabase({len(self._records)} records, {self._year_range[0]}-{self._year_range[1]})"
        else:
            return "GeoTERYTDatabase(not built)"


# ==============================================================================
# MODULE-LEVEL FUNCTION FOR LOADING COMPLETE DATABASE (NEW in v3.0)
# ==============================================================================

def load_complete_database(filepath: Union[str, Path], verbose: bool = True) -> GeoTERYTDatabase:
    """
    Load a complete GeoTERYT database from a saved file.
    
    NEW in v3.0: Restores the entire database state, including all geometries,
    from a single pickle file created by db.save_complete().
    
    Parameters:
    - filepath: Path to the saved database file (.pkl)
    - verbose: Print progress
    
    Returns:
    - GeoTERYTDatabase: Fully restored database ready to use
    """
    filepath = Path(filepath)
    
    if verbose:
        print(f"Loading complete database from {filepath}...")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Check version
    version = data.get('version', 'unknown')
    if verbose:
        print(f"  Database version: {version}")
    
    # Create new database instance
    db = GeoTERYTDatabase()
    
    # Restore basic attributes
    db._year_range = data['year_range']
    db._crs = data['crs']
    db._built = data['built']
    db._id_transitions = data.get('id_transitions', {})
    
    # Restore Poland boundary
    poland_wkb = data.get('poland_boundary_wkb')
    if poland_wkb:
        db._poland_boundary = wkb.loads(poland_wkb)
    
    # Restore records
    records_data = data['records']
    for teryt_id, rec_dict in records_data.items():
        record = TERYTRecord(
            teryt_id=rec_dict['teryt_id'],
            name=rec_dict['name'],
            name_dod=rec_dict.get('name_dod'),
            level=rec_dict.get('level'),
            kind=rec_dict.get('kind')
        )
        record.years_valid = set(rec_dict.get('years_valid', []))
        record.past_names = rec_dict.get('past_names', [])
        record.past_teryt_ids = rec_dict.get('past_teryt_ids', [])
        record.past_levels = rec_dict.get('past_levels', [])
        record.past_kinds = rec_dict.get('past_kinds', [])
        record.changes = rec_dict.get('changes', [])
        record.has_changes = rec_dict.get('has_changes', False)
        record.geometry_year = rec_dict.get('geometry_year')
        record.geometry_notes = rec_dict.get('geometry_notes')  # NEW in v3.1
        record.old_woj = rec_dict.get('old_woj')
        record.old_woj_id = rec_dict.get('old_woj_id')
        record.historical_codes = rec_dict.get('historical_codes', [])  # NEW in v3.1
        record.code_by_year = rec_dict.get('code_by_year', {})  # NEW in v3.1
        record.parent_id = rec_dict.get('parent_teryt_id')  # NEW in v3.1
        record.children_ids = rec_dict.get('child_teryt_ids', [])  # NEW
        
        # Restore geometry from WKB
        geom_wkb = rec_dict.get('geometry_wkb')
        if geom_wkb:
            record.geometry = wkb.loads(geom_wkb)
        
        # Restore geometry_best_candidate from WKB (NEW in v3.1)
        geom_cand_wkb = rec_dict.get('geometry_best_candidate_wkb')
        if geom_cand_wkb:
            record.geometry_best_candidate = wkb.loads(geom_cand_wkb)
        
        # Restore data (NEW in v4.0)
        data_dict = rec_dict.get('data')
        if data_dict:
            for key_str, series_dict in data_dict.items():
                ds = DataSeries.from_dict(series_dict)
                record.data[ds.key] = ds
        
        # Restore cross tables (NEW in v4.1)
        ct_dict = rec_dict.get('cross_tables')
        if ct_dict:
            for sid, ct_data in ct_dict.items():
                record.cross_tables[sid] = CrossTable.from_dict(ct_data)
        
        # Restore population and classification (NEW in v4.2)
        pop_dict = rec_dict.get('pop')
        if pop_dict:
            for yr, val in pop_dict.items():
                ts = pd.Timestamp(year=int(yr), month=1, day=1)
                if ts in record.pop.index:
                    record.pop[ts] = float(val)
        
        pop_class_dict = rec_dict.get('pop_class')
        if pop_class_dict:
            rows = []
            for ts_str, vals in pop_class_dict.items():
                code = vals['pop_class_code']
                rows.append({
                    'date': pd.Timestamp(ts_str),
                    'pop_class_code': code if not pd.isna(code) else np.nan,
                    'pop_class_label': str(vals['pop_class_label'])
                })
            if rows:
                pc_df = pd.DataFrame(rows).set_index('date')
                record.pop_class = pc_df
        
        db._records[teryt_id] = record
    
    # Restore indices
    db._by_year = {int(k): set(v) for k, v in data['by_year'].items()}
    db._by_name = {k: set(v) for k, v in data['by_name'].items()}
    db._by_level = {int(k): set(v) for k, v in data['by_level'].items()}
    db._by_kind = {k: set(v) for k, v in data['by_kind'].items()}
    db._by_voivodeship = {k: set(v) for k, v in data['by_voivodeship'].items()}
    
    # Restore geometry GeoDataFrames (NEW in v3.1)
    # Required for the new geometry assignment workflow
    geometries_data = data.get('geometries', {})
    if geometries_data:
        for year_str, records_list in geometries_data.items():
            year = int(year_str)
            # Reconstruct GeoDataFrame from records
            gdf_data = []
            geometries = []
            for rec in records_list:
                geom_wkb = rec.pop('geometry_wkb', None)
                gdf_data.append(rec)
                if geom_wkb:
                    geometries.append(wkb.loads(geom_wkb))
                else:
                    geometries.append(None)
            
            gdf = gpd.GeoDataFrame(gdf_data, geometry=geometries, crs=db._crs)
            db._geometries[year] = gdf
        
        if verbose:
            print(f"  ✓ Restored geometry data for years: {sorted(db._geometries.keys())}")
    
    if verbose:
        print(f"  ✓ Loaded {len(db._records)} records")
        print(f"  ✓ Year range: {db._year_range[0]} - {db._year_range[1]}")
        geom_count = sum(1 for r in db._records.values() if r.has_geometry)
        print(f"  ✓ Records with geometry: {geom_count}")
        old_woj_count = sum(1 for r in db._records.values() if r.old_woj)
        if old_woj_count > 0:
            print(f"  ✓ Records with old_woj: {old_woj_count}")
        data_count = sum(1 for r in db._records.values() if r.has_data)
        if data_count > 0:
            print(f"  ✓ Records with data: {data_count}")
        ct_count = sum(1 for r in db._records.values() if r.has_cross_tables)
        if ct_count > 0:
            print(f"  ✓ Records with cross tables: {ct_count}")
        pop_count = sum(1 for r in db._records.values() if r.pop.notna().any())
        if pop_count > 0:
            print(f"  ✓ Records with population data: {pop_count}")
        pc_count = sum(1 for r in db._records.values() if len(r.pop_class) > 0)
        if pc_count > 0:
            print(f"  ✓ Records with pop_class: {pc_count}")
    
    return db