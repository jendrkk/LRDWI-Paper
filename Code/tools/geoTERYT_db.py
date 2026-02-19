"""
GeoTERYT Database Module - Version 3.1
======================================

A comprehensive database system for Polish administrative divisions (TERYT) 
with full geometry support, historical tracking, and overlay operations.

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

Author: Generated for LRDWI-Paper project
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
            'code_by_year': self.code_by_year
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
    
    def set_old_voivodships(self, old_voiv_gdf: gpd.GeoDataFrame):
        ...
    
    def get_poland_boundary(self) -> gpd.GeoDataFrame:
        
        if self._poland_gdf is not None:
            return self._poland_gdf
        
    def get_poland_gdf(self) -> gpd.GeoDataFrame:
        
        if self._poland_boundary is not None:
            return self._poland_boundary
    
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
        ...
    
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
        print("=" * 60)
    
    # ==========================================================================
    # PERSISTENCE METHODS (NEW in v3.0)
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
                'geometry_best_candidate_wkb': record.geometry_best_candidate.wkb if record.geometry_best_candidate else None  # NEW in v3.1
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
            'version': '3.1',
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
        
        # Restore geometry from WKB
        geom_wkb = rec_dict.get('geometry_wkb')
        if geom_wkb:
            record.geometry = wkb.loads(geom_wkb)
        
        # Restore geometry_best_candidate from WKB (NEW in v3.1)
        geom_cand_wkb = rec_dict.get('geometry_best_candidate_wkb')
        if geom_cand_wkb:
            record.geometry_best_candidate = wkb.loads(geom_cand_wkb)
        
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
    
    return db