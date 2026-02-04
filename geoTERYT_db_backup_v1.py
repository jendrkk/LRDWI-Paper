"""
GeoTERYT Database Module
========================

A comprehensive database system for Polish administrative divisions (TERYT) 
with full geometry support, historical tracking, and overlay operations.

This module provides the GeoTERYTDatabase class that:
- Stores all administrative divisions from 1999-2024 with their changes
- Integrates geometry data from shapefiles (2005, 2011, 2012, 2017, etc.)
- Provides search functionality by ID, name, year, level, and kind
- Supports geometry operations (merging, best geometry selection)
- Enables overlay operations (e.g., assign gminas to pre-1999 voivodeships)

Author: Generated for LRDWI-Paper project
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
import warnings
import re
import os


# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# Pre-1999 Polish voivodeships (49 voivodeships system, 1975-1998)
# Maps voivodeship name to approximate centroid coordinates (EPSG:2180)
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


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def nuts_code_to_teryt(nuts_code: str) -> str:
    """
    Converts a NUTS code to a TERYT code.
    
    Parameters:
    - nuts_code (str): The NUTS code.
    
    Returns:
    - str: The corresponding TERYT code.
    """
    k = 3
    gmina_id = nuts_code[-k:]
    powiat_id = nuts_code[-(k+2):-k]
    if len(nuts_code) == 11:
        voivodeship_id = nuts_code[1:3]
    else:
        voivodeship_id = nuts_code[2:4]
    teryt_code = voivodeship_id + powiat_id + gmina_id
    return teryt_code


def teryt_to_short(teryt: str) -> str:
    """
    Converts a full 7-digit TERYT code to a 6-digit short version (without RODZ).
    
    Parameters:
    - teryt (str): Full 7-digit TERYT code.
    
    Returns:
    - str: 6-digit short TERYT (WOJ+POW+GMI).
    """
    return str(teryt)[:6]


def teryt_parent(teryt: str, level: int) -> str:
    """
    Gets the parent TERYT code at a higher administrative level.
    
    Parameters:
    - teryt (str): TERYT code.
    - level (int): Target level (2=voivodeship, 5=powiat).
    
    Returns:
    - str: Parent TERYT code padded with zeros.
    """
    teryt = str(teryt).zfill(7)
    if level == LEVEL_VOIVODESHIP:
        return teryt[:2] + "00000"
    elif level == LEVEL_POWIAT:
        return teryt[:4] + "000"
    else:
        return teryt


# ==============================================================================
# DATABASE RECORD CLASS
# ==============================================================================

class TERYTRecord:
    """
    Represents a single administrative division record with full metadata.
    
    Attributes:
    - teryt_id: 7-digit TERYT code (WOJ+POW+GMI+RODZ)
    - name: Name of the division
    - name_dod: Additional name/designation
    - level: Administrative level (2, 5, or 6)
    - kind: Type of division (urban, rural, urban-rural, etc.)
    - years_valid: Set of years when this division existed with this ID
    - past_names: List of previous names
    - past_teryt_ids: List of previous TERYT IDs
    - changes: List of change records
    - geometry: Shapely geometry object (optional)
    - geometry_year: Year of the geometry source
    """
    
    def __init__(self, teryt_id: str, name: str, name_dod: str = None,
                 level: int = None, kind: str = None):
        self.teryt_id = str(teryt_id).zfill(7)
        self.name = name
        self.name_dod = name_dod
        self.level = level
        self.kind = kind
        self.years_valid: set = set()
        self.past_names: List[str] = []
        self.past_teryt_ids: List[str] = []
        self.changes: List[dict] = []
        self.geometry = None
        self.geometry_year: Optional[int] = None
        
        # Additional metadata
        self.woj = self.teryt_id[:2]
        self.pow = self.teryt_id[2:4]
        self.gmi = self.teryt_id[4:6]
        self.rodz = self.teryt_id[6]
    
    def add_year(self, year: int):
        """Add a year when this division was valid."""
        self.years_valid.add(year)
    
    def add_past_name(self, name: str):
        """Add a previous name if not already present."""
        if name and name not in self.past_names and name != self.name:
            self.past_names.append(name)
    
    def add_past_teryt_id(self, teryt_id: str):
        """Add a previous TERYT ID if not already present."""
        teryt_id = str(teryt_id).zfill(7)
        if teryt_id and teryt_id not in self.past_teryt_ids and teryt_id != self.teryt_id:
            self.past_teryt_ids.append(teryt_id)
    
    def add_change(self, change: dict):
        """Add a change record."""
        self.changes.append(change)
    
    def set_geometry(self, geometry, year: int):
        """Set the geometry and its source year."""
        self.geometry = geometry
        self.geometry_year = year
    
    @property
    def first_year(self) -> Optional[int]:
        """First year this division existed."""
        return min(self.years_valid) if self.years_valid else None
    
    @property
    def last_year(self) -> Optional[int]:
        """Last year this division existed."""
        return max(self.years_valid) if self.years_valid else None
    
    @property
    def has_geometry(self) -> bool:
        """Check if geometry is available."""
        return self.geometry is not None
    
    @property
    def short_teryt(self) -> str:
        """6-digit TERYT without RODZ."""
        return self.teryt_id[:6]
    
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
            'changes': self.changes,
            'has_geometry': self.has_geometry,
            'geometry_year': self.geometry_year
        }
    
    def __repr__(self):
        return f"TERYTRecord({self.teryt_id}, {self.name}, years={self.first_year}-{self.last_year})"


# ==============================================================================
# MAIN DATABASE CLASS
# ==============================================================================

class GeoTERYTDatabase:
    """
    Comprehensive database for Polish administrative divisions with geometry support.
    
    This class provides:
    - Storage of all administrative divisions from 1999-2024
    - Historical tracking (name changes, ID changes, structural changes)
    - Geometry integration from multiple shapefile sources
    - Search functionality (by ID, name, year, level, kind)
    - Geometry operations (merging to higher levels, best geometry selection)
    - Overlay operations (assign gminas to historical voivodeships)
    
    Usage:
        db = GeoTERYTDatabase()
        db.build_from_harmonized(mega_df)
        db.load_geometries(gdf_list)
        
        # Search
        divisions = db.get_divisions_by_year(2020, level=6)
        history = db.get_unit_info('1461011')
        
        # Geometry operations
        gdf = db.to_geodataframe(year=2020, level=6)
        voivodeship_gdf = db.merge_to_level(year=2020, target_level=2)
    """
    
    def __init__(self):
        """Initialize an empty database."""
        # Main storage: teryt_id -> TERYTRecord
        self._records: Dict[str, TERYTRecord] = {}
        
        # Indices for fast lookup
        self._by_year: Dict[int, set] = {}  # year -> set of teryt_ids
        self._by_name: Dict[str, set] = {}  # lowercase name -> set of teryt_ids
        self._by_level: Dict[int, set] = {}  # level -> set of teryt_ids
        self._by_kind: Dict[str, set] = {}  # kind -> set of teryt_ids
        self._by_voivodeship: Dict[str, set] = {}  # woj code -> set of teryt_ids
        
        # Geometry storage: {year: GeoDataFrame}
        self._geometries: Dict[int, gpd.GeoDataFrame] = {}
        
        # Metadata
        self._year_range: Tuple[int, int] = (1999, 2024)
        self._crs = "EPSG:2180"
        self._built = False
    
    # ==========================================================================
    # DATABASE CONSTRUCTION
    # ==========================================================================
    
    def build_from_harmonized(self, mega_df: pd.DataFrame, verbose: bool = True):
        """
        Build the database from a harmonized TERYT mega DataFrame.
        
        Parameters:
        - mega_df (pd.DataFrame): Output from harmonize_teryt() function.
        - verbose (bool): Print progress information.
        """
        if verbose:
            print("Building GeoTERYT database from harmonized data...")
        
        # Clear existing data
        self._records.clear()
        self._by_year.clear()
        self._by_name.clear()
        self._by_level.clear()
        self._by_kind.clear()
        self._by_voivodeship.clear()
        
        # Get year range from data
        years = sorted(mega_df['year'].unique())
        self._year_range = (min(years), max(years))
        
        # Group by final TERYT ID (use the latest valid ID for each unit)
        # We need to track the evolution of each unit
        
        total_rows = len(mega_df)
        processed = 0
        
        for year in years:
            year_data = mega_df[mega_df['year'] == year]
            
            for _, row in year_data.iterrows():
                teryt_id = str(row.get('id', '')).zfill(7)
                
                # Skip invalid IDs
                if teryt_id == '0000000' or len(teryt_id) != 7:
                    continue
                
                # Get or create record
                if teryt_id not in self._records:
                    self._records[teryt_id] = TERYTRecord(
                        teryt_id=teryt_id,
                        name=row.get('NAZWA', ''),
                        name_dod=row.get('NAZWA_DOD', ''),
                        level=row.get('level'),
                        kind=row.get('kind')
                    )
                
                record = self._records[teryt_id]
                
                # Add this year
                record.add_year(year)
                
                # Update indices
                if year not in self._by_year:
                    self._by_year[year] = set()
                self._by_year[year].add(teryt_id)
                
                # Track name changes
                current_name = row.get('NAZWA', '')
                if current_name and current_name != record.name:
                    record.add_past_name(current_name)
                
                # Track changes from notes
                notes = row.get('notes', {})
                if isinstance(notes, dict) and notes.get('changes'):
                    for change in notes['changes']:
                        change_dict = {
                            'year': year,
                            'description': change
                        }
                        if change_dict not in record.changes:
                            record.add_change(change_dict)
                
                processed += 1
            
            if verbose and year % 5 == 0:
                print(f"  Processed year {year}...")
        
        # Build remaining indices
        for teryt_id, record in self._records.items():
            # Name index (lowercase)
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
            woj = record.woj
            if woj not in self._by_voivodeship:
                self._by_voivodeship[woj] = set()
            self._by_voivodeship[woj].add(teryt_id)
        
        self._built = True
        
        if verbose:
            print(f"Database built successfully!")
            print(f"  Total records: {len(self._records)}")
            print(f"  Year range: {self._year_range[0]} - {self._year_range[1]}")
            print(f"  Voivodeships: {len(self._by_level.get(2, set()))}")
            print(f"  Powiats: {len(self._by_level.get(5, set()))}")
            print(f"  Gminas: {len(self._by_level.get(6, set()))}")
    
    def load_geometries(self, gdf_dict: Dict[str, gpd.GeoDataFrame], 
                        teryt_column: str = 'teryt',
                        verbose: bool = True):
        """
        Load geometries from a dictionary of GeoDataFrames.
        
        Parameters:
        - gdf_dict (dict): Dictionary mapping year/key strings to GeoDataFrames.
            Keys should contain 4-digit year (e.g., "2017_gminy", "2005_Obszary").
        - teryt_column (str): Column name containing TERYT codes.
        - verbose (bool): Print progress information.
        """
        if verbose:
            print("Loading geometries into database...")
        
        for key, gdf in gdf_dict.items():
            # Extract year from key
            match = re.search(r'\d{4}', key)
            if not match:
                if verbose:
                    print(f"  Skipping {key}: no year found in key")
                continue
            
            year = int(match.group(0))
            
            if verbose:
                print(f"  Processing {key} (year {year})...")
            
            # Ensure CRS
            if gdf.crs is None:
                gdf = gdf.set_crs(self._crs)
            elif gdf.crs.to_string() != self._crs:
                gdf = gdf.to_crs(self._crs)
            
            # Store the GeoDataFrame
            self._geometries[year] = gdf.copy()
            
            # Link geometries to records
            if teryt_column in gdf.columns:
                geometry_linked = 0
                for _, row in gdf.iterrows():
                    teryt = str(row[teryt_column]).zfill(7)
                    
                    # For older files, TERYT might be 6 digits (without RODZ)
                    # Try to find matching records
                    if len(teryt) == 6:
                        teryt = teryt + '0'  # Try with RODZ=0
                    
                    if teryt in self._records:
                        # Only update if no geometry or newer geometry
                        record = self._records[teryt]
                        if not record.has_geometry or year > record.geometry_year:
                            record.set_geometry(row.geometry, year)
                            geometry_linked += 1
                    else:
                        # Try matching with any RODZ
                        short_teryt = teryt[:6]
                        for teryt_id, record in self._records.items():
                            if teryt_id[:6] == short_teryt:
                                if not record.has_geometry or year > record.geometry_year:
                                    record.set_geometry(row.geometry, year)
                                    geometry_linked += 1
                                    break
                
                if verbose:
                    print(f"    Linked {geometry_linked} geometries")
        
        if verbose:
            total_with_geometry = sum(1 for r in self._records.values() if r.has_geometry)
            print(f"  Total records with geometry: {total_with_geometry}/{len(self._records)}")
    
    def load_geometries_from_path(self, geometry_root: Path, verbose: bool = True):
        """
        Load geometries from a root directory containing geometry folders.
        
        Parameters:
        - geometry_root (Path): Root directory containing geometry subdirectories.
        - verbose (bool): Print progress information.
        """
        if verbose:
            print(f"Scanning for geometry files in {geometry_root}...")
        
        gdf_dict = {}
        
        for folder in geometry_root.iterdir():
            if not folder.is_dir() or folder.name.startswith('.'):
                continue
            
            # Find shapefiles
            for file in folder.iterdir():
                if file.suffix.lower() == '.shp':
                    # Check if it's a gmina-level file
                    if 'obszar' in file.name.lower() or 'gmin' in file.name.lower():
                        key = f"{folder.name}_{file.stem}"
                        try:
                            gdf = gpd.read_file(file)
                            gdf_dict[key] = gdf
                            if verbose:
                                print(f"  Loaded {key}: {len(gdf)} features")
                        except Exception as e:
                            if verbose:
                                print(f"  Error loading {file}: {e}")
        
        if gdf_dict:
            self.load_geometries(gdf_dict, verbose=verbose)
    
    # ==========================================================================
    # SEARCH METHODS
    # ==========================================================================
    
    def get_by_teryt_id(self, teryt_id: str) -> Optional[TERYTRecord]:
        """
        Get a record by its TERYT ID.
        
        Parameters:
        - teryt_id (str): 7-digit TERYT code.
        
        Returns:
        - TERYTRecord or None if not found.
        """
        teryt_id = str(teryt_id).zfill(7)
        return self._records.get(teryt_id)
    
    def get_unit_info(self, teryt_id: str) -> Optional[dict]:
        """
        Get detailed information about a unit by its TERYT ID.
        
        Parameters:
        - teryt_id (str): 7-digit TERYT code.
        
        Returns:
        - dict with unit information or None if not found.
        """
        record = self.get_by_teryt_id(teryt_id)
        return record.to_dict() if record else None
    
    def search_by_name(self, name: str, exact: bool = False) -> List[TERYTRecord]:
        """
        Search for divisions by name.
        
        Parameters:
        - name (str): Name to search for.
        - exact (bool): If True, require exact match; if False, allow partial match.
        
        Returns:
        - List of matching TERYTRecord objects.
        """
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
                               exclude_subdivisions: bool = True) -> List[TERYTRecord]:
        """
        Get all divisions valid in a specific year.
        
        Parameters:
        - year (int): Year to query.
        - level (int, optional): Filter by administrative level (2, 5, or 6).
        - kind (str, optional): Filter by kind (urban, rural, urban-rural, etc.).
        - exclude_subdivisions (bool): If True, exclude RODZ 4 and 5 (sub-gmina units).
        
        Returns:
        - List of TERYTRecord objects.
        """
        if year not in self._by_year:
            return []
        
        teryt_ids = self._by_year[year]
        
        # Apply level filter
        if level is not None:
            level_ids = self._by_level.get(level, set())
            teryt_ids = teryt_ids & level_ids
        
        # Apply kind filter
        if kind is not None:
            kind_ids = self._by_kind.get(kind, set())
            teryt_ids = teryt_ids & kind_ids
        
        # Get records
        records = [self._records[tid] for tid in teryt_ids]
        
        # Exclude subdivisions if requested
        if exclude_subdivisions:
            records = [r for r in records if r.rodz not in RODZ_SUB_DIVISIONS]
        
        return records
    
    def get_divisions_by_level(self, level: int) -> List[TERYTRecord]:
        """
        Get all divisions at a specific administrative level.
        
        Parameters:
        - level (int): Administrative level (2=voivodeship, 5=powiat, 6=gmina).
        
        Returns:
        - List of TERYTRecord objects.
        """
        teryt_ids = self._by_level.get(level, set())
        return [self._records[tid] for tid in teryt_ids]
    
    def get_gminas_in_voivodeship(self, woj_code: str, year: Optional[int] = None) -> List[TERYTRecord]:
        """
        Get all gminas in a voivodeship.
        
        Parameters:
        - woj_code (str): 2-digit voivodeship code.
        - year (int, optional): If provided, filter to gminas valid in that year.
        
        Returns:
        - List of TERYTRecord objects.
        """
        woj_code = str(woj_code).zfill(2)
        
        if woj_code not in self._by_voivodeship:
            return []
        
        teryt_ids = self._by_voivodeship[woj_code]
        
        # Filter to gminas (level 6)
        gmina_ids = self._by_level.get(LEVEL_GMINA, set())
        teryt_ids = teryt_ids & gmina_ids
        
        # Filter by year if provided
        if year is not None:
            year_ids = self._by_year.get(year, set())
            teryt_ids = teryt_ids & year_ids
        
        records = [self._records[tid] for tid in teryt_ids]
        
        # Exclude subdivisions
        records = [r for r in records if r.rodz not in RODZ_SUB_DIVISIONS]
        
        return records
    
    def get_changed_units(self, year_from: Optional[int] = None, 
                          year_to: Optional[int] = None) -> List[TERYTRecord]:
        """
        Get units that experienced changes in a time period.
        
        Parameters:
        - year_from (int, optional): Start year (inclusive).
        - year_to (int, optional): End year (inclusive).
        
        Returns:
        - List of TERYTRecord objects with changes.
        """
        results = []
        
        for record in self._records.values():
            if record.changes:
                has_relevant_change = False
                for change in record.changes:
                    change_year = change.get('year', 0)
                    if year_from is not None and change_year < year_from:
                        continue
                    if year_to is not None and change_year > year_to:
                        continue
                    has_relevant_change = True
                    break
                
                if has_relevant_change:
                    results.append(record)
        
        return results
    
    # ==========================================================================
    # GEOMETRY METHODS
    # ==========================================================================
    
    def get_geometry(self, teryt_id: str, year: Optional[int] = None):
        """
        Get geometry for a specific TERYT ID.
        
        Parameters:
        - teryt_id (str): 7-digit TERYT code.
        - year (int, optional): Preferred geometry year; if None, use best available.
        
        Returns:
        - Shapely geometry or None if not available.
        """
        record = self.get_by_teryt_id(teryt_id)
        
        if record is None:
            return None
        
        # If record has geometry directly assigned
        if record.has_geometry:
            return record.geometry
        
        # Try to find geometry from stored GeoDataFrames
        return self._find_best_geometry(teryt_id, year)
    
    def _find_best_geometry(self, teryt_id: str, target_year: Optional[int] = None):
        """
        Find the best available geometry for a TERYT ID.
        
        The "best" geometry is:
        1. Exact year match if target_year specified
        2. Closest year with available geometry
        3. Newest geometry overall
        
        Parameters:
        - teryt_id (str): TERYT code.
        - target_year (int, optional): Preferred year.
        
        Returns:
        - Shapely geometry or None.
        """
        teryt_id = str(teryt_id).zfill(7)
        short_teryt = teryt_id[:6]
        
        # Collect available geometries
        available = []
        
        for year, gdf in self._geometries.items():
            teryt_col = 'teryt' if 'teryt' in gdf.columns else 'jpt_kod_je'
            if teryt_col not in gdf.columns:
                continue
            
            # Look for exact match or short match
            mask = (gdf[teryt_col].astype(str).str.zfill(7) == teryt_id) | \
                   (gdf[teryt_col].astype(str).str[:6] == short_teryt)
            
            matches = gdf[mask]
            if len(matches) > 0:
                available.append((year, matches.iloc[0].geometry))
        
        if not available:
            return None
        
        # Sort by preference
        if target_year is not None:
            # Sort by distance from target year
            available.sort(key=lambda x: abs(x[0] - target_year))
        else:
            # Sort by recency (newest first)
            available.sort(key=lambda x: -x[0])
        
        return available[0][1]
    
    def to_geodataframe(self, year: Optional[int] = None, 
                        level: Optional[int] = None,
                        kind: Optional[str] = None,
                        exclude_subdivisions: bool = True,
                        include_all_attributes: bool = True) -> gpd.GeoDataFrame:
        """
        Convert database records to a GeoDataFrame.
        
        Parameters:
        - year (int, optional): Filter to divisions valid in this year.
        - level (int, optional): Filter by administrative level.
        - kind (str, optional): Filter by kind.
        - exclude_subdivisions (bool): Exclude RODZ 4 and 5.
        - include_all_attributes (bool): Include all record attributes.
        
        Returns:
        - GeoDataFrame with divisions and their geometries.
        """
        # Get records
        if year is not None:
            records = self.get_divisions_by_year(year, level=level, kind=kind,
                                                   exclude_subdivisions=exclude_subdivisions)
        else:
            records = list(self._records.values())
            
            if level is not None:
                records = [r for r in records if r.level == level]
            if kind is not None:
                records = [r for r in records if r.kind == kind]
            if exclude_subdivisions:
                records = [r for r in records if r.rodz not in RODZ_SUB_DIVISIONS]
        
        # Build data
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
                    'n_changes': len(record.changes),
                    'past_names': ', '.join(record.past_names) if record.past_names else None,
                    'past_teryt_ids': ', '.join(record.past_teryt_ids) if record.past_teryt_ids else None,
                })
            
            data.append(row_data)
            
            # Get geometry
            geom = self.get_geometry(record.teryt_id, year)
            geometries.append(geom)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs=self._crs)
        
        return gdf
    
    def merge_to_level(self, year: int, target_level: int) -> gpd.GeoDataFrame:
        """
        Merge geometries to a higher administrative level.
        
        For example, merge gminas (level 6) to powiats (level 5) or voivodeships (level 2).
        
        Parameters:
        - year (int): Year for source geometries.
        - target_level (int): Target administrative level (2 or 5).
        
        Returns:
        - GeoDataFrame with merged geometries at target level.
        """
        # Get gmina-level GeoDataFrame
        gdf = self.to_geodataframe(year=year, level=LEVEL_GMINA, 
                                   exclude_subdivisions=True)
        
        if len(gdf) == 0:
            return gpd.GeoDataFrame(columns=['teryt_id', 'name', 'geometry'], crs=self._crs)
        
        # Determine grouping column
        if target_level == LEVEL_VOIVODESHIP:
            gdf['group_id'] = gdf['woj']
        elif target_level == LEVEL_POWIAT:
            gdf['group_id'] = gdf['woj'] + gdf['pow']
        else:
            raise ValueError(f"Invalid target level: {target_level}")
        
        # Remove rows with missing geometry
        gdf_valid = gdf[gdf.geometry.notna()].copy()
        
        if len(gdf_valid) == 0:
            return gpd.GeoDataFrame(columns=['teryt_id', 'name', 'geometry'], crs=self._crs)
        
        # Merge geometries
        merged = gdf_valid.dissolve(by='group_id', as_index=False)
        
        # Add names from parent records
        merged_data = []
        for _, row in merged.iterrows():
            group_id = row['group_id']
            
            # Find parent record
            if target_level == LEVEL_VOIVODESHIP:
                parent_teryt = group_id + "00000"
            else:  # LEVEL_POWIAT
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
        - gminas_gdf (GeoDataFrame): GeoDataFrame with gmina geometries.
        - regions_gdf (GeoDataFrame): GeoDataFrame with region boundaries.
        - region_id_column (str): Column name for region identifier in regions_gdf.
        - method (str): Assignment method:
            - 'centroid': Assign based on where gmina centroid falls
            - 'area': Assign based on largest intersection area
            - 'contains': Assign only if region fully contains gmina
        
        Returns:
        - GeoDataFrame: gminas_gdf with added column for region assignment.
        """
        result = gminas_gdf.copy()
        result['assigned_region'] = None
        
        # Ensure same CRS
        if regions_gdf.crs != gminas_gdf.crs:
            regions_gdf = regions_gdf.to_crs(gminas_gdf.crs)
        
        for idx, gmina_row in result.iterrows():
            geom = gmina_row.geometry
            
            if geom is None or geom.is_empty:
                continue
            
            if method == 'centroid':
                centroid = geom.centroid
                for _, region_row in regions_gdf.iterrows():
                    if region_row.geometry.contains(centroid):
                        result.at[idx, 'assigned_region'] = region_row[region_id_column]
                        break
                        
            elif method == 'area':
                max_area = 0
                best_region = None
                for _, region_row in regions_gdf.iterrows():
                    try:
                        intersection = geom.intersection(region_row.geometry)
                        area = intersection.area
                        if area > max_area:
                            max_area = area
                            best_region = region_row[region_id_column]
                    except:
                        continue
                result.at[idx, 'assigned_region'] = best_region
                
            elif method == 'contains':
                for _, region_row in regions_gdf.iterrows():
                    if region_row.geometry.contains(geom):
                        result.at[idx, 'assigned_region'] = region_row[region_id_column]
                        break
        
        return result
    
    def assign_gminas_to_pre1999_voivodeships(self, year: int,
                                               pre1999_boundaries_gdf: gpd.GeoDataFrame,
                                               voivodeship_name_col: str = 'name',
                                               method: str = 'centroid') -> gpd.GeoDataFrame:
        """
        Assign gminas from a given year to pre-1999 voivodeship boundaries.
        
        This is useful for comparing data across the administrative reform
        of 1999 when Poland changed from 49 to 16 voivodeships.
        
        Parameters:
        - year (int): Year for which to get gminas.
        - pre1999_boundaries_gdf (GeoDataFrame): Pre-1999 voivodeship boundaries.
        - voivodeship_name_col (str): Column with voivodeship names in boundaries GDF.
        - method (str): Assignment method ('centroid', 'area', or 'contains').
        
        Returns:
        - GeoDataFrame: Gminas with 'pre1999_voivodeship' column added.
        """
        gminas_gdf = self.to_geodataframe(year=year, level=LEVEL_GMINA,
                                          exclude_subdivisions=True)
        
        result = self.overlay_gminas_to_regions(
            gminas_gdf, 
            pre1999_boundaries_gdf,
            region_id_column=voivodeship_name_col,
            method=method
        )
        
        result = result.rename(columns={'assigned_region': 'pre1999_voivodeship'})
        
        return result
    
    # ==========================================================================
    # EXPORT METHODS
    # ==========================================================================
    
    def to_dataframe(self, include_geometry: bool = False) -> pd.DataFrame:
        """
        Export all records to a pandas DataFrame.
        
        Parameters:
        - include_geometry (bool): If True, include geometry column (as WKT).
        
        Returns:
        - DataFrame with all records.
        """
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
        """
        Export database to a GeoPackage file.
        
        Parameters:
        - filepath (str or Path): Output file path.
        - year (int, optional): Filter to specific year.
        - level (int, optional): Filter to specific level.
        """
        gdf = self.to_geodataframe(year=year, level=level)
        gdf.to_file(filepath, driver='GPKG')
    
    def export_to_shapefile(self, filepath: Union[str, Path],
                             year: Optional[int] = None,
                             level: Optional[int] = None):
        """
        Export database to a Shapefile.
        
        Parameters:
        - filepath (str or Path): Output file path (should end with .shp).
        - year (int, optional): Filter to specific year.
        - level (int, optional): Filter to specific level.
        """
        gdf = self.to_geodataframe(year=year, level=level)
        gdf.to_file(filepath, driver='ESRI Shapefile')
    
    # ==========================================================================
    # STATISTICS AND SUMMARY
    # ==========================================================================
    
    def summary(self) -> dict:
        """
        Get a summary of the database contents.
        
        Returns:
        - dict with summary statistics.
        """
        return {
            'total_records': len(self._records),
            'year_range': self._year_range,
            'voivodeships': len(self._by_level.get(2, set())),
            'powiats': len(self._by_level.get(5, set())),
            'gminas': len(self._by_level.get(6, set())),
            'records_with_geometry': sum(1 for r in self._records.values() if r.has_geometry),
            'records_with_changes': sum(1 for r in self._records.values() if r.changes),
            'geometry_years_available': sorted(self._geometries.keys()),
            'unique_kinds': list(self._by_kind.keys()),
            'unique_voivodeships': list(self._by_voivodeship.keys()),
        }
    
    def print_summary(self):
        """Print a formatted summary of the database."""
        s = self.summary()
        print("=" * 60)
        print("GeoTERYT Database Summary")
        print("=" * 60)
        print(f"Total records:           {s['total_records']:,}")
        print(f"Year range:              {s['year_range'][0]} - {s['year_range'][1]}")
        print("-" * 60)
        print("Administrative levels:")
        print(f"  Voivodeships (2):      {s['voivodeships']}")
        print(f"  Powiats (5):           {s['powiats']}")
        print(f"  Gminas (6):            {s['gminas']}")
        print("-" * 60)
        print(f"Records with geometry:   {s['records_with_geometry']}")
        print(f"Records with changes:    {s['records_with_changes']}")
        print(f"Geometry years:          {s['geometry_years_available']}")
        print("=" * 60)
    
    def get_year_statistics(self, year: int) -> dict:
        """
        Get statistics for a specific year.
        
        Parameters:
        - year (int): Year to analyze.
        
        Returns:
        - dict with year-specific statistics.
        """
        if year not in self._by_year:
            return {'error': f'Year {year} not in database'}
        
        teryt_ids = self._by_year[year]
        records = [self._records[tid] for tid in teryt_ids]
        
        level_counts = {}
        kind_counts = {}
        
        for r in records:
            level = r.level
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
        """Check if database has been built."""
        return self._built
    
    def __len__(self):
        return len(self._records)
    
    def __repr__(self):
        if self._built:
            return f"GeoTERYTDatabase({len(self._records)} records, {self._year_range[0]}-{self._year_range[1]})"
        else:
            return "GeoTERYTDatabase(not built)"


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_database_from_files(terc_1999_path: Union[str, Path],
                                terc_2024_path: Union[str, Path],
                                changes_xml_path: Union[str, Path],
                                geometry_root: Optional[Union[str, Path]] = None,
                                verbose: bool = True) -> GeoTERYTDatabase:
    """
    Create a GeoTERYT database from source files.
    
    This is a convenience function that loads data, runs harmonization,
    and builds the database in one call.
    
    Parameters:
    - terc_1999_path: Path to 1999 TERYT CSV file.
    - terc_2024_path: Path to 2024 TERYT CSV file.
    - changes_xml_path: Path to changes XML file.
    - geometry_root: Optional path to geometry folder.
    - verbose: Print progress information.
    
    Returns:
    - GeoTERYTDatabase instance.
    """
    # Import harmonization function
    from local_utility_functions import harmonize_teryt
    
    # Load data
    terc_1999 = pd.read_csv(terc_1999_path, sep=';', encoding='utf-8')
    terc_2024 = pd.read_csv(terc_2024_path, sep=';', encoding='utf-8')
    
    # Parse XML changes (requires XML parsing)
    import xml.etree.ElementTree as ET
    tree = ET.parse(changes_xml_path)
    root = tree.getroot()
    
    # Extract changes data from XML
    changes_data = []
    for change in root.findall('.//zmiana'):
        row = {}
        for child in change:
            row[child.tag] = child.text
        changes_data.append(row)
    
    terc_changes = pd.DataFrame(changes_data)
    
    # Harmonize
    mega_df = harmonize_teryt(terc_1999, terc_2024, terc_changes)
    
    # Build database
    db = GeoTERYTDatabase()
    db.build_from_harmonized(mega_df, verbose=verbose)
    
    # Load geometries if path provided
    if geometry_root is not None:
        db.load_geometries_from_path(Path(geometry_root), verbose=verbose)
    
    return db
