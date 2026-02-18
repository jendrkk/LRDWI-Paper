from pathlib import Path
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from typing import Union

def deflate_income(
    income_series: pd.Series,
    cpi_series: pd.Series,
    ref_date: Union[str, pd.Timestamp],
    income_freq: str = 'A',
    cpi_is_rate: bool = False,
    use_exact_dates: bool = False
) -> pd.Series:
    """
    Deflate (convert to real) an income time series to a reference date using a CPI series.

    Parameters
    ----------
    income_series : pd.Series
        Nominal income values. Index should be datetimes (yearly dates accepted).
    cpi_series : pd.Series
        CPI values as:
        - Cumulative index (e.g., 100, 102.5, 105.1) if cpi_is_rate=False
        - Month-to-month ratios (e.g., 100, 102.5, 104.3 where 100=no change) if cpi_is_rate=True
        Index should be datetimes (e.g. monthly).
    ref_date : str or pd.Timestamp
        Date to which we deflate (e.g. '2020-12-31' or '2020-06-01'). Will be converted to Timestamp.
    income_freq : str, default 'A'
        Frequency of the income series: 'M' (monthly), 'Q' (quarterly), 'A' or 'Y' (annual).
        Used to map income timestamps to CPI observations (end of period) unless use_exact_dates=True.
    cpi_is_rate : bool, default False
        If True, cpi_series contains month-to-month ratios as percentages (e.g., 102.5 = 2.5% increase).
        These will be converted to a cumulative index.
    use_exact_dates : bool, default False
        If True, uses exact income dates for CPI lookup instead of period-end dates.
        Set to True for survey data where the date is the actual survey date.

    Returns
    -------
    pd.Series
        Real income series (same index as income_series) expressed in units of the reference date.
        
    Examples
    --------
    # Monthly survey data with month-to-month CPI ratios
    >>> deflated = deflate_income(income, cpi, '1990-01-01', 'M', cpi_is_rate=True, use_exact_dates=True)
    
    # Yearly income data with cumulative CPI index
    >>> deflated = deflate_income(yearly_income, cpi, '2020-12-31', 'A', cpi_is_rate=False)
    """
    # --- prepare inputs ---
    income = income_series.copy()
    cpi = cpi_series.copy()
    income.index = pd.to_datetime(income.index)
    cpi.index = pd.to_datetime(cpi.index)
    ref_date = pd.to_datetime(ref_date)

    if income.empty:
        return income.astype(float)

    # Convert CPI to cumulative index if needed
    if cpi_is_rate:
        # CPI series contains month-to-month ratios (e.g., 100 = no change, 110 = 10% increase)
        # Convert to cumulative index
        cumulative_cpi = (cpi / 100.0).cumprod() * 100.0
        
        # Normalize so that CPI at ref_date = 100
        if ref_date in cumulative_cpi.index:
            cpi_at_ref_unnormalized = cumulative_cpi.loc[ref_date]
        else:
            # Interpolate to get CPI at ref_date
            cpi_sorted = cumulative_cpi.sort_index()
            all_dates_temp = cpi_sorted.index.union([ref_date]).sort_values()
            cpi_interp = cpi_sorted.reindex(all_dates_temp).interpolate(method='time')
            cpi_at_ref_unnormalized = cpi_interp.loc[ref_date]
        
        cpi = (cumulative_cpi / cpi_at_ref_unnormalized) * 100.0
    
    # Ensure CPI sorted
    cpi = cpi.sort_index()

    # Map income timestamps to CPI lookup dates
    freq = income_freq.upper()
    if use_exact_dates:
        # Use exact income dates (for survey data)
        income_lookup_dates = pd.to_datetime(income.index)
    elif freq in ('A', 'Y'):
        # Income date -> end of year (Dec 31)
        income_lookup_dates = income.index.to_period('Y').to_timestamp('Y')
    elif freq == 'Q':
        income_lookup_dates = income.index.to_period('Q').to_timestamp('Q')
    elif freq == 'M':
        income_lookup_dates = income.index.to_period('M').to_timestamp('M')
    else:
        # Fallback: use the exact timestamp
        income_lookup_dates = pd.to_datetime(income.index)

    # Create combined index for interpolation
    all_dates = cpi.index.union(income_lookup_dates.union([ref_date])).sort_values()

    # Reindex & interpolate CPI to get values at all income lookup dates and ref_date
    cpi_full = cpi.reindex(all_dates).sort_index().interpolate(method='time')

    # Ensure we have CPI at ref_date
    if pd.isna(cpi_full.loc[ref_date]):
        raise ValueError("CPI series cannot provide a value at ref_date (check ranges).")

    cpi_at_ref = float(cpi_full.loc[ref_date])
    cpi_at_dates = cpi_full.loc[income_lookup_dates].values.astype(float)

    # Avoid division by zero
    if np.any(cpi_at_dates == 0) or cpi_at_ref == 0:
        raise ValueError("CPI contains zero values; cannot deflate.")

    # Real income in ref_date terms: real = nominal * (CPI_ref / CPI_t)
    real_values = income.values.astype(float) * (cpi_at_ref / cpi_at_dates)

    real_series = pd.Series(real_values, index=income.index, name=f"{income.name}_real_{ref_date.date()}")

    return real_series
'''
def deflate_income(income_series: pd.Series, cpi_series: pd.Series, ref_date: pd.Timestamp, freq: str) -> pd.Series:
    """
    Deflates an income time series using a CPI time series to a reference date.
    
    Parameters:
    - income_series (pd.Series): Time series of income values with datetime index.
    - cpi_series (pd.Series): Time series of CPI values with datetime index.
    - ref_date (pd.Timestamp): The reference date to which income should be deflated.
    - freq (str): Frequency of the time series ('M' for monthly, 'Q' for quarterly, 'A' for annual).
    
    Returns:
    - pd.Series: Deflated income time series.
    """
    # Ensure the indices are datetime
    income_series.index = pd.to_datetime(income_series.index)
    cpi_series.index = pd.to_datetime(cpi_series.index)
    
    if freq == 'Y':
        cpi_series = cpi_series.resample('YE').prod() * 10**(-24) * 100
    
    # shift index by one day
    cpi_series.index = cpi_series.index + pd.Timedelta(days=1)
    
    cpi_series.loc[ref_date] = 100.0

    # Before the reference date
    income_before_ref = income_series[income_series.index <= ref_date]
    # After (and including) the reference date
    income_after_ref = income_series[income_series.index >= ref_date]
    
    deflated_income = pd.Series(index=income_series.index, dtype=float)
    for idx, val in income_after_ref.items():
        cpi = cpi_series[cpi_series.index <= idx]
        cpi_compound = np.prod(cpi/100)*100
        
        deflated_income.loc[idx] = val * (100.0 / cpi_compound)
    
    for idx, val in income_before_ref.items():
        cpi = cpi_series[cpi_series.index >= idx]
        cpi_compound = np.prod(cpi/100)*100
        
        deflated_income.loc[idx] = val * (100.0 / cpi_compound)
    
    return deflated_income
''' 
def adapt_txt(file_path: Path, save_path: Path) -> pd.DataFrame:
    """
    Reads a .txt file and adapts it into a pandas DataFrame.
    
    Parameters:
    - file_path (Path): The path to the .txt file.
    
    Returns:
    - pd.DataFrame: The adapted DataFrame.
    """
    # Load test.txt file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    lines_clean = [line for line in lines if not line.startswith('>')]
    cleaned_content = ''.join(lines_clean)
    cleaned_content = cleaned_content.replace('data =', '', 1).strip()
    cleaned_content = cleaned_content.replace("'", '"')
    inline_content = cleaned_content.replace(" ", '').replace("\n",'')
    inline_content = eval(inline_content)
    
    df = pd.DataFrame.from_dict(inline_content)
    if save_path is not None and save_path.suffix == '.csv':
        df.to_csv(save_path, index=False)
    
    return df

def remove_polish_characters(text: str) -> str:
    """
    Removes Polish special characters from a given text string.
    
    Parameters:
    - text (str): The input text string.
    
    Returns:
    - str: The text string with Polish characters replaced by their non-accented counterparts.
    """
    polish_chars = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 
        'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 
        'ż': 'z', 'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 
        'Ł': 'L', 'Ń': 'N', 'Ó': 'O', 'Ś': 'S', 
        'Ź': 'Z',  'Ż': 'Z'
    }
    
    for pol_char, repl_char in polish_chars.items():
        text = text.replace(pol_char, repl_char)
    
    return text

def nuts_code_to_teryt(nuts_code: str) -> str:
    """
    Converts a NUTS code to a TERYT code by removing the first character.
    
    Parameters:
    - nuts_code (str): The NUTS code.
    
    Returns:
    - str: The corresponding TERYT code.
    """
    
    '''if len(nuts_code) == 11:
        k = 2
    elif len(nuts_code) == 12:
        k = 3
    else:
        raise ValueError("NUTS code must be either 11 or 12 characters long.")'''
    
    k=3
    
    # Code of gmina - the last 2 digits of NUTS code
    gmina_id = nuts_code[-k:]
    # Code of powiat - the 2 digits before gmina code
    powiat_id = nuts_code[-(k+2):-k]
    # Code of voivodeship
    if len(nuts_code) == 11:
        voivodeship_id = nuts_code[1:3]
    elif len(nuts_code) == 10:
        voivodeship_id = nuts_code[2:4]
        powiat_id = nuts_code[4:6]
        gmina_id = nuts_code[6:10]
    else:
        voivodeship_id = nuts_code[2:4]
    
    teryt_code = voivodeship_id + powiat_id + gmina_id
    return teryt_code
    

# ==============================================================================
# TERYT HARMONIZATION FUNCTIONS
# ==============================================================================

def encode_level(row: pd.Series, col_names: list = ['WOJ', 'POW', 'GMI']) -> int:
    """
    Encodes the administrative level based on TERYT code components.
    
    Parameters:
    - row (pd.Series): A row from a TERYT DataFrame.
    - col_names (list): Column names for voivodeship, powiat, and gmina codes.
    
    Returns:
    - int: Administrative level (2 = voivodeship, 5 = powiat, 6 = gmina), or NaN if undefined.
    """
    c0, c1, c2 = col_names
    woj = str(row[c0])
    pow_ = str(row[c1])
    gmi = str(row[c2])
    if woj != '00' and pow_ == '00' and gmi == '00':
        return 2  # Voivodeship
    elif woj != '00' and pow_ != '00' and gmi == '00':
        return 5  # County (powiat)
    elif woj != '00' and pow_ != '00' and gmi != '00':
        return 6  # Municipality (gmina)
    else:
        return np.nan  # Undefined level


def encode_kind(row: pd.Series, col_name: str = 'RODZ') -> str:
    """
    Encodes the kind/type of territorial unit based on RODZ code.
    
    Parameters:
    - row (pd.Series): A row from a TERYT DataFrame.
    - col_name (str): Column name for the RODZ code.
    
    Returns:
    - str: The kind of territorial unit.
    """
    val = str(row[col_name])
    if val == '0':
        return np.nan  # Not applicable
    mapping = {
        '1': 'urban',           # gmina miejska
        '2': 'rural',           # gmina wiejska
        '3': 'urban-rural',     # gmina miejsko-wiejska
        '4': 'town',            # miasto (in urban-rural gmina)
        '5': 'village',         # obszar wiejski (rural area in urban-rural gmina)
        '8': 'Warsaw district', # dzielnica Warszawy
        '9': 'delegatura'       # delegatura lub dzielnica miasta
    }
    return mapping.get(val, 'unknown')


def get_level_name(level: int) -> str:
    """
    Returns the name of the administrative level.
    
    Parameters:
    - level (int): Administrative level code.
    
    Returns:
    - str: Name of the level.
    """
    level_names = {
        2: 'voivodeship',
        5: 'powiat',
        6: 'gmina'
    }
    return level_names.get(level, 'unknown')


def classify_change(row: pd.Series) -> dict:
    """
    Classifies the type of change that occurred based on a row from terc_changes.
    
    Parameters:
    - row (pd.Series): A row from the terc_changes DataFrame.
    
    Returns:
    - dict: A dictionary with keys 'change_type' (list of change descriptions) and 'details'.
    """
    changes = []
    details = {}
    
    typ_korekty = row.get('TypKorekty', '')
    
    # Extract before/after values
    woj_before = str(row.get('WojPrzed', '00'))
    pow_before = str(row.get('PowPrzed', '00'))
    gmi_before = str(row.get('GmiPrzed', '00'))
    rodz_before = str(row.get('RodzPrzed', '0'))
    name_before = row.get('NazwaPrzed', None)
    name_dod_before = row.get('NazwaDodatkowaPrzed', None)
    
    woj_after = str(row.get('WojPo', '00'))
    pow_after = str(row.get('PowPo', '00'))
    gmi_after = str(row.get('GmiPo', '00'))
    rodz_after = str(row.get('RodzPo', '0'))
    name_after = row.get('NazwaPo', None)
    name_dod_after = row.get('NazwaDodatkowaPo', None)
    
    id_before = row.get('id_before', '')
    id_after = row.get('id_after', '')
    
    # Determine administrative level after change
    if woj_after != '00' and pow_after != '00' and gmi_after != '00':
        level = 6  # gmina
    elif woj_after != '00' and pow_after != '00' and gmi_after == '00':
        level = 5  # powiat
    elif woj_after != '00' and pow_after == '00' and gmi_after == '00':
        level = 2  # voivodeship
    else:
        level = 0  # unknown
    
    level_name = get_level_name(level)
    
    # Handle different change types
    if typ_korekty == 'D':  # Added (Dodano)
        # New unit was created
        parent_code = woj_after + pow_after
        if level == 6:
            parent_code = woj_after + pow_after + "00"
        elif level == 5:
            parent_code = woj_after + "0000"
        
        changes.append(f"NEW: {level_name} created (id: {id_after})")
        details['new_unit'] = True
        details['created_from_parent'] = parent_code
        
    elif typ_korekty == 'U':  # Removed (Usunięto)
        # Unit was deleted/removed
        changes.append(f"REMOVED: {level_name} deleted (id: {id_before})")
        details['removed_unit'] = True
        
        # Check if merged into another unit
        merged_into = row.get('WlaczonoDoIdentyfikatora1', None)
        if pd.notna(merged_into):
            changes.append(f"MERGED into: {merged_into}")
            details['merged_into'] = merged_into
            
    elif typ_korekty == 'M':  # Modified (Modyfikacja)
        # Check what was modified
        
        # 1. Check for RODZ (type) change
        if rodz_before != rodz_after and rodz_before != '0':
            kind_before = encode_kind({'RODZ': rodz_before})
            kind_after = encode_kind({'RODZ': rodz_after})
            changes.append(f"TYPE_CHANGE: {kind_before} -> {kind_after}")
            details['rodz_change'] = {'from': rodz_before, 'to': rodz_after}
        
        # 2. Check for name change
        if name_before != name_after and pd.notna(name_before) and pd.notna(name_after):
            changes.append(f"NAME_CHANGE: {name_before} -> {name_after}")
            details['name_change'] = {'from': name_before, 'to': name_after}
        
        # 3. Check for additional name (NAZWA_DOD) change
        if name_dod_before != name_dod_after and pd.notna(name_dod_before) and pd.notna(name_dod_after):
            changes.append(f"DESIGNATION_CHANGE: {name_dod_before} -> {name_dod_after}")
            details['designation_change'] = {'from': name_dod_before, 'to': name_dod_after}
        
        # 4. Check for parent change (voivodeship or powiat)
        if woj_before != woj_after and woj_before != '00':
            changes.append(f"VOIVODESHIP_CHANGE: {woj_before} -> {woj_after}")
            details['voivodeship_change'] = {'from': woj_before, 'to': woj_after}
        
        if pow_before != pow_after and pow_before != '00':
            changes.append(f"POWIAT_CHANGE: {pow_before} -> {pow_after}")
            details['powiat_change'] = {'from': pow_before, 'to': pow_after}
        
        # 5. Check for gmina code change
        if gmi_before != gmi_after and gmi_before != '00':
            changes.append(f"GMINA_CODE_CHANGE: {gmi_before} -> {gmi_after}")
            details['gmina_code_change'] = {'from': gmi_before, 'to': gmi_after}
        
        # 6. Check if ID changed overall
        if id_before != id_after:
            details['id_change'] = {'from': id_before, 'to': id_after}
        
        # If no specific changes detected but marked as M
        if not changes:
            changes.append(f"OTHER: modification (details unclear)")
            details['other'] = True
    
    else:
        changes.append(f"UNKNOWN: unrecognized change type '{typ_korekty}'")
        details['unknown'] = True
    
    return {
        'change_type': changes,
        'details': details,
        'level': level
    }


def prepare_changes_dataframe(changes: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the changes DataFrame by ensuring proper formatting and adding derived columns.
    
    Parameters:
    - changes (pd.DataFrame): Raw changes DataFrame from XML.
    
    Returns:
    - pd.DataFrame: Prepared changes DataFrame with standardized columns.
    """
    changes = changes.copy()
    
    # Ensure string columns with proper padding
    str_cols_2 = ['WojPrzed', 'PowPrzed', 'GmiPrzed', 'WojPo', 'PowPo', 'GmiPo']
    str_cols_1 = ['RodzPrzed', 'RodzPo']
    
    for col in str_cols_2:
        if col in changes.columns:
            changes[col] = changes[col].apply(
                lambda x: str(int(x)).zfill(2) if pd.notna(x) and x != '' else "00"
            )
    
    for col in str_cols_1:
        if col in changes.columns:
            changes[col] = changes[col].apply(
                lambda x: str(int(x)).zfill(1) if pd.notna(x) and x != '' else "0"
            )
    
    # Create or update ID columns
    changes['id_before'] = (changes['WojPrzed'] + changes['PowPrzed'] + 
                            changes['GmiPrzed'] + changes['RodzPrzed'])
    changes['id_after'] = (changes['WojPo'] + changes['PowPo'] + 
                           changes['GmiPo'] + changes['RodzPo'])
    
    # Parse dates and extract years
    changes['date_before'] = pd.to_datetime(changes['StanPrzed'], errors='coerce')
    changes['date_after'] = pd.to_datetime(changes['StanPo'], errors='coerce')
    changes['year_effective'] = changes['date_after'].dt.year
    
    return changes


def prepare_teryt_dataframe(teryt: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a TERYT DataFrame by ensuring proper formatting and adding derived columns.
    
    Parameters:
    - teryt (pd.DataFrame): Raw TERYT DataFrame.
    
    Returns:
    - pd.DataFrame: Prepared TERYT DataFrame with standardized columns.
    """
    teryt = teryt.copy()
    
    # Standardize column names (handle both original and processed data)
    col_mapping = {
        'WOJ': 'WOJ', 'POW': 'POW', 'GMI': 'GMI', 'RODZ': 'RODZ',
        'NAZWA': 'NAZWA', 'NAZWA_DOD': 'NAZWA_DOD', 'STAN_NA': 'STAN_NA'
    }
    
    # Ensure string columns with proper padding
    for col, new_col in [('WOJ', 'WOJ'), ('POW', 'POW'), ('GMI', 'GMI')]:
        if col in teryt.columns:
            teryt[col] = teryt[col].apply(
                lambda x: str(int(x)).zfill(2) if pd.notna(x) and str(x).strip() != '' else "00"
            )
    
    if 'RODZ' in teryt.columns:
        teryt['RODZ'] = teryt['RODZ'].apply(
            lambda x: str(int(x)).zfill(1) if pd.notna(x) and str(x).strip() != '' else "0"
        )
    
    # Create ID column if not exists
    if 'id' not in teryt.columns:
        teryt['id'] = teryt['WOJ'] + teryt['POW'] + teryt['GMI'] + teryt['RODZ']
    
    # Add level and kind if not present
    if 'level' not in teryt.columns:
        teryt['level'] = teryt.apply(encode_level, axis=1)
    
    if 'kind' not in teryt.columns:
        teryt['kind'] = teryt.apply(encode_kind, axis=1)
    
    return teryt


def apply_changes_to_teryt(teryt: pd.DataFrame, changes_for_year: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Applies a set of changes to a TERYT DataFrame for a specific year.
    
    Parameters:
    - teryt (pd.DataFrame): The current TERYT DataFrame.
    - changes_for_year (pd.DataFrame): Changes to apply for this year.
    - year (int): The year of changes.
    
    Returns:
    - pd.DataFrame: Updated TERYT DataFrame.
    """
    teryt = teryt.copy()
    
    # Initialize tracking columns if not present
    if 'if_changed' not in teryt.columns:
        teryt['if_changed'] = False
    if 'when_changed' not in teryt.columns:
        teryt['when_changed'] = np.nan
    if 'notes' not in teryt.columns:
        teryt['notes'] = teryt.apply(lambda x: {'number_of_changes': 0, 'changes': []}, axis=1)
    if 'historical_codes' not in teryt.columns:
        teryt['historical_codes'] = teryt.apply(lambda x: [], axis=1)
    if 'code_by_year' not in teryt.columns:
        teryt['code_by_year'] = teryt.apply(lambda x: {}, axis=1)
    
    units_to_remove = []
    new_units = []
    
    for idx, change_row in changes_for_year.iterrows():
        typ_korekty = change_row.get('TypKorekty', '')
        id_before = change_row.get('id_before', '0000000')
        id_after = change_row.get('id_after', '0000000')
        
        # Classify the change
        change_info = classify_change(change_row)
        change_descriptions = change_info['change_type']
        
        if typ_korekty == 'D':  # New unit added
            # Create new unit entry
            new_unit = {
                'WOJ': str(change_row.get('WojPo', '00')),
                'POW': str(change_row.get('PowPo', '00')),
                'GMI': str(change_row.get('GmiPo', '00')),
                'RODZ': str(change_row.get('RodzPo', '0')),
                'NAZWA': change_row.get('NazwaPo', ''),
                'NAZWA_DOD': change_row.get('NazwaDodatkowaPo', ''),
                'STAN_NA': str(year) + '-01-01',
                'id': id_after,
                'if_changed': True,
                'when_changed': year,
                'notes': {'number_of_changes': 1, 'changes': change_descriptions},
                'historical_codes': [id_after],
                'code_by_year': {year: id_after}
            }
            new_unit['level'] = encode_level(pd.Series(new_unit))
            new_unit['kind'] = encode_kind(pd.Series(new_unit))
            new_units.append(new_unit)
            
        elif typ_korekty == 'U':  # Unit removed
            # Mark for removal
            if id_before != '0000000':
                units_to_remove.append(id_before)
            
        elif typ_korekty == 'M':  # Unit modified
            # Handle M-type changes with id_before='0000000' (name-only changes)
            # In this case, we need to find the unit by id_after (which should already exist)
            if id_before == '0000000' and id_after != '0000000':
                # This is a name/designation change only - find by id_after
                mask = teryt['id'] == id_after
                if mask.any():
                    idx_to_update = teryt[mask].index[0]
                    
                    # Update name fields only
                    if pd.notna(change_row.get('NazwaPo')):
                        teryt.loc[idx_to_update, 'NAZWA'] = change_row.get('NazwaPo')
                    if pd.notna(change_row.get('NazwaDodatkowaPo')):
                        teryt.loc[idx_to_update, 'NAZWA_DOD'] = change_row.get('NazwaDodatkowaPo')
                    
                    # Update tracking columns
                    teryt.loc[idx_to_update, 'if_changed'] = True
                    teryt.loc[idx_to_update, 'when_changed'] = year
                    
                    # Update notes
                    current_notes = teryt.loc[idx_to_update, 'notes']
                    if isinstance(current_notes, dict):
                        current_notes['number_of_changes'] += 1
                        current_notes['changes'].extend(change_descriptions)
                    else:
                        teryt.loc[idx_to_update, 'notes'] = {
                            'number_of_changes': 1,
                            'changes': change_descriptions
                        }
                continue
            
            # Skip if both are invalid
            if id_before == '0000000':
                continue
            
            # Find and update the unit
            mask = teryt['id'] == id_before
            if mask.any():
                idx_to_update = teryt[mask].index[0]
                
                # For M-type changes: if id_after is 0000000, keep the original ID
                # This happens when only the name changes but not the structural codes
                effective_id_after = id_after if id_after != '0000000' else id_before
                
                # Track historical codes if ID is changing
                if id_before != effective_id_after:
                    current_hist = teryt.loc[idx_to_update, 'historical_codes']
                    if isinstance(current_hist, list):
                        if id_before not in current_hist:
                            current_hist.append(id_before)
                        if effective_id_after not in current_hist:
                            current_hist.append(effective_id_after)
                    else:
                        teryt.loc[idx_to_update, 'historical_codes'] = [id_before, effective_id_after]
                    
                    # Update code_by_year
                    current_cby = teryt.loc[idx_to_update, 'code_by_year']
                    if isinstance(current_cby, dict):
                        current_cby[year] = effective_id_after
                    else:
                        teryt.loc[idx_to_update, 'code_by_year'] = {year: effective_id_after}
                
                # Only update WOJ/POW/GMI/RODZ if id_after is valid (not 0000000)
                # Otherwise it means only the name changed
                if id_after != '0000000':
                    teryt.loc[idx_to_update, 'WOJ'] = str(change_row.get('WojPo', teryt.loc[idx_to_update, 'WOJ']))
                    teryt.loc[idx_to_update, 'POW'] = str(change_row.get('PowPo', teryt.loc[idx_to_update, 'POW']))
                    teryt.loc[idx_to_update, 'GMI'] = str(change_row.get('GmiPo', teryt.loc[idx_to_update, 'GMI']))
                    teryt.loc[idx_to_update, 'RODZ'] = str(change_row.get('RodzPo', teryt.loc[idx_to_update, 'RODZ']))
                
                if pd.notna(change_row.get('NazwaPo')):
                    teryt.loc[idx_to_update, 'NAZWA'] = change_row.get('NazwaPo')
                if pd.notna(change_row.get('NazwaDodatkowaPo')):
                    teryt.loc[idx_to_update, 'NAZWA_DOD'] = change_row.get('NazwaDodatkowaPo')
                
                # Update ID (keep original if id_after is 0000000)
                teryt.loc[idx_to_update, 'id'] = effective_id_after
                
                # Update tracking columns
                teryt.loc[idx_to_update, 'if_changed'] = True
                teryt.loc[idx_to_update, 'when_changed'] = year
                
                # Update notes
                current_notes = teryt.loc[idx_to_update, 'notes']
                if isinstance(current_notes, dict):
                    current_notes['number_of_changes'] += 1
                    current_notes['changes'].extend(change_descriptions)
                else:
                    teryt.loc[idx_to_update, 'notes'] = {
                        'number_of_changes': 1, 
                        'changes': change_descriptions
                    }
                
                # Recalculate level and kind
                teryt.loc[idx_to_update, 'level'] = encode_level(teryt.loc[idx_to_update])
                teryt.loc[idx_to_update, 'kind'] = encode_kind(teryt.loc[idx_to_update])
            else:
                # Unit not found - might have been renamed/recoded already
                # Try to find by name and approximate location
                name_match = teryt[
                    (teryt['NAZWA'] == change_row.get('NazwaPrzed', '')) &
                    (teryt['WOJ'] == str(change_row.get('WojPrzed', '')))
                ]
                if len(name_match) > 0:
                    idx_to_update = name_match.index[0]
                    old_id = teryt.loc[idx_to_update, 'id']
                    
                    # Track historical codes
                    current_hist = teryt.loc[idx_to_update, 'historical_codes']
                    if isinstance(current_hist, list):
                        if old_id not in current_hist:
                            current_hist.append(old_id)
                        if id_after not in current_hist:
                            current_hist.append(id_after)
                    else:
                        teryt.loc[idx_to_update, 'historical_codes'] = [old_id, id_after]
                    
                    # Update code_by_year
                    current_cby = teryt.loc[idx_to_update, 'code_by_year']
                    if isinstance(current_cby, dict):
                        current_cby[year] = id_after
                    else:
                        teryt.loc[idx_to_update, 'code_by_year'] = {year: id_after}
                    
                    # Apply updates
                    teryt.loc[idx_to_update, 'WOJ'] = str(change_row.get('WojPo', teryt.loc[idx_to_update, 'WOJ']))
                    teryt.loc[idx_to_update, 'POW'] = str(change_row.get('PowPo', teryt.loc[idx_to_update, 'POW']))
                    teryt.loc[idx_to_update, 'GMI'] = str(change_row.get('GmiPo', teryt.loc[idx_to_update, 'GMI']))
                    teryt.loc[idx_to_update, 'RODZ'] = str(change_row.get('RodzPo', teryt.loc[idx_to_update, 'RODZ']))
                    
                    if pd.notna(change_row.get('NazwaPo')):
                        teryt.loc[idx_to_update, 'NAZWA'] = change_row.get('NazwaPo')
                    if pd.notna(change_row.get('NazwaDodatkowaPo')):
                        teryt.loc[idx_to_update, 'NAZWA_DOD'] = change_row.get('NazwaDodatkowaPo')
                    
                    teryt.loc[idx_to_update, 'id'] = id_after
                    teryt.loc[idx_to_update, 'if_changed'] = True
                    teryt.loc[idx_to_update, 'when_changed'] = year
                    teryt.loc[idx_to_update, 'level'] = encode_level(teryt.loc[idx_to_update])
                    teryt.loc[idx_to_update, 'kind'] = encode_kind(teryt.loc[idx_to_update])
    
    # Remove deleted units
    if units_to_remove:
        teryt = teryt[~teryt['id'].isin(units_to_remove)]
    
    # Add new units
    if new_units:
        new_units_df = pd.DataFrame(new_units)
        teryt = pd.concat([teryt, new_units_df], ignore_index=True)
    
    return teryt


def harmonize_teryt(first_teryt: pd.DataFrame, last_teryt: pd.DataFrame, changes: pd.DataFrame, 
                    verbose: bool = True) -> pd.DataFrame:
    """
    Harmonizes TERYT codes over all years from the first_teryt DataFrame to the last_teryt DataFrame
    using the changes DataFrame that contains the history of TERYT code changes. The output is a mega
    DataFrame with the list of all TERYT codes that was used in every year from the year of first_teryt
    to the year of last_teryt.
    
    Parameters:
    - first_teryt (pd.DataFrame): DataFrame with the initial TERYT codes (e.g., from 1999).
    - last_teryt (pd.DataFrame): DataFrame with the latest TERYT codes (e.g., from 2024).
        This is used to determine the end year and for validation purposes.
    - changes (pd.DataFrame): DataFrame containing the history of TERYT code changes 
        (originally from XML format, with columns like TypKorekty, WojPrzed, WojPo, etc.).
    - verbose (bool): If True, print progress messages. Default is True.
    
    Returns:
    - pd.DataFrame: A mega DataFrame containing TERYT codes for all years with columns:
        - All original TERYT columns (WOJ, POW, GMI, RODZ, NAZWA, NAZWA_DOD, id, level, kind)
        - year: The year for which this administrative division state is valid
        - if_changed: Boolean indicating if the unit was involved in any changes
        - when_changed: Year of the most recent change (NaN if never changed)
        - notes: Dictionary with {'number_of_changes': int, 'changes': [list of change descriptions]}
        - historical_codes: List of all TERYT codes this unit has had over time
        - code_by_year: Dictionary mapping each year to the TERYT code used in that year
    
    Example usage:
        mega_df = harmonize_teryt(terc_1999, terc_2024, terc_changes)
        division_2010 = mega_df[mega_df['year'] == 2010]
    """
    # Prepare dataframes
    changes_prepared = prepare_changes_dataframe(changes)
    first_teryt_prepared = prepare_teryt_dataframe(first_teryt)
    
    # Determine year range
    first_year = pd.to_datetime(first_teryt_prepared['STAN_NA'].iloc[0]).year
    last_year = pd.to_datetime(last_teryt['STAN_NA'].iloc[0]).year if 'STAN_NA' in last_teryt.columns else 2024
    
    if verbose:
        print(f"Harmonizing TERYT codes from {first_year} to {last_year}...")
    
    # Initialize the mega DataFrame list
    yearly_dfs = []
    
    # Start with the first year's state
    current_teryt = first_teryt_prepared.copy()
    
    # Add tracking columns
    current_teryt['if_changed'] = False
    current_teryt['when_changed'] = np.nan
    current_teryt['notes'] = current_teryt.apply(
        lambda x: {'number_of_changes': 0, 'changes': []}, axis=1
    )
    # Initialize historical_codes with the initial ID
    current_teryt['historical_codes'] = current_teryt['id'].apply(lambda x: [x])
    # Initialize code_by_year with the first year
    current_teryt['code_by_year'] = current_teryt['id'].apply(lambda x: {first_year: x})
    
    # Store the first year
    first_year_df = current_teryt.copy()
    first_year_df['year'] = first_year
    yearly_dfs.append(first_year_df)
    
    # Process each subsequent year
    for year in range(first_year + 1, last_year + 1):
        # Get changes effective for this year
        changes_for_year = changes_prepared[changes_prepared['year_effective'] == year]
        
        if len(changes_for_year) > 0:
            if verbose:
                print(f"  Year {year}: applying {len(changes_for_year)} changes...")
            current_teryt = apply_changes_to_teryt(current_teryt, changes_for_year, year)
        else:
            if verbose:
                print(f"  Year {year}: no changes")
        
        # Update code_by_year for all units (current year's code)
        for idx in current_teryt.index:
            cby = current_teryt.loc[idx, 'code_by_year']
            current_id = current_teryt.loc[idx, 'id']
            if isinstance(cby, dict):
                cby[year] = current_id
            else:
                current_teryt.loc[idx, 'code_by_year'] = {year: current_id}
        
        # Store this year's state
        year_df = current_teryt.copy()
        year_df['year'] = year
        year_df['STAN_NA'] = f"{year}-01-01"
        yearly_dfs.append(year_df)
    
    # Concatenate all years
    mega_df = pd.concat(yearly_dfs, ignore_index=True)
    
    # Ensure consistent column order
    column_order = ['year', 'WOJ', 'POW', 'GMI', 'RODZ', 'id', 'NAZWA', 'NAZWA_DOD', 
                    'level', 'kind', 'STAN_NA', 'if_changed', 'when_changed', 'notes',
                    'historical_codes', 'code_by_year']
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in mega_df.columns]
    other_cols = [col for col in mega_df.columns if col not in column_order]
    mega_df = mega_df[column_order + other_cols]
    
    if verbose:
        print(f"\nHarmonization complete!")
        print(f"Total rows in mega DataFrame: {len(mega_df)}")
        print(f"Unique years: {mega_df['year'].nunique()}")
        print(f"Units changed at some point: {mega_df[mega_df['if_changed'] == True]['id'].nunique()}")
    
    return mega_df


def get_unit_history(mega_df: pd.DataFrame, unit_id: str = None, unit_name: str = None) -> pd.DataFrame:
    """
    Retrieves the history of a specific territorial unit across all years.
    
    Parameters:
    - mega_df (pd.DataFrame): The harmonized mega DataFrame from harmonize_teryt().
    - unit_id (str): The TERYT ID of the unit (optional if unit_name is provided).
    - unit_name (str): The name of the unit (optional if unit_id is provided).
    
    Returns:
    - pd.DataFrame: History of the unit across years where it existed.
    """
    if unit_id is not None:
        # Search by ID (might change over time)
        # First, find all IDs this unit has used
        mask = mega_df['id'] == unit_id
        if not mask.any():
            # Try partial match for units whose ID changed
            mask = mega_df['id'].str.startswith(unit_id[:6])
        return mega_df[mask].sort_values('year')
    
    elif unit_name is not None:
        # Search by name
        mask = mega_df['NAZWA'].str.contains(unit_name, case=False, na=False)
        return mega_df[mask].sort_values(['NAZWA', 'year'])
    
    else:
        raise ValueError("Either unit_id or unit_name must be provided.")


def get_changes_summary(mega_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary of all changes that occurred across the years.
    
    Parameters:
    - mega_df (pd.DataFrame): The harmonized mega DataFrame from harmonize_teryt().
    
    Returns:
    - pd.DataFrame: Summary of changes with columns: year, change_count, change_types.
    """
    # Get unique changed units per year
    changed = mega_df[mega_df['if_changed'] == True].copy()
    
    summary_data = []
    for year in sorted(changed['when_changed'].dropna().unique()):
        year_changes = changed[changed['when_changed'] == year]
        
        # Collect all change types
        all_changes = []
        for notes in year_changes['notes']:
            if isinstance(notes, dict) and 'changes' in notes:
                all_changes.extend(notes['changes'])
        
        summary_data.append({
            'year': int(year),
            'change_count': len(year_changes),
            'change_types': list(set(all_changes))
        })
    
    return pd.DataFrame(summary_data)
    