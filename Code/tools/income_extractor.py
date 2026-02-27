"""
income_extractor.py
===================
Keyword- and DATA-driven extraction of income column names from CBOS SPSS files.

Extracted column names (all may be None if not present in a given file):
  income_p_gate  – personal income gate question column (F1.0, ~5 labels, post-2008)
  income_p_num   – personal income numerical/amount column
  income_p_cat   – personal income categorical/bracket column
  income_hh_gate – HH per-capita income gate question column (post-2008)
  income_hh_num  – HH per-capita income numerical/amount column
  income_hh_cat  – HH per-capita income categorical/bracket column

Classification strategy:
  1. Label-based pre-filter: column must mention income-related keywords
  2. Label-based scope classification: personal / HH-per-capita / total-HH
  3. DATA-BASED type classification: actual values in the DataFrame determine
     whether a column is numeric (continuous amounts) or categorical (bracket codes)
  4. Gate detection via metadata (F1.0, ~5 gate-pattern labels)
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# Helper: data-based classification of column as numeric vs categorical
# ---------------------------------------------------------------------------

def _is_numeric_by_data(df_col) -> bool:
    """
    Determine whether a column contains continuous numeric amounts (True)
    or categorical bracket codes (False) by examining actual data values.

    Numeric signals:
      - Many distinct values (>= 20 after removing sentinels)
      - Values span a wide range (max > 50)

    Categorical signals:
      - Few distinct values (<= 15)
      - Values are small integers (max <= 20)
    """
    vals = df_col.dropna()
    if len(vals) == 0:
        return True  # default to numeric if no data

    # Remove common sentinel values (negative, >= 9990, >= 99990)
    clean = vals[(vals >= 0) & (vals < 9990)]
    n_unique_clean = clean.nunique()

    if n_unique_clean == 0:
        # All values are sentinels — fall back to raw
        n_unique = vals.nunique()
        val_max = vals.max()
    else:
        n_unique = n_unique_clean
        val_max = clean.max()

    # Clear numeric: many distinct values
    if n_unique >= 20:
        return True

    # Clear categorical: few values, all small
    if n_unique <= 15 and val_max <= 20:
        return False

    # Ambiguous zone: if max value > 100, likely income amounts
    if val_max > 100:
        return True

    return False  # default to categorical for small-value, few-unique columns


def _is_gate_column(col_type: str, vvl: dict) -> bool:
    """
    Return True if the column is a gate question:
    F1.0 with ~5-6 labels describing response disposition
    (e.g. "podaje kwotę", "odmawia", "trudno powiedzieć").
    """
    if col_type != 'F1.0':
        return False
    n_labs = len(vvl)
    if n_labs < 4 or n_labs > 7:
        return False
    texts = ' '.join(str(v) for v in vvl.values()).lower()
    gate_words = ['podaje', 'kwot', 'odmaw', 'brak dochodów',
                  'nie potrafi', 'nie dotyczy', 'respondent']
    if sum(w in texts for w in gate_words) >= 2:
        return True
    return False


def _is_bracket_label(label_norm: str) -> bool:
    """Return True if the column label explicitly describes a bracket question."""
    bracket_phrases = [
        'w którym przedziale',
        'przybliżone dochody',
        'a czy mógłby',
        'a czy mogłaby',
        'wskazać, w którym',
    ]
    return any(p in label_norm for p in bracket_phrases)


def _is_bracket_vvl_safe(vvl: dict) -> bool:
    """
    Conservative metadata-only fallback for bracket detection.
    Requires BOTH small group-code keys AND bracket-like label text.
    """
    if not vvl:
        return False
    keys = sorted(vvl.keys())
    positive_keys = [k for k in keys if k >= 0]
    if not positive_keys:
        return False
    max_pos = max(positive_keys)
    min_key = keys[0]
    if max_pos <= 15 and min_key >= -1:
        texts = ' '.join(str(v) for v in vvl.values()).lower()
        bracket_words = ['złot', ' zł', 'tys.', 'od ', 'do ',
                         'powyżej', 'poniżej', 'więcej', 'mniej niż', 'grup']
        if sum(w in texts for w in bracket_words) >= 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Helper: classify label as personal / HH-per-capita / total-HH
# ---------------------------------------------------------------------------

def _classify_label(label: str):
    """
    Returns (is_personal, is_hh_pc, is_total_hh, is_ankieter).

    Handles:
      • Regular notation: Pana(i)
      • 2006-era quirk: Pana[i]  → normalised to Pana(i)
      • 2017-era abbreviation: "P. MIESIĘCZNE DOCHODY"
      • Post-gate: "ANKIETER: Poniżej wpisać podaną przez respondenta kwotę"
      • HH Numerycznie:  "Kwota … na jedną osobę w gospodarstwie"
      • Personal Numerycznie:  "Kwota … dochodów NETTO respondenta"
      • Derived: "MIESIĘCZNE DOCHODY NETTO respondenta" (DochOs)
      • Derived: "PRZECIĘTNE … NA JEDNĄ OSOBĘ W P. GOSPODARSTWIE" (DochPC)
    """
    if not label:
        return False, False, False, False

    ln = label.lower()
    ln = ln.replace('pana[i]', 'pana(i)').replace('pan[i]', 'pana(i)')

    # ---- NOT income: children count, opinion, helper/auxiliary ----
    if 'dzieci' in ln:
        return False, False, False, False
    if 'pomocnicza' in ln or 'zmienna pomocnicza' in ln:
        return False, False, False, False
    if 'powinny' in ln and 'dochod' in ln:
        return False, False, False, False
    # Subjective financial well-being: "sposobowi gospodarowania dochodem"
    if 'gospodarowania' in ln:
        return False, False, False, False
    # Opinion questions: "Pana(i) zdaniem" / "Jak Pan(i) sądzi" / "według Pana(i)"
    if 'zdaniem' in ln or 'sądzi' in ln or 'według' in ln:
        return False, False, False, False
    # Verb "dochodzić" (to occur/pursue), not noun "dochód" (income)
    # "dochodzi" catches: dochodzi, dochodzić, dochodził, dochodziło, dochodziłby…
    # "dochodzeni" catches: dochodzenia, dochodzeniu
    # "dochodzą" catches: dochodzący (noise "dochodzący z fabryki")
    if 'dochodzi' in ln or 'dochodzeni' in ln or 'dochodzą' in ln:
        return False, False, False, False
    # Coping strategies: "jak postępuje się gdy dochody za niskie w stosunku do potrzeb"
    if 'postępuje' in ln:
        return False, False, False, False
    # Sufficiency opinions: "wystarczają dochody", "dochody wystarczające", "na co starcza"
    if 'starcza' in ln:
        return False, False, False, False
    # Job finding difficulty: "znalezienie pracy dającej podobne dochody"
    if 'znalezienie' in ln:
        return False, False, False, False
    # Moral evaluation: "zachowania... oceniane... dokonywanie fikcyjnych dochodów"
    if 'oceniane' in ln or 'ocenia' in ln:
        return False, False, False, False
    # Parish/church income: "informowani o dochodach parafii", "niskie dochody parafii"
    if 'parafii' in ln or 'parafian' in ln:
        return False, False, False, False

    is_ankieter = 'ankieter: poniżej wpisać' in ln

    # ---- total household income (łączne = whole HH, not per-capita) ----
    is_total_hh = (
        ('łączne' in ln and 'dochod' in ln)
        or 'łącznych dochodów' in ln
        or ('zsumowane' in ln and 'dochod' in ln)
        or ('żywiciela' in ln and 'pana(i)' not in ln)
        # "Do której grup dochodowych zalicza się Pana(i) gospodarstwo domowe?"
        or ('dochod' in ln and 'gospodarstw' in ln
            and 'na jedną' not in ln and 'na 1 osob' not in ln
            and 'przypadaj' not in ln and 'na osobę' not in ln)
    )

    # ---- HH per-capita ----
    is_hh_pc = not is_total_hh and (
        'na 1 osob' in ln
        or 'na jedną osob' in ln
        or 'przypadające na jedną' in ln
        or 'na osobę w' in ln
        or 'dochód na jedną' in ln
        or ('kwota' in ln and 'osob' in ln)
        # Derived DochPC: "PRZECIĘTNE MIESIĘCZNE DOCHODY … NA JEDNĄ OSOBĘ W P. GOSP"
        or ('przypadając' in ln and 'p. gospod' in ln)
    )

    # ---- personal ----
    is_personal = not is_hh_pc and not is_total_hh and not is_ankieter and (
        'pana(i) miesięczne dochody' in ln
        or 'pana(i) dochody' in ln
        or 'p. miesięczne dochody' in ln
        or ('miesięczne dochody' in ln and 'osob' not in ln
            and 'rodziny' not in ln and 'łączne' not in ln
            and 'zsumowane' not in ln and 'gospod' not in ln)
        or ('kwota' in ln and 'respondenta' in ln and 'osob' not in ln)
        # Derived DochOs: "MIESIĘCZNE DOCHODY NETTO respondenta"
        or ('dochody netto respondenta' in ln and 'osob' not in ln)
        # Catchall with pana(i) as primary subject
        or ('pana(i)' in ln and 'dochod' in ln and 'osob' not in ln
            and 'łączne' not in ln and 'zsumowane' not in ln
            and 'rodziny' not in ln and 'gospod' not in ln)
    )

    return is_personal, is_hh_pc, is_total_hh, is_ankieter


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_income_columns(file: str, meta, df=None) -> dict:
    """
    Scan SPSS metadata AND actual data for all income-related columns.

    Parameters
    ----------
    file : str
        Filename.
    meta : pyreadstat metadata object
        Must expose: column_names, column_labels, original_variable_types,
                     variable_value_labels.
    df : DataFrame, optional
        The actual data. When provided, data-based classification is used
        to determine numeric vs categorical (much more accurate).
        When None, falls back to conservative metadata-only heuristics.

    Returns
    -------
    dict with keys:
        income_p_gate, income_p_num, income_p_cat,
        income_hh_gate, income_hh_num, income_hh_cat
    """
    result: dict[str, Optional[str]] = {
        'income_p_gate':  None,
        'income_p_num':   None,
        'income_p_cat':   None,
        'income_hh_gate': None,
        'income_hh_num':  None,
        'income_hh_cat':  None,
    }

    col_names  = meta.column_names
    col_labels = meta.column_labels
    var_types  = meta.original_variable_types
    vvl_all    = meta.variable_value_labels

    # -----------------------------------------------------------------------
    # Pass 1: collect candidate income columns
    # -----------------------------------------------------------------------
    candidates = []

    for i, col_name in enumerate(col_names):
        label = col_labels[i] if i < len(col_labels) else ''
        if label is None:
            label = ''
        col_type = var_types.get(col_name, '')
        vvl = vvl_all.get(col_name, {})

        ln = label.lower().replace('pana[i]', 'pana(i)').replace('pan[i]', 'pana(i)')

        # --- Quick pre-filter: must mention income or be an ANKIETER marker ---
        income_context = (
            'dochod' in ln
            or 'dochów' in ln
            or 'dochód' in ln
            or 'zarobk' in ln
            or 'ankieter: poniżej wpisać' in ln
        )
        if not income_context:
            continue

        is_personal, is_hh_pc, is_total_hh, is_ankieter = _classify_label(label)

        if not (is_personal or is_hh_pc or is_total_hh or is_ankieter):
            continue

        # ---- Gate detection (metadata-based) ----
        is_gate = _is_gate_column(col_type, vvl)

        # ---- Num vs Cat classification ----
        if is_gate:
            is_num = False
            is_bracket = False
        else:
            is_bracket_by_label = _is_bracket_label(ln)

            if df is not None and col_name in df.columns:
                # DATA-BASED classification (primary, most reliable)
                is_num = _is_numeric_by_data(df[col_name])
                is_bracket = not is_num
            else:
                # METADATA-ONLY fallback
                is_bracket = is_bracket_by_label or _is_bracket_vvl_safe(vvl)
                is_num = not is_bracket

            # Override: if label explicitly says bracket, trust it
            if is_bracket_by_label:
                is_bracket = True
                is_num = False

        candidates.append({
            'idx':         i,
            'col_name':    col_name,
            'label':       label,
            'ln':          ln,
            'col_type':    col_type,
            'n_labs':      len(vvl),
            'is_personal': is_personal,
            'is_hh_pc':    is_hh_pc,
            'is_total_hh': is_total_hh,
            'is_ankieter': is_ankieter,
            'is_gate':     is_gate,
            'is_num':      is_num,
            'is_bracket':  is_bracket,
        })

    # -----------------------------------------------------------------------
    # Pass 2: resolve ANKIETER columns (scope from nearest neighbour)
    # -----------------------------------------------------------------------
    for j, c in enumerate(candidates):
        if c['is_ankieter']:
            resolved = False
            for k in range(j - 1, -1, -1):
                prev = candidates[k]
                if prev['is_personal'] and not prev['is_ankieter']:
                    c['is_personal'] = True
                    c['is_ankieter'] = False
                    resolved = True
                    break
                if prev['is_hh_pc'] and not prev['is_ankieter']:
                    c['is_hh_pc'] = True
                    c['is_ankieter'] = False
                    resolved = True
                    break
            if not resolved:
                for k in range(j + 1, len(candidates)):
                    nxt = candidates[k]
                    if nxt['is_personal'] and not nxt['is_ankieter']:
                        c['is_personal'] = True
                        c['is_ankieter'] = False
                        resolved = True
                        break
                    if nxt['is_hh_pc'] and not nxt['is_ankieter']:
                        c['is_hh_pc'] = True
                        c['is_ankieter'] = False
                        resolved = True
                        break

    # -----------------------------------------------------------------------
    # Pass 3: fill result (first-match wins per slot)
    # -----------------------------------------------------------------------
    for c in candidates:
        if c['is_total_hh'] or c['is_ankieter']:
            continue

        if c['is_personal']:
            if c['is_gate'] and result['income_p_gate'] is None:
                result['income_p_gate'] = c['col_name']
            elif c['is_bracket'] and result['income_p_cat'] is None:
                result['income_p_cat'] = c['col_name']
            elif c['is_num'] and result['income_p_num'] is None:
                result['income_p_num'] = c['col_name']

        elif c['is_hh_pc']:
            if c['is_gate'] and result['income_hh_gate'] is None:
                result['income_hh_gate'] = c['col_name']
            elif c['is_bracket'] and result['income_hh_cat'] is None:
                result['income_hh_cat'] = c['col_name']
            elif c['is_num'] and result['income_hh_num'] is None:
                result['income_hh_num'] = c['col_name']

    # -----------------------------------------------------------------------
    # Pass 4: fallback — total-HH fills empty HH slots when no per-capita
    # -----------------------------------------------------------------------
    hh_empty = (result['income_hh_num'] is None and result['income_hh_cat'] is None)
    if hh_empty:
        for c in candidates:
            if not c['is_total_hh']:
                continue
            if c['is_gate'] and result['income_hh_gate'] is None:
                result['income_hh_gate'] = c['col_name']
            elif c['is_bracket'] and result['income_hh_cat'] is None:
                result['income_hh_cat'] = c['col_name']
            elif c['is_num'] and result['income_hh_num'] is None:
                result['income_hh_num'] = c['col_name']

    return result


# ---------------------------------------------------------------------------
# Convenience: extract + print for debugging
# ---------------------------------------------------------------------------

def print_extraction(file: str, meta, df=None) -> dict:
    r = extract_income_columns(file, meta, df)
    print(f"\n{file}")
    for k, v in r.items():
        if v is not None:
            lbl = meta.column_labels[meta.column_names.index(v)]
            ctype = meta.original_variable_types.get(v, '?')
            n_labs = len(meta.variable_value_labels.get(v, {}))
            print(f"  {k:20s} = {v:12s}  [{ctype}, {n_labs} labs]  {lbl[:80]}")
        else:
            print(f"  {k:20s} = None")
    return r
