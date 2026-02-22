"""
table_extractor.py
==================
Robust extraction of numeric tables from two-page book scans (Polish statistical
yearbooks, 1980s–1990s).

Key challenges addressed
------------------------
1. Left/right page height mismatch  → split image at the detected spine and
   process each half independently; re-join rows by the printed row-number
   columns that appear on both sides.
2. Merged rows                      → detect cells that contain '\\n' and explode
   them back into separate rows.
3. Empty output files               → retry with relaxed parameters; fall back to
   full-image extraction.
4. OCR digit errors                 → translate common mis-reads (O→0, l→1, …)
   after extraction.

Validation
----------
After assembling the table we run two arithmetic consistency checks:
  • Row check  : sum(row[2:]) ≈ row[1]  (age-group columns sum to total)
  • Column check: sum(col[1:]) ≈ col[0]  (voivodeships sum to Poland total)
Results are written to a 'validation' sheet inside the same .xlsx file.
"""

from __future__ import annotations

import re
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from img2table.document import Image as Img2TableImage
from img2table.ocr import PaddleOCR
from PIL import Image as PILImage

# ── OCR engine ────────────────────────────────────────────────────────────────
ocr = PaddleOCR(lang="pl")

# ── file lists ────────────────────────────────────────────────────────────────
YEARS = list(range(1986, 1995))
file_names = [f"pop__age_{y}.png" for y in YEARS]
file_names += [f"pop__age_men_{y}.png" for y in YEARS]

in_root = Path(
    "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/"
    "Long-run dynamics of wealth inequalities/Paper/Data/GUS/data/raw pics"
)
out_root = in_root.parent / "extracted"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: image splitting
# ─────────────────────────────────────────────────────────────────────────────

def find_page_split(img_path: Path) -> int:
    """
    Detect the x-coordinate of the vertical spine between the two book pages.

    Strategy: the spine is a bright (nearly white) vertical band roughly in the
    centre of the scan.  We search in the middle-third of the image width for
    the column with the highest mean brightness.
    """
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cv2 could not read {img_path}")
    h, w = gray.shape
    x0, x1 = w // 3, 2 * w // 3
    col_means = gray[:, x0:x1].mean(axis=0)
    split_local = int(np.argmax(col_means))
    split_x = x0 + split_local
    # Keep the split within 25–75 % of the image width as a sanity guard
    split_x = max(w // 4, min(3 * w // 4, split_x))
    return split_x


def preprocess_half(pil_img: PILImage.Image) -> PILImage.Image:
    """
    Light preprocessing to improve OCR on historical scans:
      - convert to grayscale
      - mild CLAHE contrast enhancement
      - slight sharpening
    """
    img_np = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_np = clahe.apply(img_np)
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_np = cv2.filter2D(img_np, -1, kernel)
    return PILImage.fromarray(img_np).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: table extraction from a PIL image
# ─────────────────────────────────────────────────────────────────────────────

def _extract_df_from_pil(
    pil_img: PILImage.Image,
    ocr_engine,
    tmp_path: Path,
    min_confidence: int = 50,
    implicit_rows: bool = False,
) -> pd.DataFrame | None:
    """Save *pil_img* to *tmp_path* and run img2table on it."""
    pil_img.save(str(tmp_path))
    img2t = Img2TableImage(src=str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = img2t.extract_tables(
            ocr=ocr_engine,
            implicit_rows=implicit_rows,
            implicit_columns=False,
            borderless_tables=True,
            min_confidence=min_confidence,
        )
    if not tables:
        return None
    # Return the largest table found
    best = max(tables, key=lambda t: t.df.size)
    return best.df.copy()


def extract_df_with_retry(
    pil_img: PILImage.Image,
    ocr_engine,
    tmp_path: Path,
    label: str = "",
) -> pd.DataFrame | None:
    """
    Try extracting with progressively more permissive settings.
    Returns None only if all attempts fail.
    """
    attempts = [
        dict(min_confidence=50, implicit_rows=False),
        dict(min_confidence=30, implicit_rows=False),
        dict(min_confidence=30, implicit_rows=True),
    ]
    for attempt in attempts:
        df = _extract_df_from_pil(pil_img, ocr_engine, tmp_path, **attempt)
        if df is not None and df.size > 0:
            return df
        print(f"    [{label}] attempt {attempt} yielded nothing, retrying...")

    # Last resort: upscale 2x (helps when text is very small)
    w, h = pil_img.size
    upscaled = pil_img.resize((w * 2, h * 2), PILImage.LANCZOS)
    df = _extract_df_from_pil(upscaled, ocr_engine, tmp_path,
                               min_confidence=30, implicit_rows=True)
    if df is not None and df.size > 0:
        print(f"    [{label}] succeeded after 2x upscale")
        return df

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: merged-row repair
# ─────────────────────────────────────────────────────────────────────────────

def split_merged_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    img2table sometimes merges two rows into one cell (values separated by '\\n').
    This function detects such cells and explodes the row into multiple rows.
    """
    new_rows: list[list] = []
    for row in df.itertuples(index=False):
        cells = list(row)
        # Count the maximum number of newline-delimited parts in any cell
        n_parts = max(
            len(str(c).split("\n")) if pd.notna(c) else 1
            for c in cells
        )
        if n_parts == 1:
            new_rows.append(cells)
        else:
            for i in range(n_parts):
                new_row = []
                for c in cells:
                    if pd.isna(c):
                        new_row.append(np.nan)
                    else:
                        parts = str(c).split("\n")
                        new_row.append(parts[i].strip() if i < len(parts) else "")
                new_rows.append(new_row)
    result = pd.DataFrame(new_rows, columns=df.columns)
    result.reset_index(drop=True, inplace=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper: number cleaning
# ─────────────────────────────────────────────────────────────────────────────

_OCR_DIGIT_MAP = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "D": "0",
        "Q": "0",
        "l": "1",
        "I": "1",
        "|": "1",
        "!": "1",
        "S": "5",
        "s": "5",
        "G": "6",
        "B": "8",
    }
)


def clean_number(val) -> "int | float":
    """
    Convert an OCR cell value to int.

    Handles:
      - thousands-separator spaces / non-breaking spaces
      - common digit mis-reads (O->0, l->1, S->5, ...)
      - leading/trailing garbage characters
    Returns np.nan on any failure.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    if s in ("", "-", "\u2014", "\u2013", "None", "nan"):
        return np.nan
    s = s.replace(" ", "").replace("\xa0", "").replace("\n", "")
    s = s.translate(_OCR_DIGIT_MAP)
    digits = "".join(c for c in s if c.isdigit())
    if not digits:
        return np.nan
    return int(digits)


def is_name_column(series: pd.Series) -> bool:
    """True if the Series looks like a territorial-name column (mostly text)."""
    has_alpha = series.dropna().apply(
        lambda v: bool(re.search(r"[A-Za-z\u00C0-\u017E]", str(v)))
    )
    return has_alpha.mean() > 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Helper: row-number column detection and half-alignment
# ─────────────────────────────────────────────────────────────────────────────

def find_row_num_col(df: pd.DataFrame) -> "object | None":
    """
    Return the column label whose values look like 1..50 printed row indices.
    Returns None if no such column is found.
    """
    for col in df.columns:
        nums = df[col].apply(clean_number).dropna()
        if len(nums) < 35:
            continue
        int_vals = nums.dropna().astype(int)
        # Must be in range 1-60 with at least 35 distinct values
        if int_vals.min() >= 1 and int_vals.max() <= 60 and int_vals.nunique() >= 35:
            return col
    return None


def align_and_join(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge left and right page DataFrames, aligning rows via the printed row-number
    column found on each side.  Falls back to index-based concat if no such column
    is detected.
    """
    left_rn = find_row_num_col(left_df)
    right_rn = find_row_num_col(right_df)

    print(f"  Row-number column: left={left_rn}, right={right_rn}")

    left = left_df.copy()
    right = right_df.copy()

    # Assign join key
    if left_rn is not None:
        left["_rn"] = left[left_rn].apply(clean_number)
        left = left.drop(columns=[left_rn])
    else:
        left["_rn"] = range(1, len(left) + 1)

    if right_rn is not None:
        right["_rn"] = right[right_rn].apply(clean_number)
        right = right.drop(columns=[right_rn])
    else:
        right["_rn"] = range(1, len(right) + 1)

    # Drop duplicated row-number entries (keep first occurrence)
    left = left.drop_duplicates(subset=["_rn"], keep="first")
    right = right.drop_duplicates(subset=["_rn"], keep="first")

    left = left.set_index("_rn")
    right = right.set_index("_rn")

    joined = left.join(right, how="outer", lsuffix="_L", rsuffix="_R")
    joined = joined.sort_index()
    joined.columns = range(len(joined.columns))
    return joined.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run two arithmetic consistency checks on a cleaned numeric DataFrame:

    1. Row sums: for each voivodeship row (row 1 onwards), the age-group
       columns (col 2 to end) should sum to the total column (col 1).
    2. Column sums: for each numeric column (col 1 to end), the sum of
       voivodeship rows (row 1 onwards) should equal the Poland-total row (row 0).

    Returns a DataFrame of issues (empty if everything checks out).
    """
    numeric = df.copy()
    for col in numeric.columns:
        numeric[col] = numeric[col].apply(clean_number)

    tol = 0.02  # 2% relative tolerance
    issues = []

    # Row sum check
    for i in range(1, len(numeric)):
        row_total = numeric.iloc[i, 1]
        age_sum = numeric.iloc[i, 2:].sum()
        if pd.isna(row_total) or pd.isna(age_sum):
            issues.append(
                dict(check="row_sum", row=i, col="", expected=row_total,
                     got=age_sum, note="NaN present")
            )
        elif row_total != 0 and abs(row_total - age_sum) / abs(row_total) > tol:
            issues.append(
                dict(check="row_sum", row=i, col="", expected=row_total,
                     got=age_sum, note=f"diff={row_total - age_sum:+.0f}")
            )

    # Column sum check
    for col in range(1, len(numeric.columns)):
        poland_val = numeric.iloc[0, col]
        col_sum = numeric.iloc[1:, col].sum()
        if pd.isna(poland_val) or pd.isna(col_sum):
            issues.append(
                dict(check="col_sum", row="", col=col, expected=poland_val,
                     got=col_sum, note="NaN present")
            )
        elif poland_val != 0 and abs(poland_val - col_sum) / abs(poland_val) > tol:
            issues.append(
                dict(check="col_sum", row="", col=col, expected=poland_val,
                     got=col_sum, note=f"diff={poland_val - col_sum:+.0f}")
            )

    return pd.DataFrame(issues)


# ─────────────────────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────────────────────

def process_file(file_path: Path, ocr_engine, output_root: Path) -> dict:
    """
    Full pipeline for one PNG file.

    Steps
    -----
    1. Detect page spine and split image into left / right halves.
    2. Preprocess each half (contrast + sharpening).
    3. Extract tables from each half independently (with retry).
    4. Repair merged rows.
    5. Align halves by printed row-number columns.
    6. Clean OCR digit errors in numeric columns.
    7. Validate arithmetic consistency.
    8. Save data sheet + optional validation sheet to .xlsx.
    """
    status: dict = {"file": file_path.name, "ok": False, "issues": 0}
    print(f"\n{'='*60}")
    print(f"Processing: {file_path.name}")

    pil_full = PILImage.open(str(file_path))
    w, h = pil_full.size

    # 1. Split
    try:
        split_x = find_page_split(file_path)
    except Exception as exc:
        split_x = w // 2
        print(f"  Warning: page-split detection failed ({exc}); using centre ({split_x}).")
    print(f"  Page spine at x={split_x}  (image {w}x{h})")

    left_pil  = pil_full.crop((0,       0, split_x, h))
    right_pil = pil_full.crop((split_x, 0, w,       h))

    # 2. Preprocess
    left_pil  = preprocess_half(left_pil)
    right_pil = preprocess_half(right_pil)

    # 3. Extract
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_left  = Path(tmp_dir) / "left.png"
        tmp_right = Path(tmp_dir) / "right.png"

        print("  Extracting left half...")
        left_df  = extract_df_with_retry(left_pil,  ocr_engine, tmp_left,  "LEFT")
        print("  Extracting right half...")
        right_df = extract_df_with_retry(right_pil, ocr_engine, tmp_right, "RIGHT")

    # Fallback: try full image when both halves fail
    if left_df is None and right_df is None:
        print("  Both halves failed - trying full image as fallback...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_full = Path(tmp_dir) / "full.png"
            preprocessed_full = preprocess_half(pil_full)
            left_df = extract_df_with_retry(
                preprocessed_full, ocr_engine, tmp_full, "FULL"
            )
        right_df = None

    if left_df is None and right_df is None:
        print("  Extraction completely failed - skipping file.")
        status["ok"] = False
        return status

    # 4. Repair merged rows
    if left_df is not None:
        before = len(left_df)
        left_df = split_merged_rows(left_df)
        if len(left_df) > before:
            print(f"  Split {len(left_df) - before} merged rows in left half.")

    if right_df is not None:
        before = len(right_df)
        right_df = split_merged_rows(right_df)
        if len(right_df) > before:
            print(f"  Split {len(right_df) - before} merged rows in right half.")

    # 5. Align and join
    if left_df is not None and right_df is not None:
        print(f"  Left shape {left_df.shape}, Right shape {right_df.shape}")
        df = align_and_join(left_df, right_df)
        print(f"  Joined shape {df.shape}")
    elif left_df is not None:
        print("  Using left half only.")
        df = left_df
    else:
        print("  Using right half only.")
        df = right_df

    # 6. Clean numbers
    cleaned = df.copy()
    name_col_idx = None
    for i, col in enumerate(cleaned.columns):
        if is_name_column(cleaned[col]):
            name_col_idx = i
            break

    for i, col in enumerate(cleaned.columns):
        if i == name_col_idx:
            cleaned[col] = cleaned[col].apply(
                lambda v: str(v).strip() if pd.notna(v) else v
            )
        else:
            cleaned[col] = cleaned[col].apply(clean_number)

    # 7. Validate
    issues_df = validate_table(cleaned)
    n_issues = len(issues_df)
    if n_issues == 0:
        print("  Validation passed - all row/column sums consistent.")
    else:
        n_row = (issues_df["check"] == "row_sum").sum()
        n_col = (issues_df["check"] == "col_sum").sum()
        print(f"  Validation: {n_row} row-sum errors, {n_col} col-sum errors.")
    status["issues"] = n_issues

    # 8. Save
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / file_path.name.replace(".png", ".xlsx")

    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        cleaned.to_excel(writer, sheet_name="data", index=False, header=False)
        if not issues_df.empty:
            issues_df.to_excel(writer, sheet_name="validation", index=False)

    print(f"  Saved -> {out_path}  ({out_path.stat().st_size // 1024} KB)")
    status["ok"] = True
    return status


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for file_name in file_names:
        file_path = in_root / file_name
        if not file_path.exists():
            print(f"  [SKIP] {file_name} not found.")
            continue
        result = process_file(file_path, ocr, out_root)
        summary.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok   = [r for r in summary if r["ok"]]
    fail = [r for r in summary if not r["ok"]]
    print(f"  Succeeded : {len(ok)}")
    print(f"  Failed    : {len(fail)}")
    if fail:
        for r in fail:
            print(f"  FAILED: {r['file']}")
    warn = [r for r in ok if r["issues"] > 0]
    if warn:
        print(f"  With validation issues: {len(warn)}")
        for r in warn:
            print(f"  WARNING: {r['file']}  ({r['issues']} issues)")
    else:
        print("  All extracted files passed arithmetic validation.")