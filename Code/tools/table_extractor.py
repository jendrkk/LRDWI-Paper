from img2table.document import Image
from img2table.ocr import PaddleOCR
from pathlib import Path

# 1. Initialize the OCR engine
ocr = PaddleOCR(lang="pl") # "pol" for Polish characters

# 2. Load the image
in_root = Path('/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/GUS/data/raw pics')
file_name = "pop__age_1989.png"
img = Image(src=str(in_root / file_name))

out_root = in_root.parent / "extracted"
csv_name = file_name.replace('.png', '.xlsx')

# 3. Extract tables and export to CSV
# This will automatically detect the table structure
img.to_xlsx(dest=out_root / csv_name,
            ocr=ocr,
            implicit_rows=False,
            implicit_columns=False,
            borderless_tables=True,
            min_confidence=55)