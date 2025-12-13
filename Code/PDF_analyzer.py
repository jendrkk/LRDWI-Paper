"""
This module provides a tool for searching in a PDF document for specific phrases
and extracting them along with a specified number.
"""

import fitz  # PyMuPDF
import re
import os
import time
from typing import List, Tuple
import pickle
import numpy as np
import pandas as pd

PATH_PDF = "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/CBOS pdf/"

def preprocess_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a pandas DataFrame by removing empty rows and columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to preprocess.

    Returns:
    --------
    pd.DataFrame
        The preprocessed DataFrame.

    NEW COMMENT:
    If there is a cell with "Strona XX" (XX: arbitrary number) and apart of that the row is empty,
    we remove that row as it is not part of the table.
    """
    # Defensive check: require a DataFrame
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    # Work on a copy to avoid mutating caller's object
    df = df.copy()

    # Remove fully empty rows and columns first
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    # Remove rows that contain only a "Strona <number>" string (case-insensitive)
    rows_to_drop = []
    for idx in df.index:
        row = df.loc[idx]
        # collect non-empty cells in this row
        non_empty = []
        for cell in row:
            if pd.isna(cell):
                continue
            if isinstance(cell, str):
                txt = cell.strip()
                if txt == "":
                    continue
                non_empty.append(txt)
            else:
                # numeric or other non-NA value
                non_empty.append(str(cell))
        if len(non_empty) == 1:
            # Check if the single non-empty cell matches "Strona <digits>"
            if re.match(r'^\s*Strona\s*\d+\s*$', non_empty[0], flags=re.IGNORECASE):
                rows_to_drop.append(idx)
    if rows_to_drop:
        df = df.drop(index=rows_to_drop)
        df = df.dropna(how='all')  # drop if that made fully-empty rows

    # The output dataframe should have always 4 columns where:
    # Column 0: Int
    # Column 1: Label
    # Column 2: Count
    # Column 3: Percent
    # First entry of the output dataframe should be at location [0,0] in the input dataframe
    # If this is empty, we priortize the first number we find somewhat diagonally to the right
    # and below from the [0,0] position.

    if df.shape[1] < 4:
        return pd.DataFrame()  # not enough columns

    def is_integer_like(x):
        if pd.isna(x):
            return False
        if isinstance(x, (int, np.integer)):
            return True
        if isinstance(x, (float, np.floating)):
            return float(x).is_integer()
        if isinstance(x, str):
            s = x.strip()
            # Only digits (no thousands separators). If you have commas, adjust here.
            return bool(re.fullmatch(r"\d+", s))
        return False

    first_num_pos = None
    for i in range(df.shape[0]):
        # search up to diagonal (i+1 columns)
        for j in range(min(i + 1, df.shape[1])):
            cell = df.iat[i, j]
            if is_integer_like(cell):
                first_num_pos = (i, j)
                break
        if first_num_pos is not None:
            break

    if first_num_pos is None:
        return pd.DataFrame()  # no numbers found

    row, col = first_num_pos

    # Build label from cells left of the found number in the same row (non-numeric, non-empty)
    label_parts = []
    for j in range(col):
        cell = df.iat[row, j]
        if pd.isna(cell):
            continue
        if isinstance(cell, str):
            txt = cell.strip()
            if txt == "":
                continue
            # skip if cell contains numbers
            if any(ch.isdigit() for ch in txt):
                continue
            label_parts.append(txt)
        else:
            s = str(cell).strip()
            if s == "":
                continue
            if any(ch.isdigit() for ch in s):
                continue
            label_parts.append(s)

    label = ' '.join(label_parts).strip()

    count_cell = df.iat[row, col]
    if not is_integer_like(count_cell):
        return pd.DataFrame()  # can't interpret count

    # normalize count to int
    try:
        count_int = int(float(str(count_cell).strip()))
    except Exception:
        return pd.DataFrame()

    percent = None
    if (col + 1) < df.shape[1]:
        pct_cell = df.iat[row, col + 1]
        if not pd.isna(pct_cell):
            percent = pct_cell

    # Build resulting DataFrame (one-row) with the same column names as original function
    out = pd.DataFrame({
        "Dummy": [count_int],
        "Label": [label],
        "Count": [count_int],
        "Percent": [percent]
    })

    return out.reset_index(drop=True)

class PDFAnalyzer:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
    
    def get_blocks_on_page(self, page_num: int) -> List[Tuple[float, float, float, float, str, pd.DataFrame]]:
        """
        Get text blocks on a specific page.

        Parameters:
        -----------
        page_num : int
            The page number (0-based index).

        Returns:
        --------
        List[Tuple[float, float, float, float, str]]
            A list of tuples containing (x0, y0, x1, y1, text) for each block.
        """
        page = self.doc.load_page(page_num)
        blocks = page.get_text("blocks")
        return blocks
    
    def get_drawings_on_page(self, page_num: int) -> List[dict]:
        """
        Get drawing objects on a specific page.

        Parameters:
        -----------
        page_num : int
            The page number (0-based index).

        Returns:
        --------
        List[dict]
            A list of drawing objects.
        """
        page = self.doc.load_page(page_num)
        drawings = page.get_cdrawings()
        return drawings
    
    
    def get_page_size(self, page_num: int) -> Tuple[float, float]:
        """
        Get the size of a specific page.

        Parameters:
        -----------
        page_num : int
            The page number (0-based index).

        Returns:
        --------
        Tuple[float, float]
            A tuple containing (width, height) of the page.
        """
        
        page = self.doc.load_page(page_num)        
        return (page.rect.width, page.rect.height)
    
    
    def extract_questions(self) -> Tuple[List[Tuple[int, str, str, str, pd.DataFrame, pd.DataFrame]], dict]:
        """
        Extracts all questions, that is text after the word
        'Etykieta', strings starting with 'q' followed by 
        digits and a type of table that is after the phrase
        'Wartosci z etykietami' from the PDF.
        Returns a list of tuples (question_text, page_number).
        """
        results = []
        diagnostics = {'pages_with_no_q_strings': [],
                       'pages_with_multiple_q_strings': []}
        
        if_codebook = False
        
        # Auxiliary vectorized functions
        f_vstartswith = np.vectorize(lambda x,y: x.startswith(y) if isinstance(x, str) and isinstance(y,str) else x)
        f_vcompare = np.vectorize(lambda x,y: x==y)
        f_vreplace = np.vectorize(lambda x, old, new: x.replace(old, new) if isinstance(x, str) else x)
        
        for page_num in range(len(self.doc)):
            # Load the page
            page = self.doc.load_page(page_num)
            page_numer = page_num + 1  # 1-based page numbering
            
            if 'Codebook' in page.get_text():
                if_codebook = True
            if not if_codebook:
                continue  # skip pages before CODEBOOK
            print(f"[PDFAnalyzer] Processing page {page_num + 1} / {len(self.doc)}")
            # Prepare blocks
            page_blocks = self.get_blocks_on_page(page_num)
            page_blocks = np.array(page_blocks)
                        
            for i in range(len(page_blocks)):
                page_blocks[i, 4] = page_blocks[i, 4].astype(str).strip()
                page_blocks[i, 4] = page_blocks[i, 4].replace('\n', '')
                
            # Quick look up
            
            qs = np.where(f_vstartswith(page_blocks[:,4], 'q'))[0]
            types = np.where(f_vcompare(page_blocks[:,4], 'Typ'))[0]
            questions = np.where(f_vcompare(page_blocks[:,4], 'Etykieta'))[0]
            tables = np.where(f_vstartswith(page_blocks[:,4], 'Wartości z etykietami'))[0]
            continued_table_pd = None
            
            types = types if len(types) > 0 else np.where(f_vstartswith(page_blocks[:,4], 'Typ'))[0]
            questions = questions if len(questions) > 0 else np.where(f_vstartswith(page_blocks[:,4], 'Typ'))[0]
            
            # Initialize found variables
            found_q_string = ""
            found_etykieta = ""
            found_type = ""
            found_table = pd.DataFrame()
            continued_table_pd = pd.DataFrame()
            
            def find_closest_right_block(ref_idx: int, blocks: np.ndarray) -> int:
                ref_x0 = float(blocks[ref_idx, 0])
                ref_x1 = float(blocks[ref_idx, 2])
                ref_y0 = float(blocks[ref_idx, 1])
                ref_y1 = float(blocks[ref_idx, 3])
                candidates = []
                for i in range(len(blocks)):
                    if i == ref_idx:
                        continue
                    bx0 = float(blocks[i, 0])
                    by0 = float(blocks[i, 1])
                    by1 = float(blocks[i, 3])
                    # Check if block is to the right and vertically overlaps
                    if bx0 > ref_x0 and (float(min(ref_y1, by1)) - float(max(ref_y0, by0) ) > 0):
                        candidates.append((i, bx0 - ref_x1))
                if not candidates:
                    return -1
                # Return index of the closest block
                candidates.sort(key=lambda x: x[1])
                return candidates[0][0]
            
            def extract_table(page: fitz.Page, ref_idx: int, blocks: np.ndarray) -> pd.DataFrame:
                if blocks[ref_idx+1,4] == "Wartość Liczebność Procent":
                    ref_idx += 1  # move to the next block
                
                ints = np.vectorize(lambda x: x.isdigit() if isinstance(x, str) else False)(blocks[:,4])
                seq_of_rows = []
                
                i=1
                while ints[ref_idx+i]:
                    seq_of_rows += [np.mean([float(blocks[ref_idx+i,1]), float(blocks[ref_idx+i,3])] )]
                    i += 1
                
                # Check if its strictly increasing and if not remove the last element until it is
                while not all(x<y for x, y in zip(seq_of_rows, seq_of_rows[1:])):
                    seq_of_rows = seq_of_rows[:-1]
                
                if blocks[ref_idx,4] == "Wartość Liczebność Procent":
                    rect = [blocks[ref_idx-1,2].astype(float), blocks[ref_idx-1,1].astype(float),
                            blocks[:,2].astype(float).max(), max(seq_of_rows)+10]
                    table = page.find_tables(clip = rect, strategy= 'text', min_words_vertical=1, snap_tolerance=6)
                    
                else:
                    rect = [blocks[ref_idx,2].astype(float)-5, blocks[ref_idx,1].astype(float), blocks[:,2].astype(float).max(),
                            np.max(seq_of_rows)+10]
                    table = page.find_tables(clip = rect, strategy= 'text', snap_tolerance=6)
                    
                if len(table.tables) == 0 :
                    return pd.DataFrame()
                table = table.tables[0].to_pandas()
                
                return table
            
            def center_of(rect: Tuple) -> Tuple[float, float]:
                x0, y0, x1, y1 = rect
                return ((float(x0) + float(x1)) / 2, (float(y0) + float(y1)) / 2)
                
            # Check if our page has only one q string
            if len(qs) == 1:
                if len(results) > 0:
                    last_q = results[-1][1]
                    if last_q in page_blocks[qs,4]:
                        # Case where we either have only a continued table or some continued labels for the previous q
                        
                        found_q_string = page_blocks[qs,4][0]
                        
                        if len(questions) > 0:
                            found_etykieta = page_blocks[find_closest_right_block(questions[0], page_blocks),4]
                        if len(types) > 0:
                            found_type = page_blocks[find_closest_right_block(types[0], page_blocks),4]
                        if len(tables) > 0:
                            found_table = extract_table(page, tables[0], page_blocks)
                        
                        # Same q as before - means we have more labels for previous page -> add to table of the last entry
                        conitued_table = page.find_tables(strategy="text")
                        continued_table_pd = conitued_table.tables[0].to_pandas()
                        # Extract the names of the columns and use them as the new first row
                        cols = continued_table_pd.columns
                        continued_table_pd.columns = range(continued_table_pd.shape[1])
                        continued_table_pd.loc[-1] = cols  # adding a row
                        continued_table_pd.index = continued_table_pd.index + 1  # shifting index
                        continued_table_pd = continued_table_pd.sort_index()
                    else:
                        # Normal case with one q string on the page
                        found_q_string = page_blocks[qs,4][0]
                        found_etykieta = page_blocks[find_closest_right_block(questions[0], page_blocks),4]
                        found_type = page_blocks[find_closest_right_block(types[0], page_blocks),4]
                        if len(tables) > 0:
                            found_table = extract_table(page, tables[0], page_blocks)
                        
            elif len(qs) == 2:
                # Old lables and new questions on the same page - take the one with the higher number and add the old labels to old tables
                found_q_strings = page_blocks[qs,4]
                found_old_q = qs[0]
                found_q_string = found_q_strings[1]
                found_etykieta = page_blocks[find_closest_right_block(questions[0], page_blocks),4]
                found_type = page_blocks[find_closest_right_block(types[0], page_blocks),4]
                if len(tables) == 1:
                    #print(center_of(page_blocks[tables[0], :4])[1])
                    #print(center_of(page_blocks[qs[1], :4])[1])
                    if center_of(page_blocks[tables[0], :4])[1] > center_of(page_blocks[qs[1], :4])[1]:
                        #print(page_blocks[tables[0], 4])
                        #print(page_blocks[found_old_q, 4])
                        found_table = extract_table(page, tables[0], page_blocks)
                        continued_table_pd = extract_table(page, found_old_q, page_blocks)
                    else:
                        #print("\n")
                        #print("--------------------------------------------------")
                        #print(page_blocks[tables[0], 4])
                        #print("--------------------------------------------------")
                        found_table = pd.DataFrame()
                        continued_table_pd = extract_table(page, tables[0], page_blocks)
                elif len(tables) > 1:
                    #print("\n")
                    #print("--------------------------------------------------")
                    #print(tables)
                    #print(page_blocks[tables, 4])
                    #print("--------------------------------------------------")
                    found_table = extract_table(page, tables[1], page_blocks)
                    continued_table_pd = extract_table(page, tables[0], page_blocks)
                else:
                    found_table = pd.DataFrame()
                    continued_table_pd = extract_table(page, found_old_q, page_blocks)
            elif len(qs) > 2:
                print(f"[PDFAnalyzer] Warning: more than 2 q-strings found on page {page_numer}, skipping page.")
                diagnostics['pages_with_multiple_q_strings'].append(page_numer)
                continue
            else:
                print(f"[PDFAnalyzer] Warning: no q-strings found on page {page_numer}, skipping page.")
                diagnostics['pages_with_no_q_strings'].append(page_numer)
                continue
            
            print(f"Found q-string: {found_q_string}, etykieta: {found_etykieta[:10]}..., type: {found_type}, table size: {found_table.shape}, continued table size: {continued_table_pd.shape} \n")
            
            results.append((page_numer, found_q_string, found_etykieta, found_type,
                            found_table, continued_table_pd))
            
        return results, diagnostics
    
    def close(self):
        """Closes the PDF document."""
        self.doc.close()
        
def main():
    
    pdf_files = os.listdir(PATH_PDF)
    pdf_files = [f for f in pdf_files if f.lower().endswith('.pdf')]
    
    # Get only "CBOS_XXX" from the filenames, where XXX is the number and can be shorter than 3 digits
    pdf_ids = [re.match(r'CBOS_(\d+)', f).group(0) for f in pdf_files if re.match(r'CBOS_(\d+)', f)]
    
    # Zip together pdf_ids and pdf_files to ensure we have matching pairs
    pdf_id_file_map = {re.match(r'CBOS_(\d+)', f).group(0): f for f in pdf_files if re.match(r'CBOS_(\d+)', f)}
    
    results = {}
    
    start_time = time.time()
    for id in range(1,360):
        
        pdf_id = f"CBOS_{id}"
        
        if pdf_id not in pdf_id_file_map:
            continue  # skip missing files
        pdf_file = pdf_id_file_map[pdf_id]
        
        pdf_path = PATH_PDF + pdf_file  # Replace with your PDF file path
        #phrases_to_search = ["dochód", "Z ilu osób", "łączne dochody", "wykształcenie", "właścicielem gospodarstwa rolnego"]  # Example phrases
        
        print(f"Analyzing {pdf_file}...")
        
        analyzer = PDFAnalyzer(pdf_path)
        extracted_data, diagnotics = analyzer.extract_questions()
        results[pdf_file[:-4]] = extracted_data
        '''
        for item in extracted_data:
            page_num, q_string, etyk_text, typ, tables, old_table = item
            print(f"Found q-string '{q_string}' on page {page_num}")
            print(f"Type: {typ} Etykieta: {etyk_text}")
            print(f"Table of size {tables.shape} extracted and old table of size {old_table.shape} extracted.")
        '''
        analyzer.close()
        
    # Save results as json and remove .pdf from the filename
     
    print("The analysis took:", time.time() - start_time," seconds.")
    print(f"Saving results under pdf_codebook.json.")
     
    with open(f"pdf_codebook.json", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()