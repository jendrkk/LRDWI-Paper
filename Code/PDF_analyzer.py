"""
This module provides a tool for searching in a PDF document for specific phrases
and extracting them along with a specified number.
"""

import fitz  # PyMuPDF
import re
import os 
from typing import List, Tuple
import json


PATH_PDF = "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/CBOS pdf/"

class PDFAnalyzer:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def extract_phrase_with_number_and_etykieta(self, phrase: str) -> Tuple[str, str, str, str, int]:
        """
        Search the PDF for the first page that contains `phrase`. On that page:
        - Prefer left-side blocks to find a token that begins with a lowercase "q"
          immediately followed (possibly with spaces) by digits, e.g. "q12" or "q 12".
          Return the matched "q..." string and the captured number (as string).
        - Find the occurrence of the word "Etykieta" (case-insensitive). For the block
          containing that word, gather text to the right on the same line (blocks with
          overlapping vertical coordinates). From that concatenated right-side text,
          attempt to extract a substring like "[...digits...]. ...", possibly spanning
          multiple lines. Return that substring if found.
        Returns a tuple: (phrase_matched, q_number, q_found, etykieta_string, page_num).
        If nothing is found for an item, its value will be an empty string; page_num is -1
        when no page with the phrase was found.
        """
        # Defaults for not-found
        empty_result = ("", "", "", "", -1)

        # compile helpers
        phrase_re = re.compile(re.escape(phrase), re.IGNORECASE)
        # match lowercase 'q' followed by optional spaces and digits, require 'q' at token start
        q_pattern = re.compile(r'\bq\s*(\d+)\b')
        etyk_re = re.compile(r'etykieta', re.IGNORECASE)

        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            page_text = page.get_text()
            if not phrase_re.search(page_text):
                continue  # not the page we're looking for

            # Found the page containing the phrase; remember the exact matched phrase text
            m_phrase = phrase_re.search(page_text)
            phrase_text = m_phrase.group(0) if m_phrase else phrase

            q_found = ""
            q_number = ""

            # Prefer searching left-side blocks for 'q...' tokens for speed.
            try:
                blocks = page.get_text("blocks")
                # blocks: list of tuples (x0, y0, x1, y1, "text", ...)
                page_width = page.rect.width
                left_threshold = page_width * 0.4
                # sort blocks by x0 so we scan left-to-right consistently
                blocks_sorted = sorted(blocks, key=lambda b: b[0])
                left_match = None
                for b in blocks_sorted:
                    x0 = b[0]
                    block_text = b[4] if len(b) > 4 else ""
                    if x0 <= left_threshold:
                        m = q_pattern.search(block_text)
                        if m:
                            left_match = m
                            q_found = m.group(0)
                            q_number = m.group(1)
                            break
                # fallback to whole page if no left-side match
                if not left_match:
                    m = q_pattern.search(page_text)
                    if m:
                        q_found = m.group(0)
                        q_number = m.group(1)
            except Exception:
                # if blocks extraction fails, search whole page
                m = q_pattern.search(page_text)
                if m:
                    q_found = m.group(0)
                    q_number = m.group(1)

            # Now find "Etykieta" and try to extract the right-side descriptive string
            etykieta_text = ""
            try:
                blocks = page.get_text("blocks")
                # find blocks that contain "Etykieta"
                etyk_blocks = [b for b in blocks if etyk_re.search(b[4] if len(b) > 4 else "")]
                if etyk_blocks:
                    # pick the first occurrence
                    etb = etyk_blocks[0]
                    ex0, ey0, ex1, ey1 = etb[0], etb[1], etb[2], etb[3]
                    # collect candidate blocks that lie to the right and vertically overlap
                    candidates = []
                    for b in blocks:
                        bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
                        # vertical overlap
                        overlap = min(ey1, by1) - max(ey0, by0)
                        if overlap > 0 and (bx0 >= ex1 - 1):  # to the right (allow tiny tolerance)
                            candidates.append((bx0, b[4] if len(b) > 4 else ""))
                    # include the etykieta block text itself as the leftmost element
                    left_text = etb[4] if len(etb) > 4 else ""
                    # sort candidates by x0 and concatenate
                    candidates_sorted = [t for _, t in sorted(candidates, key=lambda x: x[0])]
                    combined = " ".join([left_text] + candidates_sorted).strip()
                    # Try to extract pattern like "[something containing numbers]. [some text...]" across lines
                    # look for bracketed segment containing digits followed by a dot and then some text
                    brack_pattern = re.compile(r'(\[.*?\d+.*?\]\.\s*[\s\S]+)', re.DOTALL)
                    mbr = brack_pattern.search(combined)
                    if mbr:
                        etykieta_text = mbr.group(1).strip()
                    else:
                        # fallback: look for first substring that has a number, a dot and some trailing text
                        fallback_pattern = re.compile(r'([^\n]*\d+[^\n]*\.\s*[\s\S]+)', re.DOTALL)
                        mfb = fallback_pattern.search(combined)
                        if mfb:
                            etykieta_text = mfb.group(1).strip()
                        else:
                            # last resort: if combined contains any digits, return the combined right-side text
                            if re.search(r'\d', combined):
                                etykieta_text = combined
                else:
                    # If no block-based Etykieta found, attempt a page-text based grab after the word
                    m_ety = etyk_re.search(page_text)
                    if m_ety:
                        # take the following 300 chars as candidate and try same patterns
                        start = m_ety.end()
                        snippet = page_text[start:start + 800]
                        brack_pattern = re.compile(r'(\[.*?\d+.*?\]\.\s*[\s\S]+)', re.DOTALL)
                        mbr = brack_pattern.search(snippet)
                        if mbr:
                            etykieta_text = mbr.group(1).strip()
                        else:
                            fallback_pattern = re.compile(r'([^\n]*\d+[^\n]*\.\s*[\s\S]+)', re.DOTALL)
                            mfb = fallback_pattern.search(snippet)
                            if mfb:
                                etykieta_text = mfb.group(1).strip()
            except Exception:
                # ignore errors and leave etykieta_text empty
                etykieta_text = etykieta_text or ""

            return (phrase_text, q_number, q_found, etykieta_text, page_num)

        # phrase not found in any page
        return empty_result
    
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
    
    for id in range(1,340):
        
        pdf_id = f"CBOS_{id}"
        
        if pdf_id not in pdf_id_file_map:
            continue  # skip missing files
        pdf_file = pdf_id_file_map[pdf_id]
        
        pdf_path = PATH_PDF + pdf_file  # Replace with your PDF file path
        phrases_to_search = ["dochód", "Z ilu osób", "łączne dochody", "wykształcenie", "właścicielem gospodarstwa rolnego"]  # Example phrases
        
        temp = []
        print(f"Analyzing {pdf_file}...")
        for phrase in phrases_to_search:
            analyzer = PDFAnalyzer(pdf_path)
            extracted_data = analyzer.extract_phrase_with_number_and_etykieta(phrase)
            temp.append(extracted_data)
            analyzer.close()
            if extracted_data[4] == -1:
                continue
            else:
                print(f"  Found phrase '{extracted_data[0]}' on page {extracted_data[4]} with q='{extracted_data[2]}'")
            
        results[pdf_id] = temp
        
    # Save results as json
    with open("pdf_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()