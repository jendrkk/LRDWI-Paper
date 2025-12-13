    """
    def extract_questions(self) -> List[Tuple[int, str, str, List]]:
        
        Extracts all questions, that is text after the word
        'Etykieta', strings starting with 'q' followed by 
        digits and a type of table that is after the phrase
        'Wartosci z etykietami' from the PDF.
        Returns a list of tuples (question_text, page_number).
        
        results = []
        q_pattern = re.compile(r'q(\d+)', re.IGNORECASE)
        etyk_re = re.compile(r'\bEtykieta\b', re.IGNORECASE)
        typ_re = re.compile(r'\bTyp\b', re.IGNORECASE)
        # accept both accented and non-accented spellings
        table_re = re.compile(r'Wartości z etykietami|Wartosci z etykietami', re.IGNORECASE)
        
        if_codebook = False
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            page_numer = page_num + 1  # 1-based page numbering
            page_text = page.get_text()
            if 'Codebook' in page_text:
                if_codebook = True
            if not if_codebook:
                continue  # skip pages before CODEBOOK
            
            found_q_string = None
            found_q_strings = []
            found_etykieta = None
            found_table = []
            
            # Find all 'q...' occurrences
            for m in q_pattern.finditer(page_text):
                q_text = m.group(0)
                found_q_strings.append((q_text, page_num))
            
            # For the found q string on this page, choose the q string with the highest number
            if found_q_strings:
                found_q_strings.sort(key=lambda x: int(re.search(r'\d+', x[0]).group(0)), reverse=True)
                found_q_string = found_q_strings[0][0]
            
            # Extract the question text after 'Etykieta'
            # Find all 'Etykieta' occurrences and, for the first one, locate the nearest
            # non-whitespace character geometrically to its left and return the whole
            # concatenated text that lies to the right of that character (include the
            # etykieta block and any right-side overlapping blocks).
            try:
                blocks = page.get_text("blocks")
                found_etykieta = None

                # blocks: list of tuples (x0, y0, x1, y1, "text", ...)
                etyk_blocks = [b for b in blocks if etyk_re.search(b[4] if len(b) > 4 else "")]
                if etyk_blocks:
                    # use first occurrence
                    etb = etyk_blocks[0]
                    ex0, ey0, ex1, ey1 = etb[0], etb[1], etb[2], etb[3]

                    # find left-side blocks that vertically overlap and lie to the left
                    left_candidates = [
                        b for b in blocks
                        if (min(ey1, b[3]) - max(ey0, b[1]) > 0) and (b[2] <= ex0 + 1)
                    ]
                    left_block = max(left_candidates, key=lambda b: b[2]) if left_candidates else None

                    if left_block:
                        # boundary is the right edge of the closest left block
                        boundary_x = left_block[2]
                        # collect blocks to the right of that boundary that vertically overlap
                        right_candidates = [
                            b for b in blocks
                            if (min(ey1, b[3]) - max(ey0, b[1]) > 0) and (b[0] >= boundary_x - 1)
                        ]
                        right_sorted = [t[4] for t in sorted(right_candidates, key=lambda x: x[0])]
                        found_etykieta = " ".join(right_sorted).strip()
                    else:
                        # etykieta is in a block without a separate left block:
                        # find first non-whitespace char to the left inside the same block
                        etb_text = etb[4] if len(etb) > 4 else ""
                        mloc = re.search(r'etykieta', etb_text, re.IGNORECASE)
                        if mloc:
                            start_idx = mloc.start()
                            i = start_idx - 1
                            while i >= 0 and etb_text[i].isspace():
                                i -= 1
                            if i >= 0:
                                # text that follows that left-found character (to the right)
                                tail = etb_text[i+1:].strip()
                            else:
                                tail = etb_text[start_idx:].strip()
                            # also include any right-side overlapping blocks (to the right of etyk block)
                            right_candidates = [
                                b for b in blocks
                                if (min(ey1, b[3]) - max(ey0, b[1]) > 0) and (b[0] >= ex1 - 1)
                            ]
                            right_sorted = [t[4] for t in sorted(right_candidates, key=lambda x: x[0])]
                            found_etykieta = " ".join([tail] + right_sorted).strip()
                else:
                    # fallback to plain text search: find first non-whitespace char to the left in page text
                    m = etyk_re.search(page_text)
                    if m:
                        left_slice = page_text[:m.start()]
                        j = len(left_slice) - 1
                        while j >= 0 and left_slice[j].isspace():
                            j -= 1
                        if j >= 0:
                            snippet = page_text[j+1:m.end() + 200]
                        else:
                            snippet = page_text[m.end():m.end() + 200]
                        found_etykieta = snippet.strip()

                found_etykieta = found_etykieta or ""
            except Exception:
                found_etykieta = ""
            
            # Remove the wort 'Etykieta' from the beginning if present
            if found_etykieta.lower().startswith('etykieta'):
                found_etykieta = found_etykieta[len('etykieta'):].strip()
                
            # Find all new line indications in found_etykieta and replace them with spaces
            found_etykieta = re.sub(r'\s*\n\s*', ' ', found_etykieta)
            
            # Extract 'Wartości z etykietami' tables (only the block to the right/below of the phrase until the next horizontal line)
            for m in table_re.finditer(page_text):
                try:
                    blocks = page.get_text("blocks")

                    # Locate the phrase geometry via search_for (more reliable than blocks)
                    phrase_rects = page.search_for("Wartości z etykietami") + page.search_for("Wartosci z etykietami")
                    phrase_rect = phrase_rects[0] if phrase_rects else None
                    
                    # Fallback: locate the block that contains the phrase text
                    phrase_block = None
                    if not phrase_rect:
                        for b in blocks:
                            txt = b[4] if len(b) > 4 else ""
                            if table_re.search(txt):
                                phrase_block = b
                                break
                    
                    if phrase_rect:
                        phrase_x0, phrase_y0, phrase_x1, _ = phrase_rect
                    elif phrase_block:
                        phrase_x0, phrase_y0, phrase_x1, _ = phrase_block[0], phrase_block[1], phrase_block[2], phrase_block[3]
                    else:
                        phrase_x0, phrase_y0, phrase_x1 = 0, 0, 0

                    page_width = page.rect.width

                    # Detect a long horizontal rule to stop the table (prefer the first one below the phrase)
                    stop_y = None
                    try:
                        drawings = page.get_drawings()
                        for d in drawings:
                            rect = d.get('rect') if isinstance(d, dict) else None
                            if rect:
                                w = rect[2] - rect[0]
                                h = rect[3] - rect[1]
                                if w >= page_width * 0.7 and h <= 3 and rect[1] > phrase_y0:
                                    stop_y = rect[1]
                                    break
                            items = d.get('items') if isinstance(d, dict) else None
                            if items:
                                for it in items:
                                    if not it:
                                        continue
                                    if it[0] == 'l':
                                        _, (x0, y0, x1, y1) = it
                                        w = abs(x1 - x0)
                                        h = abs(y1 - y0)
                                        if w >= page_width * 0.7 and h <= 3 and y0 > phrase_y0:
                                            stop_y = min(y0, y1)
                                            break
                                if stop_y is not None:
                                    break
                    except Exception:
                        stop_y = None

                    if stop_y is None:
                        stop_y = page.rect.height  # conservative fallback

                    start_y = max(phrase_y0 - 2, 0)

                    # Collect only the blocks that are to the right of the phrase and between start_y and stop_y
                    region_blocks = []
                    for b in blocks:
                        bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
                        if by0 < start_y - 1 or by0 > stop_y + 1:
                            continue
                        # Keep a small tolerance so we don't miss slightly left-aligned numeric column
                        if bx0 < phrase_x1 - 5:
                            continue
                        # skip the phrase block itself
                        if phrase_block and b is phrase_block:
                            continue
                        region_blocks.append(b)

                    # Group blocks into rows (visual clustering)
                    rows_map = []  # list of (y_top, [blocks_on_row])
                    v_tol = 5.0
                    for b in sorted(region_blocks, key=lambda x: x[1]):
                        by0 = b[1]
                        placed = False
                        for entry in rows_map:
                            y_top = entry[0]
                            if abs(by0 - y_top) <= v_tol:
                                entry[1].append(b)
                                placed = True
                                break
                        if not placed:
                            rows_map.append([by0, [b]])

                    table_rows = []
                    for _, blks in rows_map:
                        sorted_cols = [t for _, t in sorted([(bb[0], bb) for bb in blks], key=lambda x: x[0])]
                        texts = [ (bb[4] if len(bb) > 4 else "").strip() for bb in sorted_cols ]
                        if len(texts) > 4:
                            texts = texts[:3] + [" ".join(texts[3:])]
                        while len(texts) < 4:
                            texts.append("")
                        texts = [re.sub(r'\s+',' ', t).strip() for t in texts]
                        if any(t for t in texts):
                            table_rows.append(texts)

                    # Fallback to plain text parsing limited to nearby snippet if no rows found
                    if not table_rows:
                        start = m.end()
                        snippet = page_text[start:start + 400]
                        lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
                        for ln in lines:
                            parts = re.split(r'\s{2,}', ln)
                            if len(parts) >= 3:
                                if len(parts) == 3:
                                    parts.append("")
                                if len(parts) > 4:
                                    parts = parts[:3] + [" ".join(parts[3:])]
                                parts = [re.sub(r'\s+',' ', p).strip() for p in parts]
                                table_rows.append(parts)

                    if table_rows:
                        found_table.append(table_rows)
                except Exception:
                    # don't let one failure stop the page processing; append an empty placeholder
                    found_table.append(())
            
            results.append((page_numer, found_q_string or "", found_etykieta or "", found_table))
            
        return results
        """