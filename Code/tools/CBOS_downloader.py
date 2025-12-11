"""
Docstring for Code.CBOS_downloader
"""

import pandas as pd
import requests
import re
from datetime import date
from dateutil.relativedelta import relativedelta
from lxml import html, etree
import unicodedata
from difflib import SequenceMatcher

START = date(1990, 1, 1)
MONTHS_DICT = {
    "Styczeń": "01",
    "Luty": "02",
    "Marzec": "03",
    "Kwiecień": "04",
    "Maj": "05",
    "Czerwiec": "06",
    "Lipiec": "07",
    "Sierpień": "08",
    "Wrzesień": "09",
    "Październik": "10",
    "Listopad": "11",
    "Grudzień": "12"
}


def CBOS_API_retriver(search_id: int):
    """
    Retive API link for a file of given id from RDS Dataverse.
    
    Parameters:
    id (int): The identifier for the specific dataset to download.
    
    Returns:
    pd.DataFrame: A DataFrame containing the downloaded data.
    """
    
    url = f"https://rds.icm.edu.pl/dataverse/CBOS/?q=Aktualne+problemy+i+wydarzenia+%28{search_id}%29"
    xpath_query = '//*[@id="resultsTable"]/tbody/tr[1]/td/div/div[2]/div[1]/a/@href'
    
    # Extract the html code from the url under a given xpath:
    resp = requests.get(url)
    tree = html.fromstring(resp.content)
    try:
        url_ending = tree.xpath(xpath_query)[0]
    except IndexError:
        print(f"No dataset found for id {search_id}.")
        return None
        
    
    dataset_url = "https://rds.icm.edu.pl" + url_ending
    
    # Extract the text under a given xpath:
    resp = requests.get(dataset_url, timeout=15)
    
    html_script = resp.text
    tag = find_tags_with_phrase_lxml(html_script, "api/access/datafile/")
    
    if len(tag) == 0:
        raise ValueError("No download link found.")
    
    if len(tag) == 1:
        tag = tag[0]
    else:
        raise ValueError("Multiple download links found.")
    # Extract ALL download links from the tag using regex (avoids manual index updates and infinite loops)
    tag_str = str(tag)
    download_links = []
    files_to_download = []
    #print(tag_str)

    # find all api links including following characters up to the next quote
    for m in re.finditer(r'api/access/datafile/\d+(?:[^"]*)', tag_str):
        start = m.start()
        # find end quote after the match to capture the complete path
        end_quote = tag_str.find('"', m.end())
        if end_quote == -1:
            continue
        download_link = "https://rds.icm.edu.pl/" + tag_str[start:end_quote]
        download_links.append(download_link)

    # find all file names from DataDownload entries
    for m in re.finditer(r'DataDownload","name":"([^"]+)"', tag_str):
        files_to_download.append(m.group(1))

    # ensure lists are same length by trimming extra unmatched entries
    if len(download_links) != len(files_to_download):
        min_len = min(len(download_links), len(files_to_download))
        download_links = download_links[:min_len]
        files_to_download = files_to_download[:min_len]
    
    # Find also the first phrase "Aktualne problemy i wydarzenia" in html code and 
    # print the rest of the string after it till the next quote
    phrase = "Aktualne problemy i wydarzenia ("
    start_index = tag_str.find(phrase)
    if start_index == -1:
        # case-insensitive direct match
        m = re.search(re.escape(phrase), tag_str, flags=re.IGNORECASE)
        if m:
            start_index = m.start()

    if start_index == -1:
        # fuzzy matching: try to find a substring similar to the phrase (ignoring accents/case)

        phrase_core = "Aktualne problemy i wydarzenia"
        # token spans to be able to recover original indices
        tokens_spans = [(mo.group(0), mo.span()) for mo in re.finditer(r'\S+', tag_str)]

        def _norm(s: str) -> str:
            s = unicodedata.normalize('NFD', s)
            return "".join(ch for ch in s if unicodedata.category(ch) != 'Mn').casefold()

        best_score = 0.0
        best_i = best_j = None
        # examine windows around the phrase length
        base_words = phrase_core.split()
        min_w = max(1, len(base_words) - 1)
        max_w = len(base_words) + 2
        for w in range(min_w, max_w + 1):
            for i in range(0, len(tokens_spans) - w + 1):
                j = i + w
                candidate = " ".join(t[0] for t in tokens_spans[i:j])
                score = SequenceMatcher(None, _norm(candidate), _norm(phrase_core)).ratio()
                if score > best_score:
                    best_score = score
                    best_i, best_j = i, j

        # accept a fuzzy match if it's reasonably close
        if best_score >= 0.65 and best_i is not None:
            start_index = tokens_spans[best_i][1][0]

    if start_index != -1:
        # find the next quote after the start to extract the dataset name (same logic as original)
        end_index = tag_str.find('"', start_index + len(phrase))
        if end_index == -1:
            # fallback: find next quote after the found start index
            end_index = tag_str.find('"', start_index + 1)
        if end_index != -1:
            dataset_name = tag_str[start_index:end_index]

    if 'dataset_name' not in locals():
        dataset_name = None
    
    print(dataset_name)
    output = {"dataset_name": dataset_name,
              "files_to_download": files_to_download,
              "download_links": download_links}
    
    
    return output

def find_tags_with_phrase_lxml(html_text: str, phrase: str):
    doc = html.fromstring(html_text)
    results = []
    # find text nodes that contain phrase
    for node in doc.xpath("//*[contains(text(), $p)]", p=phrase):
        results.append(etree.tostring(node, encoding=str, with_tail=False))
    # also search attributes
    for node in doc.xpath("//*"):
        for attr_val in node.attrib.values():
            if phrase in attr_val:
                results.append(etree.tostring(node, encoding=str, with_tail=False))
                break
    # unique
    seen = set(); uniq = []
    for r in results:
        if r not in seen:
            uniq.append(r); seen.add(r)
    return uniq

def choose_download_link(links: list, filenames: list, file_identifier: str):
    """
    Choose the download link corresponding to the desired filename.
    
    Parameters:
    links (list): List of download links.
    filenames (list): List of filenames corresponding to the links.
    desired_filename (str): The filename to search for.
    
    Returns:
    str: The download link for the desired filename.
    """
    
    for link, filename in zip(links, filenames):
        if file_identifier in filename:
            return link
    raise ValueError(f"Filename {file_identifier} not found in the list.")

def download_file(url: str, output_path: str):
    """
    Download a file from a given URL and save it to the specified output path.
    
    Parameters:
    url (str): The URL of the file to download.
    output_path (str): The path where the downloaded file will be saved.
    """
    
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses

    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def facilitator(id: int, file_type: str, output_path: str):
    try:
        # Retrieve dataset information
        out = CBOS_API_retriver(id)
        print("------------------------------------------------------")
        if out is None:
            print(f"Skipping id {id} due to missing dataset.")
            return 
        # Calculate the date corresponding to the id
        current_date = START + relativedelta(months=int(id-1))
        current_date = current_date.strftime("%m_%Y")
        
        # Determine file extension based on file type
        file_extension = "pdf" if file_type == "pdf" else "tab"
        
        # Validate extracted id from dataset name 
        name = out.get("dataset_name", "Unknown Dataset")
        id_from_name = re.search(r'\((\d+)\)', name)
        if id_from_name:
            id_extracted = int(id_from_name.group(1))
            if id_extracted != id:
                print(f"Warning: Extracted id {id_extracted} does not match provided id {id}.")
        # Validate date from dataset name
        # Date from the name is after ". " and is written in words like "styczeń 1990"
        try:
            date_from_name = re.search(r'\. ([a-ząćęłńóśźż]+) (\d{4})', name, re.IGNORECASE).group(0).strip(". ")
        except AttributeError:
            print(f"Warning: Date not found in dataset name {name}.")
        # Convert month name to number
        month_name, year = date_from_name.split(" ")
        month_number = MONTHS_DICT.get(month_name.capitalize())
        if month_number is None:
            print(f"Warning: Month name {month_name} not recognized.")
        date_extracted = f"{month_number}_{year}"
        if date_extracted != current_date:
            print(f"Warning: Extracted date {date_extracted} does not match expected date {current_date}.")
            # We use the extracted date for naming
            date_of_data = date_extracted
        else:
            date_of_data = current_date
        # Choose the appropriate download link
        link = choose_download_link(out["download_links"], out["files_to_download"], file_type)
        # Download the file
        download_file(link, output_path + f"CBOS_{id}_{date_of_data}.{file_extension}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    print(f"File CBOS_{id}_{date_of_data}.{file_extension} downloaded successfully to {output_path}.")
    print("------------------------------------------------------")
    print("\n")
    

def main():
    PATH_TAB = "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/CBOS numerical/"
    PATH_PDF = "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/CBOS pdf/"
    for id in range(22, 340):  
        print(f"Downloading files for id {id}...")
        facilitator(id, "STATA", PATH_TAB)
        facilitator(id, "pdf", PATH_PDF)
    
if __name__ == "__main__":
    main()