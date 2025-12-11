"""
Docstring for Code.CBOS_downloader
"""

import pandas as pd
import requests, re, html, os
import tempfile
from lxml import html as LH
from lxml import etree
from urllib.parse import urljoin, urlparse, unquote
import shutil


def download_cbos_data(search_id: int):
    """
    Downloads data from the CBOS database based on the provided ID.
    
    Parameters:
    id (int): The identifier for the specific dataset to download.
    
    Returns:
    pd.DataFrame: A DataFrame containing the downloaded data.
    """
    
    url = f"https://rds.icm.edu.pl/dataverse/CBOS/?q=Aktualne+problemy+i+wydarzenia+%28{search_id}%29"
    xpath_query = '//*[@id="resultsTable"]/tbody/tr[1]/td/div/div[2]/div[1]/a/@href'
    
    # Extract the html code from the url under a given xpath:
    resp = requests.get(url)
    tree = LH.fromstring(resp.content)
    url_ending = tree.xpath(xpath_query)[0]
    
    page_url = "https://rds.icm.edu.pl" + url_ending
    #xpath_query_csv = '//*[@id="datasetForm:tabView:filesTable:1:j_idt644:tabularOriginalDownloadPopupButton"]'
    
    return page_url



def main():
    
    page = download_cbos_data(1)
    comp = "datasetForm:tabView:filesTable:1:j_idt644:tabularOriginalDownloadPopupButton"
    try:
        res = robust_primefaces_download(page, comp, outfile="data.dta")
        print("Downloaded to", res)
    except Exception as e:
        print("ERROR:", e)
        # ajax response saved as ajax_response.html for inspection
    
    
if __name__ == "__main__":
    main()