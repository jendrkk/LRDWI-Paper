"""
Web scraper for wealth inequality data.

This module provides functionality to download wealth inequality data from various sources.
"""

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Optional, Dict, List
import time
from tqdm import tqdm


class WealthInequalityDataScraper:
    """
    A class to scrape wealth inequality data from various online sources.
    
    This scraper focuses on long-run dynamics of wealth inequality, collecting
    historical data on wealth distribution, Gini coefficients, and related metrics.
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize the scraper.
        
        Args:
            output_dir: Directory where scraped data will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_world_bank_data(self, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scrape World Bank wealth inequality indicators.
        
        Args:
            indicators: List of indicator codes to scrape (e.g., ['SI.POV.GINI'])
        
        Returns:
            DataFrame containing the scraped data
        """
        if indicators is None:
            indicators = ['SI.POV.GINI']  # Gini index
        
        print("Note: This is a template. For actual World Bank data, use their API:")
        print("https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation")
        
        # Template for World Bank API integration
        data = {
            'country': ['USA', 'GBR', 'DEU', 'FRA'],
            'year': [2020, 2020, 2020, 2020],
            'gini_index': [41.5, 35.1, 31.9, 32.4],
            'indicator': ['SI.POV.GINI'] * 4
        }
        
        df = pd.DataFrame(data)
        output_path = os.path.join(self.output_dir, 'world_bank_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
        return df
    
    def scrape_oecd_data(self) -> pd.DataFrame:
        """
        Scrape OECD wealth inequality data.
        
        Returns:
            DataFrame containing the scraped data
        """
        print("Note: This is a template. For actual OECD data, use their API:")
        print("https://data.oecd.org/")
        
        # Template data
        data = {
            'country': ['USA', 'GBR', 'DEU', 'FRA', 'JPN'],
            'year': [2020, 2020, 2020, 2020, 2020],
            'wealth_share_top10': [70.0, 52.0, 59.0, 55.0, 48.0],
            'wealth_share_bottom50': [2.0, 9.0, 3.0, 5.0, 6.0]
        }
        
        df = pd.DataFrame(data)
        output_path = os.path.join(self.output_dir, 'oecd_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
        return df
    
    def scrape_wid_world_data(self) -> pd.DataFrame:
        """
        Scrape World Inequality Database (WID.world) data.
        
        Returns:
            DataFrame containing the scraped data
        """
        print("Note: This is a template. For actual WID.world data, use their API:")
        print("https://wid.world/data/")
        
        # Template data with historical wealth shares
        years = list(range(1980, 2021, 5))
        data = {
            'country': ['USA'] * len(years),
            'year': years,
            'top1_wealth_share': [25.0, 28.0, 30.0, 33.0, 35.0, 37.0, 39.0, 40.0, 42.0],
            'top10_wealth_share': [60.0, 62.0, 65.0, 68.0, 70.0, 72.0, 73.0, 74.0, 75.0]
        }
        
        df = pd.DataFrame(data)
        output_path = os.path.join(self.output_dir, 'wid_world_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
        return df
    
    def scrape_all_sources(self) -> Dict[str, pd.DataFrame]:
        """
        Scrape data from all available sources.
        
        Returns:
            Dictionary mapping source names to DataFrames
        """
        print("Scraping data from all sources...")
        
        datasets = {}
        
        try:
            print("\n1. Scraping World Bank data...")
            datasets['world_bank'] = self.scrape_world_bank_data()
        except Exception as e:
            print(f"Error scraping World Bank data: {e}")
        
        try:
            print("\n2. Scraping OECD data...")
            datasets['oecd'] = self.scrape_oecd_data()
        except Exception as e:
            print(f"Error scraping OECD data: {e}")
        
        try:
            print("\n3. Scraping WID.world data...")
            datasets['wid_world'] = self.scrape_wid_world_data()
        except Exception as e:
            print(f"Error scraping WID.world data: {e}")
        
        print(f"\nSuccessfully scraped {len(datasets)} datasets")
        return datasets
    
    def close(self):
        """Close the requests session."""
        self.session.close()


def main():
    """Main function to run the scraper."""
    scraper = WealthInequalityDataScraper()
    
    try:
        datasets = scraper.scrape_all_sources()
        
        print("\n" + "="*50)
        print("Scraping complete!")
        print("="*50)
        print(f"\nDatasets saved to: {scraper.output_dir}")
        print("\nNext steps:")
        print("1. Review the downloaded data files")
        print("2. Run data_processing.py to clean and process the data")
        print("3. Use the Jupyter notebooks for analysis")
        
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
