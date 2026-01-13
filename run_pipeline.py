#!/usr/bin/env python
"""
Run the complete LRDWI data pipeline.

This script executes the full workflow:
1. Data scraping
2. Data processing
3. (Notebooks should be run manually in Jupyter)
"""

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts import data_scraper, data_processing


def main():
    """Run the complete pipeline."""
    print("="*70)
    print("LRDWI Paper - Complete Data Pipeline")
    print("="*70)
    print()
    
    # Step 1: Data Scraping
    print("STEP 1: Data Scraping")
    print("-"*70)
    try:
        scraper = data_scraper.WealthInequalityDataScraper()
        datasets = scraper.scrape_all_sources()
        scraper.close()
        print("\n✓ Data scraping completed successfully\n")
    except Exception as e:
        print(f"\n✗ Error during data scraping: {e}\n")
        return 1
    
    # Step 2: Data Processing
    print("STEP 2: Data Processing")
    print("-"*70)
    try:
        processor = data_processing.WealthDataProcessor()
        processor.process_all_data()
        print("\n✓ Data processing completed successfully\n")
    except Exception as e:
        print(f"\n✗ Error during data processing: {e}\n")
        return 1
    
    # Step 3: Instructions for analysis
    print("STEP 3: Data Analysis")
    print("-"*70)
    print("To perform data analysis and create visualizations:")
    print()
    print("1. Start Jupyter:")
    print("   $ jupyter notebook")
    print()
    print("2. Open and run the notebooks:")
    print("   - notebooks/01_data_analysis.ipynb")
    print("   - notebooks/02_visualizations.ipynb")
    print()
    
    print("="*70)
    print("Pipeline completed successfully!")
    print("="*70)
    print()
    print("Output locations:")
    print(f"  • Raw data: data/raw/")
    print(f"  • Processed data: data/processed/")
    print(f"  • Figures will be saved to: output/figures/")
    print(f"  • Tables will be saved to: output/tables/")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
