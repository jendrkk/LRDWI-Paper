"""
Data processing and cleaning for wealth inequality data.

This module provides functionality to clean, process, and transform raw data
for analysis.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class WealthDataProcessor:
    """
    A class to process and clean wealth inequality data.
    """
    
    def __init__(self, raw_data_dir: str = "data/raw", 
                 processed_data_dir: str = "data/processed"):
        """
        Initialize the data processor.
        
        Args:
            raw_data_dir: Directory containing raw data files
            processed_data_dir: Directory where processed data will be saved
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(processed_data_dir, exist_ok=True)
    
    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """
        Load a raw data file.
        
        Args:
            filename: Name of the file to load
        
        Returns:
            DataFrame containing the raw data
        """
        filepath = os.path.join(self.raw_data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        print(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values
        print(f"Missing values before cleaning:\n{df.isnull().sum()}")
        
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        return df
    
    def standardize_country_names(self, df: pd.DataFrame, 
                                   country_col: str = 'country') -> pd.DataFrame:
        """
        Standardize country names across datasets.
        
        Args:
            df: Input DataFrame
            country_col: Name of the country column
        
        Returns:
            DataFrame with standardized country names
        """
        country_mapping = {
            'US': 'USA',
            'United States': 'USA',
            'UK': 'GBR',
            'United Kingdom': 'GBR',
            'Britain': 'GBR',
            'Germany': 'DEU',
            'France': 'FRA',
            'Japan': 'JPN'
        }
        
        if country_col in df.columns:
            df[country_col] = df[country_col].replace(country_mapping)
        
        return df
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple datasets on common keys.
        
        Args:
            datasets: Dictionary of DataFrames to merge
        
        Returns:
            Merged DataFrame
        """
        print("Merging datasets...")
        
        if not datasets:
            raise ValueError("No datasets provided for merging")
        
        # Start with the first dataset
        merged_df = list(datasets.values())[0].copy()
        
        # Merge subsequent datasets
        for name, df in list(datasets.items())[1:]:
            # Find common columns
            common_cols = list(set(merged_df.columns) & set(df.columns))
            
            if len(common_cols) >= 2:  # At least country and year
                print(f"Merging {name} on {common_cols}")
                merged_df = pd.merge(merged_df, df, on=common_cols, how='outer')
            else:
                print(f"Skipping {name}: insufficient common columns")
        
        print(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived metrics from the data.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with additional calculated metrics
        """
        print("Calculating derived metrics...")
        
        # Calculate wealth concentration ratio (top 10% / bottom 50%)
        if 'wealth_share_top10' in df.columns and 'wealth_share_bottom50' in df.columns:
            df['wealth_concentration_ratio'] = (
                df['wealth_share_top10'] / df['wealth_share_bottom50'].replace(0, np.nan)
            )
        
        # Calculate inequality trend (if year is present)
        if 'year' in df.columns and 'country' in df.columns:
            # Sort by country and year
            df = df.sort_values(['country', 'year'])
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = os.path.join(self.processed_data_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    
    def process_all_data(self):
        """
        Process all available raw data files.
        """
        print("="*50)
        print("Starting data processing pipeline")
        print("="*50)
        
        datasets = {}
        
        # Load all CSV files from raw data directory
        if os.path.exists(self.raw_data_dir):
            for filename in os.listdir(self.raw_data_dir):
                if filename.endswith('.csv'):
                    try:
                        df = self.load_raw_data(filename)
                        df = self.clean_data(df)
                        df = self.standardize_country_names(df)
                        datasets[filename.replace('.csv', '')] = df
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        
        if datasets:
            # Save individual processed datasets
            for name, df in datasets.items():
                self.save_processed_data(df, f"{name}_processed.csv")
            
            # Attempt to merge datasets if multiple exist
            if len(datasets) > 1:
                try:
                    merged_df = self.merge_datasets(datasets)
                    merged_df = self.calculate_derived_metrics(merged_df)
                    self.save_processed_data(merged_df, "merged_data.csv")
                except Exception as e:
                    print(f"Error merging datasets: {e}")
        
        print("\n" + "="*50)
        print("Data processing complete!")
        print("="*50)


def main():
    """Main function to run the data processor."""
    processor = WealthDataProcessor()
    processor.process_all_data()
    
    print("\nNext steps:")
    print("1. Review the processed data in data/processed/")
    print("2. Open the Jupyter notebooks for analysis")


if __name__ == "__main__":
    main()
