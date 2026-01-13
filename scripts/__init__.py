"""
LRDWI Paper - Scripts Package

This package contains scripts for web scraping, data processing, and analysis
for the Long Run Dynamics of Wealth Inequality paper.
"""

__version__ = '0.1.0'
__author__ = 'LRDWI Research Team'

from . import config
from . import data_scraper
from . import data_processing

__all__ = ['config', 'data_scraper', 'data_processing']
