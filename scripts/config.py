"""
Configuration file for LRDWI Paper project.
"""

# Data source URLs (examples - replace with actual URLs)
DATA_SOURCES = {
    'world_bank': 'https://api.worldbank.org/v2/country/all/indicator/',
    'oecd': 'https://stats.oecd.org/restsdmx/sdmx.ashx/GetData/',
    'wid_world': 'https://wid.world/api/v1/'
}

# Data indicators
INDICATORS = {
    'world_bank': {
        'gini_index': 'SI.POV.GINI',
        'poverty_ratio': 'SI.POV.DDAY'
    },
    'oecd': {
        'wealth_inequality': 'IDD',
        'income_inequality': 'INC_DISP'
    }
}

# Countries of interest (ISO3 codes)
COUNTRIES = [
    'USA',  # United States
    'GBR',  # United Kingdom
    'DEU',  # Germany
    'FRA',  # France
    'JPN',  # Japan
    'CHN',  # China
    'IND',  # India
    'BRA',  # Brazil
    'ZAF',  # South Africa
]

# Time period
START_YEAR = 1980
END_YEAR = 2023

# Output settings
OUTPUT_FIGURE_DPI = 300
OUTPUT_FIGURE_FORMAT = 'png'

# Data processing settings
MISSING_VALUE_THRESHOLD = 0.5  # Maximum proportion of missing values allowed
OUTLIER_THRESHOLD = 3  # Number of standard deviations for outlier detection
