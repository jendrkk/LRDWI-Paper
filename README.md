# Long Run Dynamics of Wealth Inequality (LRDWI) - Paper Repository

This repository contains the complete set of Python applications, files, and Jupyter notebooks for conducting web scraping (downloading), data manipulation, and data analytics for an empirical research paper on the long-run dynamics of wealth inequality.

## 📋 Overview

This project provides a comprehensive framework for:
- **Web Scraping**: Downloading wealth inequality data from multiple international sources
- **Data Processing**: Cleaning, standardizing, and merging data from different sources
- **Data Analysis**: Statistical analysis and visualization of wealth inequality trends
- **Research Output**: Generating figures and tables for academic publication

## 📁 Repository Structure

```
LRDWI-Paper/
├── data/
│   ├── raw/              # Raw data downloaded from sources
│   └── processed/        # Cleaned and processed data
├── scripts/
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration settings
│   ├── data_scraper.py   # Web scraping module
│   └── data_processing.py # Data cleaning and processing
├── notebooks/
│   ├── 01_data_analysis.ipynb    # Main analysis notebook
│   └── 02_visualizations.ipynb   # Visualization notebook
├── output/
│   ├── figures/          # Generated plots and charts
│   └── tables/           # Summary statistics and tables
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jendrkk/LRDWI-Paper.git
cd LRDWI-Paper
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Step 1: Data Collection (Web Scraping)

Run the data scraper to download wealth inequality data from various sources:

```bash
python scripts/data_scraper.py
```

This script will:
- Download data from World Bank, OECD, and WID.world
- Save raw data to `data/raw/` directory
- Generate template data for development and testing

**Note**: The current implementation uses template data. To use real data, you'll need to:
- Obtain API keys from data providers
- Update the scraper with actual API endpoints
- Configure authentication in the config file

### Step 2: Data Processing

Process and clean the downloaded data:

```bash
python scripts/data_processing.py
```

This script will:
- Load raw data files
- Clean and standardize the data
- Handle missing values and outliers
- Merge data from multiple sources
- Save processed data to `data/processed/` directory

### Step 3: Data Analysis

Open and run the Jupyter notebooks for analysis:

```bash
jupyter notebook
```

Then open:
1. **`01_data_analysis.ipynb`**: Comprehensive analysis including:
   - Exploratory data analysis
   - Time series analysis
   - Cross-country comparisons
   - Statistical tests and regressions
   - Summary statistics

2. **`02_visualizations.ipynb`**: Create publication-quality figures:
   - Time series plots
   - Comparative visualizations
   - Distribution plots
   - Multi-panel figures for papers

## 📈 Data Sources

The project is designed to work with data from:

1. **World Bank**: Gini coefficients and poverty indicators
   - API: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

2. **OECD**: Wealth and income distribution data
   - Data Portal: https://data.oecd.org/

3. **World Inequality Database (WID.world)**: Historical wealth shares
   - API: https://wid.world/data/

## 📝 Configuration

Edit `scripts/config.py` to customize:
- Data source URLs
- Countries of interest
- Time period (start/end years)
- Output settings (DPI, format)
- Data processing parameters

## 🔍 Key Features

### Web Scraping Module (`data_scraper.py`)
- Modular scraper class for different data sources
- Error handling and retry logic
- Progress tracking with tqdm
- Configurable output directories

### Data Processing Module (`data_processing.py`)
- Automated data cleaning pipeline
- Missing value imputation
- Country name standardization
- Multi-source data merging
- Derived metric calculation

### Analysis Notebooks
- **Exploratory Analysis**: Comprehensive statistical summaries
- **Time Series Analysis**: Trend detection and forecasting
- **Cross-Country Comparison**: International comparisons
- **Correlation Analysis**: Relationship between metrics
- **Visualization**: Publication-ready figures

## 📊 Output

The analysis generates:

### Figures (in `output/figures/`)
- Time series plots of inequality metrics
- Cross-country comparisons
- Distribution plots
- Correlation heatmaps
- Multi-panel figures for papers

### Tables (in `output/tables/`)
- Summary statistics
- Regression results
- Country rankings

## 🛠️ Development

### Adding New Data Sources

1. Add source configuration to `scripts/config.py`
2. Create a new scraper method in `data_scraper.py`
3. Update the processing pipeline in `data_processing.py`
4. Add analysis code to the notebooks

### Customizing Analysis

- Modify the notebooks to add new analyses
- Update visualization styles in `02_visualizations.ipynb`
- Add custom metrics in `data_processing.py`

## 📚 Dependencies

Key dependencies include:
- **Data Collection**: requests, beautifulsoup4, selenium
- **Data Processing**: pandas, numpy, openpyxl
- **Analysis**: scipy, statsmodels
- **Visualization**: matplotlib, seaborn
- **Notebooks**: jupyter, ipykernel

See `requirements.txt` for complete list with versions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaborations, please open an issue in the repository.

## 🙏 Acknowledgments

- World Bank for providing open data APIs
- OECD for economic indicators
- WID.world for historical wealth inequality data
- All contributors and researchers in the field of inequality studies

## 📖 Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{lrdwi2025,
  title={Long Run Dynamics of Wealth Inequality},
  author={[Author Names]},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/jendrkk/LRDWI-Paper}}
}
```

---

**Status**: Active Development

Last Updated: December 2025