import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple


class InequalityAnalyzer:
    """
    A comprehensive analyzer for income inequality metrics from survey data.
    
    Expected DataFrame columns:
    - 'survey file': Survey identifier
    - 'survey year': Year of survey
    - 'survey month': Month of survey
    - 'income_hh': Household income
    - 'weight': Survey weights
    - 'household_size': Number of people in household
    """
    
    def __init__(self, df: pd.DataFrame, income_col: str = 'income_hh', 
                 weight_col: str = 'weight', hh_size_col: str = 'household_size',
                 year_col: str = 'survey year', month_col: str = 'survey month',
                 file_col: str = 'survey file'):
        """
        Initialize the analyzer with a dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The survey data
        income_col : str
            Column name for household income
        weight_col : str
            Column name for survey weights
        hh_size_col : str
            Column name for household size
        year_col : str
            Column name for survey year
        month_col : str
            Column name for survey month
        file_col : str
            Column name for survey file identifier
        """
        self.df = df.copy()
        self.income_col = income_col
        self.weight_col = weight_col
        self.hh_size_col = hh_size_col
        self.year_col = year_col
        self.month_col = month_col
        self.file_col = file_col
        
        # Results storage
        self.time_series = None
        self.gini_series = None
        self.palma_series = None
        self.income_groups = None
        
        # Automatically compute basic time series upon initialization
        self._compute_basic_timeseries()
    
    def _get_files_sorted(self):
        """Get sorted list of survey files."""
        files = set(self.df[self.file_col].values)
        return sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    def _clean_income_data(self, df_file):
        """Clean income data by removing invalid values."""
        income = pd.to_numeric(df_file[self.income_col], errors='coerce')
        weights = pd.to_numeric(df_file[self.weight_col], errors='coerce')
        hh_size = pd.to_numeric(df_file[self.hh_size_col], errors='coerce')
        year = df_file[self.year_col].values[0]
        
        '''
        # Remove values above threshold (depends on year)
        threshold = 9991 if year < 2001 else 99991
        mask = income <= threshold
        income = income[mask]
        weights = weights[mask]
        hh_size = hh_size[mask]
        
        # Remove negative values
        mask = income >= 0
        income = income[mask]
        weights = weights[mask]
        hh_size = hh_size[mask]
        '''
        
        # Remove NaN values
        mask = ~np.isnan(income)
        income = income[mask]
        weights = weights[mask]
        hh_size = hh_size[mask]
        
        return income, weights, hh_size
    
    def _compute_basic_timeseries(self):
        """Compute basic time series metrics (means, medians) for each survey."""
        files = self._get_files_sorted()
        
        results = {
            'date': [],
            'mean_hh': [],
            'mean_total': [],
            'weighted_mean_hh': [],
            'weighted_mean_total': [],
            'median_hh': [],
            'median_total': [],
            'weighted_median_hh': [],
            'weighted_median_total': []
        }
        
        for file in files:
            df_file = self.df[self.df[self.file_col] == file]
            year = df_file[self.year_col].values[0]
            month = df_file[self.month_col].values[0]
            date = pd.to_datetime(f"{year}-{month}-01")
            results['date'].append(date)
            
            income, weights, hh_size = self._clean_income_data(df_file)
            
            # Total income (household income * household size)
            income_total = income * hh_size
            mask_total = ~np.isnan(income_total)
            income_total = income_total[mask_total]
            weights_total = weights[mask_total]
            
            # Compute metrics
            results['mean_hh'].append(income.mean())
            results['mean_total'].append(income_total.mean())
            results['median_hh'].append(income.median())
            results['median_total'].append(income_total.median())
            
            # Weighted metrics
            if weights.sum() > 0:
                results['weighted_mean_hh'].append(np.average(income, weights=weights))
                results['weighted_median_hh'].append(self._weighted_quantile(income, weights, 0.5))
            else:
                results['weighted_mean_hh'].append(np.nan)
                results['weighted_median_hh'].append(np.nan)
            
            if weights_total.sum() > 0:
                results['weighted_mean_total'].append(np.average(income_total, weights=weights_total))
                results['weighted_median_total'].append(self._weighted_quantile(income_total, weights_total, 0.5))
            else:
                results['weighted_mean_total'].append(np.nan)
                results['weighted_median_total'].append(np.nan)
        
        self.time_series = pd.DataFrame(results)
        self.time_series.set_index('date', inplace=True)
        
    def _weighted_quantile(self, values, weights, quantile):
        """Compute weighted quantile."""
        if len(values) == 0 or weights.sum() == 0:
            return np.nan
        
        sorted_indices = np.argsort(values)
        sorted_values = values.iloc[sorted_indices].values if isinstance(values, pd.Series) else values[sorted_indices]
        sorted_weights = weights.iloc[sorted_indices].values if isinstance(weights, pd.Series) else weights[sorted_indices]
        
        cumsum = np.cumsum(sorted_weights)
        cutoff = quantile * cumsum[-1]
        
        return sorted_values[cumsum >= cutoff][0] if np.any(cumsum >= cutoff) else sorted_values[-1]
    
    @staticmethod
    def _gini_coefficient(x):
        """Calculate Gini coefficient."""
        n = len(x)
        if n == 0:
            return np.nan
        cumulative_x = np.cumsum(np.sort(x))
        sum_x = cumulative_x[-1]
        if sum_x == 0:
            return 0.0
        gini = (n + 1 - 2 * np.sum(cumulative_x) / sum_x) / n
        return gini
    
    @staticmethod
    def _palma_ratio(x):
        """Calculate Palma ratio (top 10% / bottom 40%)."""
        n = len(x)
        if n == 0:
            return np.nan
        sorted_x = np.sort(x)
        bottom_40 = np.sum(sorted_x[:int(0.4 * n)])
        top_10 = np.sum(sorted_x[int(0.9 * n):])
        if bottom_40 == 0:
            return np.nan
        return top_10 / bottom_40
    
    def compute_gini(self, use_total_income: bool = False):
        """
        Compute Gini coefficient for each survey period.
        
        Parameters:
        -----------
        use_total_income : bool
            If True, use total household income (income * household_size)
            If False, use household income directly
            
        Returns:
        --------
        pd.Series : Gini coefficients indexed by date
        """
        files = self._get_files_sorted()
        
        dates = []
        ginis = []
        
        for file in files:
            df_file = self.df[self.df[self.file_col] == file]
            year = df_file[self.year_col].values[0]
            month = df_file[self.month_col].values[0]
            date = pd.to_datetime(f"{year}-{month}-01")
            dates.append(date)
            
            income, weights, hh_size = self._clean_income_data(df_file)
            
            if use_total_income:
                income = income * hh_size
                income = income[~np.isnan(income)]
            
            gini = self._gini_coefficient(income.values)
            ginis.append(gini if gini > 0 else np.nan)
        
        self.gini_series = pd.Series(ginis, index=dates)
        return self.gini_series
    
    def compute_palma(self, use_total_income: bool = False):
        """
        Compute Palma ratio for each survey period.
        
        Parameters:
        -----------
        use_total_income : bool
            If True, use total household income (income * household_size)
            If False, use household income directly
            
        Returns:
        --------
        pd.Series : Palma ratios indexed by date
        """
        files = self._get_files_sorted()
        
        dates = []
        palmas = []
        
        for file in files:
            df_file = self.df[self.df[self.file_col] == file]
            year = df_file[self.year_col].values[0]
            month = df_file[self.month_col].values[0]
            date = pd.to_datetime(f"{year}-{month}-01")
            dates.append(date)
            
            income, weights, hh_size = self._clean_income_data(df_file)
            
            if use_total_income:
                income = income * hh_size
                income = income[~np.isnan(income)]
            
            palma = self._palma_ratio(income.values)
            palmas.append(palma)
        
        self.palma_series = pd.Series(palmas, index=dates)
        return self.palma_series
    
    def compute_income_groups(self, groups: Optional[Dict[str, Tuple[float, float]]] = None,
                            weighted: bool = True, use_total_income: bool = True):
        """
        Compute mean income for different income groups.
        
        Parameters:
        -----------
        groups : dict, optional
            Dictionary mapping group names to (lower_percentile, upper_percentile) tuples.
            Default: {'bottom_50': (0, 0.5), 'middle_40': (0.5, 0.9), 
                     'top_10': (0.9, 1.0), 'top_1': (0.99, 1.0)}
        weighted : bool
            If True, use weighted income values
        use_total_income : bool
            If True, use total household income (income * household_size)
            
        Returns:
        --------
        pd.DataFrame : Mean incomes for each group indexed by date
        """
        if groups is None:
            groups = {
                'bottom_50': (0, 0.5),
                'middle_40': (0.5, 0.9),
                'top_10': (0.9, 1.0),
                'top_1': (0.99, 1.0)
            }
        
        files = self._get_files_sorted()
        
        results = {'date': []}
        for group_name in groups.keys():
            results[group_name] = []
        
        for file in files:
            df_file = self.df[self.df[self.file_col] == file]
            year = df_file[self.year_col].values[0]
            month = df_file[self.month_col].values[0]
            date = pd.to_datetime(f"{year}-{month}-01")
            results['date'].append(date)
            
            income, weights, hh_size = self._clean_income_data(df_file)
            
            if use_total_income:
                income = income * hh_size
                mask = ~np.isnan(income)
                income = income[mask]
                weights = weights[mask]
            
            # Apply weights if requested
            if weighted:
                weighted_income = income * weights
            else:
                weighted_income = income
            
            # Remove NaN values
            weighted_income = weighted_income[~np.isnan(weighted_income)]
            
            # Sort income
            sorted_income = np.sort(weighted_income)
            
            # Compute mean for each group
            for group_name, (lower, upper) in groups.items():
                lower_idx = int(lower * len(sorted_income))
                upper_idx = int(upper * len(sorted_income))
                if upper_idx == lower_idx:
                    upper_idx = len(sorted_income)
                
                group_income = sorted_income[lower_idx:upper_idx]
                results[group_name].append(group_income.mean() if len(group_income) > 0 else np.nan)
        
        self.income_groups = pd.DataFrame(results)
        self.income_groups.set_index('date', inplace=True)
        
        return self.income_groups
    
    def compute_percentile_ratio(self, top_pct: float = 0.1, bottom_pct: float = 0.5,
                                use_total_income: bool = False):
        """
        Compute ratio between top and bottom percentiles.
        
        Parameters:
        -----------
        top_pct : float
            Top percentile (e.g., 0.1 for top 10%)
        bottom_pct : float
            Bottom percentile (e.g., 0.5 for bottom 50%)
        use_total_income : bool
            If True, use total household income
            
        Returns:
        --------
        pd.Series : Ratios indexed by date
        """
        files = self._get_files_sorted()
        
        dates = []
        ratios = []
        
        for file in files:
            df_file = self.df[self.df[self.file_col] == file]
            year = df_file[self.year_col].values[0]
            month = df_file[self.month_col].values[0]
            date = pd.to_datetime(f"{year}-{month}-01")
            dates.append(date)
            
            income, weights, hh_size = self._clean_income_data(df_file)
            
            if use_total_income:
                income = income * hh_size
                income = income[~np.isnan(income)]
            
            sorted_income = np.sort(income)
            
            bottom_mean = sorted_income[:int(bottom_pct * len(sorted_income))].mean()
            top_mean = sorted_income[int((1 - top_pct) * len(sorted_income)):].mean()
            
            ratio = top_mean / bottom_mean if bottom_mean > 0 else np.nan
            ratios.append(ratio)
        
        return pd.Series(ratios, index=dates)
    
    def resample_to_annual(self, metric: str = 'all', method: str = 'mean'):
        """
        Resample time series data to annual frequency.
        
        Parameters:
        -----------
        metric : str
            Which metric to resample: 'all', 'time_series', 'gini', 'palma', 'income_groups'
        method : str
            Aggregation method: 'mean', 'median', 'first', 'last'
            
        Returns:
        --------
        dict or pd.DataFrame : Resampled data
        """
        results = {}
        
        if metric in ['all', 'time_series'] and self.time_series is not None:
            results['time_series'] = self._resample_df(self.time_series, method)
        
        if metric in ['all', 'gini'] and self.gini_series is not None:
            results['gini'] = self._resample_series(self.gini_series, method)
        
        if metric in ['all', 'palma'] and self.palma_series is not None:
            results['palma'] = self._resample_series(self.palma_series, method)
        
        if metric in ['all', 'income_groups'] and self.income_groups is not None:
            results['income_groups'] = self._resample_df(self.income_groups, method)
        
        return results if metric == 'all' else results.get(metric)
    
    def _resample_series(self, series, method):
        """Resample a pandas Series to annual frequency."""
        if method == 'mean':
            return series.resample('Y').mean()
        elif method == 'median':
            return series.resample('Y').median()
        elif method == 'first':
            return series.resample('Y').first()
        elif method == 'last':
            return series.resample('Y').last()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _resample_df(self, df, method):
        """Resample a pandas DataFrame to annual frequency."""
        if method == 'mean':
            return df.resample('Y').mean()
        elif method == 'median':
            return df.resample('Y').median()
        elif method == 'first':
            return df.resample('Y').first()
        elif method == 'last':
            return df.resample('Y').last()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_summary(self):
        """
        Get a summary of all computed metrics.
        
        Returns:
        --------
        dict : Summary of available metrics
        """
        summary = {
            'time_series_available': self.time_series is not None,
            'gini_available': self.gini_series is not None,
            'palma_available': self.palma_series is not None,
            'income_groups_available': self.income_groups is not None,
            'n_observations': len(self.df),
            'n_survey_files': len(self._get_files_sorted()),
            'date_range': (self.time_series.index.min(), self.time_series.index.max()) if self.time_series is not None else None
        }
        return summary
        