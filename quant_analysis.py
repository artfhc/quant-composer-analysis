"""
Quantitative analysis utilities using quantstats library.
"""

import inspect
import logging
from typing import Dict, Any, Optional, Set

import pandas as pd
import quantstats as qs

# Configure logging
logger = logging.getLogger(__name__)


class QuantAnalysisError(Exception):
    """Custom exception for quantitative analysis errors."""
    pass


# Constants for QuantStats functions to skip
SKIPPED_FUNCTIONS: Set[str] = {
    'comp', 'compsum', 'distribution', 'implied_volatility', 
    'monthly_returns', 'to_drawdown_series', 'outliers', 
    'remove_outliers', 'rolling_sharpe', 'rolling_sortino',
    'rolling_volatility', 'pct_rank'
}


def get_portfolio_values(df: pd.DataFrame) -> pd.Series:
    """
    Extract portfolio values from returns DataFrame.
    
    Args:
        df: DataFrame with portfolio returns
        
    Returns:
        Series of portfolio values
        
    Raises:
        QuantAnalysisError: If DataFrame is invalid or empty
    """
    if not isinstance(df, pd.DataFrame):
        raise QuantAnalysisError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise QuantAnalysisError("DataFrame is empty")
    
    if df.shape[1] == 0:
        raise QuantAnalysisError("DataFrame has no columns")
    
    try:
        # Filter out rows where all values are zero (non-trading days)
        filtered_df = df.loc[(df != 0).any(axis=1)]
        
        if filtered_df.empty:
            logger.warning("All rows contain only zeros, returning original first column")
            return df.iloc[:, 0]
        
        result = filtered_df.iloc[:, 0]
        logger.debug(f"Extracted portfolio values: {len(result)} data points")
        return result
        
    except Exception as e:
        raise QuantAnalysisError(f"Failed to extract portfolio values: {e}") from e


def calculate_quantstats_metrics(dict_return: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate quantstats metrics for all portfolios.
    
    Args:
        dict_return: Dictionary mapping symphony ID to returns DataFrame
        
    Returns:
        Dictionary mapping symphony ID to calculated metrics
        
    Raises:
        QuantAnalysisError: If input data is invalid
    """
    if not isinstance(dict_return, dict):
        raise QuantAnalysisError("dict_return must be a dictionary")
    
    if not dict_return:
        logger.warning("Empty dictionary provided")
        return {}
    
    try:
        # Extend pandas with quantstats functionality
        qs.extend_pandas()
        
        # Get available quantstats functions
        function_names = [
            f for f in dir(qs.stats) 
            if not f.startswith('_') and f not in SKIPPED_FUNCTIONS
        ]
        
        logger.info(f"Calculating {len(function_names)} metrics for {len(dict_return)} portfolios")
        
        dict_quant_stats = {}
        successful_calculations = 0
        failed_calculations = 0
        
        for symphony_id, returns_df in dict_return.items():
            if not isinstance(returns_df, pd.DataFrame):
                logger.warning(f"Skipping {symphony_id}: not a DataFrame")
                continue
                
            try:
                portfolio_values = get_portfolio_values(returns_df)
                temp_dict = {}
                
                for function_name in function_names:
                    try:
                        func = getattr(qs.stats, function_name)
                        args = set(inspect.getfullargspec(func).args)
                        
                        # Only calculate metrics that take returns/prices and don't need benchmark
                        if ('returns' in args or 'prices' in args) and 'benchmark' not in args:
                            result = func(portfolio_values)
                            temp_dict[function_name] = result
                            successful_calculations += 1
                            
                    except Exception as e:
                        logger.debug(f"Failed to calculate {function_name} for {symphony_id}: {e}")
                        temp_dict[function_name] = None
                        failed_calculations += 1
                
                dict_quant_stats[symphony_id] = temp_dict
                
            except QuantAnalysisError as e:
                logger.warning(f"Skipping {symphony_id}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {symphony_id}: {e}")
                continue
        
        logger.info(f"Completed: {successful_calculations} successful, {failed_calculations} failed calculations")
        return dict_quant_stats
        
    except Exception as e:
        raise QuantAnalysisError(f"Failed to calculate quantstats metrics: {e}") from e


def calculate_oos_stats(
    dict_return: Dict[str, pd.DataFrame], 
    sid_to_oos: Dict[str, pd.Timestamp]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate out-of-sample (OOS) statistics for portfolios.
    
    Args:
        dict_return: Dictionary mapping symphony ID to returns DataFrame
        sid_to_oos: Dictionary mapping symphony ID to OOS start date
        
    Returns:
        Dictionary mapping symphony ID to OOS metrics
        
    Raises:
        QuantAnalysisError: If input data is invalid
    """
    if not isinstance(dict_return, dict):
        raise QuantAnalysisError("dict_return must be a dictionary")
    
    if not isinstance(sid_to_oos, dict):
        raise QuantAnalysisError("sid_to_oos must be a dictionary")
    
    if not dict_return or not sid_to_oos:
        logger.warning("Empty input dictionaries provided")
        return {}
    
    try:
        # Extend pandas with quantstats functionality
        qs.extend_pandas()
        
        # Get available quantstats functions
        function_names = [
            f for f in dir(qs.stats) 
            if not f.startswith('_') and f not in SKIPPED_FUNCTIONS
        ]
        
        # Find symphonies that have both return data and OOS dates
        eligible_symphonies = set(dict_return.keys()) & set(sid_to_oos.keys())
        logger.info(f"Calculating OOS metrics for {len(eligible_symphonies)} eligible symphonies")
        
        dict_quant_oos_stats = {}
        successful_calculations = 0
        failed_calculations = 0
        
        for symphony_id in eligible_symphonies:
            try:
                returns_df = dict_return[symphony_id]
                oos_start_date = sid_to_oos[symphony_id]
                
                if not isinstance(returns_df, pd.DataFrame):
                    logger.warning(f"Skipping {symphony_id}: not a DataFrame")
                    continue
                
                # Filter data from OOS start date onwards
                try:
                    oos_data = returns_df.loc[oos_start_date:]
                    if oos_data.empty:
                        logger.warning(f"No OOS data for {symphony_id} from {oos_start_date}")
                        continue
                except KeyError:
                    logger.warning(f"OOS start date {oos_start_date} not found in data for {symphony_id}")
                    continue
                
                portfolio_values = get_portfolio_values(oos_data)
                temp_dict = {}
                
                for function_name in function_names:
                    try:
                        func = getattr(qs.stats, function_name)
                        args = set(inspect.getfullargspec(func).args)
                        
                        # Only calculate metrics that take returns/prices and don't need benchmark
                        if ('returns' in args or 'prices' in args) and 'benchmark' not in args:
                            result = func(portfolio_values)
                            temp_dict[function_name] = result
                            successful_calculations += 1
                            
                    except Exception as e:
                        logger.debug(f"Failed to calculate OOS {function_name} for {symphony_id}: {e}")
                        temp_dict[function_name] = None
                        failed_calculations += 1
                
                dict_quant_oos_stats[symphony_id] = temp_dict
                
            except Exception as e:
                logger.error(f"Unexpected error processing OOS for {symphony_id}: {e}")
                continue
        
        logger.info(f"OOS calculations completed: {successful_calculations} successful, {failed_calculations} failed")
        return dict_quant_oos_stats
        
    except Exception as e:
        raise QuantAnalysisError(f"Failed to calculate OOS statistics: {e}") from e