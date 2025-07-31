"""
Data processing utilities for Discord exports and symphony data.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


class InvalidDiscordDataError(DataProcessingError):
    """Exception raised when Discord export data is invalid."""
    pass


def get_symphonies(filename: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Extract symphony URLs and metadata from Discord export JSON file.
    
    Args:
        filename: Path to Discord export JSON file
        
    Returns:
        Dictionary of symphony data keyed by symphony ID
        
    Raises:
        InvalidDiscordDataError: If Discord export format is invalid
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Discord export file not found: {filename}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in Discord export file: {e}") from e
    
    if not isinstance(data, dict) or 'messages' not in data:
        raise InvalidDiscordDataError("Discord export missing 'messages' key")
    
    messages = data['messages']
    if not isinstance(messages, list):
        raise InvalidDiscordDataError("Discord 'messages' should be a list")
    
    symphonies_dict = {}
    processed_count = 0
    
    try:
        embeds = []
        for message in messages:
            if not isinstance(message, dict) or 'author' not in message:
                logger.warning(f"Skipping invalid message: {message}")
                continue
                
            author_name = message.get('author', {}).get('name', 'Unknown')
            
            for embed in message.get('embeds', []):
                if not isinstance(embed, dict):
                    continue
                    
                # Extract author from embed fields if available
                fields = embed.get('fields', [])
                author_fields = [field for field in fields 
                               if isinstance(field, dict) and field.get('name') == 'Author']
                if author_fields:
                    author_name = author_fields[0].get('value', author_name)
                    
                embeds.append((embed, author_name))

        for embed, author_name in embeds:
            embed_url = embed.get('url')
            if not embed_url or 'app.composer.trade/symphony' not in embed_url:
                continue
                
            try:
                symphony_id = get_symphony_id(embed_url)
                symphonies_dict[symphony_id] = {
                    'title': embed.get('title', 'Unknown Title'),
                    'url': embed_url,
                    'timestamp': embed.get('timestamp', ''),
                    'id': symphony_id,
                    'author': author_name
                }
                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to process embed URL {embed_url}: {e}")
                continue
        
        logger.info(f"Processed {processed_count} symphonies from {len(messages)} messages")
        return symphonies_dict
        
    except Exception as e:
        raise DataProcessingError(f"Failed to process Discord export: {e}") from e


def get_symphony_id(url: str) -> str:
    """Extract symphony ID from Composer URL."""
    if url.endswith('/details'):
        return url.split('/')[-2]
    else:
        return url.split('/')[-1]


def symphonies_to_df(symphonies_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert symphony dictionary to pandas DataFrame.
    
    Args:
        symphonies_dict: Dictionary of symphony data
        
    Returns:
        DataFrame with symphony information
        
    Raises:
        DataProcessingError: If dictionary format is invalid
    """
    if not isinstance(symphonies_dict, dict):
        raise DataProcessingError("symphonies_dict must be a dictionary")
    
    if not symphonies_dict:
        logger.warning("Empty symphonies dictionary provided")
        return pd.DataFrame(columns=["symphony_sid", "title", "url", "timestamp", "name"])
    
    # Use list comprehension for better performance
    try:
        data = {
            "symphony_sid": [value['id'] for value in symphonies_dict.values()],
            "title": [value.get('title', 'Unknown') for value in symphonies_dict.values()],
            "url": [value['url'] for value in symphonies_dict.values()],
            "timestamp": [value.get('timestamp', '') for value in symphonies_dict.values()],
            "name": [value.get('author', 'Unknown') for value in symphonies_dict.values()]
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with {len(df)} symphonies")
        return df
        
    except KeyError as e:
        raise DataProcessingError(f"Missing required key in symphony data: {e}") from e
    except Exception as e:
        raise DataProcessingError(f"Failed to create DataFrame: {e}") from e


def response_to_dataframe(symphony_data_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of symphony API responses to DataFrame.
    
    Args:
        symphony_data_list: List of symphony API response dictionaries
        
    Returns:
        DataFrame with symphony response data
        
    Raises:
        DataProcessingError: If input data is invalid
    """
    if not isinstance(symphony_data_list, list):
        raise DataProcessingError("symphony_data_list must be a list")
    
    if not symphony_data_list:
        logger.warning("Empty symphony data list provided")
        return pd.DataFrame()
    
    try:
        symphony_dict = {}
        for i, symphony_data in enumerate(symphony_data_list):
            if not isinstance(symphony_data, dict):
                logger.warning(f"Skipping non-dict item at index {i}: {type(symphony_data)}")
                continue
                
            for key, value in symphony_data.items():
                if key not in symphony_dict:
                    symphony_dict[key] = []
                symphony_dict[key].append(value)
        
        if not symphony_dict:
            logger.warning("No valid symphony data found")
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(symphony_dict)
        logger.info(f"Created DataFrame with {len(df)} symphony responses")
        return df
        
    except Exception as e:
        raise DataProcessingError(f"Failed to convert responses to DataFrame: {e}") from e


def convert_trading_date(date_int: Union[int, str]) -> datetime:
    """
    Convert integer trading date to datetime object.
    
    Args:
        date_int: Integer or string representing days since epoch
        
    Returns:
        Datetime object
        
    Raises:
        ValueError: If date_int cannot be converted to integer
        DataProcessingError: If conversion fails
    """
    try:
        days_since_epoch = int(date_int)
        if days_since_epoch < 0:
            raise ValueError(f"Invalid date integer: {date_int} (must be >= 0)")
            
        # Unix epoch: 1970-01-01
        epoch = datetime(1970, 1, 1)
        result = epoch + timedelta(days=days_since_epoch)
        
        # Sanity check: result should be reasonable
        if result.year < 1970 or result.year > 2100:
            logger.warning(f"Unusual date result: {result} from input {date_int}")
            
        return result
        
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert {date_int} to trading date: {e}") from e
    except Exception as e:
        raise DataProcessingError(f"Unexpected error converting trading date: {e}") from e


def get_backtest_and_symphony_name(jsond: Dict[str, Any]) -> tuple:
    """
    Extract backtest data and symphony information from API response.
    
    Args:
        jsond: Backtest API response JSON
        
    Returns:
        Tuple of (allocations_df, returns_df, stats, symphony_name, id)
    """
    id = list(jsond['legend'].keys())[0]
    symphony_name = jsond['legend'][id]['name']

    holdings = jsond["last_market_days_holdings"]
    start_date = convert_trading_date(jsond["first_day"])
    end_date = convert_trading_date(jsond["last_market_day"])

    tickers = []
    for ticker in holdings:
        tickers.append(ticker)

    allocations = jsond["tdvm_weights"]
    date_range = pd.date_range(start=start_date, end=end_date)
    df_allocations = pd.DataFrame(0.0, index=date_range, columns=tickers)

    for ticker in allocations:
        for date_int in allocations[ticker]:
            trading_date = convert_trading_date(date_int)
            percent = allocations[ticker][date_int]
            df_allocations.at[trading_date, ticker] = percent

    capital_return = jsond['dvm_capital']
    df_return = pd.DataFrame(0.0, index=date_range, columns=['capital'])
    for date_int in capital_return[id]:
        trading_date = convert_trading_date(date_int)
        df_return.at[trading_date, 'capital'] = capital_return[id][date_int]

    stats = jsond['stats']
    
    return df_allocations, df_return, stats, symphony_name, id


def convert_sid_dict_to_df(sid_to_name: Dict[str, str], sid_to_values: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert symphony ID dictionaries to DataFrame.
    
    Args:
        sid_to_name: Mapping of symphony ID to name
        sid_to_values: Mapping of symphony ID to statistics
        
    Returns:
        DataFrame with symphony statistics
    """
    columns = set()
    df_dict = {'sid': [], 'name': []}
    
    for sid, values_in_dict in sid_to_values.items():
        columns = columns.union(values_in_dict.keys())
    
    for column in columns:
        df_dict[column] = []
    
    for sid, name in sid_to_name.items():
        if sid not in sid_to_values:
            print(f'SID {sid} {name} not in backtest.')
            continue
        backtest_stats = sid_to_values[sid]
        df_dict['sid'].append(sid)
        df_dict['name'].append(name)
        for column in columns:
            if column in backtest_stats:
                df_dict[column].append(backtest_stats[column])
            else:
                df_dict[column].append(None)
    
    return pd.DataFrame(df_dict)


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries, combining values for matching keys.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            result[key].update(value)
        else:
            result[key] = value
    return result


def extract_date(row: pd.Series) -> pd.Timestamp:
    """Extract date from DataFrame row."""
    return pd.to_datetime(row['last_semantic_update_at'][:10])


def get_date(row: pd.Series, key: str) -> str:
    """Get date string from DataFrame row, with fallback."""
    return '2024-12-28' if isinstance(row[key], float) else row[key][:10]


def extract_start_date(row: pd.Series) -> pd.Timestamp:
    """Extract start date from DataFrame row."""
    return pd.to_datetime(get_date(row, 'last_semantic_update_at'))


def extract_last_date(row: pd.Series) -> pd.Timestamp:
    """Extract last date from DataFrame row."""
    return pd.to_datetime(get_date(row, 'last_backtest_last_market_day'))