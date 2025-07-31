"""
Composer.trade API utility functions for fetching symphony and backtest data.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout, ConnectionError
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)

# Constants
COMPOSER_BASE_URL = "https://backtest-api.composer.trade/api"
DEFAULT_RETRY_ATTEMPTS = 3
RATE_LIMIT_DELAY = 10
ERROR_RETRY_DELAY = 1
DEFAULT_TIMEOUT = 30


class ComposerAPIError(Exception):
    """Custom exception for Composer API errors."""
    pass


class RateLimitError(ComposerAPIError):
    """Exception raised when API rate limit is exceeded."""
    pass


class InvalidSymphonyURLError(ComposerAPIError):
    """Exception raised when symphony URL is invalid."""
    pass


def get_symphony_id(url: str) -> str:
    """Extract symphony ID from Composer URL.
    
    Args:
        url: Composer symphony URL or symphony ID
        
    Returns:
        Symphony ID extracted from URL
        
    Raises:
        InvalidSymphonyURLError: If URL format is invalid
    """
    if not url or not isinstance(url, str):
        raise InvalidSymphonyURLError(f"Invalid URL: {url}")
    
    # If it's already just an ID, return it
    if '/' not in url:
        return url.strip()
    
    try:
        parsed = urlparse(url)
        path_parts = [part for part in parsed.path.split('/') if part]
        
        if not path_parts:
            raise InvalidSymphonyURLError(f"No path in URL: {url}")
        
        # Handle /symphony/{id}/details format
        if len(path_parts) >= 2 and path_parts[-1] == 'details':
            return path_parts[-2]
        
        # Handle /symphony/{id} format
        return path_parts[-1]
        
    except Exception as e:
        raise InvalidSymphonyURLError(f"Failed to parse URL {url}: {e}") from e


def fetch_symphony(
    id: str, 
    retry: int = DEFAULT_RETRY_ATTEMPTS,
    timeout: int = DEFAULT_TIMEOUT
) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Fetch symphony data from Composer API with retry logic.
    
    Args:
        id: Symphony ID or URL
        retry: Number of retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, status_code, data)
        
    Raises:
        ComposerAPIError: If API request fails after all retries
        InvalidSymphonyURLError: If symphony ID/URL is invalid
    """
    if retry <= 0:
        logger.error(f"Max retries exceeded for symphony {id}")
        return False, 404, {}
    
    try:
        ok, status_code, data = _fetch_symphony(id, timeout)
        
        if status_code == 429:
            logger.warning(f"Rate limited for symphony {id}, retrying in {RATE_LIMIT_DELAY}s")
            time.sleep(RATE_LIMIT_DELAY)
            return fetch_symphony(id, retry - 1, timeout)
        
        if ok:
            logger.info(f"Successfully fetched symphony {id}")
        else:
            logger.warning(f"Failed to fetch symphony {id}, status: {status_code}")
            
        return ok, status_code, data
        
    except (RequestException, ConnectionError, Timeout) as e:
        logger.warning(f"Network error fetching symphony {id}: {e}, retrying in {ERROR_RETRY_DELAY}s")
        time.sleep(ERROR_RETRY_DELAY)
        return fetch_symphony(id, retry - 1, timeout)
    
    except Exception as e:
        logger.error(f"Unexpected error fetching symphony {id}: {e}")
        raise ComposerAPIError(f"Failed to fetch symphony {id}: {e}") from e


def _fetch_symphony(id: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[bool, int, Dict[str, Any]]:
    """Internal function to fetch symphony data.
    
    Args:
        id: Symphony ID or URL
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, status_code, response_data)
        
    Raises:
        InvalidSymphonyURLError: If symphony ID is invalid
        RequestException: If HTTP request fails
    """
    symphony_id = get_symphony_id(id)
    url = f"{COMPOSER_BASE_URL}/v1/public/symphonies/{symphony_id}"
    
    logger.debug(f"Fetching symphony data from: {url}")
    
    # Configure session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        response = session.get(url, timeout=timeout)
        
        # Handle empty response
        if not response.content:
            return False, response.status_code, {}
            
        response_data = response.json()
        return response.ok, response.status_code, response_data
        
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Invalid JSON response for symphony {symphony_id}: {e}")
        return False, 500, {}
    except RequestException as e:
        logger.error(f"Request failed for symphony {symphony_id}: {e}")
        raise


def fetch_backtest_raw(
    id: str, 
    start_date: str, 
    end_date: str, 
    retry: int = DEFAULT_RETRY_ATTEMPTS,
    timeout: int = DEFAULT_TIMEOUT,
    capital: int = 100000,
    slippage_percent: float = 0.0005
) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Fetch backtest data from Composer API with retry logic.
    
    Args:
        id: Symphony ID or URL
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        retry: Number of retry attempts
        timeout: Request timeout in seconds
        capital: Initial capital for backtest
        slippage_percent: Slippage percentage for backtest
        
    Returns:
        Tuple of (success, status_code, data)
        
    Raises:
        ComposerAPIError: If API request fails after all retries
        ValueError: If date format is invalid
    """
    if retry <= 0:
        logger.error(f"Max retries exceeded for backtest {id}")
        return False, 404, {}
    
    # Validate date format
    try:
        from datetime import datetime
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD: {e}") from e
    
    try:
        ok, status_code, data = _fetch_backtest_raw(
            id, start_date, end_date, timeout, capital, slippage_percent
        )
        
        if status_code == 429:
            logger.warning(f"Rate limited for backtest {id}, retrying in {RATE_LIMIT_DELAY}s")
            time.sleep(RATE_LIMIT_DELAY)
            return fetch_backtest_raw(id, start_date, end_date, retry - 1, timeout, capital, slippage_percent)
        
        if ok:
            logger.info(f"Successfully fetched backtest for symphony {id}")
        else:
            logger.warning(f"Failed to fetch backtest for symphony {id}, status: {status_code}")
            
        return ok, status_code, data
        
    except (RequestException, ConnectionError, Timeout) as e:
        logger.warning(f"Network error fetching backtest {id}: {e}, retrying in {ERROR_RETRY_DELAY}s")
        time.sleep(ERROR_RETRY_DELAY)
        return fetch_backtest_raw(id, start_date, end_date, retry - 1, timeout, capital, slippage_percent)
    
    except Exception as e:
        logger.error(f"Unexpected error fetching backtest {id}: {e}")
        raise ComposerAPIError(f"Failed to fetch backtest {id}: {e}") from e


def _fetch_backtest_raw(
    id: str, 
    start_date: str, 
    end_date: str,
    timeout: int = DEFAULT_TIMEOUT,
    capital: int = 100000,
    slippage_percent: float = 0.0005
) -> Tuple[bool, int, Dict[str, Any]]:
    """Internal function to fetch backtest data.
    
    Args:
        id: Symphony ID or URL
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        timeout: Request timeout in seconds
        capital: Initial capital for backtest
        slippage_percent: Slippage percentage for backtest
        
    Returns:
        Tuple of (success, status_code, response_data)
        
    Raises:
        InvalidSymphonyURLError: If symphony ID is invalid
        RequestException: If HTTP request fails
    """
    symphony_id = get_symphony_id(id)
    
    payload = {
        "capital": capital,
        "apply_reg_fee": True,
        "apply_taf_fee": True,
        "backtest_version": "v2",
        "slippage_percent": slippage_percent,
        "start_date": start_date,
        "end_date": end_date,
    }

    url = f"{COMPOSER_BASE_URL}/v2/public/symphonies/{symphony_id}/backtest"
    logger.debug(f"Fetching backtest: {symphony_id} {start_date} to {end_date}")
    
    # Configure session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        response = session.post(url, json=payload, timeout=timeout)
        
        # Handle empty response
        if not response.content:
            return False, response.status_code, {}
            
        response_data = response.json()
        return response.ok, response.status_code, response_data
        
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Invalid JSON response for backtest {symphony_id}: {e}")
        return False, 500, {}
    except RequestException as e:
        logger.error(f"Request failed for backtest {symphony_id}: {e}")
        raise