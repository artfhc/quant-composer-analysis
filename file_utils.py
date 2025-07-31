"""
File I/O utilities for managing JSON and CSV files.
"""

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


class FileUtilsError(Exception):
    """Custom exception for file utility errors."""
    pass


def write_json(json_data: Any, filename: Union[str, Path]) -> None:
    """
    Write JSON data to file with directory creation.
    
    Args:
        json_data: Data to write as JSON
        filename: Output file path
        
    Raises:
        FileUtilsError: If writing fails
        TypeError: If json_data is not JSON serializable
    """
    file_path = Path(filename)
    
    try:
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Test JSON serialization first
        json.dumps(json_data, ensure_ascii=False)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
            
        logger.debug(f"Successfully wrote JSON to {file_path}")
        
    except TypeError as e:
        raise TypeError(f"Data is not JSON serializable: {e}") from e
    except (OSError, IOError) as e:
        raise FileUtilsError(f"Failed to write JSON file {filename}: {e}") from e
    except Exception as e:
        raise FileUtilsError(f"Unexpected error writing JSON: {e}") from e


def read_json(filename: Union[str, Path]) -> Any:
    """
    Read JSON data from file.
    
    Args:
        filename: Input file path
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        FileUtilsError: If reading fails
    """
    file_path = Path(filename)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {filename}")
    
    if not file_path.is_file():
        raise FileUtilsError(f"Path is not a file: {filename}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.debug(f"Successfully read JSON from {file_path}")
        return data
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {filename}: {e}") from e
    except (OSError, IOError) as e:
        raise FileUtilsError(f"Failed to read JSON file {filename}: {e}") from e
    except Exception as e:
        raise FileUtilsError(f"Unexpected error reading JSON: {e}") from e


def get_csv_name(filename: str, end_date: Optional[str] = None, base_dir: str = "bin") -> str:
    """
    Generate CSV filename with date-based directory structure.
    
    Args:
        filename: Base filename (without extension)
        end_date: End date string, defaults to today
        base_dir: Base directory for output files
        
    Returns:
        Full path for CSV file
        
    Raises:
        FileUtilsError: If directory creation fails
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    if end_date is None:
        _, end_date = get_start_end_date()
    
    # Sanitize filename
    safe_filename = filename.replace('/', '_').replace('\\', '_')
    
    try:
        path = Path(base_dir) / f"SYMPHONIES-{end_date}" / f"{safe_filename}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Generated CSV path: {path}")
        return str(path)
        
    except (OSError, IOError) as e:
        raise FileUtilsError(f"Failed to create directory structure: {e}") from e
    except Exception as e:
        raise FileUtilsError(f"Unexpected error generating CSV path: {e}") from e


def get_start_end_date() -> Tuple[str, str]:
    """
    Get default start and end dates for data collection.
    
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    today = date.today().strftime('%Y-%m-%d')
    start_date = '2000-01-01'
    end_date = today
    return start_date, end_date