"""
Input validation and security utilities for Literature Mapper.
"""

import re
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Security constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB default
ALLOWED_FILE_EXTENSIONS = {'.pdf'}

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format with future-proof patterns.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    api_key = api_key.strip()
    
    # Basic length checks
    if len(api_key) < 20 or len(api_key) > 200:
        return False
    
    # Current Gemini format or flexible future format
    patterns = [
        r'^AIza[0-9A-Za-z_-]{35}$',  # Current Gemini format
        r'^[A-Za-z0-9_-]{32,128}$',  # Future-proof flexible format
    ]
    
    for pattern in patterns:
        if re.match(pattern, api_key):
            return True
    
    return False

def validate_directory_path(path: Path, check_writable: bool = True) -> bool:
    """
    Validate directory path for corpus operations.
    """
    try:
        path = Path(path).resolve()
        
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        
        if not path.is_dir():
            return False
        
        if not os.access(path, os.R_OK):
            return False
        
        if check_writable and not os.access(path, os.W_OK):
            return False
        
        return True
        
    except Exception:
        return False

def validate_pdf_file(file_path: Path, max_size: int = MAX_FILE_SIZE) -> bool:
    """
    Validate PDF file for processing.
    """
    try:
        path = Path(file_path)
        
        if not path.exists() or not path.is_file():
            return False
        
        # Check file extension
        if path.suffix.lower() not in ALLOWED_FILE_EXTENSIONS:
            return False
        
        # Check file size
        file_size = path.stat().st_size
        if file_size == 0 or file_size > max_size:
            return False
        
        # Check read permission
        if not os.access(path, os.R_OK):
            return False
        
        return True
        
    except Exception:
        return False

def validate_json_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean AI JSON response with strict requirements.
    """
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
    
    # Required fields
    required_fields = ['title', 'authors', 'year', 'core_argument', 'methodology', 
                      'theoretical_framework', 'contribution_to_field']
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    validated_data = {}
    
    # Title - must exist and be non-empty
    title = str(data['title']).strip()
    if not title:
        raise ValueError("Title cannot be empty")
    validated_data['title'] = clean_text(title)
    
    # Authors - must have at least one non-empty author
    authors = data['authors']
    if not isinstance(authors, list) or not authors:
        raise ValueError("Must have at least one author")
    
    cleaned_authors = [clean_text(str(author)) for author in authors if str(author).strip()]
    if not cleaned_authors:
        raise ValueError("Must have at least one valid author")
    validated_data['authors'] = cleaned_authors
    
    # Year - must be valid integer in reasonable range
    year = data['year']
    if year is not None:
        try:
            year_int = int(year)
            if year_int < 1900 or year_int > 2030:
                raise ValueError(f"Year {year_int} must be between 1900 and 2030")
            validated_data['year'] = year_int
        except (ValueError, TypeError):
            validated_data['year'] = None
    else:
        validated_data['year'] = None
    
    # Required text fields - must be non-empty
    text_fields = ['core_argument', 'methodology', 'theoretical_framework', 'contribution_to_field']
    for field in text_fields:
        value = str(data[field]).strip()
        if not value:
            raise ValueError(f"Required field '{field}' cannot be empty")
        validated_data[field] = clean_text(value)
    
    # Optional fields
    optional_fields = ['journal', 'abstract_short', 'key_concepts', 'doi', 'citation_count']
    
    for field in optional_fields:
        if field in data and data[field] is not None:
            if field == 'key_concepts':
                if isinstance(data[field], list):
                    validated_data[field] = [clean_text(str(concept)) for concept in data[field] 
                                           if str(concept).strip()]
                else:
                    validated_data[field] = []
            elif field == 'citation_count':
                try:
                    count = int(data[field])
                    validated_data[field] = count if count >= 0 else None
                except:
                    validated_data[field] = None
            else:
                validated_data[field] = clean_text(str(data[field]))
        else:
            validated_data[field] = None
    
    return validated_data

def clean_text(text: str, max_length: int = 5000) -> str:
    """Clean and normalize text input."""
    if not text or not isinstance(text, str):
        return ""
    
    text = str(text).strip()
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def validate_search_params(column: str, query: str) -> tuple[str, str]:
    """
    Validate search parameters.
    """
    searchable_fields = {
        'title', 'core_argument', 'methodology', 'theoretical_framework',
        'contribution_to_field', 'journal', 'abstract_short'
    }
    
    if not column or column not in searchable_fields:
        raise ValueError(f"Invalid search column '{column}'. Allowed: {', '.join(searchable_fields)}")
    
    if not query or not isinstance(query, str):
        raise ValueError("Search query cannot be empty")
    
    cleaned_query = clean_text(query.strip())
    if not cleaned_query:
        raise ValueError("Search query cannot be empty after cleaning")
    
    return column, cleaned_query

def validate_update_params(paper_ids: List[int], updates: Dict[str, Any]) -> tuple[List[int], Dict[str, Any]]:
    """
    Validate paper update parameters.
    """
    # Validate paper IDs
    if not paper_ids or not isinstance(paper_ids, list):
        raise ValueError("Paper IDs must be a non-empty list")
    
    for i, pid in enumerate(paper_ids):
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError(f"Paper ID at position {i} must be a positive integer")
    
    # Validate updates
    if not updates or not isinstance(updates, dict):
        raise ValueError("Updates must be a non-empty dictionary")
    
    updatable_fields = {
        'title', 'year', 'journal', 'abstract_short', 'core_argument',
        'methodology', 'theoretical_framework', 'contribution_to_field',
        'doi', 'citation_count'
    }
    
    invalid_fields = set(updates.keys()) - updatable_fields
    if invalid_fields:
        raise ValueError(f"Invalid fields: {', '.join(sorted(invalid_fields))}")
    
    # Clean update values
    cleaned_updates = {}
    for field, value in updates.items():
        if field == 'year' and value is not None:
            try:
                year_int = int(value)
                if year_int < 1900 or year_int > 2030:
                    raise ValueError(f"Year {year_int} must be between 1900 and 2030")
                cleaned_updates[field] = year_int
            except (ValueError, TypeError):
                raise ValueError(f"Year must be an integer")
        elif field == 'citation_count' and value is not None:
            try:
                count = int(value)
                if count < 0:
                    raise ValueError(f"Citation count must be non-negative")
                cleaned_updates[field] = count
            except (ValueError, TypeError):
                raise ValueError(f"Citation count must be an integer")
        else:
            # Text fields
            if value is not None:
                cleaned_value = clean_text(str(value))
                if not cleaned_value and field in ['title', 'core_argument', 'methodology', 
                                                  'theoretical_framework', 'contribution_to_field']:
                    raise ValueError(f"Required field '{field}' cannot be empty")
                cleaned_updates[field] = cleaned_value
            else:
                cleaned_updates[field] = None
    
    return paper_ids, cleaned_updates

# Export validation functions
__all__ = [
    'validate_api_key',
    'validate_directory_path', 
    'validate_pdf_file',
    'validate_json_response',
    'clean_text',
    'validate_search_params',
    'validate_update_params'
]