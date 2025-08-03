"""
Input validation and security utilities for Literature Mapper.
"""

import re
import os
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Security constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB default
ALLOWED_FILE_EXTENSIONS = {'.pdf'}

# Flexible API key validation patterns
API_KEY_PATTERNS = [
    r'^AIza[0-9A-Za-z_-]{35}$',  # Current Gemini format
    r'^[A-Za-z0-9_-]{32,128}$',  # Future-proof flexible format
]

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format with future-proof patterns.
    
    Performance: Fast validation, safe to call frequently.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    api_key = api_key.strip()
    
    # Basic length checks
    if len(api_key) < 20 or len(api_key) > 200:
        return False
    
    # Check against known patterns
    for pattern in API_KEY_PATTERNS:
        if re.match(pattern, api_key):
            return True
    
    # Basic character validation for unknown formats
    if re.match(r'^[A-Za-z0-9_-]+$', api_key):
        logger.warning("API key format not recognized but appears valid (length: %d)", len(api_key))
        return True
    
    return False

def validate_directory_path(path: Path, check_writable: bool = True) -> bool:
    """
    Validate directory path for corpus operations.
    
    Performance: Fast file system checks.
    """
    try:
        path = Path(path).resolve()
        
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        
        if not path.is_dir():
            logger.error("Path is not a directory: %s", path)
            return False
        
        if not os.access(path, os.R_OK):
            logger.error("No read permission: %s", path)
            return False
        
        if check_writable and not os.access(path, os.W_OK):
            logger.error("No write permission: %s", path)
            return False
        
        return True
        
    except Exception as e:
        logger.error("Directory validation failed: %s", e)
        return False

def validate_pdf_file(file_path: Path, max_size: int = MAX_FILE_SIZE) -> bool:
    """
    Validate PDF file for processing.
    
    Performance: Fast file checks, safe for batch operations.
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
        
        # Basic MIME type check (if available)
        try:
            import mimetypes
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type != 'application/pdf':
                return False
        except ImportError:
            pass  # Skip MIME check if not available
        
        return True
        
    except Exception as e:
        logger.error("PDF validation failed: %s", e)
        return False

def validate_json_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean AI JSON response.
    
    Validation Rules:
    - Required fields with type checking
    - Year range: 1900-2030 
    - Text cleaning and fallback values
    
    Performance: Moderate - processes all fields.
    """
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
    
    # Required fields with validation
    required_fields = {
        'title': str,
        'authors': list,
        'year': (int, type(None)),
        'core_argument': str,
        'methodology': str,
        'theoretical_framework': str,
        'contribution_to_field': str
    }
    
    validated_data = {}
    
    # Validate required fields
    for field, expected_type in required_fields.items():
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        
        value = data[field]
        
        if field == 'title':
            if not value or not str(value).strip():
                raise ValueError("Title cannot be empty")
            validated_data[field] = clean_text(str(value))
            
        elif field == 'authors':
            if not value:
                validated_data[field] = ["Unknown Author"]
            else:
                cleaned_authors = [clean_text(str(author)) for author in value if str(author).strip()]
                validated_data[field] = cleaned_authors if cleaned_authors else ["Unknown Author"]
                    
        elif field == 'year':
            if value is not None:
                try:
                    year_int = int(value)
                    if year_int < 1900 or year_int > 2030:
                        raise ValueError(f"Year {year_int} must be between 1900 and 2030")
                    validated_data[field] = year_int
                except (ValueError, TypeError):
                    validated_data[field] = None
            else:
                validated_data[field] = None
                
        else:
            # String fields
            if not value or not str(value).strip():
                validated_data[field] = "Not specified"
            else:
                validated_data[field] = clean_text(str(value))
    
    # Handle optional fields
    optional_fields = ['journal', 'abstract_short', 'key_concepts', 'doi', 'citation_count']
    
    for field in optional_fields:
        if field in data:
            value = data[field]
            
            if field == 'key_concepts':
                if isinstance(value, list):
                    validated_data[field] = [clean_text(str(concept)) for concept in value if str(concept).strip()]
                else:
                    validated_data[field] = []
            elif field == 'abstract_short':
                if value:
                    cleaned_text = clean_text(str(value))
                    word_count = len(cleaned_text.split())
                    if word_count > 35:
                        logger.warning("Abstract too long: %d words", word_count)
                    validated_data[field] = cleaned_text
                else:
                    validated_data[field] = None
            elif field == 'doi':
                if value and validate_doi(str(value)):
                    validated_data[field] = str(value).strip()
                else:
                    validated_data[field] = None
            elif field == 'citation_count':
                if isinstance(value, int) and value >= 0:
                    validated_data[field] = value
                else:
                    validated_data[field] = None
            else:
                # journal and other string fields
                validated_data[field] = clean_text(str(value)) if value else None
        else:
            validated_data[field] = None
    
    return validated_data

def validate_processing_result(result) -> bool:
    """
    Validate ProcessingResult dataclass structure.
    
    Performance: Fast validation.
    """
    try:
        required_attrs = ['processed', 'failed', 'skipped']
        if not all(hasattr(result, attr) for attr in required_attrs):
            return False
        
        # Check all counts are non-negative integers
        for attr in required_attrs:
            value = getattr(result, attr)
            if not isinstance(value, int) or value < 0:
                return False
        
        return True
    except Exception:
        return False

def validate_doi(doi: str) -> bool:
    """Validate DOI format."""
    if not doi or not isinstance(doi, str):
        return False
    
    # Basic DOI pattern: 10.xxxx/xxxxx
    doi_pattern = r'^10\.\d{4,}/[-._;()/:\w\[\]]+$'
    return bool(re.match(doi_pattern, doi.strip()))

def clean_text(text: str, max_length: int = 5000) -> str:
    """Clean and normalize text input."""
    if not text or not isinstance(text, str):
        return ""
    
    text = str(text).strip()
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning("Text truncated to %d characters", max_length)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def validate_search_params(column: str, query: str) -> tuple[str, str]:
    """
    Validate search parameters.
    
    Rules: Column must be whitelisted, query non-empty after cleaning.
    Performance: Very fast.
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
    
    if len(cleaned_query) > 500:
        raise ValueError("Search query too long (max 500 characters)")
    
    return column, cleaned_query

def validate_update_params(paper_ids: List[int], updates: Dict[str, Any]) -> tuple[List[int], Dict[str, Any]]:
    """
    Validate paper update parameters.
    
    Rules: Valid IDs, whitelisted fields, proper types.
    Performance: Fast validation.
    """
    # Validate paper IDs
    if not paper_ids or not isinstance(paper_ids, list):
        raise ValueError("Paper IDs must be a non-empty list")
    
    for i, pid in enumerate(paper_ids):
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError(f"Paper ID at position {i} must be a positive integer, got: {pid}")
    
    if len(paper_ids) > 1000:
        raise ValueError(f"Cannot update more than 1000 papers at once, got: {len(paper_ids)}")
    
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
    
    # Clean and validate update values
    cleaned_updates = {}
    for field, value in updates.items():
        if field == 'year':
            if value is not None:
                try:
                    year_int = int(value)
                    if year_int < 1900 or year_int > 2030:
                        raise ValueError(f"Year {year_int} must be between 1900 and 2030")
                    cleaned_updates[field] = year_int
                except (ValueError, TypeError):
                    raise ValueError(f"Year must be an integer, got: {value}")
            else:
                cleaned_updates[field] = None
        elif field == 'citation_count':
            if value is not None:
                try:
                    count = int(value)
                    if count < 0:
                        raise ValueError(f"Citation count must be non-negative, got: {count}")
                    cleaned_updates[field] = count
                except (ValueError, TypeError):
                    raise ValueError(f"Citation count must be an integer, got: {value}")
            else:
                cleaned_updates[field] = None
        elif field == 'doi':
            if value is not None and not validate_doi(str(value)):
                raise ValueError(f"Invalid DOI format: {value}")
            cleaned_updates[field] = str(value).strip() if value else None
        else:
            # Text fields
            if value is not None:
                cleaned_value = clean_text(str(value))
                if not cleaned_value and field in ['title', 'core_argument', 'methodology', 'theoretical_framework', 'contribution_to_field']:
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
    'validate_processing_result',
    'validate_doi',
    'clean_text',
    'validate_search_params',
    'validate_update_params'
]