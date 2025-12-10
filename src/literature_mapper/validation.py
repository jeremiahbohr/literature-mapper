"""
Input validation and data cleaning utilities.
"""

import re
import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_NODE_TYPES = {
    'paper', 'author', 'concept', 'method', 'finding', 
    'institution', 'hypothesis', 'limitation', 'task', 'dataset', 'metric', 'source',
    'challenge', 'problem_statement'
}

def validate_api_key(api_key: str) -> bool:
    """Validate Gemini API key format."""
    if not api_key or not isinstance(api_key, str):
        return False
    # Basic check for AIza... format
    return api_key.startswith("AIza") and len(api_key) > 20

def validate_pdf_file(file_path: str, max_size: int) -> bool:
    """Validate PDF file existence and size."""
    try:
        path = os.path.abspath(file_path)
        if not os.path.exists(path):
            return False
        
        if not os.path.isfile(path):
            return False
            
        if not path.lower().endswith('.pdf'):
            return False
            
        size = os.path.getsize(path)
        if size == 0 or size > max_size:
            return False
            
        return True
    except Exception:
        return False

def validate_directory_path(dir_path: str) -> bool:
    """Validate directory existence and permissions."""
    try:
        path = os.path.abspath(dir_path)
        if not os.path.exists(path):
            return False
        
        if not os.path.isdir(path):
            return False
        
        # Check read permission
        if not os.access(path, os.R_OK):
            return False
        
        return True
        
    except Exception:
        return False

def validate_json_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean AI JSON response.
    Returns cleaned, validated data or raises ValueError.
    """
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
    
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

def validate_manual_entry(title: str, authors: List[str], year: int) -> tuple[str, List[str], int]:
    """
    Validate manual entry parameters.
    Returns cleaned values or raises ValueError.
    """
    # Clean and validate title
    clean_title = clean_text(title)
    if not clean_title:
        raise ValueError("Title cannot be empty")
    
    # Clean and validate authors
    clean_authors = [clean_text(author) for author in authors if author and str(author).strip()]
    if not clean_authors:
        raise ValueError("At least one author required")
    
    # Validate year
    if not isinstance(year, int) or not (1900 <= year <= 2030):
        raise ValueError("Year must be an integer between 1900 and 2030")
    
    return clean_title, clean_authors, year

def validate_search_params(column: str, query: str) -> tuple[str, str]:
    """Validate search parameters."""
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
        raise ValueError("Search query cannot be empty")
        
    return column, cleaned_query

def validate_update_params(field: str, value: Any) -> tuple[str, Any]:
    """
    Validate update parameters.
    Returns cleaned values or raises ValueError.
    """
    updatable_fields = {
        'title', 'journal', 'year', 'core_argument', 'methodology',
        'theoretical_framework', 'contribution_to_field', 'abstract_short',
        'doi', 'citation_count'
    }
    
    if field not in updatable_fields:
        raise ValueError(f"Field '{field}' cannot be updated")
    
    if field == 'year':
        try:
            year = int(value)
            if not (1900 <= year <= 2030):
                raise ValueError("Year must be between 1900 and 2030")
            return field, year
        except (ValueError, TypeError):
            raise ValueError("Year must be a valid integer")
            
    elif field == 'citation_count':
        try:
            count = int(value)
            if count < 0:
                raise ValueError("Citation count cannot be negative")
            return field, count
        except (ValueError, TypeError):
            raise ValueError("Citation count must be a valid integer")
            
    else:
        cleaned_value = clean_text(str(value))
        if not cleaned_value:
            raise ValueError(f"Value for '{field}' cannot be empty")
        return field, cleaned_value

def validate_kg_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Knowledge Graph extraction response.
    
    Ensures nodes and edges conform to schema:
    - Nodes must have id, type, and label
    - Node types must be in ALLOWED_NODE_TYPES  
    - Edges must reference valid node IDs
    - Edge types are normalized to uppercase
    
    Args:
        data: Raw KG response from LLM
        
    Returns:
        Cleaned dict with valid nodes and edges only
    """
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
        
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    
    if not isinstance(nodes, list):
        raise ValueError("Nodes must be a list")
    if not isinstance(edges, list):
        raise ValueError("Edges must be a list")
        
    # Validate and collect nodes
    valid_node_ids = set()
    valid_nodes = []
    
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            logger.warning(f"Skipping invalid node {i}: not a dict")
            continue
        
        # Required fields
        node_id = node.get('id')
        node_type = node.get('type')
        node_label = node.get('label')
        
        if not node_id:
            logger.warning(f"Skipping node {i}: missing id")
            continue
            
        if not node_type:
            logger.warning(f"Skipping node {i} (id={node_id}): missing type")
            continue
            
        if not node_label:
            logger.warning(f"Skipping node {i} (id={node_id}): missing label")
            continue
        
        # Normalize and validate type
        node_type_lower = str(node_type).lower().strip()
        if node_type_lower not in ALLOWED_NODE_TYPES:
            logger.warning(f"Skipping node {i} (id={node_id}): invalid type '{node_type}'")
            continue
        
        # Clean label
        clean_label = clean_text(str(node_label), max_length=500)
        if not clean_label:
            logger.warning(f"Skipping node {i} (id={node_id}): empty label after cleaning")
            continue
        
        # Build validated node
        validated_node = {
            'id': str(node_id),
            'type': node_type_lower,
            'label': clean_label,
        }
        
        # Optional fields
        if 'confidence' in node:
            try:
                conf = float(node['confidence'])
                validated_node['confidence'] = max(0.0, min(1.0, conf))  # Clamp to [0, 1]
            except (ValueError, TypeError):
                pass  # Skip invalid confidence
                
        if 'subtype' in node and node['subtype']:
            validated_node['subtype'] = str(node['subtype']).strip()
        
        valid_nodes.append(validated_node)
        valid_node_ids.add(str(node_id))
    
    # Validate edges
    valid_edges = []
    
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            logger.warning(f"Skipping invalid edge {i}: not a dict")
            continue
        
        source = edge.get('source')
        target = edge.get('target')
        edge_type = edge.get('type')
        
        if not source or not target or not edge_type:
            logger.warning(f"Skipping edge {i}: missing source, target, or type")
            continue
        
        source_str = str(source)
        target_str = str(target)
        
        # Validate connectivity
        if source_str not in valid_node_ids:
            logger.warning(f"Skipping edge {i}: source '{source}' not found in valid nodes")
            continue
            
        if target_str not in valid_node_ids:
            logger.warning(f"Skipping edge {i}: target '{target}' not found in valid nodes")
            continue
        
        # Normalize edge type to uppercase
        edge_type_normalized = str(edge_type).upper().strip()
        edge_type_normalized = re.sub(r'[^A-Z_]', '_', edge_type_normalized)
        
        if not edge_type_normalized:
            logger.warning(f"Skipping edge {i}: invalid edge type '{edge_type}'")
            continue
        
        valid_edges.append({
            'source': source_str,
            'target': target_str,
            'type': edge_type_normalized,
        })
    
    logger.info(f"Validated KG: {len(valid_nodes)}/{len(nodes)} nodes, {len(valid_edges)}/{len(edges)} edges")
            
    return {"nodes": valid_nodes, "edges": valid_edges}


# Export main functions
__all__ = [
    'validate_api_key',
    'validate_directory_path',
    'validate_pdf_file',
    'validate_json_response',
    'validate_manual_entry',
    'validate_search_params',
    'validate_update_params',
    'validate_kg_response',
    'clean_text',
    'ALLOWED_NODE_TYPES'
]