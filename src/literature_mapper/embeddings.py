"""
Embedding service for semantic search capabilities.
Handles vector generation and similarity calculations.
"""

import logging
import numpy as np
from typing import List, Optional, Union
from google.genai import types
from .gemini_client import get_client
from .exceptions import APIError
from .config import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Handles generation of vector embeddings using Gemini API.
    """
    
    def __init__(self, api_key: str, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize embedding generator.
        
        Args:
            api_key: Gemini API key
            model_name: Name of the embedding model to use
        """
        if not api_key:
            raise ValueError("API key is required for embeddings")
            
        self.api_key = api_key
        self.model_name = model_name
        self.client = get_client(api_key)
        
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding vector for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of the embedding vector, or None if failed
        """
        if not text or not text.strip():
            return None
            
        retries = 3
        delay = 1
        
        for attempt in range(retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                
                if result.embeddings:
                    return np.array(result.embeddings[0].values, dtype=np.float32)
                return None
                
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    if attempt < retries - 1:
                        import time
                        time.sleep(delay)
                        delay *= 2
                        continue
                logger.error("Failed to generate embedding: %s", e)
                return None
            
    def generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Generate embedding vector for a search query.
        Uses 'RETRIEVAL_QUERY' task type for better matching.
        
        Args:
            query: Search query text
            
        Returns:
            Numpy array of the embedding vector
        """
        if not query or not query.strip():
            return None
            
        retries = 3
        delay = 1
        
        for attempt in range(retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=query,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                    ),
                )
                
                if result.embeddings:
                    return np.array(result.embeddings[0].values, dtype=np.float32)
                return None
                
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    if attempt < retries - 1:
                        import time
                        time.sleep(delay)
                        delay *= 2
                        continue
                logger.error("Failed to generate query embedding: %s", e)
                return None

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity score (-1.0 to 1.0)
    """
    if v1 is None or v2 is None:
        return 0.0
        
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return np.dot(v1, v2) / (norm1 * norm2)
