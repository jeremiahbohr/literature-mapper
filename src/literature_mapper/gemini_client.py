"""
Centralized Gemini client factory for Literature Mapper.

All modules should use get_client() instead of creating their own
genai.Client instances.  The client is cached per API key.
"""

import os
import logging
from typing import Optional

from google import genai

logger = logging.getLogger(__name__)

_client_cache: dict[str, genai.Client] = {}


def get_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Return a cached genai.Client for the given API key.

    If *api_key* is None the GEMINI_API_KEY environment variable is used.
    """
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "No Gemini API key supplied and GEMINI_API_KEY env var is not set."
        )

    if key not in _client_cache:
        _client_cache[key] = genai.Client(api_key=key)
        logger.debug("Created new genai.Client (key …%s)", key[-4:])

    return _client_cache[key]
