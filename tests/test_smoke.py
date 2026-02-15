"""
Smoke tests for Literature Mapper Gemini API integration.

These tests require a valid GEMINI_API_KEY environment variable.
They are skipped automatically when the key is unavailable.
"""

import os
import pytest
import numpy as np

requires_api_key = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — skipping live API test",
)


@requires_api_key
def test_embed_hello_world():
    """Embedding 'hello world' should return a non-empty float vector."""
    from literature_mapper.embeddings import EmbeddingGenerator

    gen = EmbeddingGenerator(api_key=os.environ["GEMINI_API_KEY"])
    vec = gen.generate_embedding("hello world")

    assert vec is not None, "Embedding returned None"
    assert isinstance(vec, np.ndarray), f"Expected np.ndarray, got {type(vec)}"
    assert len(vec) > 0, "Embedding vector is empty"


@requires_api_key
def test_generate_one_token():
    """A minimal generate_content call should return non-empty text."""
    from google.genai import types
    from literature_mapper.gemini_client import get_client
    from literature_mapper.config import DEFAULT_MODEL

    client = get_client(os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents="Say hello.",
        config=types.GenerateContentConfig(max_output_tokens=5),
    )

    assert response.text, "generate_content returned empty text"
