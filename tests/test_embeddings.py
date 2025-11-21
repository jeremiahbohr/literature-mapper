import pytest
from unittest.mock import Mock, patch
import numpy as np
from literature_mapper.embeddings import EmbeddingGenerator, cosine_similarity

@pytest.fixture
def mock_genai():
    with patch('literature_mapper.embeddings.genai') as mock:
        yield mock

def test_embedding_generator_init(mock_genai):
    generator = EmbeddingGenerator("fake_key")
    mock_genai.configure.assert_called_with(api_key="fake_key")
    assert generator.model_name == "models/text-embedding-004"

def test_generate_embedding_success(mock_genai):
    mock_genai.embed_content.return_value = {'embedding': [0.1, 0.2, 0.3]}
    generator = EmbeddingGenerator("fake_key")
    
    vector = generator.generate_embedding("test text")
    
    assert vector is not None
    assert isinstance(vector, np.ndarray)
    assert len(vector) == 3
    assert vector[0] == 0.1
    
    mock_genai.embed_content.assert_called_with(
        model="models/text-embedding-004",
        content="test text",
        task_type="retrieval_document"
    )

def test_generate_embedding_failure(mock_genai):
    mock_genai.embed_content.side_effect = Exception("API Error")
    generator = EmbeddingGenerator("fake_key")
    
    vector = generator.generate_embedding("test text")
    assert vector is None

def test_generate_query_embedding(mock_genai):
    mock_genai.embed_content.return_value = {'embedding': [0.1, 0.2, 0.3]}
    generator = EmbeddingGenerator("fake_key")
    
    vector = generator.generate_query_embedding("query")
    
    mock_genai.embed_content.assert_called_with(
        model="models/text-embedding-004",
        content="query",
        task_type="retrieval_query"
    )

def test_cosine_similarity():
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    assert cosine_similarity(v1, v2) == 1.0
    
    v3 = np.array([0, 1, 0])
    assert cosine_similarity(v1, v3) == 0.0
    
    v4 = np.array([-1, 0, 0])
    assert cosine_similarity(v1, v4) == -1.0
