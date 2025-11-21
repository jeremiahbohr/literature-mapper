import pytest
from unittest.mock import Mock, patch
import numpy as np
from literature_mapper.mapper import LiteratureMapper
from literature_mapper.database import Paper, KGNode, get_db_session

@pytest.fixture
def mock_mapper(tmp_path):
    with patch('literature_mapper.mapper.EmbeddingGenerator') as MockGen, \
         patch('literature_mapper.mapper.AIAnalyzer') as MockAnalyzer, \
         patch('literature_mapper.mapper.genai') as MockGenAI:
        
        # Setup mock genai to pass validation
        MockGenAI.list_models.return_value = []
        
        # Key must be >32 chars
        mapper = LiteratureMapper(str(tmp_path), api_key="AIzaSyDFAKEKEY123456789012345678901234567890")
        mapper.embedding_generator = Mock()
        return mapper

def test_search_corpus_semantic(mock_mapper):
    # Setup mock embeddings
    mock_mapper.embedding_generator.generate_query_embedding.return_value = np.array([1.0, 0.0])
    
    # Setup mock DB session and data
    with patch('literature_mapper.mapper.get_db_session') as mock_get_session:
        session = Mock()
        mock_get_session.return_value.__enter__.return_value = session
        
        # Mock nodes with vectors
        node1 = Mock(spec=KGNode)
        node1.vector = np.array([1.0, 0.0]) # Perfect match
        node1.label = "Node 1"
        node1.type = "concept"
        node1.source_paper_id = 1
        
        node2 = Mock(spec=KGNode)
        node2.vector = np.array([0.0, 1.0]) # No match
        node2.label = "Node 2"
        node2.type = "concept"
        node2.source_paper_id = 2
        
        session.query.return_value.filter.return_value.all.return_value = [node1, node2]
        
        # Mock paper retrieval
        paper1 = Mock(spec=Paper)
        paper1.id = 1
        paper1.title = "Paper 1"
        paper1.year = 2023
        
        session.query.return_value.get.side_effect = lambda id: paper1 if id == 1 else None
        
        # Execute search
        results = mock_mapper.search_corpus("query", semantic=True)
        
        # Verify
        assert len(results) == 1
        assert results[0]['id'] == 1
        assert results[0]['match_type'] == "semantic"
        assert results[0]['match_score'] == 1.0
        assert "Node 1" in results[0]['match_context']

def test_search_corpus_semantic_no_embedding(mock_mapper):
    mock_mapper.embedding_generator.generate_query_embedding.return_value = None
    
    with patch('literature_mapper.mapper.get_db_session'):
        results = mock_mapper.search_corpus("query", semantic=True)
        assert len(results) == 0

def test_search_corpus_fallback_keyword(mock_mapper):
    # If semantic=False, should use keyword search
    with patch('literature_mapper.mapper.get_db_session') as mock_get_session:
        session = Mock()
        mock_get_session.return_value.__enter__.return_value = session
        
        paper = Mock(spec=Paper)
        paper.id = 1
        paper.title = "Keyword Match"
        paper.year = 2023
        
        session.query.return_value.filter.return_value.limit.return_value.all.return_value = [paper]
        
        results = mock_mapper.search_corpus("query", semantic=False)
        
        assert len(results) == 1
        assert results[0]['match_type'] == "keyword"
