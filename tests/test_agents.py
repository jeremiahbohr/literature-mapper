import pytest
from unittest.mock import Mock, patch
from literature_mapper.agents import ArgumentAgent, ValidationAgent
from literature_mapper.mapper import LiteratureMapper

@pytest.fixture
def mock_nodes():
    return [
        {"match_context": "Finding A", "match_score": 0.9},
        {"match_context": "Finding B", "match_score": 0.8}
    ]

@pytest.fixture
def mock_agent_api():
    with patch('literature_mapper.agents.genai') as mock_genai:
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        yield mock_model

def test_argument_agent_synthesis(mock_agent_api, mock_nodes):
    agent = ArgumentAgent("fake_key", "model")
    mock_agent_api.generate_content.return_value.text = "Synthesized answer."
    
    response = agent.synthesize("query", mock_nodes)
    
    assert response == "Synthesized answer."
    mock_agent_api.generate_content.assert_called_once()
    
def test_argument_agent_empty_context(mock_agent_api):
    agent = ArgumentAgent("fake_key", "model")
    response = agent.synthesize("query", [])
    assert "No relevant information" in response

def test_validation_agent_critique(mock_agent_api, mock_nodes):
    agent = ValidationAgent("fake_key", "model")
    mock_response = """
    ```json
    {
        "verdict": "SUPPORTED",
        "explanation": "Evidence supports it.",
        "citations": ["Paper 1"]
    }
    ```
    """
    mock_agent_api.generate_content.return_value.text = mock_response
    
    response = agent.validate_hypothesis("hypothesis", mock_nodes)
    
    assert response["verdict"] == "SUPPORTED"
    assert response["explanation"] == "Evidence supports it."
    assert response["citations"] == ["Paper 1"]

def test_validation_agent_empty_context(mock_agent_api):
    agent = ValidationAgent("fake_key", "model")
    response = agent.validate_hypothesis("hypothesis", [])
    assert response["verdict"] == "NOVEL"

def test_mapper_integration(tmp_path):
    with patch('literature_mapper.mapper.EmbeddingGenerator'), \
         patch('literature_mapper.mapper.ArgumentAgent') as MockArgAgent, \
         patch('literature_mapper.mapper.ValidationAgent') as MockValAgent, \
         patch('literature_mapper.mapper.AIAnalyzer'), \
         patch('literature_mapper.mapper.genai'):
         
        mapper = LiteratureMapper(str(tmp_path), api_key="AIzaSyDFAKEKEY123456789012345678901234567890")
        
        # Mock search results
        mapper.search_corpus = Mock(return_value=[{"match_context": "ctx"}])
        
        # Test synthesis
        mapper.synthesize_answer("query")
        MockArgAgent.return_value.synthesize.assert_called_with("query", [{"match_context": "ctx"}])
        
        # Test validation
        mapper.validate_hypothesis("hypothesis")
        MockValAgent.return_value.validate_hypothesis.assert_called_with("hypothesis", [{"match_context": "ctx"}])
