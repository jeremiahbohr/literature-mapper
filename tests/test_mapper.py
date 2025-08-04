"""
Simple, focused test suite for Literature Mapper.
Tests core functionality without over-engineering.
"""

import pytest
import tempfile
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
import shutil

from literature_mapper import LiteratureMapper, ValidationError, APIError, PDFProcessingError
from literature_mapper.validation import validate_api_key, validate_json_response


@pytest.fixture
def temp_corpus():
    """Create temporary test directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_api_key():
    """Set up mock API key."""
    test_key = "AIza" + "x" * 35
    old_key = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = test_key
    yield test_key
    if old_key:
        os.environ["GEMINI_API_KEY"] = old_key
    else:
        os.environ.pop("GEMINI_API_KEY", None)


@pytest.fixture
def sample_pdf():
    """Get a sample PDF from fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    pdf_files = list(fixtures_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files in tests/fixtures")
    return pdf_files[0]


@pytest.fixture
def sample_analysis():
    """Sample AI response with all required fields."""
    return {
        "title": "Test Paper Title",
        "authors": ["Author One", "Author Two"],
        "year": 2024,
        "journal": "Test Journal",
        "abstract_short": "This is a test abstract with exactly twenty five words to meet the strict validation requirements for testing purposes only.",
        "core_argument": "This paper argues something important",
        "methodology": "Experimental study",
        "theoretical_framework": "Some framework",
        "contribution_to_field": "Novel contribution",
        "key_concepts": ["concept1", "concept2"],
        "doi": None,
        "citation_count": None
    }


@pytest.fixture
def mapper_instance(temp_corpus, mock_api_key):
    """Create mapper with robust cleanup for Windows."""
    mapper = None
    try:
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                mock_model.return_value.generate_content.return_value = Mock(text="test")
                mapper = LiteratureMapper(str(temp_corpus))
                yield mapper
    finally:
        if mapper:
            try:
                # Force close all connections
                mapper.db_session.close()
                mapper.db_session.bind.dispose()
                # Small delay for Windows file system
                time.sleep(0.1)
            except:
                pass


class TestBasicFunctionality:
    """Test core features work."""
    
    def test_package_imports(self):
        """Test package can be imported."""
        from literature_mapper import LiteratureMapper
        assert LiteratureMapper is not None
    
    def test_api_key_validation(self):
        """Test API key validation works."""
        assert validate_api_key("AIza" + "x" * 35)  # Valid
        assert not validate_api_key("invalid")      # Invalid
        assert not validate_api_key("")             # Empty
    
    def test_mapper_needs_api_key(self, temp_corpus):
        """Test mapper requires API key."""
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
        
        with pytest.raises(ValidationError, match="API key missing"):
            LiteratureMapper(str(temp_corpus))
    
    def test_mapper_creation(self, mapper_instance):
        """Test mapper can be created."""
        mapper = mapper_instance
        assert mapper.corpus_path.exists()
        assert mapper.model_name == "gemini-2.5-flash"


class TestPDFProcessing:
    """Test PDF handling."""
    
    def test_pdf_text_extraction(self, mapper_instance, sample_pdf):
        """Test PDF text extraction works."""
        mapper = mapper_instance
        
        # Copy PDF to test location
        test_pdf = mapper.corpus_path / "test.pdf"
        shutil.copy(sample_pdf, test_pdf)
        
        # Extract text
        text = mapper.pdf_processor.extract_text(test_pdf)
        assert isinstance(text, str)
        assert len(text) > 50  # Should have some content
    
    def test_rejects_non_pdf(self, mapper_instance):
        """Test rejects non-PDF files."""
        mapper = mapper_instance
        
        # Create fake file
        fake_file = mapper.corpus_path / "fake.txt"
        fake_file.write_text("Not a PDF")
        
        with pytest.raises(PDFProcessingError):
            mapper.pdf_processor.extract_text(fake_file)


class TestDatabaseOperations:
    """Test database functionality."""
    
    def test_manual_entry(self, mapper_instance):
        """Test adding papers manually."""
        mapper = mapper_instance
        
        # Add a paper
        mapper.add_manual_entry(
            title="Test Paper",
            authors=["Test Author"],
            year=2024,
            core_argument="Test argument",
            methodology="Test method",
            theoretical_framework="Test framework",
            contribution_to_field="Test contribution"
        )
        
        # Check it was added
        df = mapper.get_all_analyses()
        assert len(df) == 1
        assert df.iloc[0]["title"] == "Test Paper"
    
    def test_export_csv(self, mapper_instance):
        """Test CSV export works."""
        mapper = mapper_instance
        
        # Add a paper
        mapper.add_manual_entry(
            title="Export Test",
            authors=["Author"],
            year=2024,
            core_argument="Argument",
            methodology="Method",
            theoretical_framework="Framework",
            contribution_to_field="Contribution"
        )
        
        # Export to CSV
        csv_path = mapper.corpus_path / "export.csv"
        mapper.export_to_csv(str(csv_path))
        
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0


class TestValidation:
    """Test input validation."""
    
    def test_json_validation(self, sample_analysis):
        """Test AI response validation."""
        # Valid response
        result = validate_json_response(sample_analysis)
        assert result["title"] == sample_analysis["title"]
        
        # Invalid response - missing title
        invalid = sample_analysis.copy()
        del invalid["title"]
        with pytest.raises(ValueError, match="Missing required field"):
            validate_json_response(invalid)
        
        # Invalid response - empty title
        invalid = sample_analysis.copy()
        invalid["title"] = ""
        with pytest.raises(ValueError, match="Title cannot be empty"):
            validate_json_response(invalid)


class TestEndToEnd:
    """Test complete workflows."""
    
    def test_full_pipeline(self, temp_corpus, mock_api_key, sample_pdf, sample_analysis):
        """Test processing a PDF end-to-end."""
        mapper = None
        try:
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel') as mock_model:
                    # Mock AI response with proper format
                    mock_response = Mock()
                    mock_response.text = json.dumps(sample_analysis)
                    mock_model.return_value.generate_content.return_value = mock_response
                    
                    mapper = LiteratureMapper(str(temp_corpus))
                    
                    # Copy PDF to corpus
                    test_pdf = temp_corpus / "paper.pdf"
                    shutil.copy(sample_pdf, test_pdf)
                    
                    # Process papers
                    result = mapper.process_new_papers()
                    
                    # Should have processed 1 paper
                    assert result.processed == 1
                    assert result.failed == 0
                    
                    # Should be in database
                    df = mapper.get_all_analyses()
                    assert len(df) == 1
                    assert df.iloc[0]["title"] == sample_analysis["title"]
        finally:
            if mapper:
                try:
                    mapper.db_session.close()
                    mapper.db_session.bind.dispose()
                    time.sleep(0.1)  # Windows file system delay
                except:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])