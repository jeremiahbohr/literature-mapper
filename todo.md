# Project Todo List: literature-mapper

## Core Features (MVP) ✅ COMPLETED
- [x] Implement `LiteratureMapper` class structure in `mapper.py`
- [x] Implement database creation and session management in `database.py`
- [x] Define the database schema (normalized with proper relationships)
- [x] Implement PDF text extraction with comprehensive error handling
- [x] Implement the core AI analysis prompt in `ai_prompts.py`
- [x] Implement the main `process_new_papers` logic with retry and validation
- [x] Implement `export_to_csv` function
- [x] Implement `add_manual_entry` function with validation
- [x] Build out CLI with `typer` and `rich` (5 focused commands)
- [x] **BONUS:** Future-proof model support (works with any Gemini model)
- [x] **BONUS:** Comprehensive error handling and user-friendly messages
- [x] **BONUS:** Modern Python practices (type hints, dataclasses, clean APIs)

## Immediate Next Steps (Testing & Polish)
- [ ] **Create test suite** with pytest covering:
  - [ ] PDF processing with various file types and edge cases
  - [ ] AI response parsing and validation
  - [ ] Database operations and relationships
  - [ ] CLI command functionality
  - [ ] Error handling scenarios
- [ ] **Package distribution** setup:
  - [ ] Test installation from PyPI test server
  - [ ] Verify all dependencies work correctly
  - [ ] Test on different Python versions (3.8, 3.9, 3.10, 3.11, 3.12)
- [ ] **Documentation improvements**:
  - [ ] Add docstring examples to key functions
  - [ ] Create simple tutorial notebook
  - [ ] Add troubleshooting guide for common PDF issues

## Potential Future Enhancements
- [ ] **External Metadata Integration:**
  - [ ] Semantic Scholar API integration for citation counts and related papers
  - [ ] CrossRef API for DOI validation and metadata enrichment
  - [ ] arXiv API for preprint information
- [ ] **Analysis Extensions:**
  - [ ] Basic network analysis of citation relationships
  - [ ] Concept co-occurrence analysis
  - [ ] Temporal analysis of research trends
  - [ ] Export to bibliographic formats (BibTeX, RIS)
- [ ] **User Interface Options:**
  - [ ] Simple Streamlit web interface for non-technical users
  - [ ] Jupyter notebook widgets for interactive analysis
  - [ ] Export to visualization tools (Gephi, Cytoscape)
- [ ] **Advanced Processing:**
  - [ ] Support for other document formats (Word, HTML)
  - [ ] OCR integration for scanned PDFs
  - [ ] Multi-language support
  - [ ] Custom field extraction based on research domain

## Code Quality & Maintenance
- [ ] **Performance optimization:**
  - [ ] Async processing for large batches
  - [ ] Database query optimization
  - [ ] Memory usage optimization for large corpora
- [ ] **Monitoring & observability:**
  - [ ] Processing metrics and statistics
  - [ ] API usage tracking and cost estimation
  - [ ] Health checks for long-running processes
- [ ] **Extensibility:**
  - [ ] Plugin system for custom analysis functions
  - [ ] Configuration templates for different research domains
  - [ ] Integration with reference managers (Zotero, Mendeley)
- [ ] **Architecture refactoring (if needed):**
  - [ ] Consider splitting `LiteratureMapper` into focused components (`CorpusManager`, `DocumentProcessor`, `AIAnalyzer`, `DatabaseManager`) if we see evidence of:
    - Users needing programmatic access to individual components
    - Support for multiple AI providers or database backends
    - Testing becoming genuinely difficult
    - Building web interfaces that need different composition
  - [ ] Evaluate clearer module hierarchy (core/, processing/, storage/, interface/)
  - [ ] Standardize return types (consistent use of dataclasses vs dicts)

## Research & Experimental Features
- [ ] **Advanced AI capabilities:**
  - [ ] Experiment with different model combinations
  - [ ] Custom fine-tuning for domain-specific analysis
  - [ ] Multi-modal analysis (figures, tables, citations)
- [ ] **Data science integration:**
  - [ ] Integration with pandas ecosystem (plotly, seaborn)
  - [ ] Export to analysis platforms (Observable, Kaggle)
  - [ ] API for programmatic access to corpus data

## Known Limitations & Technical Debt
- [ ] **Current limitations to address:**
  - Single-threaded processing (could benefit from parallelization)
  - No progress persistence (restart from beginning if interrupted)
  - Limited PDF parsing (struggles with complex layouts)
  - No deduplication logic (same paper from different sources)
- [ ] **Potential refactoring:**
  - Consider async/await patterns for I/O operations
  - Evaluate Pydantic models for data validation
  - Consider SQLAlchemy 2.0 async support
  - Evaluate alternatives to pypdf for better text extraction

## Deployment & Distribution
- [ ] **Production readiness:**
  - [ ] CI/CD pipeline setup
  - [ ] Automated testing on multiple platforms
  - [ ] Performance benchmarking
- [ ] **Community building:**
  - [ ] Example datasets and tutorials
  - [ ] Academic paper about the tool

---

## Project Status: **BETA READY** 🎉

The core functionality is complete and robust. The package successfully:
- Processes PDFs with comprehensive error handling
- Extracts structured data using AI with model optimization
- Stores data in a normalized, queryable database
- Provides both Python API and CLI interfaces
- Works with current and future Gemini models
- Follows modern Python development practices

**Ready for:** Real-world testing, user feedback, and iterative improvements based on actual usage patterns.