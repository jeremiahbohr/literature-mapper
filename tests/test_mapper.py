import os
from pathlib import Path
import shutil
import tempfile
from io import BytesIO

import pandas as pd
import pytest

from literature_mapper.mapper import LiteratureMapper
from literature_mapper.exceptions import ValidationError, PDFProcessingError, APIError

# ---------------------------------------------------------------------------
# Helper to create an isolated mapper instance for each test
# ---------------------------------------------------------------------------

# Patch _setup_api globally so tests don't need a real key or network
from literature_mapper.mapper import LiteratureMapper
LiteratureMapper._setup_api = lambda self, api_key=None: None


def _make_mapper(tmp_path: Path) -> LiteratureMapper:
    """Instantiate LiteratureMapper with patched API + AI layers for tests."""
    # ---- 1. Disable real API validation ---- #
    from literature_mapper import mapper as mapper_mod
    mapper_mod.LiteratureMapper._setup_api = lambda self, api_key=None: None  # type: ignore[attr-defined]

    # ---- 2. Stub AI analysis to deterministic data ---- #
    def _stub_analysis(_self, _text):
        return {
            "title": "Stub Title",
            "authors": ["Stub Author"],
            "year": 2020,
            "journal": None,
            "abstract_short": "word " * 25,  # 25 words
            "core_argument": "Stub thesis.",
            "methodology": "Stub method",
            "theoretical_framework": "N/A",
            "key_concepts": ["concept"],
            "contribution_to_field": "Adds nothing",
            "doi": None,
            "citation_count": None,
        }

    from literature_mapper.mapper import AIAnalyzer  # late import after monkeypatch above

    AIAnalyzer.analyze = _stub_analysis  # type: ignore[assignment]

    # ---- 3. Bypass scalars() incompat on some SQLAlchemy versions ---- #
    mapper_mod.LiteratureMapper._get_existing_papers = lambda self: set()  # type: ignore[attr-defined]

    # ---- 4. Build mapper in isolated corpus ---- #
    os.environ["GEMINI_API_KEY"] = "unit-test-key"
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    return mapper_mod.LiteratureMapper(str(corpus_dir), model_name="gemini-2.5-flash")


fixtures_dir = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Utility to add a manual paper quickly
# ---------------------------------------------------------------------------

def _add_sample_paper(mapper: LiteratureMapper, title: str = "Sample", year: int = 2020):
    mapper.add_manual_entry(
        title=title,
        authors=["Author1"],
        year=year,
        core_argument="Arg",
        methodology="Method",
        theoretical_framework="TF",
        contribution_to_field="Contribution",
        key_concepts=["Concept1"],
    )
    from literature_mapper.database import Paper
    return mapper.db_session.query(Paper).filter_by(title=title, year=year).first().id

# ---------------------------------------------------------------------------
# PDF extraction & processing pipeline(tmp_path: Path):
    mapper = _make_mapper(tmp_path)
    pid = _add_sample_paper(mapper, "Old", 2020)
    mapper.update_papers([pid], {"title": "New"})
    assert mapper.get_all_analyses().iloc[0]["title"] == "New"
    mapper.update_papers([pid], {"journal": "J", "methodology": "M2"})
    row = mapper.get_all_analyses().iloc[0]
    assert row["journal"] == "J" and row["methodology"] == "M2"


@pytest.mark.xfail(reason="Relationship columns not yet supported for updates")
def test_update_relationship_fields(tmp_path: Path):
    mapper = _make_mapper(tmp_path)
    pid = _add_sample_paper(mapper, "Rel", 2021)
    mapper.update_papers([pid], {"authors": ["B"]})


def test_update_validation_cases(tmp_path: Path):
    mapper = _make_mapper(tmp_path)
    pid = _add_sample_paper(mapper, "Val", 2022)
    # Non‑existent ID
    with pytest.raises(ValidationError):
        mapper.update_papers([9999], {"title": "x"})
    # Invalid field
    with pytest.raises(ValidationError):
        mapper.update_papers([pid], {"bogus": "x"})
    # Empty params
    with pytest.raises(ValidationError):
        mapper.update_papers([], {"title": "x"})
    with pytest.raises(ValidationError):
        mapper.update_papers([pid], {})


@pytest.mark.xfail(reason="Value validation not yet implemented on update_papers")
def test_update_invalid_values(tmp_path: Path):
    mapper = _make_mapper(tmp_path)
    pid = _add_sample_paper(mapper, "Bad", 2023)
    with pytest.raises(ValidationError):
        mapper.update_papers([pid], {"year": 1800})
    with pytest.raises(ValidationError):
        mapper.update_papers([pid], {"citation_count": -1})

# ---------------------------------------------------------------------------
# process_new_papers skip / fail paths
# ---------------------------------------------------------------------------

def test_process_skip_and_fail(tmp_path: Path, monkeypatch):
    mapper = _make_mapper(tmp_path)
    sample_pdf = fixtures_dir / "attention.pdf"
    # Place two PDFs in corpus
    for i in range(2):
        shutil.copy(sample_pdf, tmp_path / "corpus" / f"{i}.pdf")

    # Patch extractor to raise on first file (skip), patch AI to raise on second (fail)
    calls = {"count": 0}

    original_extract = mapper.pdf_processor.extract_text

    def fake_extract(path):
        if calls["count"] == 0:
            calls["count"] += 1
            raise PDFProcessingError("bad", pdf_path=path, error_type="extraction")
        return original_extract(path)

    monkeypatch.setattr(mapper.pdf_processor, "extract_text", fake_extract)

    def fake_analyze(text):
        raise APIError("AI boom")

    monkeypatch.setattr(mapper.ai_analyzer, "analyze", fake_analyze)

    result = mapper.process_new_papers()
    assert result.skipped == 1 and result.failed == 1 and result.processed == 0

# ---------------------------------------------------------------------------
# export_to_csv
# ---------------------------------------------------------------------------

def test_export_csv_empty_and_populated(tmp_path: Path):
    mapper = _make_mapper(tmp_path)
    out_path = tmp_path / "export.csv"
    # Empty
    mapper.export_to_csv(str(out_path))
    assert pd.read_csv(out_path).empty
    # Two papers with distinct authors/concepts
    mapper.add_manual_entry(title="A1", authors=["X"], year=2020,
                            core_argument="a", methodology="m", theoretical_framework="t",
                            contribution_to_field="c", key_concepts=["K1"], journal="J")
    mapper.add_manual_entry(title="A2", authors=["X"], year=2021,
                            core_argument="a", methodology="m", theoretical_framework="t",
                            contribution_to_field="c", key_concepts=["K1", "K2"], journal="J")
    mapper.export_to_csv(str(out_path))
    df = pd.read_csv(out_path)
    assert set(df["title"]) == {"A1", "A2"}
    row_a2 = df[df["title"] == "A2"].iloc[0]
    assert "K2" in row_a2["key_concepts"]

# ---------------------------------------------------------------------------
# statistics tests (distinct counts)
# ---------------------------------------------------------------------------

def test_statistics_distinct(tmp_path: Path):
    mapper = _make_mapper(tmp_path)
    mapper.add_manual_entry(title="P1", authors=["Same"], year=2020,
                            core_argument="a", methodology="m", theoretical_framework="t",
                            contribution_to_field="c", key_concepts=["K"], journal="J")
    mapper.add_manual_entry(title="P2", authors=["Same"], year=2021,
                            core_argument="a", methodology="m", theoretical_framework="t",
                            contribution_to_field="c", key_concepts=["K"], journal="J")
    stats = mapper.get_statistics()
    assert stats.total_papers == 2
    assert stats.total_authors == 1  # distinct
    assert stats.total_concepts == 1

# ---------------------------------------------------------------------------
# PDFProcessor oversize guard
# ---------------------------------------------------------------------------

def test_pdfprocessor_max_file_size(tmp_path: Path):
    mapper = _make_mapper(tmp_path)
    # Create a tiny PDF in fixtures; we force max_file_size=10 bytes to trigger size check
    small_pdf = fixtures_dir / "attention.pdf"
    proc = mapper.pdf_processor.__class__(max_file_size=10)  # new instance with low limit
    with pytest.raises(PDFProcessingError):
        proc.extract_text(small_pdf)
