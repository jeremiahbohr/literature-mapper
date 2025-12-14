import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, Text, ForeignKey, Index, UniqueConstraint, Float
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from pathlib import Path
import logging
import os
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager
from .exceptions import DatabaseError

logger = logging.getLogger(__name__)

Base = declarative_base()

@dataclass
class DatabaseInfo:
    """Information about the database state and contents."""
    exists: bool
    path: str
    size_mb: float
    table_counts: dict[str, int]
    is_healthy: bool
    error: Optional[str] = None


# ===========================================================
#               TEMPORAL EXTENSION MODELS
# ===========================================================

class ConceptTemporalStats(Base):
    """
    Year-by-year temporal statistics for each concept.
    Supports temporal clustering, era detection, wave analysis,
    and citations-per-year trajectory analysis.
    """
    __tablename__ = "concept_temporal_stats"

    concept_id = Column(Integer, ForeignKey("concepts.id", ondelete="CASCADE"), primary_key=True)
    year = Column(Integer, primary_key=True)

    paper_count = Column(Integer, nullable=False, default=0)
    citation_sum = Column(Integer, nullable=True)
    citations_per_year = Column(Float, nullable=True)

    concept = relationship("Concept", back_populates="temporal_stats")

    __table_args__ = (
        Index("idx_cts_concept", "concept_id"),
        Index("idx_cts_year", "year"),
    )

    def __repr__(self):
        return f"<ConceptTemporalStats(concept={self.concept_id}, year={self.year}, papers={self.paper_count})>"


# ===========================================================
#               CORE TABLES
# ===========================================================

class Paper(Base):
    """Main papers table storing core paper information."""
    __tablename__ = 'papers'

    id = Column(Integer, primary_key=True)
    pdf_path = Column(String, nullable=True)
    title = Column(String, nullable=False)
    year = Column(Integer, nullable=False)
    journal = Column(String, nullable=True)

    abstract_short = Column(Text, nullable=True)
    core_argument = Column(Text, nullable=False)
    methodology = Column(Text, nullable=False)
    theoretical_framework = Column(Text, nullable=False)
    contribution_to_field = Column(Text, nullable=False)

    doi = Column(String, nullable=True)
    arxiv_id = Column(String, nullable=True)

    citation_count = Column(Integer, nullable=True)
    citations_per_year = Column(Float, nullable=True)

    authors = relationship("Author", secondary="paper_authors", back_populates="papers", order_by=sa.text("paper_authors.rowid"))
    concepts = relationship("Concept", secondary="paper_concepts", back_populates="papers")

    __table_args__ = (
        Index('idx_paper_year', 'year'),
        Index('idx_paper_title', 'title'),
        Index('idx_paper_pdf_path', 'pdf_path'),
        UniqueConstraint('title', 'year', name='uq_paper_title_year'),
        UniqueConstraint('pdf_path', name='uq_paper_pdf_path'),
    )

    def __repr__(self):
        return f"<Paper(id={self.id}, title='{self.title[:50]}...', year={self.year})>"


class Author(Base):
    """Normalized author names with relationships to papers."""
    __tablename__ = 'authors'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    canonical_name = Column(String, nullable=True)

    papers = relationship("Paper", secondary="paper_authors", back_populates="authors")
    aliases = relationship("AuthorAlias", back_populates="canonical_author")

    def __repr__(self):
        return f"<Author(id={self.id}, name='{self.name}')>"


class Concept(Base):
    """Key concepts and keywords extracted from papers."""
    __tablename__ = 'concepts'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    canonical_name = Column(String, nullable=True)

    # =======================
    # NEW TEMPORAL FIELDS
    # =======================
    first_year = Column(Integer, nullable=True)
    last_year = Column(Integer, nullable=True)
    peak_year = Column(Integer, nullable=True)
    trend_slope = Column(Float, nullable=True)

    papers = relationship("Paper", secondary="paper_concepts", back_populates="concepts")
    aliases = relationship("ConceptAlias", back_populates="canonical_concept")

    # NEW relationship to temporal stats
    temporal_stats = relationship("ConceptTemporalStats",
                                  back_populates="concept",
                                  cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Concept(id={self.id}, name='{self.name}')>"


class PaperAuthor(Base):
    __tablename__ = 'paper_authors'
    paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), primary_key=True)
    author_id = Column(Integer, ForeignKey('authors.id', ondelete='CASCADE'), primary_key=True)


class PaperConcept(Base):
    __tablename__ = 'paper_concepts'
    paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), primary_key=True)
    concept_id = Column(Integer, ForeignKey('concepts.id', ondelete='CASCADE'), primary_key=True)


class ConceptAlias(Base):
    __tablename__ = 'concept_aliases'

    id = Column(Integer, primary_key=True)
    alias = Column(String, nullable=False, unique=True)
    canonical_id = Column(Integer, ForeignKey('concepts.id', ondelete='CASCADE'), nullable=False)

    canonical_concept = relationship("Concept", back_populates="aliases")

    __table_args__ = (
        Index('idx_concept_alias', 'alias'),
    )


class AuthorAlias(Base):
    __tablename__ = 'author_aliases'

    id = Column(Integer, primary_key=True)
    alias = Column(String, nullable=False, unique=True)
    canonical_id = Column(Integer, ForeignKey('authors.id', ondelete='CASCADE'), nullable=False)

    canonical_author = relationship("Author", back_populates="aliases")

    __table_args__ = (
        Index('idx_author_alias', 'alias'),
    )


class KGNode(Base):
    __tablename__ = 'kg_nodes'

    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)
    label = Column(String, nullable=False)

    source_paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), nullable=False)
    paper = relationship("Paper", backref="nodes")

    vector = Column(sa.PickleType, nullable=True)
    embedding_model = Column(String, nullable=True)

    claim_confidence = Column(Float, nullable=True)
    claim_type = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint('type', 'label', name='uq_kg_node_type_label'),
        Index('idx_kg_node_label', 'label'),
    )


class KGEdge(Base):
    __tablename__ = 'kg_edges'

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('kg_nodes.id', ondelete='CASCADE'), nullable=False)
    target_id = Column(Integer, ForeignKey('kg_nodes.id', ondelete='CASCADE'), nullable=False)
    type = Column(String, nullable=False)
    source_paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), nullable=False)

    source = relationship("KGNode", foreign_keys=[source_id])
    target = relationship("KGNode", foreign_keys=[target_id])

    __table_args__ = (
        Index('idx_kg_edge_source', 'source_id'),
        Index('idx_kg_edge_target', 'target_id'),
    )


class Citation(Base):
    __tablename__ = 'citations'

    id = Column(Integer, primary_key=True)
    source_paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), nullable=False)

    title = Column(String, nullable=False)
    author = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    openalex_id = Column(String, nullable=True)

    source_paper = relationship("Paper", backref="citations")

    __table_args__ = (
        Index('idx_citation_title', 'title'),
        Index('idx_citation_author', 'author'),
    )


class IntellectualEdge(Base):
    __tablename__ = 'intellectual_edges'

    id = Column(Integer, primary_key=True)
    source_paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), nullable=False)
    target_paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), nullable=False)

    relation_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    evidence = Column(Text, nullable=True)

    source_paper = relationship("Paper", foreign_keys=[source_paper_id], backref="outgoing_relations")
    target_paper = relationship("Paper", foreign_keys=[target_paper_id], backref="incoming_relations")

    __table_args__ = (
        Index('idx_intellectual_edge_source', 'source_paper_id'),
        Index('idx_intellectual_edge_target', 'target_paper_id'),
        Index('idx_intellectual_edge_type', 'relation_type'),
        UniqueConstraint('source_paper_id', 'target_paper_id', 'relation_type', name='uq_intellectual_edge'),
    )


# ===========================================================
#               ENGINE + SESSION MANAGEMENT
# ===========================================================

def _create_engine(corpus_path: Path):
    corpus_path.mkdir(parents=True, exist_ok=True)

    if not os.access(corpus_path, os.W_OK):
        raise DatabaseError(f"No write permission for directory {corpus_path}")

    db_path = corpus_path / "corpus.db"

    engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={'timeout': 30},
        pool_timeout=20,
        echo=False
    )

    @sa.event.listens_for(engine, 'connect')
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.close()

    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        Base.metadata.create_all(engine)
        logger.info("Database initialized at %s", db_path)
        return engine
    except Exception as e:
        raise DatabaseError(f"Failed to initialize database: {e}")


@contextmanager
def get_db_session(corpus_path: Path):
    engine = _create_engine(corpus_path)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()


def get_database_info(corpus_path: Path) -> DatabaseInfo:
    db_path = corpus_path / "corpus.db"

    if not db_path.exists():
        return DatabaseInfo(
            exists=False,
            path=str(db_path),
            size_mb=0.0,
            table_counts={},
            is_healthy=False
        )

    try:
        size_bytes = db_path.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 2)

        engine = sa.create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            table_counts = {}
            is_healthy = True
            error_msg = None

            try:
                papers_count = conn.execute(sa.text("SELECT COUNT(*) FROM papers")).fetchone()[0]
                authors_count = conn.execute(sa.text("SELECT COUNT(*) FROM authors")).fetchone()[0]
                concepts_count = conn.execute(sa.text("SELECT COUNT(*) FROM concepts")).fetchone()[0]
                kg_nodes_count = conn.execute(sa.text("SELECT COUNT(*) FROM kg_nodes")).fetchone()[0]
                kg_edges_count = conn.execute(sa.text("SELECT COUNT(*) FROM kg_edges")).fetchone()[0]

                table_counts = {
                    'papers': papers_count,
                    'authors': authors_count,
                    'concepts': concepts_count,
                    'kg_nodes': kg_nodes_count,
                    'kg_edges': kg_edges_count
                }

            except Exception as e:
                is_healthy = False
                error_msg = f"Database query failed: {e}"
                table_counts = {}

        engine.dispose()

        return DatabaseInfo(
            exists=True,
            path=str(db_path),
            size_mb=size_mb,
            table_counts=table_counts,
            is_healthy=is_healthy,
            error=error_msg
        )

    except Exception as e:
        logger.error("Failed to get database info: %s", e)
        return DatabaseInfo(
            exists=True,
            path=str(db_path),
            size_mb=0.0,
            table_counts={},
            is_healthy=False,
            error=str(e)
        )


__all__ = [
    'Base', 'Paper', 'Author', 'AuthorAlias',
    'Concept', 'ConceptAlias', 'PaperAuthor', 'PaperConcept',
    'KGNode', 'KGEdge', 'Citation', 'IntellectualEdge',
    'ConceptTemporalStats',
    'DatabaseInfo', 'get_db_session', 'get_database_info'
]
