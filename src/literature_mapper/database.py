import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, Text, ForeignKey, Index
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from pathlib import Path
import logging
import os
from dataclasses import dataclass
from typing import Optional
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

class Paper(Base):
    """Main papers table storing core paper information."""
    __tablename__ = 'papers'

    id = Column(Integer, primary_key=True)
    pdf_path = Column(String, unique=True, nullable=True)  # Nullable for manual entries
    title = Column(String, nullable=False)
    year = Column(Integer, nullable=False)
    journal = Column(String, nullable=True)
    abstract_short = Column(Text, nullable=True)
    core_argument = Column(Text, nullable=False)
    methodology = Column(Text, nullable=False)
    theoretical_framework = Column(Text, nullable=False)
    contribution_to_field = Column(Text, nullable=False)
    doi = Column(String, nullable=True)
    citation_count = Column(Integer, nullable=True)
    
    # Relationships
    authors = relationship("Author", secondary="paper_authors", back_populates="papers")
    concepts = relationship("Concept", secondary="paper_concepts", back_populates="papers")
    
    # Performance indexes only - no unique constraints on title/year
    __table_args__ = (
        Index('idx_paper_year', 'year'),
        Index('idx_paper_title', 'title'),
    )
    
    def __repr__(self):
        return f"<Paper(id={self.id}, title='{self.title[:50]}...', year={self.year})>"


class Author(Base):
    """Normalized author names with relationships to papers."""
    __tablename__ = 'authors'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    
    # Relationships
    papers = relationship("Paper", secondary="paper_authors", back_populates="authors")
    
    def __repr__(self):
        return f"<Author(id={self.id}, name='{self.name}')>"


class Concept(Base):
    """Key concepts and keywords extracted from papers."""
    __tablename__ = 'concepts'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    
    # Relationships
    papers = relationship("Paper", secondary="paper_concepts", back_populates="concepts")
    
    def __repr__(self):
        return f"<Concept(id={self.id}, name='{self.name}')>"


class PaperAuthor(Base):
    """Junction table linking papers to their authors."""
    __tablename__ = 'paper_authors'
    
    paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), primary_key=True)
    author_id = Column(Integer, ForeignKey('authors.id', ondelete='CASCADE'), primary_key=True)


class PaperConcept(Base):
    """Junction table linking papers to their key concepts."""
    __tablename__ = 'paper_concepts'
    
    paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), primary_key=True)
    concept_id = Column(Integer, ForeignKey('concepts.id', ondelete='CASCADE'), primary_key=True)


def get_db_session(corpus_path: Path):
    """
    Create database session with optimal SQLite configuration.
    
    Args:
        corpus_path: Path to the corpus directory
        
    Returns:
        SQLAlchemy session object
        
    Raises:
        DatabaseError: If database creation or connection fails
    """
    try:
        # Ensure directory exists and is writable
        corpus_path.mkdir(parents=True, exist_ok=True)
        
        if not os.access(corpus_path, os.W_OK):
            raise DatabaseError(f"No write permission for directory {corpus_path}")
        
        # Create database path
        db_path = corpus_path / "corpus.db"
        
        # Create SQLite engine with optimizations
        engine = sa.create_engine(
            f"sqlite:///{db_path}",
            connect_args={'timeout': 30},
            pool_timeout=20,
            echo=False
        )
        
        # Configure SQLite for optimal performance
        @sa.event.listens_for(engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.close()
        
        # Test connection and create tables
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        
        Base.metadata.create_all(engine)
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        logger.info("Database initialized at %s", db_path)
        return session
        
    except Exception as e:
        raise DatabaseError(f"Failed to initialize database: {e}")


def get_database_info(corpus_path: Path) -> DatabaseInfo:
    """
    Get information about the database state.
    
    Args:
        corpus_path: Path to the corpus directory
        
    Returns:
        DatabaseInfo with database status and contents
    """
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
        # Get file size
        size_bytes = db_path.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 2)
        
        # Get table counts using a temporary connection
        engine = sa.create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            table_counts = {}
            is_healthy = True
            error_msg = None
            
            try:
                papers_count = conn.execute(sa.text("SELECT COUNT(*) FROM papers")).fetchone()[0]
                authors_count = conn.execute(sa.text("SELECT COUNT(*) FROM authors")).fetchone()[0]
                concepts_count = conn.execute(sa.text("SELECT COUNT(*) FROM concepts")).fetchone()[0]
                
                table_counts = {
                    'papers': papers_count,
                    'authors': authors_count,
                    'concepts': concepts_count
                }
                
            except Exception as e:
                is_healthy = False
                error_msg = f"Database query failed: {e}"
                table_counts = {}
        
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


# Export main classes and functions
__all__ = [
    'Base', 'Paper', 'Author', 'Concept', 'PaperAuthor', 'PaperConcept',
    'DatabaseInfo', 'get_db_session', 'get_database_info'
]