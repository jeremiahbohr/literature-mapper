import pytest
import sqlalchemy as sa
from literature_mapper.ai_prompts import get_kg_prompt
from literature_mapper.validation import validate_kg_response
from literature_mapper.database import KGNode, KGEdge, Base, Paper

def test_get_kg_prompt():
    prompt = get_kg_prompt("Test Paper")
    assert "Test Paper" in prompt
    assert '"nodes":' in prompt
    assert '"edges":' in prompt
    assert '"metric"' in prompt
    assert '"institution"' in prompt

def test_validate_kg_response_valid():
    data = {
        "nodes": [
            {"id": "paper", "type": "paper", "label": "Test Paper"},
            {"id": "c1", "type": "concept", "label": "AI"},
            {"id": "m1", "type": "metric", "label": "BLEU"},
            {"id": "i1", "type": "institution", "label": "DeepMind"}
        ],
        "edges": [
            {"source": "paper", "target": "c1", "type": "discusses"},
            {"source": "paper", "target": "m1", "type": "evaluates_on"}
        ]
    }
    result = validate_kg_response(data)
    assert len(result['nodes']) == 4
    assert len(result['edges']) == 2

def test_validate_kg_response_invalid_type():
    data = {
        "nodes": [
            {"id": "n1", "type": "invalid_type", "label": "Test"}
        ],
        "edges": []
    }
    with pytest.raises(ValueError, match="invalid type"):
        validate_kg_response(data)

def test_validate_kg_response_missing_edge_target():
    data = {
        "nodes": [
            {"id": "n1", "type": "concept", "label": "Test"}
        ],
        "edges": [
            {"source": "n1", "target": "missing", "type": "rel"}
        ]
    }
    with pytest.raises(ValueError, match="not found in nodes"):
        validate_kg_response(data)

def test_kg_database_models():
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sa.orm.sessionmaker(bind=engine)
    session = Session()
    
    # Create dummy paper
    p = Paper(
        title="Test", year=2024, core_argument="Arg", 
        methodology="Meth", theoretical_framework="Frame", 
        contribution_to_field="Contrib"
    )
    session.add(p)
    session.commit()
    
    # Create nodes
    n1 = KGNode(type="concept", label="AI", source_paper_id=p.id)
    session.add(n1)
    session.commit()
    
    # Test unique constraint
    n2 = KGNode(type="concept", label="AI", source_paper_id=p.id)
    session.add(n2)
    with pytest.raises(sa.exc.IntegrityError):
        session.commit()
    session.rollback()
    
    # Test edge creation
    n2 = KGNode(type="method", label="DL", source_paper_id=p.id)
    session.add(n2)
    session.commit()
    
    edge = KGEdge(source_id=n1.id, target_id=n2.id, type="uses", source_paper_id=p.id)
    session.add(edge)
    session.commit()
    
    assert edge.id is not None
