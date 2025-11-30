"""
Visualization module for Literature Mapper.
Exports Knowledge Graph to GEXF format for use in Gephi.
"""

import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from sqlalchemy import text
from .database import get_db_session

logger = logging.getLogger(__name__)

def _get_graph_data(session, mode: str, threshold: float):
    """
    Helper to fetch nodes and edges based on mode and threshold.
    Returns: (nodes_dict, edges_list)
    """
    # 1. Calculate Threshold
    result = session.execute(text("SELECT COUNT(*) FROM papers"))
    total_papers = result.scalar()
    
    if total_papers == 0:
        logger.warning("No papers in corpus.")
        min_weight = 1
    else:
        min_weight = max(1, int(total_papers * threshold))
        
    logger.info(f"Total papers: {total_papers}. Minimum edge weight: {min_weight}")
    
    nodes = {}
    edges = []
    
    # --- MODE: SEMANTIC (Default) ---
    if mode == 'semantic':
        # Fetch Nodes
        nodes_query = text("SELECT id, label, type FROM kg_nodes")
        nodes_result = session.execute(nodes_query)
        nodes = {row.id: {'label': row.label, 'type': row.type} for row in nodes_result}
        
        # Fetch Edges
        edges_query = text("""
            SELECT source_id, target_id, type, COUNT(*) as weight 
            FROM kg_edges 
            GROUP BY source_id, target_id, type 
            HAVING weight >= :min_weight
        """)
        edges_result = session.execute(edges_query, {"min_weight": min_weight})
        edges = list(edges_result)

    # --- MODE: AUTHORS (Invisible College) ---
    elif mode == 'authors':
        # Fetch Authors as Nodes
        nodes_query = text("SELECT id, name FROM authors")
        nodes_result = session.execute(nodes_query)
        nodes = {row.id: {'label': row.name, 'type': 'author'} for row in nodes_result}
        
        # Fetch Co-authorship Edges (Self-join on paper_authors)
        edges_query = text("""
            SELECT a1.author_id as source_id, a2.author_id as target_id, 'co_authored' as type, COUNT(*) as weight
            FROM paper_authors a1
            JOIN paper_authors a2 ON a1.paper_id = a2.paper_id
            WHERE a1.author_id < a2.author_id
            GROUP BY a1.author_id, a2.author_id
            HAVING weight >= :min_weight
        """)
        edges_result = session.execute(edges_query, {"min_weight": min_weight})
        edges = list(edges_result)

    # --- MODE: CONCEPTS (Topic Landscape) ---
    elif mode == 'concepts' or mode == 'river':
        # Fetch Concepts as Nodes
        nodes_query = text("SELECT id, name FROM concepts")
        nodes_result = session.execute(nodes_query)
        nodes = {row.id: {'label': row.name, 'type': 'concept'} for row in nodes_result}
        
        # Fetch Co-occurrence Edges
        edges_query = text("""
            SELECT c1.concept_id as source_id, c2.concept_id as target_id, 'co_occurs' as type, COUNT(*) as weight
            FROM paper_concepts c1
            JOIN paper_concepts c2 ON c1.paper_id = c2.paper_id
            WHERE c1.concept_id < c2.concept_id
            GROUP BY c1.concept_id, c2.concept_id
            HAVING weight >= :min_weight
        """)
        edges_result = session.execute(edges_query, {"min_weight": min_weight})
        edges = list(edges_result)
        
        # River Mode: Add Time Intervals to Nodes
        if mode == 'river':
            # Find the first year each concept appeared
            time_query = text("""
                SELECT pc.concept_id, MIN(p.year) as start_year
                FROM paper_concepts pc
                JOIN papers p ON pc.paper_id = p.id
                GROUP BY pc.concept_id
            """)
            time_result = session.execute(time_query)
            for row in time_result:
                if row.concept_id in nodes and row.start_year:
                    nodes[row.concept_id]['start'] = str(row.start_year)

    # --- MODE: SIMILARITY (Paper Similarity) ---
    elif mode == 'similarity':
        # Fetch Papers as Nodes
        nodes_query = text("SELECT id, title, year FROM papers")
        nodes_result = session.execute(nodes_query)
        nodes = {row.id: {'label': f"{row.title[:30]}... ({row.year})", 'type': 'paper'} for row in nodes_result}
        
        # Calculate Jaccard Similarity based on shared concepts
        # This is expensive in SQL, so we do it in Python for small corpora
        # 1. Get concepts for each paper
        paper_concepts = {}
        pc_query = text("SELECT paper_id, concept_id FROM paper_concepts")
        for row in session.execute(pc_query):
            if row.paper_id not in paper_concepts:
                paper_concepts[row.paper_id] = set()
            paper_concepts[row.paper_id].add(row.concept_id)
        
        # 2. Compare all pairs (O(N^2) - be careful with large corpora)
        # Only compare if they share at least one concept
        paper_ids = list(paper_concepts.keys())
        import itertools
        
        for p1, p2 in itertools.combinations(paper_ids, 2):
            c1 = paper_concepts[p1]
            c2 = paper_concepts[p2]
            intersection = len(c1.intersection(c2))
            union = len(c1.union(c2))
            
            if union > 0:
                jaccard = intersection / union
                # Scale Jaccard (0-1) to integer weight for GEXF (e.g. * 10)
                # Threshold: e.g. 0.1 threshold means > 0.1 similarity
                if jaccard >= threshold:
                    edges.append({
                        'source_id': p1,
                        'target_id': p2,
                        'type': 'similar_to',
                        'weight': int(jaccard * 10)
                    })

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return nodes, edges


def export_to_gexf(corpus_path: str, output_path: str, threshold: float = 0.1, mode: str = 'semantic'):
    """
    Export the Knowledge Graph to GEXF format.
    """
    corpus_path = Path(corpus_path).resolve()
    output_path = Path(output_path).resolve()
    
    logger.info(f"Exporting {mode} graph to {output_path} (threshold={threshold})")
    
    with get_db_session(corpus_path) as session:
        nodes, edges = _get_graph_data(session, mode, threshold)
        
        logger.info(f"Found {len(edges)} edges meeting threshold.")
        
        # Filter Nodes (Only keep nodes that have edges)
        active_node_ids = set()
        for edge in edges:
            # Handle both object (SQLAlchemy) and dict (Python) edge formats
            sid = edge.source_id if hasattr(edge, 'source_id') else edge['source_id']
            tid = edge.target_id if hasattr(edge, 'target_id') else edge['target_id']
            active_node_ids.add(sid)
            active_node_ids.add(tid)
            
        logger.info(f"Found {len(active_node_ids)} active nodes.")
        
        # Build GEXF XML
        gexf = ET.Element('gexf', {
            'xmlns': 'http://www.gexf.net/1.2draft',
            'version': '1.2'
        })
        
        # <meta>
        meta = ET.SubElement(gexf, 'meta')
        ET.SubElement(meta, 'creator').text = "Literature Mapper"
        ET.SubElement(meta, 'description').text = f"{mode.title()} Graph (Threshold: {threshold})"
        
        # <graph mode="static" defaultedgetype="undirected">
        edge_type = 'directed' if mode == 'semantic' else 'undirected'
        graph = ET.SubElement(gexf, 'graph', {
            'mode': 'static',
            'defaultedgetype': edge_type
        })
        
        # <attributes class="node">
        attributes = ET.SubElement(graph, 'attributes', {'class': 'node'})
        ET.SubElement(attributes, 'attribute', {'id': '0', 'title': 'type', 'type': 'string'})
        
        if mode == 'river':
             ET.SubElement(attributes, 'attribute', {'id': '1', 'title': 'start', 'type': 'integer'})
        
        # <nodes>
        nodes_elem = ET.SubElement(graph, 'nodes')
        for node_id in active_node_ids:
            if node_id not in nodes:
                continue 
                
            node_data = nodes[node_id]
            node_elem = ET.SubElement(nodes_elem, 'node', {
                'id': str(node_id),
                'label': node_data['label']
            })
            
            attvalues = ET.SubElement(node_elem, 'attvalues')
            ET.SubElement(attvalues, 'attvalue', {'for': '0', 'value': node_data['type']})
            
            if mode == 'river' and 'start' in node_data:
                ET.SubElement(attvalues, 'attvalue', {'for': '1', 'value': node_data['start']})
            
        # <edges>
        edges_elem = ET.SubElement(graph, 'edges')
        for i, edge in enumerate(edges):
            sid = edge.source_id if hasattr(edge, 'source_id') else edge['source_id']
            tid = edge.target_id if hasattr(edge, 'target_id') else edge['target_id']
            etype = edge.type if hasattr(edge, 'type') else edge['type']
            eweight = edge.weight if hasattr(edge, 'weight') else edge['weight']

            ET.SubElement(edges_elem, 'edge', {
                'id': str(i),
                'source': str(sid),
                'target': str(tid),
                'label': etype,
                'weight': str(eweight)
            })
            
        # Write to file
        xml_str = minidom.parseString(ET.tostring(gexf)).toprettyxml(indent="  ")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
            
        logger.info(f"Successfully wrote GEXF to {output_path}")




