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

def _get_graph_data(session, mode: str, threshold: float, before_year: int = None, after_year: int = None):
    """
    Helper to fetch nodes and edges based on mode and threshold.
    
    Args:
        session: Database session
        mode: Graph mode (semantic, authors, concepts, river, similarity)
        threshold: Minimum edge weight threshold
        before_year: Only include data from papers published before this year
        after_year: Only include data from papers published on or after this year
        
    Returns: (nodes_dict, edges_list)
    """
    # Build year filter clause for SQL queries
    year_conditions = []
    year_params = {}
    if before_year is not None:
        year_conditions.append("p.year < :before_year")
        year_params["before_year"] = before_year
    if after_year is not None:
        year_conditions.append("p.year >= :after_year")
        year_params["after_year"] = after_year
    year_filter = " AND ".join(year_conditions) if year_conditions else "1=1"
    
    # 1. Calculate Threshold (based on filtered paper count)
    if year_conditions:
        count_query = text(f"SELECT COUNT(*) FROM papers p WHERE {year_filter}")
        result = session.execute(count_query, year_params)
    else:
        result = session.execute(text("SELECT COUNT(*) FROM papers"))
    total_papers = result.scalar()
    
    if total_papers == 0:
        logger.warning("No papers in corpus (or matching year filter).")
        min_weight = 1
    else:
        min_weight = max(1, int(total_papers * threshold))
        
    year_desc = ""
    if before_year or after_year:
        year_desc = f" (years: {after_year or '...'}-{before_year or '...'})"
    logger.info(f"Total papers: {total_papers}{year_desc}. Minimum edge weight: {min_weight}")
    
    nodes = {}
    edges = []
    
    # --- MODE: SEMANTIC (Default) ---
    if mode == 'semantic':
        # Fetch Nodes (all nodes, edges will determine which are active)
        nodes_query = text("SELECT id, label, type FROM kg_nodes")
        nodes_result = session.execute(nodes_query)
        nodes = {row.id: {'label': row.label, 'type': row.type} for row in nodes_result}
        
        # Fetch Edges (filtered by source paper year)
        if year_conditions:
            edges_query = text(f"""
                SELECT e.source_id, e.target_id, e.type, COUNT(*) as weight 
                FROM kg_edges e
                JOIN papers p ON e.source_paper_id = p.id
                WHERE {year_filter}
                GROUP BY e.source_id, e.target_id, e.type 
                HAVING weight >= :min_weight
            """)
            edges_result = session.execute(edges_query, {"min_weight": min_weight, **year_params})
        else:
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
        # 1. Fetch Authors and Map to Canonical Names
        # We want to merge Author A and Author B if they map to the same canonical name.
        nodes_query = text("SELECT id, name, canonical_name FROM authors")
        nodes_result = session.execute(nodes_query)
        
        id_to_label = {}
        for row in nodes_result:
            # Use canonical_name if present, else name
            label = row.canonical_name if row.canonical_name else row.name
            id_to_label[row.id] = label
            
            # Initialize node entry (keyed by label to ensure uniqueness)
            if label not in nodes:
                nodes[label] = {'label': label, 'type': 'author'}

        # 2. Fetch Raw Edges (Pre-aggregated by DB ID)
        # We will re-aggregate using canonical labels in Python
        if year_conditions:
            edges_query = text(f"""
                SELECT a1.author_id as source_id, a2.author_id as target_id, 'co_authored' as type, COUNT(*) as weight
                FROM paper_authors a1
                JOIN paper_authors a2 ON a1.paper_id = a2.paper_id
                JOIN papers p ON a1.paper_id = p.id
                WHERE a1.author_id < a2.author_id AND {year_filter}
                GROUP BY a1.author_id, a2.author_id
            """)
            edges_result = session.execute(edges_query, year_params)
        else:
            edges_query = text("""
                SELECT a1.author_id as source_id, a2.author_id as target_id, 'co_authored' as type, COUNT(*) as weight
                FROM paper_authors a1
                JOIN paper_authors a2 ON a1.paper_id = a2.paper_id
                WHERE a1.author_id < a2.author_id
                GROUP BY a1.author_id, a2.author_id
            """)
            edges_result = session.execute(edges_query)
            
        # 3. Aggregate Edges by Canonical Label
        edge_map = {} # (source_label, target_label) -> weight
        
        for row in edges_result:
            u_label = id_to_label.get(row.source_id)
            v_label = id_to_label.get(row.target_id)
            
            if not u_label or not v_label:
                continue
            if u_label == v_label:
                continue # Skip self-loops caused by merging
                
            # Sort to ensure undirected uniqueness
            if u_label > v_label:
                u_label, v_label = v_label, u_label
                
            pair = (u_label, v_label)
            if pair not in edge_map:
                edge_map[pair] = 0
            edge_map[pair] += row.weight
            
        # 4. Filter by Threshold and Format
        for (u, v), w in edge_map.items():
            if w >= min_weight:
                edges.append({
                    'source_id': u, # Keys in 'nodes' are now labels
                    'target_id': v,
                    'type': 'co_authored',
                    'weight': w
                })

    # --- MODE: CONCEPTS (Topic Landscape) ---
    elif mode == 'concepts' or mode == 'river':
        # 1. Fetch Concepts and Map to Canonical Names
        nodes_query = text("SELECT id, name, canonical_name FROM concepts")
        nodes_result = session.execute(nodes_query)
        
        id_to_label = {}
        for row in nodes_result:
            label = row.canonical_name if row.canonical_name else row.name
            id_to_label[row.id] = label
            if label not in nodes:
                nodes[label] = {'label': label, 'type': 'concept'}
        
        # 2. Fetch Raw Edges
        if year_conditions:
            edges_query = text(f"""
                SELECT c1.concept_id as source_id, c2.concept_id as target_id, 'co_occurs' as type, COUNT(*) as weight
                FROM paper_concepts c1
                JOIN paper_concepts c2 ON c1.paper_id = c2.paper_id
                JOIN papers p ON c1.paper_id = p.id
                WHERE c1.concept_id < c2.concept_id AND {year_filter}
                GROUP BY c1.concept_id, c2.concept_id
            """)
            edges_result = session.execute(edges_query, year_params)
        else:
            edges_query = text("""
                SELECT c1.concept_id as source_id, c2.concept_id as target_id, 'co_occurs' as type, COUNT(*) as weight
                FROM paper_concepts c1
                JOIN paper_concepts c2 ON c1.paper_id = c2.paper_id
                WHERE c1.concept_id < c2.concept_id
                GROUP BY c1.concept_id, c2.concept_id
            """)
            edges_result = session.execute(edges_query)
        
        # 3. Aggregate Edges
        edge_map = {}
        for row in edges_result:
            u_label = id_to_label.get(row.source_id)
            v_label = id_to_label.get(row.target_id)
            
            if not u_label or not v_label:
                continue
            if u_label == v_label:
                continue
                
            if u_label > v_label:
                u_label, v_label = v_label, u_label
                
            pair = (u_label, v_label)
            if pair not in edge_map:
                edge_map[pair] = 0
            edge_map[pair] += row.weight
            
        # 4. Filter by Threshold
        for (u, v), w in edge_map.items():
            if w >= min_weight:
                edges.append({
                    'source_id': u,
                    'target_id': v,
                    'type': 'co_occurs',
                    'weight': w
                })
        
        # River Mode: Add Time Intervals to Nodes (respecting year filter)
        if mode == 'river':
            if year_conditions:
                time_query = text(f"""
                    SELECT pc.concept_id, MIN(p.year) as start_year
                    FROM paper_concepts pc
                    JOIN papers p ON pc.paper_id = p.id
                    WHERE {year_filter}
                    GROUP BY pc.concept_id
                """)
                time_result = session.execute(time_query, year_params)
            else:
                time_query = text("""
                    SELECT pc.concept_id, MIN(p.year) as start_year
                    FROM paper_concepts pc
                    JOIN papers p ON pc.paper_id = p.id
                    GROUP BY pc.concept_id
                """)
                time_result = session.execute(time_query)
            for row in time_result:
                label = id_to_label.get(row.concept_id)
                if label in nodes and row.start_year:
                    # Update if earlier year found for this canonical concept
                    current_start = nodes[label].get('start')
                    if current_start is None or row.start_year < int(current_start):
                        nodes[label]['start'] = str(row.start_year)

    # --- MODE: SIMILARITY (Paper Similarity) ---
    elif mode == 'similarity':
        # Fetch Papers as Nodes (filtered by year)
        if year_conditions:
            nodes_query = text(f"SELECT id, title, year FROM papers p WHERE {year_filter}")
            nodes_result = session.execute(nodes_query, year_params)
        else:
            nodes_query = text("SELECT id, title, year FROM papers")
            nodes_result = session.execute(nodes_query)
        nodes = {row.id: {'label': f"{row.title[:30]}... ({row.year})", 'type': 'paper'} for row in nodes_result}
        
        # Calculate Jaccard Similarity based on shared concepts
        # This is expensive in SQL, so we do it in Python for small corpora
        # 1. Get concepts for each paper (only for papers in nodes)
        paper_concepts = {}
        if year_conditions:
            pc_query = text(f"""
                SELECT pc.paper_id, pc.concept_id 
                FROM paper_concepts pc
                JOIN papers p ON pc.paper_id = p.id
                WHERE {year_filter}
            """)
            for row in session.execute(pc_query, year_params):
                if row.paper_id not in paper_concepts:
                    paper_concepts[row.paper_id] = set()
                paper_concepts[row.paper_id].add(row.concept_id)
        else:
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


def export_to_gexf(
    corpus_path: str, 
    output_path: str, 
    threshold: float = 0.1, 
    mode: str = 'semantic',
    before_year: int = None,
    after_year: int = None,
):
    """
    Export the Knowledge Graph to GEXF format.
    
    Args:
        corpus_path: Path to corpus directory
        output_path: Path for output GEXF file
        threshold: Minimum edge weight threshold
        mode: Graph mode (semantic, authors, concepts, river, similarity)
        before_year: Only include data from papers published before this year
        after_year: Only include data from papers published on or after this year
    """
    corpus_path = Path(corpus_path).resolve()
    output_path = Path(output_path).resolve()
    
    year_desc = ""
    if before_year or after_year:
        year_desc = f", years: {after_year or '...'}-{before_year or '...'}"
    logger.info(f"Exporting {mode} graph to {output_path} (threshold={threshold}{year_desc})")
    
    with get_db_session(corpus_path) as session:
        nodes, edges = _get_graph_data(session, mode, threshold, before_year, after_year)
        
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
        attributes = ET.SubElement(graph, 'attributes', {'class': 'node', 'mode': 'static'})
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




