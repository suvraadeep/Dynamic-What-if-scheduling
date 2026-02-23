"""
Builds a directed acyclic graph (DAG) of activity dependencies using networkx.
"""

import pandas as pd
import networkx as nx
from typing import Optional
from data_loader import DataLoader


def build_dag(project_id: str, loader: Optional[DataLoader] = None) -> nx.DiGraph:
    """
    Build a dependency DAG for the given project.

    Nodes : activity IDs (strings)
    Edges : (predecessor_id → successor_id)  [depends_on direction]
    Node attributes: all activity fields
    """
    if loader is None:
        loader = DataLoader()

    acts = loader.get_project_activities(project_id)
    if acts.empty:
        return nx.DiGraph()

    G = nx.DiGraph()

    # Add all activities as nodes
    for _, row in acts.iterrows():
        G.add_node(row["id"], **row.to_dict())

    # Add dependency edges
    for _, row in acts.iterrows():
        dep = row.get("depends_on")
        if dep and not (isinstance(dep, float)) and str(dep).strip():
            dep = str(dep).strip()
            if dep in G.nodes:
                # Edge: dep (predecessor) → row["id"] (successor)
                G.add_edge(dep, row["id"])

    # Validate: warn if cycle found (shouldn't happen in real data)
    if not nx.is_directed_acyclic_graph(G):
        print(f"⚠️  Cycle detected in DAG for {project_id}!")

    return G


def get_topological_order(G: nx.DiGraph) -> list:
    """Return activities in topological order (starts before ends)."""
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return list(G.nodes)


def get_descendants(G: nx.DiGraph, activity_id: str) -> list:
    """Return all downstream activities (BFS)."""
    try:
        return list(nx.descendants(G, activity_id))
    except nx.NodeNotFound:
        return []


def get_ancestors(G: nx.DiGraph, activity_id: str) -> list:
    """Return all upstream activities."""
    try:
        return list(nx.ancestors(G, activity_id))
    except nx.NodeNotFound:
        return []


def get_activity_depth(G: nx.DiGraph, activity_id: str) -> int:
    """Depth of activity in the DAG from source nodes."""
    try:
        sources = [n for n in G.nodes if G.in_degree(n) == 0]
        max_depth = 0
        for src in sources:
            try:
                path_length = nx.shortest_path_length(G, src, activity_id)
                max_depth = max(max_depth, path_length)
            except nx.NetworkXNoPath:
                pass
        return max_depth
    except Exception:
        return 0
