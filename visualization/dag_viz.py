

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from typing import Optional, List
from data_loader import DataLoader
from engine.dag_builder import build_dag


def build_dag_figure(project_id: str,
                     loader: Optional[DataLoader] = None,
                     critical_path_ids: Optional[List[str]] = None,
                     predictions_df: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Build an interactive Plotly DAG figure with delay heatmap.
    Nodes colored by schedule_variance (green=on-time → red=very late).
    Critical path edges highlighted in red.
    """
    if loader is None:
        loader = DataLoader()
    if critical_path_ids is None:
        critical_path_ids = []

    G = build_dag(project_id, loader=loader)
    if G.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="No dependency data", showarrow=False)

    # Layout
    try:
        pos = nx.drawing.nx_pydot.pydot_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.planar_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42)

    acts = loader.get_project_activities(project_id)
    act_map = {str(row["id"]): row.to_dict() for _, row in acts.iterrows()}

    pred_map = {}
    if predictions_df is not None and not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            pred_map[str(row.get("activity_id", ""))] = row.to_dict()

    # Delay coloring (schedule_variance_days): green=0 → red=14+
    def delay_color(act_id: str) -> str:
        act = act_map.get(act_id, {})
        var_raw = act.get("schedule_variance_days", 0)
        var = 0
        try:
            var = float(var_raw or 0)
        except Exception:
            var = 0

        if var <= 0:
            return "#22c55e"   # green
        elif var <= 3:
            return "#86efac"   # light green
        elif var <= 7:
            return "#fbbf24"   # amber
        elif var <= 14:
            return "#f97316"   # orange
        else:
            return "#ef4444"   # red

    # Build edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos.get(edge[0], (0, 0))
        x1, y1 = pos.get(edge[1], (0, 0))
        is_critical = edge[0] in critical_path_ids and edge[1] in critical_path_ids
        color = "#ef4444" if is_critical else "#64748b"
        width = 3 if is_critical else 1.5

        edge_traces.append(go.Scatter(
            x=[x0, (x0 + x1) / 2, x1, None],
            y=[y0, (y0 + y1) / 2, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="none",
            showlegend=False,
        ))

    # Build node trace
    node_x, node_y, node_colors, node_text, node_hover = [], [], [], [], []
    for node in G.nodes():
        x, y = pos.get(node, (0, 0))
        node_x.append(x)
        node_y.append(y)
        node_colors.append(delay_color(node))
        act = act_map.get(node, {})
        name = act.get("name", node)[:20]
        prog = float(act.get("progress", 0) or 0)
        var = float(act.get("schedule_variance_days", 0) or 0)
        border = "★ " if node in critical_path_ids else ""
        node_text.append(f"{border}{name}")
        node_hover.append(
            f"<b>{act.get('name', node)}</b><br>"
            f"Status: {act.get('status', '-')}<br>"
            f"Progress: {prog:.0f}%<br>"
            f"Schedule Variance: {var:+.0f} days<br>"
            f"ID: {node}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=22,
            color=node_colors,
            line=dict(color="#1e293b", width=2),
            symbol="circle",
        ),
        showlegend=False,
    )

    # Legend items
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color=c), name=lbl)
        for lbl, c in [
            ("On time (0 days)", "#22c55e"),
            ("Slightly late (1-3d)", "#86efac"),
            ("Moderately late (4-7d)", "#fbbf24"),
            ("Late (8-14d)", "#f97316"),
            ("Very late (>14d)", "#ef4444"),
            ("★ Critical path", "#ef4444"),
        ]
    ]

    fig = go.Figure(data=edge_traces + [node_trace] + legend_traces)
    fig.update_layout(
        title=dict(text=f"🕸️ Dependency DAG — {project_id}", font=dict(size=18)),
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=550,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
        margin=dict(t=60, l=20, r=20, b=20),
    )
    return fig
