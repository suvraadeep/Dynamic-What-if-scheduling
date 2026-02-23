

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional
from data_loader import DataLoader

STATUS_COLORS = {
    "completed": "#22c55e",       
    "in_progress": "#f59e0b",     
    "not_started": "#94a3b8",     
    "critical": "#ef4444",        
}

TIMELINE_COLORS = {
    "planned": "rgba(99,102,241,0.6)",     
    "actual": "rgba(34,197,94,0.7)",       
    "forecasted": "rgba(245,158,11,0.7)",  
}


def build_gantt(project_id: str,
                loader: Optional[DataLoader] = None,
                predictions_df: Optional[pd.DataFrame] = None,
                critical_path_ids: Optional[list] = None,
                today: Optional[datetime] = None) -> go.Figure:
    """
    Build a Plotly Gantt chart for the given project.

    Bars per activity (3 tracks):
    - Top: Planned (planned_start → planned_end)
    - Mid: Actual (actual_start → actual_end or today)
    - Bot: Forecasted (today → ensemble_end from predictions)
    """
    if loader is None:
        loader = DataLoader()
    if today is None:
        today = datetime(2024, 6, 1)
    if critical_path_ids is None:
        critical_path_ids = []

    today_ts = pd.Timestamp(today)
    acts = loader.get_project_activities(project_id)
    if acts.empty:
        return go.Figure().add_annotation(text="No activities found", showarrow=False)

    # Merge predictions
    pred_map = {}
    if predictions_df is not None and not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            pred_map[row.get("activity_id", "")] = row.to_dict()

    # Sort by planned start
    acts = acts.sort_values("planned_start_date", na_position="last")

    fig = go.Figure()
    y_labels = []
    legend_added = set()

    def add_bar(name, x0, x1, y_pos, color, showlegend=False):
        if pd.isna(x0) or pd.isna(x1) or x0 >= x1:
            return
        leg = name not in legend_added
        if leg:
            legend_added.add(name)
        fig.add_trace(go.Bar(
            x=[(x1 - x0).total_seconds() * 1000],
            base=[x0],
            y=[y_pos],
            orientation="h",
            name=name,
            marker_color=color,
            showlegend=showlegend or leg,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Start: {x0.strftime('%Y-%m-%d')}<br>"
                f"End: {x1.strftime('%Y-%m-%d')}<br>"
                f"Duration: {(x1-x0).days} days<extra></extra>"
            ),
            width=0.25,
        ))

    for idx, (_, row) in enumerate(acts.iterrows()):
        act_id = str(row["id"])
        act_name = row.get("name", act_id)
        status = str(row.get("status", "not_started"))
        is_critical = act_id in critical_path_ids

        # Y position: 3 tracks per activity
        base_y = idx * 3

        y_label = f"{'🔴 ' if is_critical else ''}{act_name[:35]}"
        y_labels.extend([
            f"{y_label} (planned)",
            f"{y_label} (actual)",
            f"{y_label} (forecast)",
        ])

        planned_s = pd.to_datetime(row.get("planned_start_date"), errors="coerce")
        planned_e = pd.to_datetime(row.get("planned_end_date"), errors="coerce")
        actual_s = pd.to_datetime(row.get("actual_start_date"), errors="coerce")
        actual_e = pd.to_datetime(row.get("actual_end_date"), errors="coerce")

        # Planned bar
        if not pd.isna(planned_s) and not pd.isna(planned_e):
            color = TIMELINE_COLORS["planned"]
            add_bar("Planned", planned_s, planned_e, base_y, color, showlegend=True)

        # Actual bar
        if not pd.isna(actual_s):
            act_end_display = actual_e if not pd.isna(actual_e) else today_ts
            color = STATUS_COLORS.get(status, STATUS_COLORS["not_started"])
            add_bar("Actual", actual_s, act_end_display, base_y + 1, color, showlegend=True)

        # Forecasted bar (from predictions)
        pred = pred_map.get(act_id, {})
        ens_end = pred.get("ensemble_end")
        if ens_end and status in ("in_progress", "not_started"):
            ens_ts = pd.Timestamp(ens_end)
            fc_start = today_ts
            if not pd.isna(actual_s):
                fc_start = actual_s if pd.isna(actual_e) else actual_e
            add_bar("Forecasted", fc_start, ens_ts, base_y + 2,
                    TIMELINE_COLORS["forecasted"], showlegend=True)

    # Today line
    fig.add_vline(
        x=today_ts.timestamp() * 1000,
        line_color="red", line_dash="dash", line_width=2,
        annotation_text="Today",
        annotation_position="top right",
    )

    n_acts = len(acts)
    fig.update_layout(
        title=dict(text=f"📅 Project Schedule — {project_id}", font=dict(size=18)),
        xaxis=dict(title="Date", type="date"),
        yaxis=dict(
            title="",
            tickvals=list(range(0, n_acts * 3, 3)),
            ticktext=[acts.iloc[i].get("name", "")[:35]
                      for i in range(min(n_acts, len(acts)))],
            autorange="reversed",
        ),
        height=max(400, n_acts * 55),
        barmode="overlay",
        legend=dict(orientation="h", y=1.05),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
    )
    return fig
