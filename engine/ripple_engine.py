"""
Propagates delays through the dependency DAG using BFS.
Computes cascade impact on all downstream activities.
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from data_loader import DataLoader
from engine.dag_builder import build_dag, get_descendants


class RippleEngine:
    """
    Given a delay delta on one activity, propagate the effect to all
    downstream activities and compute the new project end date.
    """

    def __init__(self, G: nx.DiGraph, loader: Optional[DataLoader] = None):
        self.G = G
        self.loader = loader or DataLoader()

    def _get_activity(self, activity_id: str) -> Dict:
        """Return node attribute dict for an activity."""
        if activity_id in self.G.nodes:
            return dict(self.G.nodes[activity_id])
        return {}

    def _to_ts(self, val) -> Optional[pd.Timestamp]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            return pd.Timestamp(val)
        except Exception:
            return None

    def propagate_delay(self, affected_activity_id: str, delta_days: int,
                        reference_date: Optional[datetime] = None) -> Dict:
        """
        Simulate delaying `affected_activity_id` by `delta_days` and compute
        the cascade effect on all downstream activities.

        Returns
        -------
        dict with:
            - affected_activity_id
            - delta_days
            - cascade_table : DataFrame of impacted activities
            - new_project_end : pd.Timestamp or None
            - original_project_end : pd.Timestamp or None
            - total_project_delay_days : int
        """
        if reference_date is None:
            reference_date = datetime(2024, 6, 1)
        today = pd.Timestamp(reference_date)

        # Get all descendants (will be affected)
        downstream = get_descendants(self.G, affected_activity_id)

        # Compute original project end (max of all leaf node ends)
        leaf_nodes = [n for n in self.G.nodes if self.G.out_degree(n) == 0]
        original_project_end = None
        for node in leaf_nodes:
            end = self._to_ts(self.G.nodes[node].get("planned_end_date"))
            if end and (original_project_end is None or end > original_project_end):
                original_project_end = end

        # Compute shifted dates using topological traversal
        # shifted_ends dict: activity_id → new_end_date
        shifted_ends = {}
        shifted_starts = {}

        # The directly affected activity shifts by delta_days in its end
        act_data = self._get_activity(affected_activity_id)
        orig_end = self._to_ts(
            act_data.get("forecasted_end_date") or act_data.get("planned_end_date")
        )
        if orig_end is None:
            orig_end = self._to_ts(act_data.get("planned_end_date"))
        if orig_end:
            shifted_ends[affected_activity_id] = orig_end + timedelta(days=delta_days)
        else:
            shifted_ends[affected_activity_id] = today + timedelta(days=delta_days)

        # BFS propagation
        try:
            topo_order = list(nx.topological_sort(self.G))
        except Exception:
            topo_order = [affected_activity_id] + downstream

        cascade_rows = []

        for node_id in topo_order:
            if node_id not in downstream and node_id != affected_activity_id:
                continue

            node_data = self._get_activity(node_id)
            original_start = self._to_ts(
                node_data.get("planned_start_date")
            )
            original_end = self._to_ts(
                node_data.get("forecasted_end_date") or node_data.get("planned_end_date")
            )
            planned_dur = node_data.get(
                "planned_duration_days",
                (original_end - original_start).days if original_start and original_end else 14
            )

            # New start = max(original planned start, max of all predecessor new ends)
            pred_ends = []
            for pred_id in self.G.predecessors(node_id):
                if pred_id in shifted_ends:
                    pred_ends.append(shifted_ends[pred_id])
                else:
                    pred_data = self._get_activity(pred_id)
                    pred_end = self._to_ts(
                        pred_data.get("forecasted_end_date") or pred_data.get("planned_end_date")
                    )
                    if pred_end:
                        pred_ends.append(pred_end)

            if pred_ends:
                new_start = max(pred_ends)
                if original_start:
                    new_start = max(new_start, original_start)
            else:
                new_start = original_start or today

            new_end = new_start + timedelta(days=int(planned_dur or 14))
            shifted_starts[node_id] = new_start
            shifted_ends[node_id] = new_end

            cascade_delay = 0
            if original_end:
                cascade_delay = (new_end - original_end).days

            if node_id != affected_activity_id:
                cascade_rows.append({
                    "activity_id": node_id,
                    "activity_name": node_data.get("name", node_id),
                    "original_start": original_start,
                    "original_end": original_end,
                    "new_start": new_start,
                    "new_end": new_end,
                    "cascade_delay_days": cascade_delay,
                    "has_open_issues": node_data.get("issue_count", 0) > 0,
                })

        # New project end
        new_project_end = None
        for node in leaf_nodes:
            end = shifted_ends.get(node) or self._to_ts(
                self.G.nodes[node].get("planned_end_date")
            )
            if end and (new_project_end is None or end > new_project_end):
                new_project_end = end

        total_project_delay = 0
        if original_project_end and new_project_end:
            total_project_delay = (new_project_end - original_project_end).days

        cascade_df = pd.DataFrame(cascade_rows) if cascade_rows else pd.DataFrame(
            columns=["activity_id", "activity_name", "original_start", "original_end",
                     "new_start", "new_end", "cascade_delay_days", "has_open_issues"]
        )

        return {
            "affected_activity_id": affected_activity_id,
            "delta_days": delta_days,
            "cascade_table": cascade_df,
            "new_project_end": new_project_end,
            "original_project_end": original_project_end,
            "total_project_delay_days": total_project_delay,
            "num_activities_affected": len(cascade_rows),
        }

    def get_high_impact_activities(self, top_n: int = 5) -> pd.DataFrame:
        """
        Find activities whose 1-day delay causes the most downstream impact.
        """
        rows = []
        for node_id in self.G.nodes:
            descendants = get_descendants(self.G, node_id)
            rows.append({
                "activity_id": node_id,
                "activity_name": self.G.nodes[node_id].get("name", node_id),
                "downstream_count": len(descendants),
                "status": self.G.nodes[node_id].get("status", "unknown"),
            })
        df = pd.DataFrame(rows).sort_values("downstream_count", ascending=False)
        return df.head(top_n)
