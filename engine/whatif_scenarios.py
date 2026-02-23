"""
Stores and evaluates 4 types of what-if scenarios with side-by-side comparison.
"""

import pandas as pd
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import networkx as nx
from data_loader import DataLoader
from engine.dag_builder import build_dag
from engine.ripple_engine import RippleEngine


SCENARIO_TYPES = ["delay", "resource_boost", "issue_resolved", "parallelize"]


class WhatIfScenarioEngine:
    """
    Manages scenarios and evaluates their impact.
    """

    def __init__(self, project_id: str, loader: Optional[DataLoader] = None,
                 reference_date: Optional[datetime] = None):
        self.project_id = project_id
        self.loader = loader or DataLoader()
        self.reference_date = reference_date or datetime(2024, 6, 1)
        self.G = build_dag(project_id, loader=self.loader)
        self.ripple = RippleEngine(self.G, loader=self.loader)
        self._scenarios: List[Dict] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario Builders
    # ──────────────────────────────────────────────────────────────────────────

    def scenario_delay(self, activity_id: str, delay_days: int,
                       description: str = "") -> Dict:
        """What-if: Activity X is delayed by N days."""
        result = self.ripple.propagate_delay(
            activity_id, delay_days, reference_date=self.reference_date
        )
        # Cost impact: rough estimate from BOQ
        boq = self.loader.get_project_boq(self.project_id)
        daily_rate = (boq["total_cost"].sum() / 120) if not boq.empty else 50000
        cost_impact = daily_rate * result["total_project_delay_days"]

        scenario = {
            "scenario_id": str(uuid.uuid4())[:8],
            "type": "delay",
            "description": description or f"Activity {activity_id} delayed by {delay_days} days",
            "modified_activities": [activity_id],
            "delta_input": delay_days,
            "num_activities_affected": result["num_activities_affected"],
            "original_project_end": result["original_project_end"],
            "new_project_end": result["new_project_end"],
            "total_project_delay_days": result["total_project_delay_days"],
            "days_saved": -result["total_project_delay_days"],
            "cost_impact_inr": round(cost_impact, 0),
            "cascade_table": result["cascade_table"],
        }
        self._scenarios.append(scenario)
        return scenario

    def scenario_resource_boost(self, activity_id: str, duration_reduction_pct: float = 25.0,
                                description: str = "") -> Dict:
        """
        What-if: Add resources to activity Y → reduce its duration by X%.
        Propagates the earlier completion to all downstream.
        """
        act_data = dict(self.G.nodes.get(activity_id, {}))
        planned_dur = float(act_data.get("planned_duration_days", 14) or 14)
        reduction_days = int(planned_dur * duration_reduction_pct / 100)

        # A negative delay = finishing early
        result = self.ripple.propagate_delay(
            activity_id, -reduction_days, reference_date=self.reference_date
        )
        boq = self.loader.get_project_boq(self.project_id)
        daily_rate = (boq["total_cost"].sum() / 120) if not boq.empty else 50000
        overtime_premium = 1.4  # 40% overtime/extra crew premium
        extra_cost = daily_rate * reduction_days * overtime_premium

        scenario = {
            "scenario_id": str(uuid.uuid4())[:8],
            "type": "resource_boost",
            "description": description or f"Add resources to {activity_id} → reduce by {duration_reduction_pct:.0f}%",
            "modified_activities": [activity_id],
            "delta_input": -reduction_days,
            "num_activities_affected": result["num_activities_affected"],
            "original_project_end": result["original_project_end"],
            "new_project_end": result["new_project_end"],
            "total_project_delay_days": result["total_project_delay_days"],
            "days_saved": -result["total_project_delay_days"],
            "cost_impact_inr": round(extra_cost, 0),
            "cascade_table": result["cascade_table"],
        }
        self._scenarios.append(scenario)
        return scenario

    def scenario_issue_resolved(self, issue_id: str, description: str = "") -> Dict:
        """
        What-if: Issue Z is resolved → estimate delay days saved.
        """
        issues = self.loader.issues
        issue = issues[issues["id"] == issue_id]
        if issue.empty:
            return {"error": f"Issue {issue_id} not found"}

        row = issue.iloc[0]
        activity_id = str(row.get("activity_id", ""))
        delay_impact = float(row.get("delay_impact_days", 0) or 0)

        # Negative delta = resolving the issue saves those days
        result = self.ripple.propagate_delay(
            activity_id, -int(delay_impact), reference_date=self.reference_date
        )

        scenario = {
            "scenario_id": str(uuid.uuid4())[:8],
            "type": "issue_resolved",
            "description": description or f"Issue {issue_id} resolved → saves {delay_impact:.0f} days",
            "modified_activities": [activity_id],
            "delta_input": -int(delay_impact),
            "num_activities_affected": result["num_activities_affected"],
            "original_project_end": result["original_project_end"],
            "new_project_end": result["new_project_end"],
            "total_project_delay_days": result["total_project_delay_days"],
            "days_saved": -result["total_project_delay_days"],
            "cost_impact_inr": 0,
            "cascade_table": result["cascade_table"],
        }
        self._scenarios.append(scenario)
        return scenario

    def scenario_parallelize(self, act_a_id: str, act_b_id: str,
                             description: str = "") -> Dict:
        """
        What-if: Run A and B concurrently (remove A→B dependency).
        B's start moves to A's start, saving B's original waiting time.
        """
        G_copy = self.G.copy()
        days_saved = 0

        if G_copy.has_edge(act_a_id, act_b_id):
            G_copy.remove_edge(act_a_id, act_b_id)
            # B's new start = same as A's start
            a_data = dict(self.G.nodes.get(act_a_id, {}))
            b_data = dict(self.G.nodes.get(act_b_id, {}))
            a_start = b_data.get("planned_start_date") or b_data.get("actual_start_date")
            b_orig_start = b_data.get("planned_start_date")
            b_planned_dur = float(b_data.get("planned_duration_days", 14) or 14)
            a_planned_dur = float(a_data.get("planned_duration_days", 14) or 14)

            if a_start and b_orig_start:
                import pandas as _pd
                a_s = _pd.Timestamp(a_start) if not _pd.isna(a_start) else None
                b_s = _pd.Timestamp(b_orig_start) if not _pd.isna(b_orig_start) else None
                if a_s and b_s:
                    days_saved = max(0, int((b_s - a_s).days))

        # Ripple from a with -days_saved on B
        ripple_b = RippleEngine(G_copy, loader=self.loader)
        result = ripple_b.propagate_delay(
            act_b_id, -days_saved, reference_date=self.reference_date
        )

        scenario = {
            "scenario_id": str(uuid.uuid4())[:8],
            "type": "parallelize",
            "description": description or f"Run {act_a_id} and {act_b_id} in parallel",
            "modified_activities": [act_a_id, act_b_id],
            "delta_input": -days_saved,
            "num_activities_affected": result["num_activities_affected"],
            "original_project_end": result["original_project_end"],
            "new_project_end": result["new_project_end"],
            "total_project_delay_days": result["total_project_delay_days"],
            "days_saved": days_saved,
            "cost_impact_inr": 0,
            "cascade_table": result["cascade_table"],
        }
        self._scenarios.append(scenario)
        return scenario

    # ──────────────────────────────────────────────────────────────────────────
    # Comparison
    # ──────────────────────────────────────────────────────────────────────────

    def get_scenario_comparison(self) -> pd.DataFrame:
        """Return all stored scenarios as a comparison table (no cascade_table column)."""
        if not self._scenarios:
            return pd.DataFrame()
        rows = []
        for s in self._scenarios:
            row = {k: v for k, v in s.items() if k != "cascade_table"}
            rows.append(row)
        return pd.DataFrame(rows)

    def clear_scenarios(self):
        self._scenarios = []

    def get_scenarios(self) -> List[Dict]:
        return self._scenarios
