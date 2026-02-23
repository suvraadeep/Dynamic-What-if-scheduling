

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from data_loader import DataLoader
from engine.dag_builder import build_dag, get_topological_order

REFERENCE_DATE = datetime(2024, 6, 1)


class ScheduleOptimizer:
    """
    CPM calculator + rule-based optimization suggestions.
    """

    def __init__(self, project_id: str, loader: Optional[DataLoader] = None,
                 today: Optional[datetime] = None):
        self.project_id = project_id
        self.loader = loader or DataLoader()
        self.today = pd.Timestamp(today or REFERENCE_DATE)
        self.G = build_dag(project_id, loader=self.loader)
        self._cpm_results: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Critical Path Method (CPM)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_cpm(self) -> pd.DataFrame:
        """
        Compute ES, EF, LS, LF, and Float for every activity.
        Returns a DataFrame with one row per activity + CPM columns.
        """
        acts = self.loader.get_project_activities(self.project_id)
        if acts.empty:
            return pd.DataFrame()

        topo = get_topological_order(self.G)
        act_map = {row["id"]: row.to_dict() for _, row in acts.iterrows()}

        # Duration lookup
        def dur(act_id):
            d = act_map.get(act_id, {}).get("planned_duration_days", 14)
            return max(int(d or 14), 1)

        # --- Forward pass (Early Start, Early Finish) ---
        ES = {}
        EF = {}
        start_day = 0
        for node in topo:
            preds = list(self.G.predecessors(node))
            if not preds:
                ES[node] = 0
            else:
                ES[node] = max(EF.get(p, 0) for p in preds)
            EF[node] = ES[node] + dur(node)

        # Project duration = max EF across all leaf nodes
        project_duration = max(EF.values()) if EF else 0

        # --- Backward pass (Late Start, Late Finish, Float) ---
        LS = {}
        LF = {}
        for node in reversed(topo):
            succs = list(self.G.successors(node))
            if not succs:
                LF[node] = project_duration
            else:
                LF[node] = min(LS.get(s, project_duration) for s in succs)
            LS[node] = LF[node] - dur(node)

        # Float = LS - ES
        rows = []
        for node in topo:
            if node not in act_map:
                continue
            a = act_map[node]
            float_val = LS.get(node, 0) - ES.get(node, 0)
            is_critical = float_val <= 0

            rows.append({
                "activity_id": node,
                "activity_name": a.get("name", node),
                "status": a.get("status", "unknown"),
                "planned_duration_days": dur(node),
                "early_start_day": ES.get(node, 0),
                "early_finish_day": EF.get(node, 0),
                "late_start_day": LS.get(node, 0),
                "late_finish_day": LF.get(node, 0),
                "total_float_days": float_val,
                "is_critical_path": is_critical,
                "progress": float(a.get("progress", 0) or 0),
                "schedule_variance_days": int(a.get("schedule_variance_days", 0) or 0),
                "category": a.get("category", ""),
            })

        self._cpm_results = pd.DataFrame(rows)
        return self._cpm_results

    def get_critical_path(self) -> List[str]:
        """Return list of activity IDs on the critical path (float ≤ 0)."""
        if self._cpm_results is None:
            self.compute_cpm()
        if self._cpm_results is None or self._cpm_results.empty:
            return []
        return self._cpm_results[self._cpm_results["is_critical_path"]]["activity_id"].tolist()

    # ──────────────────────────────────────────────────────────────────────────
    # Rule-Based Suggestions
    # ──────────────────────────────────────────────────────────────────────────

    def generate_suggestions(self, predictions_df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Apply 6 rule-based optimization rules and return suggestion cards.
        """
        if self._cpm_results is None:
            self.compute_cpm()

        cpm = self._cpm_results
        if cpm is None or cpm.empty:
            return []

        acts = self.loader.get_project_activities(self.project_id)
        if acts.empty:
            return []

        all_issues = self.loader.get_activity_issues(project_id=self.project_id)
        if not all_issues.empty and "status" in all_issues.columns:
            open_issues = all_issues[all_issues["status"] == "open"]
        else:
            open_issues = pd.DataFrame()

        suggestions: List[Dict] = []

        act_map = {row["id"]: row.to_dict() for _, row in acts.iterrows()}
        cpm_map = {row["activity_id"]: row.to_dict() for _, row in cpm.iterrows()}

        # Merge predictions if available
        pred_map = {}
        if predictions_df is not None and not predictions_df.empty:
            for _, row in predictions_df.iterrows():
                pred_map[row.get("activity_id", "")] = row.to_dict()

        downstream_count = {n: len(list(nx.descendants(self.G, n))) for n in self.G.nodes}

        # Rule 1: Slow critical activity → increase crew
        for _, cpm_row in cpm.iterrows():
            if not cpm_row["is_critical_path"]:
                continue
            act_id = cpm_row["activity_id"]
            act = act_map.get(act_id, {})
            progress = float(act.get("progress", 0) or 0)
            if act.get("status") != "in_progress":
                continue
            elapsed = max(1, (self.today - pd.Timestamp(
                act.get("actual_start_date") or act.get("planned_start_date") or self.today
            )).days)
            actual_rate = progress / elapsed
            planned_dur = max(cpm_row["planned_duration_days"], 1)
            planned_rate = 100 / planned_dur
            if actual_rate < 0.5 * planned_rate:
                suggestions.append({
                    "type": "ACTION",
                    "priority": "🔴 CRITICAL",
                    "activity_id": act_id,
                    "activity_name": cpm_row["activity_name"],
                    "rule": "Slow Critical Activity",
                    "suggestion": (
                        f"**{cpm_row['activity_name']}** is on the critical path and running at "
                        f"{actual_rate:.1f}%/day vs planned {planned_rate:.1f}%/day. "
                        "→ **Increase crew size or shift to overtime.**"
                    ),
                    "estimated_savings_days": int(planned_dur * 0.2),
                })

        # Rule 2: High schedule variance + many downstream → escalate
        for _, cpm_row in cpm.iterrows():
            act_id = cpm_row["activity_id"]
            act = act_map.get(act_id, {})
            var = int(act.get("schedule_variance_days", 0) or 0)
            down = downstream_count.get(act_id, 0)
            if var > 5 and down > 3:
                suggestions.append({
                    "type": "ALERT",
                    "priority": "🔴 HIGH",
                    "activity_id": act_id,
                    "activity_name": cpm_row["activity_name"],
                    "rule": "High Impact Delay",
                    "suggestion": (
                        f"**{cpm_row['activity_name']}** is {var} days late and has "
                        f"{down} downstream activities. "
                        "→ **Escalate immediately — cascading delay risk.**"
                    ),
                    "estimated_savings_days": 0,
                })

        # Rule 3: Material delay issue on upcoming activity → pre-order
        for _, cpm_row in cpm.iterrows():
            act_id = cpm_row["activity_id"]
            if act_map.get(act_id, {}).get("status") == "not_started":
                act_issues = open_issues[open_issues["activity_id"] == act_id] if not open_issues.empty else pd.DataFrame()
                material_issues = act_issues[act_issues.get("category", pd.Series()) == "material_delay"] if not act_issues.empty else pd.DataFrame()
                if not material_issues.empty:
                    suggestions.append({
                        "type": "PREVENTIVE",
                        "priority": "🟡 MEDIUM",
                        "activity_id": act_id,
                        "activity_name": cpm_row["activity_name"],
                        "rule": "Material Delay Risk",
                        "suggestion": (
                            f"**{cpm_row['activity_name']}** (not started) has open material delay issues. "
                            "→ **Pre-order materials now to avoid blocking this activity.**"
                        ),
                        "estimated_savings_days": 3,
                    })

        # Rule 4: Two non-dependent activities → suggest parallel
        nodes = list(self.G.nodes)
        for i, a1 in enumerate(nodes):
            for a2 in nodes[i + 1:]:
                if (not self.G.has_edge(a1, a2) and not self.G.has_edge(a2, a1)
                        and not nx.has_path(self.G, a1, a2)
                        and not nx.has_path(self.G, a2, a1)):
                    a1_stat = act_map.get(a1, {}).get("status")
                    a2_stat = act_map.get(a2, {}).get("status")
                    if a1_stat == "not_started" and a2_stat == "not_started":
                        a1_name = cpm_map.get(a1, {}).get("activity_name", a1)
                        a2_name = cpm_map.get(a2, {}).get("activity_name", a2)
                        dur_saved = min(
                            cpm_map.get(a1, {}).get("planned_duration_days", 10),
                            cpm_map.get(a2, {}).get("planned_duration_days", 10),
                        )
                        suggestions.append({
                            "type": "OPTIMIZATION",
                            "priority": "🟢 OPPORTUNITY",
                            "activity_id": f"{a1}+{a2}",
                            "activity_name": f"{a1_name} + {a2_name}",
                            "rule": "Parallelization Opportunity",
                            "suggestion": (
                                f"**{a1_name}** and **{a2_name}** have no dependencies. "
                                f"→ **Run in parallel — potential savings: ~{dur_saved} days.**"
                            ),
                            "estimated_savings_days": dur_saved,
                        })
                        break  # Only one parallelization suggestion per node

        # Rule 5: Stalled activity (0 progress for 3+ consecutive days)
        all_updates = self.loader.daily_updates
        if not all_updates.empty:
            all_updates = all_updates.copy()
            all_updates["date"] = pd.to_datetime(all_updates["date"], errors="coerce")
            for _, act_row in acts.iterrows():
                if act_row.get("status") != "in_progress":
                    continue
                act_id = str(act_row["id"])
                upd = all_updates[all_updates["activity_id"] == act_id].sort_values("date")
                if len(upd) >= 3:
                    inc_col = "daily_increment" if "daily_increment" in upd.columns else None
                    if inc_col:
                        last3 = upd.tail(3)[inc_col].astype(float)
                        if (last3 <= 0.1).all():
                            a_name = act_row.get("name", act_id)
                            suggestions.append({
                                "type": "ALERT",
                                "priority": "🔴 HIGH",
                                "activity_id": act_id,
                                "activity_name": a_name,
                                "rule": "Stalled Activity",
                                "suggestion": (
                                    f"**{a_name}** has shown zero progress for 3+ days. "
                                    "→ **Investigate immediately — possible blockage.**"
                                ),
                                "estimated_savings_days": 0,
                            })

        # Rule 6: Activity ahead of schedule → reallocate resources
        for _, cpm_row in cpm.iterrows():
            if cpm_row["total_float_days"] > 10 and cpm_row["is_critical_path"] is False:
                act_id = cpm_row["activity_id"]
                act = act_map.get(act_id, {})
                if act.get("status") == "in_progress":
                    suggestions.append({
                        "type": "OPTIMIZATION",
                        "priority": "🟢 OPPORTUNITY",
                        "activity_id": act_id,
                        "activity_name": cpm_row["activity_name"],
                        "rule": "Resource Reallocation",
                        "suggestion": (
                            f"**{cpm_row['activity_name']}** has {cpm_row['total_float_days']} days of float. "
                            "→ **Consider reallocating some resources to critical path activities.**"
                        ),
                        "estimated_savings_days": 2,
                    })

        # De-duplicate by activity_id + rule
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            key = (s["activity_id"], s["rule"])
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)

        # Sort: CRITICAL first, then HIGH, then others
        priority_order = {"🔴 CRITICAL": 0, "🔴 HIGH": 1, "🟡 MEDIUM": 2, "🟢 OPPORTUNITY": 3}
        unique_suggestions.sort(key=lambda x: priority_order.get(x["priority"], 9))

        return unique_suggestions[:15]  # cap at 15


if __name__ == "__main__":
    from data_loader import DataLoader
    dl = DataLoader()
    opt = ScheduleOptimizer("proj_008", loader=dl)
    cpm = opt.compute_cpm()
    print("CPM Results:")
    print(cpm[["activity_name", "total_float_days", "is_critical_path"]].to_string())
    print("\nSuggestions:")
    for s in opt.generate_suggestions():
        print(f"[{s['priority']}] {s['activity_name']}: {s['suggestion'][:80]}...")
