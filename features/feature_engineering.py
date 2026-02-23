"""
Computes 12 engineered features per activity for ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

REFERENCE_DATE = datetime(2024, 6, 1)

ISSUE_SEVERITY_WEIGHTS = {
    "design_change": 3.0,
    "inspection_fail": 2.0,
    "scope_creep": 2.0,
    "weather": 1.0,
    "material_delay": 1.5,
    "labor_shortage": 1.5,
    "equipment_breakdown": 1.0,
    "safety": 0.5,
}

SEVERITY_MULTIPLIER = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.5}


def engineer_features(
    activities: pd.DataFrame,
    loader,
    today: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Compute all 12 features for every activity in the dataframe.

    Parameters
    ----------
    activities : pd.DataFrame  — activities to featurize
    loader     : DataLoader    — for accessing updates, issues, boq, etc.
    today      : datetime      — reference date (defaults to REFERENCE_DATE)

    Returns
    -------
    pd.DataFrame with original columns + 12 new feature columns
    """
    if today is None:
        today = REFERENCE_DATE

    today = pd.Timestamp(today)
    df = activities.copy()

    # ── Ensure date columns are Timestamps ──────────────────────────────────
    date_cols = ["planned_start_date", "planned_end_date",
                 "actual_start_date", "actual_end_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── Feature 1: planned_duration ──────────────────────────────────────────
    df["planned_duration"] = (
        df["planned_end_date"] - df["planned_start_date"]
    ).dt.days.clip(lower=1)

    # ── Feature 2: elapsed_days ──────────────────────────────────────────────
    def elapsed(row):
        start = row.get("actual_start_date") or row.get("planned_start_date")
        if pd.isna(start):
            return 0
        start = pd.Timestamp(start)
        if row["status"] == "completed" and not pd.isna(row.get("actual_end_date")):
            return max(1, (pd.Timestamp(row["actual_end_date"]) - start).days)
        return max(1, (today - start).days)

    df["elapsed_days"] = df.apply(elapsed, axis=1)

    # ── Feature 3: progress_rate (% per day) ────────────────────────────────
    prog = df.get("progress", pd.Series(0, index=df.index))
    df["progress"] = pd.to_numeric(prog, errors="coerce").fillna(0)
    df["progress_rate"] = (df["progress"] / df["elapsed_days"]).clip(0, 20)

    # ── Feature 4: schedule_variance (days late at start) ───────────────────
    if "schedule_variance_days" in df.columns:
        df["schedule_variance"] = pd.to_numeric(
            df["schedule_variance_days"], errors="coerce").fillna(0)
    else:
        def sch_var(row):
            planned = row.get("planned_start_date")
            actual = row.get("actual_start_date")
            if pd.isna(planned) or pd.isna(actual):
                return 0
            return (pd.Timestamp(actual) - pd.Timestamp(planned)).days
        df["schedule_variance"] = df.apply(sch_var, axis=1)

    # ── Feature 5: delay_ratio (actual/planned — for completed) ─────────────
    def delay_ratio(row):
        if "actual_duration_days" in row and not pd.isna(row["actual_duration_days"]):
            pd_dur = max(row["planned_duration"], 1)
            return row["actual_duration_days"] / pd_dur
        return 1.0

    df["delay_ratio"] = df.apply(delay_ratio, axis=1)

    # ── Features 6 & 7: issue_count + issue_severity_score ──────────────────
    all_issues = loader.issues
    if not all_issues.empty:
        def issue_stats(activity_id):
            iss = all_issues[all_issues["activity_id"] == activity_id]
            open_iss = iss[iss["status"] == "open"]
            count = len(open_iss)
            score = 0.0
            for _, row in open_iss.iterrows():
                cat_w = ISSUE_SEVERITY_WEIGHTS.get(row.get("category", ""), 1.0)
                sev_m = SEVERITY_MULTIPLIER.get(row.get("severity", "medium"), 1.0)
                score += cat_w * sev_m
            return count, score

        issue_data = df["id"].apply(lambda aid: pd.Series(
            issue_stats(aid), index=["issue_count", "issue_severity_score"]
        ))
        df["issue_count"] = issue_data["issue_count"]
        df["issue_severity_score"] = issue_data["issue_severity_score"]
    else:
        df["issue_count"] = 0
        df["issue_severity_score"] = 0.0

    # ── Feature 8: boq_complexity ────────────────────────────────────────────
    all_boq = loader.boq
    if not all_boq.empty:
        def boq_complexity(activity_id):
            b = all_boq[all_boq["activity_id"] == activity_id]
            if b.empty:
                return 0.0
            count_score = len(b)
            if "total_price" in b.columns and "total_cost" in b.columns:
                variance = (b["total_price"] - b["total_cost"]).sum() / max(b["total_cost"].sum(), 1)
                return count_score + variance * 0.1
            return count_score

        df["boq_complexity"] = df["id"].apply(boq_complexity)
    else:
        df["boq_complexity"] = 0.0

    # ── Feature 9: parent_delay (binary) ────────────────────────────────────
    def parent_delayed(row):
        pred_id = row.get("depends_on")
        if not pred_id or (isinstance(pred_id, float) and np.isnan(pred_id)):
            return 0
        pred_mask = df["id"] == pred_id
        if pred_mask.any():
            pred_row = df[pred_mask].iloc[0]
            return 1 if pred_row.get("schedule_variance", 0) > 2 else 0
        return 0

    df["parent_delay"] = df.apply(parent_delayed, axis=1)

    # ── Feature 10: historical_avg_delay (by category) ─────────────────────
    completed_acts = df[df["status"] == "completed"].copy()
    if len(completed_acts) > 0:
        hist_delay = (
            completed_acts.groupby("category")["delay_ratio"]
            .mean()
            .reset_index(name="historical_avg_delay")
        )
        df = df.merge(hist_delay, on="category", how="left")
        df["historical_avg_delay"] = df["historical_avg_delay"].fillna(1.0)
    else:
        df["historical_avg_delay"] = 1.0

    # ── Features 11 & 12: progress_velocity_7d + progress_acceleration ──────
    all_updates = loader.daily_updates
    if not all_updates.empty:
        all_updates = all_updates.copy()
        all_updates["date"] = pd.to_datetime(all_updates["date"], errors="coerce")
        if "daily_increment" not in all_updates.columns and "reported_progress" in all_updates.columns:
            all_updates = all_updates.sort_values(["activity_id", "date"])
            all_updates["daily_increment"] = (
                all_updates.groupby("activity_id")["reported_progress"].diff().fillna(0)
            )

        def velocity_and_accel(activity_id):
            upd = all_updates[all_updates["activity_id"] == activity_id].sort_values("date")
            if upd.empty:
                return 0.0, 0.0
            recent = upd.tail(14)
            vel_14 = recent["daily_increment"].mean() if len(recent) > 0 else 0
            vel_7 = upd.tail(7)["daily_increment"].mean() if len(upd) >= 7 else vel_14
            prev_7 = upd.iloc[-14:-7]["daily_increment"].mean() if len(upd) >= 14 else vel_7
            accel = vel_7 - prev_7
            return float(vel_7), float(accel)

        vel_data = df["id"].apply(lambda aid: pd.Series(
            velocity_and_accel(aid), index=["progress_velocity_7d", "progress_acceleration"]
        ))
        df["progress_velocity_7d"] = vel_data["progress_velocity_7d"]
        df["progress_acceleration"] = vel_data["progress_acceleration"]
    else:
        df["progress_velocity_7d"] = df["progress_rate"]
        df["progress_acceleration"] = 0.0

    return df


FEATURE_COLS = [
    "planned_duration", "elapsed_days", "progress_rate", "schedule_variance",
    "delay_ratio", "issue_count", "issue_severity_score", "boq_complexity",
    "parent_delay", "historical_avg_delay", "progress_velocity_7d", "progress_acceleration",
]

TARGET_COL = "delay_ratio"

CATEGORY_COLS = ["category", "project_type"]


def get_ml_ready(df: pd.DataFrame):
    """
    Returns X (features), y (target) arrays for ML training.
    Only uses completed activities with non-null targets.
    """
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    # Encode categorical columns
    for cat_col in CATEGORY_COLS:
        if cat_col in df.columns:
            le = LabelEncoder()
            df[f"{cat_col}_enc"] = le.fit_transform(df[cat_col].astype(str))

    feat_cols = FEATURE_COLS + [f"{c}_enc" for c in CATEGORY_COLS if c in df.columns]
    feat_cols = [c for c in feat_cols if c in df.columns]

    y_col = TARGET_COL
    mask = df[y_col].notna() & df["status"].isin(["completed"])
    return df[mask][feat_cols], df[mask][y_col], feat_cols
