import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "data.db")

DATE_COLS = {
    "projects": ["planned_start", "planned_end", "actual_start", "actual_end"],
    "activities": ["planned_start_date", "planned_end_date", "actual_start_date",
                   "actual_end_date", "forecasted_start_date", "forecasted_end_date"],
    "daily_updates": ["date"],
    "issues": ["date_raised"],
    "resources": ["start_date", "end_date"],
}

REFERENCE_DATE = datetime(2024, 6, 1)  # "today" for in-progress activities


class DataLoader:
    """Loads and caches all project data."""

    def __init__(self, db_path: str = None, use_csv: bool = False):
        self.db_path = db_path or DB_PATH
        self.use_csv = use_csv
        self._cache = {}
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            self._engine = create_engine(f"sqlite:///{self.db_path}")
        return self._engine

    def _load_table(self, table_name: str) -> pd.DataFrame:
        if table_name in self._cache:
            return self._cache[table_name]

        df = None
        # Try DB first
        if not self.use_csv and os.path.exists(self.db_path):
            try:
                engine = self._get_engine()
                df = pd.read_sql_table(table_name, engine)
            except Exception:
                df = None

        # Fallback to CSV
        if df is None:
            csv_map = {
                "projects": "projects.csv",
                "activities": "activities.csv",
                "daily_updates": "daily_updates.csv",
                "issues": "issues.csv",
                "boq": "boq.csv",
                "resources": "resources.csv",
            }
            fname = csv_map.get(table_name)
            if fname:
                fpath = os.path.join(DATA_DIR, fname)
                if os.path.exists(fpath):
                    df = pd.read_csv(fpath)

        if df is None:
            df = pd.DataFrame()
            return df

        # Parse dates
        for col in DATE_COLS.get(table_name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Normalize column names for projects table
        if table_name == "projects":
            rename = {
                "planned_start": "planned_start_date",
                "planned_end": "planned_end_date",
                "actual_start": "actual_start_date",
                "actual_end": "actual_end_date",
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns and v not in df.columns})

        self._cache[table_name] = df
        return df

    # ── Public accessors ──────────────────────────────────────────────────────

    @property
    def projects(self) -> pd.DataFrame:
        return self._load_table("projects")

    @property
    def activities(self) -> pd.DataFrame:
        return self._load_table("activities")

    @property
    def daily_updates(self) -> pd.DataFrame:
        return self._load_table("daily_updates")

    @property
    def issues(self) -> pd.DataFrame:
        return self._load_table("issues")

    @property
    def boq(self) -> pd.DataFrame:
        return self._load_table("boq")

    @property
    def resources(self) -> pd.DataFrame:
        return self._load_table("resources")

    # ── Filtered views ────────────────────────────────────────────────────────

    def get_historical_activities(self) -> pd.DataFrame:
        """Completed activities — used for training ML models."""
        acts = self.activities
        if acts.empty:
            return acts
        return acts[acts["status"] == "completed"].copy()

    def get_active_activities(self, project_id: str = None) -> pd.DataFrame:
        """In-progress and not-started activities for a project (or all)."""
        acts = self.activities
        if acts.empty:
            return acts
        mask = acts["status"].isin(["in_progress", "not_started"])
        if project_id:
            mask &= acts["project_id"] == project_id
        return acts[mask].copy()

    def get_inprogress_projects(self) -> pd.DataFrame:
        projs = self.projects
        if projs.empty:
            return projs
        return projs[projs["status"] == "in_progress"].copy()

    def get_project_activities(self, project_id: str) -> pd.DataFrame:
        acts = self.activities
        return acts[acts["project_id"] == project_id].copy()

    def get_activity_updates(self, activity_id: str) -> pd.DataFrame:
        upd = self.daily_updates
        return upd[upd["activity_id"] == activity_id].sort_values("date").copy()

    def get_activity_issues(self, activity_id: str = None, project_id: str = None) -> pd.DataFrame:
        iss = self.issues
        if activity_id:
            iss = iss[iss["activity_id"] == activity_id]
        if project_id:
            iss = iss[iss["project_id"] == project_id]
        return iss.copy()

    def get_project_boq(self, project_id: str) -> pd.DataFrame:
        b = self.boq
        return b[b["project_id"] == project_id].copy()

    def get_activity_boq(self, activity_id: str) -> pd.DataFrame:
        b = self.boq
        return b[b["activity_id"] == activity_id].copy()

    def get_all_data(self) -> dict:
        return {
            "projects": self.projects,
            "activities": self.activities,
            "daily_updates": self.daily_updates,
            "issues": self.issues,
            "boq": self.boq,
            "resources": self.resources,
        }

    def reference_date(self) -> datetime:
        """The 'today' reference for in-progress predictions."""
        return REFERENCE_DATE

    def clear_cache(self):
        self._cache = {}


# Singleton for convenience
_loader = None

def get_loader(db_path: str = None, use_csv: bool = False) -> DataLoader:
    global _loader
    if _loader is None:
        _loader = DataLoader(db_path=db_path, use_csv=use_csv)
    return _loader


if __name__ == "__main__":
    dl = DataLoader()
    print("Projects:", len(dl.projects))
    print("Activities:", len(dl.activities))
    print("Daily Updates:", len(dl.daily_updates))
    print("Issues:", len(dl.issues))
    print("BOQ:", len(dl.boq))
    hist = dl.get_historical_activities()
    print(f"Historical (completed) activities: {len(hist)}")
    active = dl.get_active_activities()
    print(f"Active activities: {len(active)}")
