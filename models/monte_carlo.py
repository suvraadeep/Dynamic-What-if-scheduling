"""
models/monte_carlo.py
----------------------
Monte Carlo simulation for activity completion dates.
Samples from fitted distribution of daily progress increments.
Outputs P50, P80, P90 completion dates and full distributions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
import warnings
warnings.filterwarnings("ignore")

from data_loader import DataLoader
from features.feature_engineering import engineer_features

REFERENCE_DATE = datetime(2024, 6, 1)
N_SIMULATIONS = 1000
LOOKBACK_DAYS = 14


class MonteCarloSimulator:
    """
    Per-activity Monte Carlo simulation using last-N-day progress increments.
    """

    def __init__(self, loader: Optional[DataLoader] = None, today: Optional[datetime] = None,
                 n_sims: int = N_SIMULATIONS):
        self.loader = loader or DataLoader()
        self.today = pd.Timestamp(today or REFERENCE_DATE)
        self.n_sims = n_sims

    # ──────────────────────────────────────────────────────────────────────────
    # Core simulation for a single activity
    # ──────────────────────────────────────────────────────────────────────────

    def _get_increment_samples(self, activity_id: str) -> np.ndarray:
        """Get recent daily increments for the activity."""
        upd = self.loader.get_activity_updates(activity_id)
        if upd.empty:
            return np.array([3.0])  # default 3% per day

        upd = upd.sort_values("date")
        if "daily_increment" in upd.columns:
            increments = pd.to_numeric(upd["daily_increment"], errors="coerce").dropna()
        elif "reported_progress" in upd.columns:
            upd["_inc"] = upd["reported_progress"].diff().fillna(upd["reported_progress"].iloc[0])
            increments = upd["_inc"].clip(lower=0)
        else:
            return np.array([3.0])

        # Use last LOOKBACK_DAYS
        recent = increments.tail(LOOKBACK_DAYS).values
        recent = recent[recent >= 0]
        return recent if len(recent) > 0 else np.array([3.0])

    def simulate_activity(self, activity_id: str, current_progress: float,
                          max_days: int = 365) -> Dict:
        """
        Run N Monte Carlo simulations for one activity.

        Returns
        -------
        dict with:
            - completion_days_samples: array of simulated days-to-complete
            - p50, p80, p90: percentile completion dates
            - mean, std: distribution stats
        """
        samples = self._get_increment_samples(activity_id)

        if len(samples) < 2:
            # Use uniform distribution around the mean
            mean_inc = float(samples.mean()) if len(samples) > 0 else 3.0
            samples_for_dist = np.clip(
                np.random.normal(mean_inc, mean_inc * 0.3, 50), 0.1, 15
            )
        else:
            samples_for_dist = samples

        # Fit distribution parameters (mean + std for truncated normal)
        mu = np.mean(samples_for_dist)
        sigma = max(np.std(samples_for_dist), 0.5)

        completion_days = []
        remaining = 100.0 - current_progress

        for _ in range(self.n_sims):
            pct_remaining = remaining
            days = 0
            while pct_remaining > 0 and days < max_days:
                # Sample daily increment from fitted distribution (truncated normal)
                inc = np.random.normal(mu, sigma)
                inc = max(0.1, min(inc, 20))  # clip [0.1, 20]
                pct_remaining -= inc
                days += 1
            completion_days.append(days)

        completion_days = np.array(completion_days)

        # Convert to actual dates
        p50_days = int(np.percentile(completion_days, 50))
        p80_days = int(np.percentile(completion_days, 80))
        p90_days = int(np.percentile(completion_days, 90))

        return {
            "activity_id": activity_id,
            "current_progress": current_progress,
            "n_simulations": self.n_sims,
            "mean_days_to_complete": float(np.mean(completion_days)),
            "std_days_to_complete": float(np.std(completion_days)),
            "p50_days": p50_days,
            "p80_days": p80_days,
            "p90_days": p90_days,
            "p50_date": self.today + timedelta(days=p50_days),
            "p80_date": self.today + timedelta(days=p80_days),
            "p90_date": self.today + timedelta(days=p90_days),
            "completion_days_distribution": completion_days,
            "increment_mean": mu,
            "increment_std": sigma,
            "samples_used": len(samples_for_dist),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Batch simulation for all active activities
    # ──────────────────────────────────────────────────────────────────────────

    def simulate_all(self, project_id: Optional[str] = None) -> pd.DataFrame:
        """Run Monte Carlo for all in-progress activities."""
        if project_id:
            acts = self.loader.get_project_activities(project_id)
            acts = acts[acts["status"].isin(["in_progress", "not_started"])]
        else:
            acts = self.loader.get_active_activities()

        if acts.empty:
            return pd.DataFrame()

        results = []
        for _, row in acts.iterrows():
            progress = float(row.get("progress", 0))
            sim = self.simulate_activity(
                activity_id=str(row["id"]),
                current_progress=progress,
            )
            sim["activity_name"] = row.get("name", "")
            sim["project_id"] = row.get("project_id", "")
            sim["status"] = row.get("status", "")
            sim["planned_end_date"] = row.get("planned_end_date")
            # Drop the full distribution array from the summary table
            sim.pop("completion_days_distribution", None)
            results.append(sim)

        df = pd.DataFrame(results)
        return df

    def get_distribution_for_plot(self, activity_id: str, current_progress: float) -> np.ndarray:
        """Return the raw completion_days array for histogram plotting."""
        sim = self.simulate_activity(activity_id, current_progress)
        return sim["completion_days_distribution"]


if __name__ == "__main__":
    from data_loader import DataLoader
    dl = DataLoader()
    mc = MonteCarloSimulator(loader=dl, n_sims=500)
    results = mc.simulate_all(project_id="proj_008")
    print(results[["activity_id", "activity_name", "current_progress",
                   "p50_date", "p80_date", "p90_date"]].to_string())
