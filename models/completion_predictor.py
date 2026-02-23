"""
Three prediction methods for activity completion dates + ensemble.

Method A : Earned Value / Linear Extrapolation (baseline)
Method B : GradientBoostingRegressor delay-multiplier model
Ensemble : 0.4 × A + 0.6 × B (calibrated weights)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import joblib
import os

from data_loader import DataLoader
from features.feature_engineering import engineer_features, get_ml_ready, FEATURE_COLS, CATEGORY_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
REFERENCE_DATE = datetime(2024, 6, 1)


class CompletionPredictor:
    """
    Trains and runs the three-method completion date predictor.
    """

    def __init__(self, loader: Optional[DataLoader] = None, today: Optional[datetime] = None):
        self.loader = loader or DataLoader()
        self.today = pd.Timestamp(today or REFERENCE_DATE)
        self.model_B = None
        self.feature_cols = None
        self.label_encoders = {}
        self._trained = False

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, force: bool = False) -> dict:
        """Train GBR on historical (completed) activities. Returns train metrics."""
        if self._trained and not force:
            return {}

        hist = self.loader.get_historical_activities()
        if hist.empty:
            print("⚠️  No historical data found for training.")
            return {}

        # Feature engineering
        feats = engineer_features(hist, self.loader, today=self.today)
        X, y, feat_cols = get_ml_ready(feats)

        if len(X) < 5:
            print(f"⚠️  Only {len(X)} training examples — skipping training.")
            return {}

        self.feature_cols = feat_cols

        # Gradient Boosting (primary Method B model)
        self.model_B = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=3,
            subsample=0.8,
            random_state=42,
        )
        self.model_B.fit(X, y)

        # Cross-val score
        cv_scores = cross_val_score(
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            X, y, cv=min(5, len(X)//2), scoring="neg_mean_absolute_error"
        )
        train_pred = self.model_B.predict(X)
        train_mae = mean_absolute_error(y, train_pred)
        self._trained = True

        metrics = {
            "n_train": len(X),
            "features": feat_cols,
            "train_mae_multiplier": round(train_mae, 4),
            "cv_mae_mean": round(-cv_scores.mean(), 4),
            "cv_mae_std": round(cv_scores.std(), 4),
        }

        # Feature importances
        self.feature_importances_ = pd.Series(
            self.model_B.feature_importances_, index=feat_cols
        ).sort_values(ascending=False)

        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Method A: Earned Value / Linear Extrapolation
    # ──────────────────────────────────────────────────────────────────────────

    def predict_method_A(self, row: pd.Series) -> Optional[datetime]:
        """
        remaining_work = 100 - progress
        days_to_complete = remaining_work / progress_rate  (with smoothing)
        predicted_end = today + days_to_complete
        """
        progress = float(row.get("progress", 0))
        if progress >= 100:
            end = row.get("actual_end_date")
            return pd.Timestamp(end) if not pd.isna(end) else self.today

        start = row.get("actual_start_date") or row.get("planned_start_date")
        if pd.isna(start):
            return None
        start = pd.Timestamp(start)

        elapsed = max(1, (self.today - start).days)
        progress_rate = progress / elapsed  # % per day

        # Smooth with planned rate
        planned_dur = float(row.get("planned_duration", 30) or 30)
        planned_rate = 100 / planned_dur
        # Weighted smoothing — trust actual more if > 5 days elapsed
        w = min(elapsed / 14, 0.85)
        blended_rate = w * progress_rate + (1 - w) * planned_rate
        blended_rate = max(blended_rate, 0.5)  # floor: 0.5% per day

        remaining = 100 - progress
        days_left = remaining / blended_rate
        return self.today + timedelta(days=round(days_left))

    # ──────────────────────────────────────────────────────────────────────────
    # Method B: GradientBoosting delay-multiplier → predicted end date
    # ──────────────────────────────────────────────────────────────────────────

    def predict_method_B(self, row: pd.Series) -> Optional[datetime]:
        """
        Uses trained GBR to predict delay_multiplier.
        predicted_end = actual_start + planned_duration × predicted_multiplier
        """
        if not self._trained or self.model_B is None:
            return None

        # Build feature vector
        X_row = {}
        for col in self.feature_cols:
            X_row[col] = row.get(col, 0)

        X_df = pd.DataFrame([X_row])
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0)

        multiplier = float(self.model_B.predict(X_df)[0])
        multiplier = max(0.8, min(multiplier, 5.0))  # clip unreasonable values

        start = row.get("actual_start_date") or row.get("planned_start_date")
        if pd.isna(start):
            return None
        start = pd.Timestamp(start)

        planned_dur = float(row.get("planned_duration", 30) or 30)
        # Adjust for already-elapsed progress
        progress = float(row.get("progress", 0))
        remaining_fraction = (100 - progress) / 100
        remaining_days = planned_dur * multiplier * remaining_fraction

        return self.today + timedelta(days=round(remaining_days))

    # ──────────────────────────────────────────────────────────────────────────
    # Ensemble: A + B
    # ──────────────────────────────────────────────────────────────────────────

    def predict_ensemble(self, row: pd.Series,
                         weight_A: float = 0.4, weight_B: float = 0.6) -> Optional[datetime]:
        """Weighted average of Methods A and B."""
        a = self.predict_method_A(row)
        b = self.predict_method_B(row)

        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a

        a_ts = pd.Timestamp(a)
        b_ts = pd.Timestamp(b)
        days_a = (a_ts - self.today).days
        days_b = (b_ts - self.today).days
        blended_days = weight_A * days_a + weight_B * days_b
        return self.today + timedelta(days=max(0, round(blended_days)))

    # ──────────────────────────────────────────────────────────────────────────
    # Predict all active activities
    # ──────────────────────────────────────────────────────────────────────────

    def predict_all(self, project_id: Optional[str] = None) -> pd.DataFrame:
        """
        Run all 3 predictions on in-progress / not-started activities.
        Returns a DataFrame with one row per activity and prediction columns.
        """
        if not self._trained:
            self.train()

        if project_id:
            acts = self.loader.get_project_activities(project_id)
            acts = acts[acts["status"].isin(["in_progress", "not_started"])]
        else:
            acts = self.loader.get_active_activities()

        if acts.empty:
            return pd.DataFrame()

        feats = engineer_features(acts, self.loader, today=self.today)

        results = []
        for _, row in feats.iterrows():
            a_end = self.predict_method_A(row)
            b_end = self.predict_method_B(row) if self._trained else None
            ens_end = self.predict_ensemble(row) if self._trained else a_end

            delay_mult_pred = None
            if self._trained and self.feature_cols:
                X_row = {col: row.get(col, 0) for col in self.feature_cols}
                X_df = pd.DataFrame([X_row])
                for col in X_df.columns:
                    X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0)
                delay_mult_pred = round(float(self.model_B.predict(X_df)[0]), 3)

            results.append({
                "activity_id": row["id"],
                "activity_name": row["name"],
                "project_id": row["project_id"],
                "status": row["status"],
                "progress": row.get("progress", 0),
                "planned_end_date": row.get("planned_end_date"),
                "methodA_end": a_end,
                "methodB_end": b_end,
                "ensemble_end": ens_end,
                "delay_multiplier_pred": delay_mult_pred,
                "progress_rate": round(row.get("progress_rate", 0), 3),
                "schedule_variance": row.get("schedule_variance", 0),
                "issue_count": row.get("issue_count", 0),
                "issue_severity_score": round(row.get("issue_severity_score", 0), 2),
                "is_critical": False,  # set after CPM computation
            })

        return pd.DataFrame(results)

    def save_model(self):
        if self.model_B:
            joblib.dump({"model": self.model_B, "features": self.feature_cols}, MODEL_PATH)

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            data = joblib.load(MODEL_PATH)
            self.model_B = data["model"]
            self.feature_cols = data["features"]
            self._trained = True


if __name__ == "__main__":
    from data_loader import DataLoader
    dl = DataLoader()
    cp = CompletionPredictor(loader=dl)
    metrics = cp.train()
    print("Training metrics:", metrics)
    results = cp.predict_all()
    print(results[["activity_id", "activity_name", "progress", "methodA_end",
                   "methodB_end", "ensemble_end"]].to_string())
