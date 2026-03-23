"""Phase 11 ML + risk scoring engine for URL phishing detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.url_analysis.feature_fusion_engine import FeatureFusionEngine

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


DEFAULT_DATASET_PATH = Path("app/url_analysis/datasets/dataset_final.csv")
DEFAULT_MODEL_DIR = Path("app/url_analysis/models")


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float with a safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _risk_band(value: float) -> str:
    """Map normalized risk value into Low/Medium/High buckets."""
    if value >= 0.7:
        return "High"
    if value >= 0.4:
        return "Medium"
    return "Low"


@dataclass
class TrainingSummary:
    """Summary for model training and validation metrics."""

    total_rows: int
    cleaned_rows: int
    removed_rows: int
    xgboost_available: bool
    xgboost_metrics: dict[str, float] | None
    random_forest_metrics: dict[str, float]
    primary_model: str
    model_dir: str


class URLMLRiskEngine:
    """Train and run phishing probability + risk scoring over fused URL features."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.dataset_path = Path(dataset_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.feature_schema = list(FeatureFusionEngine.FEATURE_SCHEMA)

        self.rf_model: RandomForestClassifier | None = None
        self.xgb_model: Any = None
        self.primary_model_name = "random_forest"

    def _feature_columns_present(self, frame: pd.DataFrame) -> list[str]:
        """Return the subset of schema columns that exist in the DataFrame."""
        return [column for column in self.feature_schema if column in frame.columns]

    def _clean_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset rows before training.

        Rules:
        - Keep only label in {0, 1}
        - Drop rows with conversion/runtime error flags
        - Drop rows with all-zero feature vectors (failed conversion fallbacks)
        """
        df = frame.copy()

        if "label" not in df.columns:
            raise ValueError("Dataset must contain a 'label' column")

        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df[df["label"].isin([0, 1])]

        if "analysis_error" in df.columns:
            normalized = df["analysis_error"].fillna("").astype(str).str.strip()
            df = df[normalized == ""]

        if "sandbox_error" in df.columns:
            normalized = df["sandbox_error"].fillna("").astype(str).str.strip().str.lower()
            df = df[(normalized == "") | (normalized == "none")]

        present_features = self._feature_columns_present(df)
        if not present_features:
            raise ValueError("No fusion feature columns found in dataset")

        df[present_features] = df[present_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        non_zero_mask = (df[present_features].abs().sum(axis=1) > 0.0)
        df = df[non_zero_mask]

        if df.empty:
            raise ValueError("No usable rows remain after dataset cleaning")

        return df.reset_index(drop=True)

    def _train_random_forest(self, x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train baseline RandomForest model."""
        model = RandomForestClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(x_train, y_train)
        return model

    def _train_xgboost(self, x_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train primary XGBoost model when package is available."""
        if XGBClassifier is None:
            return None

        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def _evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Evaluate a classifier and return key metrics."""
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 6),
        }

        try:
            metrics["auc"] = round(float(roc_auc_score(y_test, probabilities)), 6)
        except ValueError:
            metrics["auc"] = 0.0

        return metrics

    def train(self, dataset_path: str | Path | None = None) -> TrainingSummary:
        """Train XGBoost(primary) + RandomForest(baseline) with dataset cleaning."""
        source = Path(dataset_path) if dataset_path is not None else self.dataset_path
        if not source.exists():
            raise FileNotFoundError(f"Training dataset not found: {source}")

        raw = pd.read_csv(source)
        cleaned = self._clean_dataset(raw)

        features = self._feature_columns_present(cleaned)
        x = cleaned[features]
        y = cleaned["label"].astype(int)

        if len(cleaned) < 20:
            test_size = 0.3
        else:
            test_size = 0.2

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if y.nunique() > 1 else None,
        )

        self.rf_model = self._train_random_forest(x_train, y_train)
        rf_metrics = self._evaluate_model(self.rf_model, x_test, y_test)

        self.xgb_model = self._train_xgboost(x_train, y_train)
        xgb_metrics: dict[str, float] | None = None
        if self.xgb_model is not None:
            xgb_metrics = self._evaluate_model(self.xgb_model, x_test, y_test)

        if xgb_metrics is not None and xgb_metrics.get("auc", 0.0) >= rf_metrics.get("auc", 0.0):
            self.primary_model_name = "xgboost"
        else:
            self.primary_model_name = "random_forest"

        self._save_models(features)

        return TrainingSummary(
            total_rows=int(len(raw)),
            cleaned_rows=int(len(cleaned)),
            removed_rows=int(len(raw) - len(cleaned)),
            xgboost_available=self.xgb_model is not None,
            xgboost_metrics=xgb_metrics,
            random_forest_metrics=rf_metrics,
            primary_model=self.primary_model_name,
            model_dir=str(self.model_dir),
        )

    def _save_models(self, feature_columns: list[str]) -> None:
        """Persist trained models and metadata."""
        if self.rf_model is None:
            raise ValueError("RandomForest model is not trained")

        joblib.dump(self.rf_model, self.model_dir / "url_rf_model.joblib")
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, self.model_dir / "url_xgb_model.joblib")

        metadata = {
            "feature_columns": feature_columns,
            "primary_model": self.primary_model_name,
        }
        joblib.dump(metadata, self.model_dir / "url_model_metadata.joblib")

    def load(self) -> None:
        """Load trained models and metadata from disk."""
        metadata_path = self.model_dir / "url_model_metadata.joblib"
        rf_path = self.model_dir / "url_rf_model.joblib"
        xgb_path = self.model_dir / "url_xgb_model.joblib"

        if not metadata_path.exists() or not rf_path.exists():
            raise FileNotFoundError("Model artifacts not found. Run training first.")

        metadata = joblib.load(metadata_path)
        self.primary_model_name = str(metadata.get("primary_model", "random_forest"))

        self.rf_model = joblib.load(rf_path)
        self.xgb_model = joblib.load(xgb_path) if xgb_path.exists() else None

    def _active_model(self) -> Any:
        """Return active primary model object."""
        if self.primary_model_name == "xgboost" and self.xgb_model is not None:
            return self.xgb_model
        if self.rf_model is not None:
            return self.rf_model
        raise ValueError("No model loaded/trained")

    def predict_probability_from_vector(self, feature_vector: list[float]) -> float:
        """Predict phishing probability from a fused feature vector."""
        if len(feature_vector) != len(self.feature_schema):
            raise ValueError(
                f"Invalid feature vector length={len(feature_vector)} expected={len(self.feature_schema)}"
            )

        model = self._active_model()
        probability = model.predict_proba([feature_vector])[0][1]
        return round(float(probability), 6)

    def score_risk(
        self,
        phishing_probability: float,
        sub_scores: dict[str, float],
        cookie_score: float,
    ) -> dict[str, Any]:
        """Compute final composite risk score and textual level.

        Uses weighted score formula aligned with current code structure and extends
        with model probability for better ranking stability.
        """
        url_score = _to_float(sub_scores.get("url", 0.0))
        content_score = _to_float(sub_scores.get("content", 0.0))
        infra_score = _to_float(sub_scores.get("infra", 0.0))
        behavior_score = _to_float(sub_scores.get("behavior", 0.0))
        cookie_risk = _to_float(cookie_score, 0.0)

        weighted_subscore = (
            0.3 * url_score
            + 0.25 * content_score
            + 0.2 * cookie_risk
            + 0.15 * infra_score
            + 0.1 * behavior_score
        )

        final_risk = 0.6 * _to_float(phishing_probability) + 0.4 * weighted_subscore
        final_risk = max(0.0, min(round(final_risk, 6), 1.0))

        return {
            "risk_score": final_risk,
            "risk_level": _risk_band(final_risk),
            "components": {
                "url_score": round(url_score, 6),
                "content_score": round(content_score, 6),
                "cookie_score": round(cookie_risk, 6),
                "infra_score": round(infra_score, 6),
                "behavior_score": round(behavior_score, 6),
            },
        }

    def predict_from_phase_payload(self, phase_payload: dict[str, Any]) -> dict[str, Any]:
        """Predict phishing probability and risk from live phase payload output."""
        fused = phase_payload.get("fused_features")
        if not isinstance(fused, dict):
            fused = FeatureFusionEngine().fuse_features(phase_payload)

        feature_vector = fused.get("feature_vector", [])
        sub_scores = fused.get("sub_scores", {})

        cookie_features = phase_payload.get("cookie_features", {})
        cookie_score = 0.0
        if isinstance(cookie_features, dict):
            cookie_score = _to_float(cookie_features.get("cookie_risk_score", 0.0), 0.0)

        probability = self.predict_probability_from_vector(feature_vector)
        risk = self.score_risk(probability, sub_scores, cookie_score)

        return {
            "phishing_probability": probability,
            "model": self.primary_model_name,
            "risk": risk,
        }


def main() -> None:
    """Train Phase 11 models from hardcoded dataset path."""
    engine = URLMLRiskEngine()
    summary = engine.train()

    print(
        {
            "total_rows": summary.total_rows,
            "cleaned_rows": summary.cleaned_rows,
            "removed_rows": summary.removed_rows,
            "xgboost_available": summary.xgboost_available,
            "xgboost_metrics": summary.xgboost_metrics,
            "random_forest_metrics": summary.random_forest_metrics,
            "primary_model": summary.primary_model,
            "model_dir": summary.model_dir,
        }
    )


if __name__ == "__main__":
    main()
