"""Loading a trained model and producing predictions."""

import json
import logging
from pathlib import Path
from typing import Any, Literal, Self

import joblib
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from dropout_service.config import get_settings
from dropout_service.ml.schema import ALL_FEATURES, StudentFeatures

logger = logging.getLogger(__name__)

RiskLevel = Literal["bajo", "medio", "alto"]


class PredictionResult(BaseModel):
    """A single student's dropout assessment."""

    deserta: bool = Field(..., description="True when dropout probability meets the threshold")
    probabilidad: float = Field(..., ge=0, le=1, description="Estimated dropout probability")
    riesgo: RiskLevel = Field(..., description="Risk band derived from the probability")


class Predictor:
    """A fitted pipeline plus the metadata needed to interpret its output.

    Holds everything in memory, so a prediction touches neither the filesystem nor the
    database. Construct once per process via :meth:`from_disk` and reuse it.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        metadata: dict[str, Any] | None = None,
        threshold: float | None = None,
        risk_medium: float | None = None,
        risk_high: float | None = None,
    ) -> None:
        settings = get_settings()
        self.pipeline = pipeline
        self.metadata = metadata or {}
        self.threshold = (
            threshold
            if threshold is not None
            else float(self.metadata.get("decision_threshold", 0.5))
        )
        self.risk_medium = (
            risk_medium if risk_medium is not None else settings.risk_threshold_medium
        )
        self.risk_high = risk_high if risk_high is not None else settings.risk_threshold_high

    @classmethod
    def from_disk(cls, model_path: Path | None = None, metrics_path: Path | None = None) -> Self:
        """Load a pipeline and its metadata from disk.

        Raises:
            FileNotFoundError: if the model file does not exist.
        """
        settings = get_settings()
        model_path = model_path or settings.model_path
        metrics_path = metrics_path or settings.metrics_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"No model at {model_path}. Train one first: uv run scripts/train.py"
            )

        pipeline = joblib.load(model_path)
        metadata: dict[str, Any] = {}
        if metrics_path.exists():
            metadata = json.loads(metrics_path.read_text())
        else:
            logger.warning("No metrics at %s; falling back to a 0.50 threshold", metrics_path)

        predictor = cls(pipeline=pipeline, metadata=metadata)
        logger.info(
            "Loaded %s (version %s, threshold %.4f)",
            metadata.get("model_name", "model"),
            metadata.get("model_version", "unknown"),
            predictor.threshold,
        )
        return predictor

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def risk_level(self, probability: float) -> RiskLevel:
        """Band a probability. These cut-points are institutional policy, not statistics."""
        if probability < self.risk_medium:
            return "bajo"
        if probability < self.risk_high:
            return "medio"
        return "alto"

    def predict(self, students: list[StudentFeatures]) -> list[PredictionResult]:
        """Assess one or more students, preserving input order.

        A single prediction is ``predict([student])[0]`` -- one code path, so batch and
        single requests can never diverge in behaviour.
        """
        if not students:
            return []

        frame = pd.DataFrame([s.model_dump() for s in students]).reindex(columns=ALL_FEATURES)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]

        return [
            PredictionResult(
                deserta=bool(p >= self.threshold),
                probabilidad=float(p),
                riesgo=self.risk_level(float(p)),
            )
            for p in probabilities
        ]
