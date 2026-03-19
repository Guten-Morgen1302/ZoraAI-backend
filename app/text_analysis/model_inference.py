from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from app.text_analysis.preprocessing import validate_sms_text_quality


def _read_env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


CALIBRATION_TEMPERATURE = max(0.05, _read_env_float("SMS_CALIBRATION_TEMPERATURE", 1.6))
PLATT_A = _read_env_float("SMS_PLATT_A", 1.0)
PLATT_B = _read_env_float("SMS_PLATT_B", 0.0)
DRIFT_EMA_ALPHA = min(1.0, max(0.001, _read_env_float("SMS_CALIBRATION_EMA_ALPHA", 0.05)))
DRIFT_ALERT_THRESHOLD = max(0.01, _read_env_float("SMS_CALIBRATION_DRIFT_THRESHOLD", 0.2))


class CalibrationDriftMonitor:
    """Tracks confidence calibration drift statistics over time."""

    def __init__(self):
        self._lock = Lock()

    def update(self, *, stats_path: Path, raw_confidence: float, calibrated_confidence: float) -> dict[str, Any]:
        with self._lock:
            existing: dict[str, Any] = {}
            if stats_path.exists():
                try:
                    existing = json.loads(stats_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}

            total_predictions = int(existing.get("total_predictions") or 0) + 1
            cumulative_abs_delta = float(existing.get("cumulative_abs_delta") or 0.0)
            cumulative_abs_delta += abs(raw_confidence - calibrated_confidence)

            previous_ema = float(existing.get("ema_abs_delta") or 0.0)
            abs_delta = abs(raw_confidence - calibrated_confidence)
            ema_abs_delta = (DRIFT_EMA_ALPHA * abs_delta) + ((1.0 - DRIFT_EMA_ALPHA) * previous_ema)

            stats = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "total_predictions": total_predictions,
                "average_abs_delta": round(cumulative_abs_delta / total_predictions, 6),
                "cumulative_abs_delta": round(cumulative_abs_delta, 6),
                "ema_abs_delta": round(ema_abs_delta, 6),
                "drift_threshold": round(DRIFT_ALERT_THRESHOLD, 6),
                "drift_alert": bool(ema_abs_delta >= DRIFT_ALERT_THRESHOLD),
                "last_raw_confidence": round(raw_confidence, 6),
                "last_calibrated_confidence": round(calibrated_confidence, 6),
                "temperature": round(CALIBRATION_TEMPERATURE, 6),
                "platt_a": round(PLATT_A, 6),
                "platt_b": round(PLATT_B, 6),
            }

            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            return stats


class SMSModelInferenceAPI:
    """Loads SMS model artifacts once and serves predictions."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._max_length = 128
        self._load_lock = Lock()
        self._model_dir: Path | None = None
        self._drift_monitor = CalibrationDriftMonitor()

    def _candidate_model_dirs(self) -> list[Path]:
        base_dir = Path(__file__).resolve().parent
        return [
            base_dir / "sms_model",
            base_dir / "sms_phishing_model",
            base_dir / "sms_analyzer" / "sms_model",
            base_dir / "sms_analyzer" / "sms_phishing_model",
        ]

    def _resolve_model_dir(self) -> Path:
        for path in self._candidate_model_dirs():
            if path.exists() and path.is_dir():
                config_file = path / "config.json"
                tokenizer_file = path / "tokenizer_config.json"
                if config_file.exists() and tokenizer_file.exists():
                    return path

        checked = "\n".join(str(p) for p in self._candidate_model_dirs())
        raise FileNotFoundError(
            "Could not find a valid SMS model directory. "
            "Expected model artifacts (config.json + tokenizer_config.json) in one of:\n"
            f"{checked}"
        )

    def _load_model_once(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return

            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "transformers and torch are required for SMS model inference. "
                    "Install them with 'pip install transformers torch'."
                ) from exc

            self._model_dir = self._resolve_model_dir()
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_dir)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)

    def predict(self, text: str) -> dict[str, Any]:
        normalized_text = validate_sms_text_quality(text)

        self._load_model_once()

        import torch

        encoded = self._tokenizer(
            normalized_text,
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(self._device)
            for key, value in encoded.items()
            if key != "token_type_ids"
        }

        self._model.eval()
        with torch.no_grad():
            logits = self._model(**encoded).logits
            raw_probabilities = torch.softmax(logits, dim=-1)[0]
            temperature = torch.tensor(CALIBRATION_TEMPERATURE, device=logits.device)
            calibrated_logits = logits / temperature
            calibrated_probabilities = torch.softmax(calibrated_logits, dim=-1)[0]

            predicted_index = int(torch.argmax(calibrated_probabilities).item())

            # Optional Platt scaling for binary classifiers.
            if calibrated_probabilities.shape[-1] == 2:
                margin = calibrated_logits[0, 1] - calibrated_logits[0, 0]
                positive_probability = torch.sigmoid((PLATT_A * margin) + PLATT_B).item()
                if positive_probability >= 0.5:
                    predicted_index = 1
                    calibrated_confidence = positive_probability
                else:
                    predicted_index = 0
                    calibrated_confidence = 1.0 - positive_probability
            else:
                calibrated_confidence = float(calibrated_probabilities[predicted_index].item())

        id_to_label = getattr(self._model.config, "id2label", None) or {}
        predicted_label = id_to_label.get(predicted_index, str(predicted_index))
        raw_confidence = float(raw_probabilities[predicted_index].item())

        calibration_stats: dict[str, Any] | None = None
        if self._model_dir is not None:
            stats_path = self._model_dir / "calibration_stats.json"
            calibration_stats = self._drift_monitor.update(
                stats_path=stats_path,
                raw_confidence=raw_confidence,
                calibrated_confidence=calibrated_confidence,
            )

        # Match the training script's output contract.
        return {
            "label": predicted_label,
            "confidence": round(float(calibrated_confidence), 4),
            "raw_confidence": round(raw_confidence, 4),
            "calibration": {
                "temperature": round(CALIBRATION_TEMPERATURE, 4),
                "platt_a": round(PLATT_A, 4),
                "platt_b": round(PLATT_B, 4),
                "drift": calibration_stats,
            },
        }


_sms_model_api = SMSModelInferenceAPI()


def predict_sms_text(text: str) -> dict[str, Any]:
    """Public function for in-code predictions from the trained SMS model."""
    return _sms_model_api.predict(text)
