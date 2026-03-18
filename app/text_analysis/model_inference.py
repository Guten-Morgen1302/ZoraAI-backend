from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any


class SMSModelInferenceAPI:
    """Loads SMS model artifacts once and serves predictions."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._max_length = 128
        self._load_lock = Lock()
        self._model_dir: Path | None = None

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
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        self._load_model_once()

        import torch

        encoded = self._tokenizer(
            text,
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
            probabilities = torch.softmax(logits, dim=-1)[0]
            predicted_index = int(torch.argmax(probabilities).item())

        id_to_label = getattr(self._model.config, "id2label", None) or {}
        predicted_label = id_to_label.get(predicted_index, str(predicted_index))
        confidence = float(probabilities[predicted_index].item())

        # Match the training script's output contract.
        return {
            "label": predicted_label,
            "confidence": round(confidence, 4),
        }


_sms_model_api = SMSModelInferenceAPI()


def predict_sms_text(text: str) -> dict[str, Any]:
    """Public function for in-code predictions from the trained SMS model."""
    return _sms_model_api.predict(text)
