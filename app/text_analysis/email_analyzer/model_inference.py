from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("zora.text_analysis.email_model_inference")


class EmailModelInferenceAPI:
    """Loads trained email NLP model once and serves phishing risk predictions."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._max_length = 256
        self._load_lock = Lock()
        self._model_dir: Path | None = None

    def _candidate_model_dirs(self) -> list[Path]:
        base_dir = Path(__file__).resolve().parent
        return [
            base_dir / "email_model",
            base_dir / "email_phishing_model",
            base_dir,
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
            "Could not find a valid email model directory. "
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
                    "transformers and torch are required for email model inference. "
                    "Install them with 'pip install transformers torch'."
                ) from exc

            self._model_dir = self._resolve_model_dir()
            logger.info("Loading email model from %s", self._model_dir)
            print(f"[email-nlp] Loading model/tokenizer from: {self._model_dir}")

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_dir)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)

            logger.info("Email NLP model loaded on device=%s", self._device)
            print(f"[email-nlp] Model loaded on device: {self._device}")

    @staticmethod
    def _resolve_phishing_index(model_config: Any, num_classes: int) -> int:
        id2label = getattr(model_config, "id2label", None) or {}
        label2id = getattr(model_config, "label2id", None) or {}

        for idx, label in id2label.items():
            if isinstance(label, str) and "phish" in label.lower():
                return int(idx)

        for label, idx in label2id.items():
            if isinstance(label, str) and "phish" in label.lower():
                return int(idx)

        # Binary fallback: class index 1 is typically phishing.
        if num_classes == 2:
            return 1
        return 0

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Email model input text must be a non-empty string")

        self._load_model_once()

        import torch

        model_text = text.strip()
        encoded = self._tokenizer(
            model_text,
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
        predicted_label = str(id_to_label.get(predicted_index, str(predicted_index)))
        confidence = float(probabilities[predicted_index].item())

        phishing_index = self._resolve_phishing_index(self._model.config, int(probabilities.shape[-1]))
        phishing_probability = float(probabilities[phishing_index].item())

        logger.info(
            "Email NLP prediction label=%s confidence=%.4f phishing_probability=%.4f",
            predicted_label,
            confidence,
            phishing_probability,
        )
        print(
            "[email-nlp] Prediction "
            f"label={predicted_label} confidence={confidence:.4f} phishing_probability={phishing_probability:.4f}"
        )

        return {
            "label": predicted_label,
            "confidence": round(confidence, 4),
            "phishing_probability": round(phishing_probability, 4),
            "risk_score": round(phishing_probability, 4),
            "model_dir": str(self._model_dir) if self._model_dir else None,
        }


_email_model_api = EmailModelInferenceAPI()


def predict_email_text(text: str) -> dict[str, Any]:
    """Public function for in-code predictions from the trained email NLP model."""
    return _email_model_api.predict(text)
