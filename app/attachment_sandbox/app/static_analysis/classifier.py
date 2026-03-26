from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np

from app.static_analysis.mime_detector import detect_mime, get_file_category
from app.static_analysis.extractor import extract_base_features
from app.static_analysis.pe_parser import extract_pe_features
from app.static_analysis.pdf_parser import extract_pdf_features
from app.static_analysis.office_parser import extract_office_features

logger = logging.getLogger(__name__)

# ── Canonical feature columns ──────────────────────────────────────────────
# Order matters — the model is trained against this exact sequence.
FEATURE_COLS: list[str] = [
    # Base features
    "file_size",
    "entropy",
    "strings_count",
    "has_ip_pattern",
    "has_registry_keys",
    "has_powershell",
    "has_base64_blob",
    # PE features
    "max_section_entropy",
    "section_count",
    "has_suspicious_section",
    "import_count",
    "suspicious_api_count",
    "has_overlay",
    # PDF features
    "page_count",
    "has_javascript",
    "has_embedded_files",
    "has_launch_action",
    "has_suspicious_urls",
    # Office features
    "has_macros",
    "has_auto_open",
    "has_external_links",
    "has_dde",
    "has_obfuscated_strings",
]


def build_feature_vector(feature_dict: dict[str, Any]) -> np.ndarray:
    """Convert a flat feature dict to a numpy array in FEATURE_COLS order."""
    vec = []
    for col in FEATURE_COLS:
        val = feature_dict.get(col, 0)
        # Convert booleans to int for the model
        if isinstance(val, bool):
            val = int(val)
        vec.append(float(val))
    return np.array(vec, dtype=np.float32)


def load_model() -> Any:
    """Load the legacy XGBoost classifier from disk; return None if not yet trained."""
    default_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "static_classifier.pkl")
    )

    env_path = os.environ.get("STATIC_MODEL_PATH")
    if env_path:
        model_path = env_path if os.path.isabs(env_path) else os.path.abspath(env_path)
    else:
        model_path = default_path

    if not os.path.isfile(model_path):
        logger.info("Static XGBoost model not found at %s", model_path)
        return None

    try:
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        return model
    except Exception:
        logger.exception("Failed to load static model from %s", model_path)
        return None


def load_thrember_model() -> Any:
    """Load the Thrember LightGBM classifier from disk; return None if not found."""
    # Resolves to c:\D\SL\Hacks\ZoraAI\ZoraAI-backend\app\attachment-sandbox\models\EMBER2024_PE.model
    default_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "EMBER2024_PE.model")
    model_path = os.environ.get("STATIC_THREMBER_MODEL_PATH", os.path.abspath(default_path))

    if not os.path.isfile(model_path):
        logger.info("Thrember model not found at %s", model_path)
        return None

    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_path)
        logger.info("Thrember LightGBM model successfully loaded from %s", model_path)
        return model
    except Exception:
        logger.exception("Failed to load Thrember model from %s", model_path)
        return None


# ── Module-level lazy singletons ────────────────────────────────────────────
_MODEL: Any | None = None
_MODEL_LOADED = False

_THREMBER_MODEL: Any | None = None
_THREMBER_MODEL_LOADED = False

def _get_model() -> Any:
    """Return the cached XGBoost model instance (may be None)."""
    global _MODEL, _MODEL_LOADED
    if not _MODEL_LOADED:
        _MODEL = load_model()
        _MODEL_LOADED = True
    return _MODEL

def _get_thrember_model() -> Any:
    """Return the cached Thrember LightGBM instance (may be None)."""
    global _THREMBER_MODEL, _THREMBER_MODEL_LOADED
    if not _THREMBER_MODEL_LOADED:
        _THREMBER_MODEL = load_thrember_model()
        _THREMBER_MODEL_LOADED = True
    return _THREMBER_MODEL


def predict(file_path: str) -> tuple[float, dict[str, Any]]:
    """Run the full static-analysis pipeline on *file_path* and return (probability, features)."""
    # Step 1 — base features (always runs to populate the UI)
    feature_dict: dict[str, Any] = extract_base_features(file_path)

    # Step 2 — file-type–specific features (strictly for UI display / legacy modeling)
    mime = feature_dict.get("mime_type", detect_mime(file_path))
    category = get_file_category(mime)

    pe_feats = extract_pe_features(file_path) if category == "pe" else {}
    pdf_feats = extract_pdf_features(file_path) if category == "pdf" else {}
    office_feats = extract_office_features(file_path) if category == "office" else {}

    # Merge — type-specific dicts overlay base dict
    feature_dict.update(pe_feats)
    feature_dict.update(pdf_feats)
    feature_dict.update(office_feats)

    # Step 3 — THREMBER PATH (Only for PE files)
    if category == "pe":
        thrember_model = _get_thrember_model()
        if thrember_model is not None:
            try:
                # Ensure the thrember module is loadable
                try:
                    from thrember.model import predict_sample
                except ImportError:
                    import sys
                    sandbox_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    if sandbox_root not in sys.path:
                        sys.path.insert(0, sandbox_root)
                    from thrember.model import predict_sample
                
                with open(file_path, "rb") as f:
                    file_data = f.read()
                
                proba = predict_sample(thrember_model, file_data)
                return float(round(proba, 6)), feature_dict
            except Exception:
                logger.exception("Thrember Model prediction failed for %s. Falling back to XGBoost...", file_path)

    # Step 4 — LEGACY XGBOOST PATH (For PDFs, Office files, or if Thrember fails)
    vec = build_feature_vector(feature_dict)
    model = _get_model()
    if model is None:
        return 0.0, feature_dict

    try:
        proba = model.predict_proba(vec.reshape(1, -1))[0][1]
        return float(round(proba, 6)), feature_dict
    except Exception:
        logger.exception("XGBoost Model prediction failed for %s", file_path)
        return 0.0, feature_dict
