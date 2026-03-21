"""Common URL analysis orchestration across all intelligence layers."""

from __future__ import annotations

from typing import Any

from app.url_analysis.domain_intelligence import extract_domain_features
from app.url_analysis.feature_extractor import extract_url_features
from app.url_analysis.tls_intelligence import extract_tls_features


def extract_phase_1_features(input_value: str) -> dict[str, Any]:
    """Run phase 1 features (URL lexical + DNS/WHOIS intelligence)."""
    return {
        "url_features": extract_url_features(input_value),
        "domain_features": extract_domain_features(input_value),
    }


def extract_phase_2_features(input_value: str) -> dict[str, Any]:
    """Run phase 2 features (phase 1 + TLS/SSL intelligence)."""
    payload: dict[str, Any] = extract_phase_1_features(input_value)
    payload["tls_features"] = extract_tls_features(input_value)
    return payload


def extract_all_features(input_value: str) -> dict[str, Any]:
    """Run the complete layered URL analysis pipeline."""
    return extract_phase_2_features(input_value)
