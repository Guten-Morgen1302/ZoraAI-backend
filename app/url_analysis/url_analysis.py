"""Common URL analysis orchestration across all intelligence layers."""

from __future__ import annotations

from typing import Any

from app.url_analysis.cookie_analyzer import analyze_cookies
from app.url_analysis.domain_intelligence import extract_domain_features
from app.url_analysis.feature_fusion_engine import FeatureFusionEngine
from app.url_analysis.feature_extractor import extract_url_features
from app.url_analysis.homoglyph_detector import extract_homoglyph_features
from app.url_analysis.sandbox_analyzer import analyze_url as analyze_url_in_sandbox
from app.url_analysis.tls_intelligence import extract_tls_features


_FEATURE_FUSION_ENGINE = FeatureFusionEngine()


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


def extract_phase_3_features(input_value: str) -> dict[str, Any]:
    """Run phase 3 features (phase 2 + homoglyph/punycode intelligence)."""
    payload: dict[str, Any] = extract_phase_2_features(input_value)
    payload["homoglyph_features"] = extract_homoglyph_features(input_value)
    return payload


async def extract_phase_4_features_async(input_value: str) -> dict[str, Any]:
    """Run phase 4 features (phase 3 + headless sandbox intelligence)."""
    payload: dict[str, Any] = extract_phase_3_features(input_value)
    sandbox_features = await analyze_url_in_sandbox(input_value)

    cookie_features = analyze_cookies(
        cookies=sandbox_features.get("cookies"),
        cookies_before_login=sandbox_features.get("cookies_before_login"),
        cookies_after_login=sandbox_features.get("cookies_after_login"),
    )

    sandbox_features["cookie_analysis"] = cookie_features
    payload["sandbox_features"] = sandbox_features
    payload["cookie_features"] = cookie_features
    payload["phishing_behavior_features"] = sandbox_features.get(
        "phishing_behavior_analysis", {}
    )
    payload["fingerprint_beacon_features"] = sandbox_features.get(
        "fingerprint_beacon_analysis", {}
    )
    payload["fused_features"] = _FEATURE_FUSION_ENGINE.fuse_features(payload)
    return payload


def extract_all_features(input_value: str) -> dict[str, Any]:
    """Run complete sync pipeline up to phase 3 (without async sandbox)."""
    return extract_phase_3_features(input_value)


async def extract_all_features_async(input_value: str) -> dict[str, Any]:
    """Run the complete layered URL analysis pipeline including sandbox phase."""
    return await extract_phase_4_features_async(input_value)
