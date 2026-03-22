"""Feature fusion engine for building ML-ready cybersecurity feature vectors."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sklearn.preprocessing import MinMaxScaler


def _safe_bool(value: Any, default: int = -1) -> int:
    """Convert bool-like values into integer representation with safe default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return 1
        if lowered in {"0", "false", "no", "n", "off"}:
            return 0
    return default


def _safe_float(value: Any, default: float = -1.0) -> float:
    """Convert arbitrary value to float with safe fallback."""
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_len(value: Any, default: float = -1.0) -> float:
    """Length helper that tolerates missing/non-sized values."""
    if value is None:
        return default
    if isinstance(value, (list, tuple, set, dict, str)):
        return float(len(value))
    return default


def _encode_risk_level(value: Any) -> float:
    """Encode risk category as numeric value for ML consumption."""
    lowered = str(value or "").strip().lower()
    if lowered == "low":
        return 0.0
    if lowered == "medium":
        return 1.0
    if lowered == "high":
        return 2.0
    return -1.0


class FeatureFusionEngine:
    """Fuse multi-phase security analysis outputs into a fixed feature vector."""

    FEATURE_SCHEMA: tuple[str, ...] = (
        # URL lexical features
        "url_length",
        "url_num_dots",
        "url_num_hyphens",
        "url_has_ip",
        "url_num_suspicious_keywords",
        "url_entropy",
        # DNS/WHOIS and infra features
        "infra_has_mx_records",
        "infra_num_a_records",
        "infra_fast_flux_detected",
        "infra_suspicious_ttl",
        "tls_has_https",
        "tls_certificate_valid",
        "tls_is_self_signed",
        "tls_days_to_expiry",
        "tls_hsts_enabled",
        # Homoglyph/content features
        "homoglyph_is_punycode",
        "homoglyph_mixed_scripts",
        "homoglyph_brand_similarity_score",
        "homoglyph_is_attack",
        # Redirect/IFrame/CSP behavior-content crossover
        "redirect_count",
        "redirect_multi_domain",
        "redirect_suspicious",
        "iframe_count",
        "iframe_external",
        "iframe_suspicious",
        "csp_present",
        "csp_issue_count",
        "csp_risk_level",
        # Fingerprinting + beaconing behavior
        "fingerprint_method_count",
        "fingerprint_score",
        "fingerprint_risk_level",
        "beacon_call_count",
        "beacon_suspicious_requests",
        "beacon_external_requests",
        "beacon_risk_level",
    )

    FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
        "url_length": (-1.0, 4096.0),
        "url_num_dots": (-1.0, 30.0),
        "url_num_hyphens": (-1.0, 50.0),
        "url_has_ip": (-1.0, 1.0),
        "url_num_suspicious_keywords": (-1.0, 12.0),
        "url_entropy": (-1.0, 8.0),
        "infra_has_mx_records": (-1.0, 1.0),
        "infra_num_a_records": (-1.0, 50.0),
        "infra_fast_flux_detected": (-1.0, 1.0),
        "infra_suspicious_ttl": (-1.0, 1.0),
        "tls_has_https": (-1.0, 1.0),
        "tls_certificate_valid": (-1.0, 1.0),
        "tls_is_self_signed": (-1.0, 1.0),
        "tls_days_to_expiry": (-1.0, 3650.0),
        "tls_hsts_enabled": (-1.0, 1.0),
        "homoglyph_is_punycode": (-1.0, 1.0),
        "homoglyph_mixed_scripts": (-1.0, 1.0),
        "homoglyph_brand_similarity_score": (-1.0, 1.0),
        "homoglyph_is_attack": (-1.0, 1.0),
        "redirect_count": (-1.0, 20.0),
        "redirect_multi_domain": (-1.0, 1.0),
        "redirect_suspicious": (-1.0, 1.0),
        "iframe_count": (-1.0, 30.0),
        "iframe_external": (-1.0, 1.0),
        "iframe_suspicious": (-1.0, 1.0),
        "csp_present": (-1.0, 1.0),
        "csp_issue_count": (-1.0, 10.0),
        "csp_risk_level": (-1.0, 2.0),
        "fingerprint_method_count": (-1.0, 12.0),
        "fingerprint_score": (-1.0, 12.0),
        "fingerprint_risk_level": (-1.0, 2.0),
        "beacon_call_count": (-1.0, 200.0),
        "beacon_suspicious_requests": (-1.0, 500.0),
        "beacon_external_requests": (-1.0, 500.0),
        "beacon_risk_level": (-1.0, 2.0),
    }

    SUBSCORE_GROUPS: dict[str, tuple[str, ...]] = {
        "url": (
            "url_length",
            "url_num_dots",
            "url_num_hyphens",
            "url_has_ip",
            "url_num_suspicious_keywords",
            "url_entropy",
        ),
        "infra": (
            "infra_has_mx_records",
            "infra_num_a_records",
            "infra_fast_flux_detected",
            "infra_suspicious_ttl",
            "tls_has_https",
            "tls_certificate_valid",
            "tls_is_self_signed",
            "tls_days_to_expiry",
            "tls_hsts_enabled",
        ),
        "content": (
            "homoglyph_is_punycode",
            "homoglyph_mixed_scripts",
            "homoglyph_brand_similarity_score",
            "homoglyph_is_attack",
            "iframe_count",
            "iframe_external",
            "iframe_suspicious",
            "csp_present",
            "csp_issue_count",
            "csp_risk_level",
        ),
        "behavior": (
            "redirect_count",
            "redirect_multi_domain",
            "redirect_suspicious",
            "fingerprint_method_count",
            "fingerprint_score",
            "fingerprint_risk_level",
            "beacon_call_count",
            "beacon_suspicious_requests",
            "beacon_external_requests",
            "beacon_risk_level",
        ),
    }

    def __init__(self) -> None:
        self._scaler = MinMaxScaler()
        mins = [self.FEATURE_BOUNDS[name][0] for name in self.FEATURE_SCHEMA]
        maxs = [self.FEATURE_BOUNDS[name][1] for name in self.FEATURE_SCHEMA]
        # Fit on deterministic bounds so single-sample transforms are stable.
        self._scaler.fit([mins, maxs])

    def _extract_raw_feature_map(self, phase_payload: dict[str, Any]) -> dict[str, float]:
        """Map multi-phase payload into fixed raw (unscaled) numeric features."""
        url_features = phase_payload.get("url_features", {})
        domain_features = phase_payload.get("domain_features", {})
        tls_features = phase_payload.get("tls_features", {})
        homoglyph_features = phase_payload.get("homoglyph_features", {})

        phishing_features = phase_payload.get("phishing_behavior_features", {})
        redirect = phishing_features.get("redirect_analysis", {}) if isinstance(phishing_features, dict) else {}
        iframe = phishing_features.get("iframe_analysis", {}) if isinstance(phishing_features, dict) else {}
        csp = phishing_features.get("csp_analysis", {}) if isinstance(phishing_features, dict) else {}

        fpb_features = phase_payload.get("fingerprint_beacon_features", {})
        fp = fpb_features.get("fingerprinting", {}) if isinstance(fpb_features, dict) else {}
        beacon = fpb_features.get("beaconing", {}) if isinstance(fpb_features, dict) else {}

        raw: dict[str, float] = {
            "url_length": _safe_float(url_features.get("url_length")),
            "url_num_dots": _safe_float(url_features.get("num_dots")),
            "url_num_hyphens": _safe_float(url_features.get("num_hyphens")),
            "url_has_ip": float(_safe_bool(url_features.get("has_ip"))),
            "url_num_suspicious_keywords": _safe_float(url_features.get("num_suspicious_keywords")),
            "url_entropy": _safe_float(url_features.get("entropy")),
            "infra_has_mx_records": float(_safe_bool(domain_features.get("has_mx_records"))),
            "infra_num_a_records": _safe_float(domain_features.get("num_a_records")),
            "infra_fast_flux_detected": float(_safe_bool(domain_features.get("fast_flux_detected"))),
            "infra_suspicious_ttl": float(_safe_bool(domain_features.get("suspicious_ttl"))),
            "tls_has_https": float(_safe_bool(tls_features.get("has_https"))),
            "tls_certificate_valid": float(_safe_bool(tls_features.get("certificate_valid"))),
            "tls_is_self_signed": float(_safe_bool(tls_features.get("is_self_signed"))),
            "tls_days_to_expiry": _safe_float(tls_features.get("days_to_expiry")),
            "tls_hsts_enabled": float(_safe_bool(tls_features.get("hsts_enabled"))),
            "homoglyph_is_punycode": float(_safe_bool(homoglyph_features.get("is_punycode"))),
            "homoglyph_mixed_scripts": float(_safe_bool(homoglyph_features.get("mixed_scripts"))),
            "homoglyph_brand_similarity_score": _safe_float(homoglyph_features.get("brand_similarity_score")),
            "homoglyph_is_attack": float(_safe_bool(homoglyph_features.get("is_homoglyph_attack"))),
            "redirect_count": _safe_float(redirect.get("count")),
            "redirect_multi_domain": 1.0 if _safe_len(redirect.get("domains")) > 1 else 0.0,
            "redirect_suspicious": float(_safe_bool(redirect.get("suspicious"))),
            "iframe_count": _safe_float(iframe.get("count")),
            "iframe_external": float(_safe_bool(iframe.get("external_iframes"))),
            "iframe_suspicious": float(_safe_bool(iframe.get("suspicious_iframe"))),
            "csp_present": float(_safe_bool(csp.get("present"))),
            "csp_issue_count": _safe_len(csp.get("issues")),
            "csp_risk_level": _encode_risk_level(csp.get("risk_level")),
            "fingerprint_method_count": _safe_len(fp.get("methods")),
            "fingerprint_score": _safe_float(fp.get("score")),
            "fingerprint_risk_level": _encode_risk_level(fp.get("risk")),
            "beacon_call_count": _safe_len(beacon.get("beacon_calls")),
            "beacon_suspicious_requests": _safe_float(beacon.get("suspicious_requests")),
            "beacon_external_requests": _safe_float(beacon.get("external_requests")),
            "beacon_risk_level": _encode_risk_level(beacon.get("risk")),
        }

        # Clamp to configured bounds to prevent out-of-range distortion.
        for name, value in raw.items():
            lo, hi = self.FEATURE_BOUNDS[name]
            if value < lo:
                raw[name] = lo
            elif value > hi:
                raw[name] = hi

        return raw

    def _normalize_feature_map(self, raw_map: dict[str, float]) -> dict[str, float]:
        """Normalize raw features into [0, 1] using fitted MinMaxScaler."""
        ordered = [raw_map[name] for name in self.FEATURE_SCHEMA]
        transformed = self._scaler.transform([ordered])[0]
        return {
            name: float(max(0.0, min(1.0, round(transformed[idx], 6))))
            for idx, name in enumerate(self.FEATURE_SCHEMA)
        }

    def _compute_sub_scores(self, normalized_map: dict[str, float]) -> dict[str, float]:
        """Compute grouped sub-scores from normalized features."""
        result: dict[str, float] = {}
        for group_name, feature_names in self.SUBSCORE_GROUPS.items():
            values = [normalized_map.get(name, 0.0) for name in feature_names]
            score = sum(values) / len(values) if values else 0.0
            result[group_name] = round(float(score), 6)
        return result

    def fuse_features(self, phase_payload: dict[str, Any]) -> dict[str, Any]:
        """Fuse cybersecurity phase outputs into ML-ready feature vector and sub-scores."""
        payload = phase_payload if isinstance(phase_payload, dict) else {}

        raw_map = self._extract_raw_feature_map(payload)
        normalized_map = self._normalize_feature_map(raw_map)
        feature_vector = [normalized_map[name] for name in self.FEATURE_SCHEMA]
        sub_scores = self._compute_sub_scores(normalized_map)

        return {
            "feature_vector": feature_vector,
            "sub_scores": {
                "url": sub_scores.get("url", 0.0),
                "infra": sub_scores.get("infra", 0.0),
                "content": sub_scores.get("content", 0.0),
                "behavior": sub_scores.get("behavior", 0.0),
            },
        }

    def save_features_to_csv(
        self,
        feature_payload: dict[str, Any],
        csv_path: str | Path,
        *,
        label: int | float | None = None,
        append: bool = True,
    ) -> None:
        """Persist fused features to CSV for ML dataset creation."""
        fused = self.fuse_features(feature_payload)
        vector = fused.get("feature_vector", [])

        if not isinstance(vector, list) or len(vector) != len(self.FEATURE_SCHEMA):
            raise ValueError("Invalid feature vector generated by fusion engine")

        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or not append

        mode = "a" if append else "w"
        with path.open(mode=mode, encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)

            header = list(self.FEATURE_SCHEMA)
            if label is not None:
                header.append("label")
            if write_header:
                writer.writerow(header)

            row = [float(value) for value in vector]
            if label is not None:
                row.append(float(label))
            writer.writerow(row)
