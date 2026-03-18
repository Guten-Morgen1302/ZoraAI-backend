from __future__ import annotations

from dataclasses import dataclass


MALICIOUS_LABELS = {"spam", "phishing", "scam", "fraud", "malicious"}


def _normalize_label(label: str | None) -> str:
    if not label:
        return ""
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def _is_malicious_label(label: str | None) -> bool:
    normalized = _normalize_label(label)
    return normalized in MALICIOUS_LABELS


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


@dataclass
class ThreatScoreResult:
    risk_score: float
    fraud_type: str
    confidence: float
    flags: list[str]
    explanation: str
    nlp_score: float
    similarity_score: float
    stylometry_score: float


def score_sms_threat(
    *,
    nlp_label: str | None,
    nlp_confidence: float,
    similarity_score: float,
    stylometry_score: float,
    url_risk_score: float,
    urgency_score: float,
    matched_label: str | None,
    similarity_high_risk: bool,
) -> ThreatScoreResult:
    nlp_confidence = _clamp(float(nlp_confidence))
    similarity_score = _clamp(float(similarity_score))
    stylometry_score = _clamp(float(stylometry_score))
    url_risk_score = _clamp(float(url_risk_score))
    urgency_score = _clamp(float(urgency_score))

    nlp_is_malicious = _is_malicious_label(nlp_label)
    nlp_score = nlp_confidence if nlp_is_malicious else (1.0 - nlp_confidence)

    final_score = (0.4 * nlp_score) + (0.3 * similarity_score) + (0.3 * stylometry_score)
    final_score = _clamp(final_score)

    flags: list[str] = []
    if urgency_score >= 0.4:
        flags.append("urgent language")
    if similarity_high_risk or _is_malicious_label(matched_label):
        flags.append("known scam pattern")
    if url_risk_score >= 0.25:
        flags.append("suspicious url")
    if nlp_is_malicious and nlp_confidence >= 0.65:
        flags.append(f"nlp model indicates {_normalize_label(nlp_label)}")

    if final_score >= 0.7:
        fraud_type = "sms_phishing"
    elif final_score >= 0.4:
        fraud_type = "suspicious_sms"
    else:
        fraud_type = "safe"

    confidence = _clamp((final_score + max(nlp_confidence, similarity_score, stylometry_score)) / 2.0)

    reason_parts: list[str] = []
    if "known scam pattern" in flags:
        reason_parts.append("Matches known phishing/scam vectors")
    if "urgent language" in flags:
        reason_parts.append("uses urgency tone")
    if "suspicious url" in flags:
        reason_parts.append("contains suspicious URL patterns")
    if not reason_parts:
        reason_parts.append("no strong fraud indicators were detected")

    explanation = " + ".join(reason_parts)

    return ThreatScoreResult(
        risk_score=round(final_score, 4),
        fraud_type=fraud_type,
        confidence=round(confidence, 4),
        flags=flags,
        explanation=explanation,
        nlp_score=round(nlp_score, 4),
        similarity_score=round(similarity_score, 4),
        stylometry_score=round(stylometry_score, 4),
    )