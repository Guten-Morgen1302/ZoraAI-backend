from __future__ import annotations

from dataclasses import dataclass


MALICIOUS_LABELS = {"spam", "phishing", "scam", "fraud", "malicious"}


def _normalize_label(label: str | None) -> str:
    if not label:
        return ""
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def _is_malicious_label(label: str | None) -> bool:
    return _normalize_label(label) in MALICIOUS_LABELS


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


@dataclass
class EmailThreatScoreResult:
    final_score: float
    nlp_score: float
    similarity_score: float
    stylometry_score: float
    confidence: float
    fraud_type: str


def score_email_threat(
    *,
    nlp_label: str | None,
    nlp_confidence: float,
    similarity_score: float,
    stylometry_score: float,
) -> EmailThreatScoreResult:
    nlp_confidence = _clamp(float(nlp_confidence))
    similarity_score = _clamp(float(similarity_score))
    stylometry_score = _clamp(float(stylometry_score))

    nlp_is_malicious = _is_malicious_label(nlp_label)
    nlp_score = nlp_confidence if nlp_is_malicious else (1.0 - nlp_confidence)

    # Final Score Formula:
    # final_score = 0.4 * nlp_score + 0.3 * similarity_score + 0.3 * stylometry_score
    final_score = (0.4 * nlp_score) + (0.3 * similarity_score) + (0.3 * stylometry_score)
    final_score = _clamp(final_score)

    confidence = _clamp((final_score + max(nlp_score, similarity_score, stylometry_score)) / 2.0)

    if final_score >= 0.7:
        fraud_type = "email_phishing"
    elif final_score >= 0.4:
        fraud_type = "suspicious_email"
    else:
        fraud_type = "safe"

    return EmailThreatScoreResult(
        final_score=round(final_score, 4),
        nlp_score=round(nlp_score, 4),
        similarity_score=round(similarity_score, 4),
        stylometry_score=round(stylometry_score, 4),
        confidence=round(confidence, 4),
        fraud_type=fraud_type,
    )
