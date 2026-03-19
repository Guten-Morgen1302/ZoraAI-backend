from __future__ import annotations

from dataclasses import dataclass

from app.text_analysis.llm_reasoner import analyze_with_llm


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
    llm_enhanced: bool
    llm_explanation: str | None
    nlp_score: float
    similarity_score: float
    stylometry_score: float


def _signals_conflict(
    *,
    nlp_is_malicious: bool,
    nlp_confidence: float,
    similarity_score: float,
    stylometry_score: float,
) -> bool:
    # Conflicting examples:
    # 1) NLP is very sure it is safe, but behavior/similarity looks malicious.
    # 2) NLP is very sure it is malicious, but other signals are weak.
    safe_but_other_high = (not nlp_is_malicious) and nlp_confidence >= 0.8 and (
        similarity_score >= 0.75 or stylometry_score >= 0.75
    )
    malicious_but_other_low = nlp_is_malicious and nlp_confidence >= 0.8 and (
        similarity_score <= 0.35 and stylometry_score <= 0.35
    )
    return safe_but_other_high or malicious_but_other_low


def score_sms_threat(
    *,
    sms_text: str,
    nlp_label: str | None,
    nlp_confidence: float,
    similarity_score: float,
    stylometry_score: float,
    url_risk_score: float,
    url_flags: list[str] | None,
    rule_flags: list[str] | None,
    urgency_score: float,
    matched_label: str | None,
    similarity_high_risk: bool,
    force_llm_explanation: bool = False,
) -> ThreatScoreResult:
    nlp_confidence = _clamp(float(nlp_confidence))
    similarity_score = _clamp(float(similarity_score))
    stylometry_score = _clamp(float(stylometry_score))
    url_risk_score = _clamp(float(url_risk_score))
    urgency_score = _clamp(float(urgency_score))

    nlp_is_malicious = _is_malicious_label(nlp_label)
    nlp_score = nlp_confidence if nlp_is_malicious else (1.0 - nlp_confidence)
    url_flags = url_flags or []
    rule_flags = rule_flags or []

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

    llm_enhanced = False
    llm_explanation: str | None = None
    llm_result: dict | None = None
    should_call_llm = force_llm_explanation or (nlp_confidence < 0.7) or _signals_conflict(
        nlp_is_malicious=nlp_is_malicious,
        nlp_confidence=nlp_confidence,
        similarity_score=similarity_score,
        stylometry_score=stylometry_score,
    )

    if should_call_llm:
        llm_result = analyze_with_llm(
            {
                "sms_text": sms_text,
                "nlp_prediction": nlp_label,
                "nlp_confidence": nlp_confidence,
                "url_flags": url_flags,
                "stylometry_score": stylometry_score,
                "similarity_score": similarity_score,
                "rule_flags": rule_flags,
            }
        )
        llm_label = _normalize_label(llm_result.get("final_label"))
        llm_confidence = _clamp(float(llm_result.get("confidence") or 0.0))
        llm_explanation = str(llm_result.get("explanation") or "").strip() or None

        if llm_label in {"safe", "spam", "phishing", "scam"} and llm_confidence > 0:
            llm_enhanced = True
            llm_score = llm_confidence if llm_label in MALICIOUS_LABELS else (1.0 - llm_confidence)
            final_score = _clamp((0.7 * final_score) + (0.3 * llm_score))
            confidence = _clamp((confidence + llm_confidence) / 2.0)

            if llm_label == "safe":
                fraud_type = "safe"
            elif llm_label == "spam":
                fraud_type = "sms_spam"
            elif llm_label == "scam":
                fraud_type = "sms_scam"
            else:
                fraud_type = "sms_phishing"

            if llm_explanation:
                explanation = llm_explanation

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
        llm_enhanced=llm_enhanced,
        llm_explanation=llm_explanation,
        nlp_score=round(nlp_score, 4),
        similarity_score=round(similarity_score, 4),
        stylometry_score=round(stylometry_score, 4),
    )