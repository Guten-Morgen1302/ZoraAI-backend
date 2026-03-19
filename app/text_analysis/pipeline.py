from __future__ import annotations

import logging
import re
import string
from urllib.parse import urlparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.text_analysis.preprocessing import preprocess_text
from app.text_analysis.url_analyzer import analyze_urls

try:
    import tldextract
except ImportError:  # pragma: no cover - optional runtime dependency
    tldextract = None

try:
    import spacy
except ImportError:  # pragma: no cover - optional runtime dependency
    spacy = None

logger = logging.getLogger("zora.text_pipeline")

URL_PATTERN = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
URL_OR_DOMAIN_PATTERN = re.compile(
    r"(?i)\b((?:https?://|www\.)[^\s<>'\"]+|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>'\"]*)?)"
)
WHITESPACE_PATTERN = re.compile(r"\s+")
WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)

URGENCY_PHRASES = (
    "urgent",
    "immediately",
    "verify now",
    "account suspended",
    "limited time",
)

_NLP_MODEL: Any | None = None
_NLP_DISABLED = False


def _deduplicate_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _get_spacy_model():
    global _NLP_MODEL, _NLP_DISABLED

    if _NLP_DISABLED:
        return None
    if _NLP_MODEL is not None:
        return _NLP_MODEL
    if spacy is None:
        _NLP_DISABLED = True
        logger.warning("spaCy is not installed; entity extraction will return empty entities")
        return None

    try:
        _NLP_MODEL = spacy.load("en_core_web_sm")
    except OSError:
        _NLP_DISABLED = True
        logger.warning(
            "spaCy model 'en_core_web_sm' is not available; entity extraction will return empty entities"
        )
        return None

    return _NLP_MODEL


def clean_text(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    return preprocess_text(raw_text)["clean_text"]


def extract_urls(text: str) -> dict[str, Any]:
    raw_urls = [match.group(1) for match in URL_OR_DOMAIN_PATTERN.finditer(text)]

    urls: list[str] = []
    for raw_url in raw_urls:
        cleaned_url = raw_url.rstrip(".,;!?)")
        if cleaned_url.startswith("www."):
            cleaned_url = f"https://{cleaned_url}"
        elif not cleaned_url.startswith(("http://", "https://")):
            cleaned_url = f"https://{cleaned_url}"
        urls.append(cleaned_url)

    urls = _deduplicate_preserve_order(urls)

    extracted_domains: list[str] = []
    for url in urls:
        if tldextract is not None:
            parsed = tldextract.extract(url)
            domain = ".".join(part for part in [parsed.domain, parsed.suffix] if part)
        else:
            domain = urlparse(url).netloc.lower()
        if domain:
            extracted_domains.append(domain)

    domains = _deduplicate_preserve_order(extracted_domains)

    return {
        "urls": urls,
        "domain": domains[0] if domains else None,
        "domains": domains,
        "url_count": len(urls),
    }


def detect_urgency(cleaned_text: str) -> float:
    phrase_hits = sum(1 for phrase in URGENCY_PHRASES if phrase in cleaned_text)
    urgency_score = phrase_hits / len(URGENCY_PHRASES)
    return round(min(1.0, urgency_score), 4)


def extract_stylometry(raw_text: str, cleaned_text: str) -> dict[str, float | int]:
    words = WORD_PATTERN.findall(cleaned_text)
    text_without_urls = URL_PATTERN.sub("", cleaned_text)
    sentence_count = len([segment for segment in SENTENCE_SPLIT_PATTERN.split(text_without_urls) if segment.strip()])

    alphabetic_chars = [char for char in raw_text if char.isalpha()]
    uppercase_chars = [char for char in alphabetic_chars if char.isupper()]

    punctuation_count = sum(1 for char in raw_text if char in string.punctuation)
    non_space_count = sum(1 for char in raw_text if not char.isspace())

    avg_word_length = (sum(len(word) for word in words) / len(words)) if words else 0.0
    uppercase_ratio = (len(uppercase_chars) / len(alphabetic_chars)) if alphabetic_chars else 0.0
    punctuation_density = (punctuation_count / non_space_count) if non_space_count else 0.0

    emoji_count = len(EMOJI_PATTERN.findall(raw_text))

    return {
        "word_count": len(words),
        "sentence_count": sentence_count,
        "average_word_length": round(avg_word_length, 4),
        "uppercase_ratio": round(uppercase_ratio, 4),
        "punctuation_density": round(punctuation_density, 4),
        "emoji_count": emoji_count,
    }


def extract_entities(cleaned_text: str) -> dict[str, list[str]]:
    nlp = _get_spacy_model()
    if nlp is None:
        return {"ORG": [], "PERSON": [], "LOCATION": []}

    doc = nlp(cleaned_text)

    orgs = _deduplicate_preserve_order([ent.text for ent in doc.ents if ent.label_ == "ORG"])
    persons = _deduplicate_preserve_order([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    locations = _deduplicate_preserve_order(
        [ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC", "FAC"}]
    )

    return {"ORG": orgs, "PERSON": persons, "LOCATION": locations}


def process_text(raw_message: str) -> dict[str, Any]:
    preprocessing_output = preprocess_text(raw_message)
    cleaned_text = preprocessing_output["clean_text"]
    urls = preprocessing_output["urls"]
    url_features = extract_urls(cleaned_text)
    url_risk = analyze_urls(urls)
    sanitized_urls = list(url_risk.get("sanitized_urls") or urls)
    urgency_score = detect_urgency(cleaned_text)
    stylometry_features = extract_stylometry(raw_text=raw_message, cleaned_text=cleaned_text)
    entities = extract_entities(cleaned_text)

    return {
        "clean_text": cleaned_text,
        "cleaned_text": cleaned_text,
        "urls": sanitized_urls,
        "phones": preprocessing_output["phones"],
        "emails": preprocessing_output["emails"],
        "domain": url_features["domain"],
        "url_count": url_features["url_count"],
        "url_risk_score": url_risk["url_risk_score"],
        "url_risk_flags": url_risk["flags"],
        "urgency_score": urgency_score,
        "stylometry_features": stylometry_features,
        "entities": entities,
    }


@dataclass
class PreprocessResult:
    link_count: int
    urgency_score: float
    features: dict[str, Any]


class TextPreprocessingPipeline:
    """Production-ready text preprocessing pipeline wrapper."""

    def run(self, text: str) -> PreprocessResult:
        features = process_text(text)
        return PreprocessResult(
            link_count=int(features["url_count"]),
            urgency_score=float(features["urgency_score"]),
            features=features,
        )
