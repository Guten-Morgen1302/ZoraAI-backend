from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SUSPICIOUS_URL_PATTERN = re.compile(
    r"https?://(?:\d{1,3}\.){3}\d{1,3}|"
    r"https?://[^\s/]+\.(?:ru|cn|tk|top|xyz|pw|cc|ws)\b",
    re.IGNORECASE,
)
_URL_EXTRACT_PATTERN = re.compile(
    r"https?://[^\s\"'<>()[\]{}]+",
    re.IGNORECASE,
)


def _zeroed_pdf_features() -> dict[str, Any]:
    """Return a dict of PDF features all set to safe defaults."""
    return {
        "page_count": 0,
        "has_javascript": False,
        "has_embedded_files": False,
        "has_launch_action": False,
        "has_suspicious_urls": False,
        "extracted_urls": [],
        "extracted_url_count": 0,
    }


def extract_pdf_features(file_path: str) -> dict[str, Any]:
    """Extract PDF-specific features using pdfminer.six."""
    features = _zeroed_pdf_features()

    try:
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfpage import PDFPage

        # ── Page count ──────────────────────────────────────────────────
        with open(file_path, "rb") as fh:
            parser = PDFParser(fh)
            doc = PDFDocument(parser)
            pages = list(PDFPage.create_pages(doc))
            features["page_count"] = len(pages)

        # ── Raw bytes scan for suspicious objects ───────────────────────
        with open(file_path, "rb") as fh:
            raw = fh.read()

        raw_text = raw.decode("latin-1", errors="ignore")

        features["has_javascript"] = bool(
            re.search(r"/JavaScript|/JS\s", raw_text, re.IGNORECASE)
        )
        features["has_embedded_files"] = bool(
            re.search(r"/EmbeddedFile|/FileAttachment", raw_text, re.IGNORECASE)
        )
        features["has_launch_action"] = bool(
            re.search(r"/Launch|/Action\s*/S\s*/Launch", raw_text, re.IGNORECASE)
        )
        features["has_suspicious_urls"] = bool(
            _SUSPICIOUS_URL_PATTERN.search(raw_text)
        )

        extracted_urls: list[str] = []
        seen_urls: set[str] = set()
        for match in _URL_EXTRACT_PATTERN.findall(raw_text):
            cleaned = match.rstrip("),.;\"'")
            if cleaned and cleaned not in seen_urls:
                seen_urls.add(cleaned)
                extracted_urls.append(cleaned)
            if len(extracted_urls) >= 25:
                break

        features["extracted_urls"] = extracted_urls
        features["extracted_url_count"] = len(extracted_urls)

    except ImportError as exc:
        logger.warning("PDF parser dependency unavailable for %s: %s", file_path, exc)
    except Exception:
        logger.exception("PDF parsing failed for %s", file_path)

    return features
