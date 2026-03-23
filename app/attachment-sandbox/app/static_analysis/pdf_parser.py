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


def _zeroed_pdf_features() -> dict[str, Any]:
    """Return a dict of PDF features all set to safe defaults."""
    return {
        "page_count": 0,
        "has_javascript": False,
        "has_embedded_files": False,
        "has_launch_action": False,
        "has_suspicious_urls": False,
    }


def extract_pdf_features(file_path: str) -> dict[str, Any]:
    """Extract PDF-specific features using pdfminer.six."""
    features = _zeroed_pdf_features()

    try:
        from pdfminer.high_level import extract_text
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

    except Exception:
        logger.exception("PDF parsing failed for %s", file_path)

    return features
