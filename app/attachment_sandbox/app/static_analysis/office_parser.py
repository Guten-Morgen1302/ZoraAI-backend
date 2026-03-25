from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Patterns that indicate obfuscated VBA strings (e.g. Chr() concatenation)
_OBFUSCATION_PATTERN = re.compile(
    r"(?:Chr\(\d+\)\s*[&+]\s*){3,}|"             # Chr(72) & Chr(101) & ...
    r'(?:"\w"\s*[&+]\s*){5,}|'                     # "H" & "e" & "l" & ...
    r"(?:Replace|StrReverse|Mid\$?)\s*\(",          # String manipulation calls
    re.IGNORECASE,
)

_DDE_PATTERN = re.compile(
    r"DDE|DDEAUTO|\\ddeauto|\\dde\b", re.IGNORECASE
)

_EXTERNAL_LINK_PATTERN = re.compile(
    r"https?://|ftp://|\\\\[A-Za-z0-9]", re.IGNORECASE
)

_AUTOOPEN_KEYWORDS: set[str] = {
    "autoopen", "auto_open", "autoexec", "auto_close",
    "document_open", "document_close", "workbook_open",
}


def _zeroed_office_features() -> dict[str, Any]:
    """Return a dict of Office features all set to safe defaults."""
    return {
        "has_macros": False,
        "has_auto_open": False,
        "has_external_links": False,
        "has_dde": False,
        "has_obfuscated_strings": False,
    }


def extract_office_features(file_path: str) -> dict[str, Any]:
    """Extract Office document features using oletools (olevba)."""
    features = _zeroed_office_features()

    try:
        from oletools.olevba import VBA_Parser

        vba_parser = VBA_Parser(file_path)

        if vba_parser.detect_vba_macros():
            features["has_macros"] = True

            for _, _, vba_code in vba_parser.extract_macros():
                code_lower = vba_code.lower()

                # Check for auto-execution triggers
                for keyword in _AUTOOPEN_KEYWORDS:
                    if keyword in code_lower:
                        features["has_auto_open"] = True
                        break

                # Check for external links / network calls
                if _EXTERNAL_LINK_PATTERN.search(vba_code):
                    features["has_external_links"] = True

                # Check for DDE
                if _DDE_PATTERN.search(vba_code):
                    features["has_dde"] = True

                # Check for obfuscated strings
                if _OBFUSCATION_PATTERN.search(vba_code):
                    features["has_obfuscated_strings"] = True

        vba_parser.close()

    except Exception:
        logger.exception("Office parsing failed for %s", file_path)

    return features
