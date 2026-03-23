from __future__ import annotations

import logging
import mimetypes
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Extension-based fallback map ────────────────────────────────────────────
_EXT_MIME_MAP: dict[str, str] = {
    ".exe": "application/x-dosexec",
    ".dll": "application/x-dosexec",
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".zip": "application/zip",
    ".rar": "application/x-rar-compressed",
    ".7z": "application/x-7z-compressed",
    ".tar": "application/x-tar",
    ".gz": "application/gzip",
    ".py": "text/x-python",
    ".js": "application/javascript",
    ".ps1": "text/x-powershell",
    ".bat": "text/x-msdos-batch",
    ".vbs": "text/vbscript",
    ".sh": "text/x-shellscript",
}

# ── MIME → category mapping ─────────────────────────────────────────────────
_PE_MIMES = {"application/x-dosexec", "application/x-msdownload", "application/vnd.microsoft.portable-executable"}
_PDF_MIMES = {"application/pdf"}
_OFFICE_MIMES = {
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-office",
}
_ARCHIVE_MIMES = {
    "application/zip",
    "application/x-rar-compressed",
    "application/x-7z-compressed",
    "application/x-tar",
    "application/gzip",
    "application/x-bzip2",
}
_SCRIPT_MIMES = {
    "text/x-python",
    "application/javascript",
    "text/x-powershell",
    "text/x-msdos-batch",
    "text/vbscript",
    "text/x-shellscript",
}


def detect_mime(file_path: str) -> str:
    """Return the MIME type of *file_path*, falling back to extension sniffing."""
    try:
        import magic  # python-magic

        mime: str = magic.from_file(file_path, mime=True)
        if mime and mime != "application/octet-stream":
            return mime
    except Exception:
        logger.debug("python-magic failed for %s, falling back to extension", file_path)

    # Extension-based fallback
    ext = os.path.splitext(file_path)[1].lower()
    fallback: Optional[str] = _EXT_MIME_MAP.get(ext)
    if fallback:
        return fallback

    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or "application/octet-stream"


def get_file_category(mime: str) -> str:
    """Map a MIME type string to a high-level file category."""
    if mime in _PE_MIMES:
        return "pe"
    if mime in _PDF_MIMES:
        return "pdf"
    if mime in _OFFICE_MIMES:
        return "office"
    if mime in _ARCHIVE_MIMES:
        return "archive"
    if mime in _SCRIPT_MIMES:
        return "script"
    return "unknown"
