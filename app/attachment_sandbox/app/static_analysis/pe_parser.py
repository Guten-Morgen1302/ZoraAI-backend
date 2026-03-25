from __future__ import annotations

import logging
from typing import Any

from app.static_analysis.extractor import compute_entropy

logger = logging.getLogger(__name__)

# Win32 APIs commonly abused by malware
_SUSPICIOUS_APIS: set[str] = {
    "VirtualAllocEx",
    "VirtualAlloc",
    "WriteProcessMemory",
    "CreateRemoteThread",
    "NtUnmapViewOfSection",
    "URLDownloadToFile",
    "URLDownloadToFileA",
    "URLDownloadToFileW",
    "WinExec",
    "ShellExecuteA",
    "ShellExecuteW",
    "CreateProcessA",
    "CreateProcessW",
    "InternetOpenA",
    "InternetOpenUrlA",
    "HttpSendRequestA",
    "LoadLibraryA",
    "GetProcAddress",
    "SetWindowsHookEx",
    "RegSetValueEx",
    "RegSetValueExA",
    "IsDebuggerPresent",
    "NtSetInformationThread",
}

# Section names often injected by packers
_SUSPICIOUS_SECTIONS: set[str] = {
    "UPX0", "UPX1", "UPX2", ".aspack", ".adata",
    ".nsp0", ".nsp1", ".perplex", ".yP", ".petite",
}


def _zeroed_pe_features() -> dict[str, Any]:
    """Return a dict of PE features all set to safe defaults."""
    return {
        "max_section_entropy": 0.0,
        "section_count": 0,
        "has_suspicious_section": False,
        "import_count": 0,
        "suspicious_api_count": 0,
        "has_overlay": False,
    }


def extract_pe_features(file_path: str) -> dict[str, Any]:
    """Extract PE-specific features from an EXE/DLL file."""
    features = _zeroed_pe_features()

    try:
        import pefile

        pe = pefile.PE(file_path, fast_load=False)

        # ── Section analysis ────────────────────────────────────────────
        section_entropies: list[float] = []
        for section in pe.sections:
            name = section.Name.rstrip(b"\x00").decode("ascii", errors="ignore")
            section_entropies.append(compute_entropy(section.get_data()))
            if name in _SUSPICIOUS_SECTIONS:
                features["has_suspicious_section"] = True

        features["section_count"] = len(pe.sections)
        features["max_section_entropy"] = round(max(section_entropies), 4) if section_entropies else 0.0

        # ── Import analysis ─────────────────────────────────────────────
        total_imports = 0
        suspicious_count = 0
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    total_imports += 1
                    if imp.name:
                        api_name = imp.name.decode("ascii", errors="ignore")
                        if api_name in _SUSPICIOUS_APIS:
                            suspicious_count += 1

        features["import_count"] = total_imports
        features["suspicious_api_count"] = suspicious_count

        # ── Overlay detection ───────────────────────────────────────────
        overlay_offset = pe.get_overlay_data_start_offset()
        features["has_overlay"] = overlay_offset is not None and overlay_offset > 0

        pe.close()

    except Exception:
        logger.exception("PE parsing failed for %s", file_path)

    return features
