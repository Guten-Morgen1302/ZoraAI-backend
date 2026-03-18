from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class UrlFraudRecord:
    text: str
    label: str
    source_file: str
