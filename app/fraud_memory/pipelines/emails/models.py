from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EmailFraudRecord:
    text: str
    subject: str
    sender: str
    urls: str
    label: str
    source_file: str
