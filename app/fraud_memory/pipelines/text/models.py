from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TextFraudRecord:
    text: str
    label: str
    source_file: str
    dataset_type: str
