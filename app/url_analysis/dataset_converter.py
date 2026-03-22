"""Dataset conversion utilities for URL-analysis ML training pipelines.

Converts source datasets such as `urls_dataset_small.csv` into URL-analysis
feature CSVs aligned with `FeatureFusionEngine.FEATURE_SCHEMA`.
"""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.url_analysis.feature_fusion_engine import FeatureFusionEngine
from app.url_analysis.url_analysis import extract_phase_4_features_async


# Update these paths when running conversion for a different dataset.
INPUT_CSV_PATH = Path("app/fraud_memory/data/data_url_phishing/urls_dataset_1.csv")
OUTPUT_DIR = Path("app/fraud_memory/data/data_url_phishing")
OUTPUT_FILE_NAME = "urls_processed_30.csv"

# Per-URL live analysis timeout (seconds) for full phase-4 execution.
PER_URL_ANALYSIS_TIMEOUT_SEC = 90.0


def _safe_float(value: Any, default: float = -1.0) -> float:
    """Convert value into float with safe fallback."""
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    """Convert value into int with safe fallback."""
    parsed = _safe_float(value, default=float(default))
    try:
        return int(parsed)
    except (TypeError, ValueError):
        return default


def _extract_row_url(row: dict[str, Any]) -> str:
    """Extract URL value from source row with strict cleanup."""
    return str(row.get("URL") or "").strip()


@dataclass
class ConversionSummary:
    """Result summary for dataset conversion operations."""

    input_path: str
    output_path: str
    total_rows: int
    converted_rows: int
    skipped_rows: int
    error_rows: int
    errors: list[str]


class URLDatasetToFusionCSVConverter:
    """Convert external URL datasets into URL-analysis fusion CSV format."""

    def __init__(self) -> None:
        self._fusion_engine = FeatureFusionEngine()

    async def _analyze_url_phase_payload(self, url: str) -> dict[str, Any]:
        """Run full phase-4 URL analysis for one URL with timeout guard."""
        return await asyncio.wait_for(
            extract_phase_4_features_async(url),
            timeout=PER_URL_ANALYSIS_TIMEOUT_SEC,
        )

    def _read_rows(self, input_csv_path: Path) -> list[dict[str, Any]]:
        """Read CSV rows with fallback encodings to handle messy datasets."""
        if not input_csv_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {input_csv_path}")

        encodings = ("utf-8-sig", "utf-8", "latin-1")
        last_error: Exception | None = None
        for encoding in encodings:
            try:
                with input_csv_path.open("r", encoding=encoding, newline="") as handle:
                    reader = csv.DictReader(handle)
                    return [dict(row) for row in reader]
            except UnicodeDecodeError as exc:
                last_error = exc
                continue

        raise ValueError(f"Unable to decode CSV file: {input_csv_path}; last_error={last_error}")

    def convert(
        self,
        *,
        input_csv_path: str | Path,
        output_csv_path: str | Path,
        fail_on_row_error: bool = False,
        include_metadata: bool = True,
    ) -> ConversionSummary:
        """Convert source dataset rows into fusion CSV using live URL analysis."""
        input_path = Path(input_csv_path)
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self._read_rows(input_path)
        if not rows:
            raise ValueError(f"Input dataset is empty: {input_path}")

        feature_header = list(self._fusion_engine.FEATURE_SCHEMA)
        if include_metadata:
            base_header = [
                "source_url",
                "source_domain",
                "label",
                "analysis_error",
                "sandbox_error",
            ]
        else:
            base_header = ["label", "analysis_error", "sandbox_error"]
        header = base_header + feature_header

        total_rows = 0
        converted_rows = 0
        skipped_rows = 0
        error_rows = 0
        errors: list[str] = []

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)

            with asyncio.Runner() as runner:
                for idx, row in enumerate(rows, start=2):
                    total_rows += 1

                    try:
                        url = _extract_row_url(row)
                        if not url:
                            skipped_rows += 1
                            errors.append(f"line {idx}: skipped row with empty URL")
                            continue

                        label = _safe_int(row.get("label"), default=-1)
                        if label not in {0, 1}:
                            label = -1

                        phase_payload = runner.run(self._analyze_url_phase_payload(url))
                        fused = phase_payload.get("fused_features")
                        if not isinstance(fused, dict):
                            fused = self._fusion_engine.fuse_features(phase_payload)

                        vector = fused.get("feature_vector", [])
                        if not isinstance(vector, list) or len(vector) != len(feature_header):
                            raise ValueError("invalid fused feature vector size")

                        parsed = urlparse(url)
                        domain = (parsed.hostname or str(row.get("Domain") or "").strip()).lower()
                        sandbox_features = phase_payload.get("sandbox_features", {})
                        sandbox_error = ""
                        if isinstance(sandbox_features, dict):
                            sandbox_error = str(sandbox_features.get("error", "") or "")

                        output_row: list[Any] = []
                        if include_metadata:
                            output_row.extend([url, domain, label, "", sandbox_error])
                        else:
                            output_row.extend([label, "", sandbox_error])
                        output_row.extend(float(value) for value in vector)

                        writer.writerow(output_row)
                        converted_rows += 1

                    except Exception as exc:
                        error_rows += 1
                        message = f"line {idx}: {exc.__class__.__name__}: {exc}"
                        errors.append(message)

                        # Persist error row with default feature vector for traceability.
                        fallback_vector = [0.0] * len(feature_header)
                        url = _extract_row_url(row)
                        parsed = urlparse(url) if url else None
                        domain = (
                            (parsed.hostname if parsed is not None else "")
                            or str(row.get("Domain") or "").strip().lower()
                        )
                        label = _safe_int(row.get("label"), default=-1)
                        if label not in {0, 1}:
                            label = -1

                        error_row: list[Any] = []
                        if include_metadata:
                            error_row.extend([url, domain, label, message, "conversion_failed"])
                        else:
                            error_row.extend([label, message, "conversion_failed"])
                        error_row.extend(fallback_vector)
                        writer.writerow(error_row)

                        if fail_on_row_error:
                            raise RuntimeError(message) from exc

        return ConversionSummary(
            input_path=str(input_path),
            output_path=str(output_path),
            total_rows=total_rows,
            converted_rows=converted_rows,
            skipped_rows=skipped_rows,
            error_rows=error_rows,
            errors=errors,
        )


def main() -> None:
    """Run conversion using hardcoded input and output paths."""
    input_path = INPUT_CSV_PATH
    output_path = OUTPUT_DIR / OUTPUT_FILE_NAME

    converter = URLDatasetToFusionCSVConverter()
    summary = converter.convert(
        input_csv_path=input_path,
        output_csv_path=output_path,
        fail_on_row_error=False,
        include_metadata=True,
    )

    print(
        {
            "input_path": summary.input_path,
            "output_path": summary.output_path,
            "total_rows": summary.total_rows,
            "converted_rows": summary.converted_rows,
            "skipped_rows": summary.skipped_rows,
            "error_rows": summary.error_rows,
            "errors": summary.errors[:10],
        }
    )


if __name__ == "__main__":
    main()
