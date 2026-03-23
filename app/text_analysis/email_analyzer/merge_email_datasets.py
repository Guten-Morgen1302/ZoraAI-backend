from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd

TEXT_COLUMNS = ("text", "body", "message", "content")
LABEL_COLUMNS = ("label", "Label", "is_spam", "target", "class")

PHISHING_VALUES = {"1", "1.0", "true", "yes", "phishing", "scam", "spam", "fraud", "malicious"}
GENUINE_VALUES = {"0", "0.0", "false", "no", "genuine", "safe", "ham", "legitimate", "benign"}


def _set_max_csv_field_size() -> None:
    max_size = sys.maxsize
    while max_size > 0:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _extract_first(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    for col in columns:
        value = _normalize_text(row.get(col))
        if value:
            return value
    return ""


def _normalize_label(row: dict[str, Any]) -> int | None:
    raw_value = ""
    for col in LABEL_COLUMNS:
        raw_value = _normalize_text(row.get(col))
        if raw_value:
            break

    if not raw_value:
        return None

    lowered = raw_value.lower()
    if lowered in PHISHING_VALUES:
        return 1
    if lowered in GENUINE_VALUES:
        return 0

    try:
        numeric = float(lowered)
    except ValueError:
        return None

    if numeric == 1.0:
        return 1
    if numeric == 0.0:
        return 0
    return None


def _read_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"processed": 0, "kept": 0, "skipped": 0}
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fp:
        reader = csv.DictReader(fp)
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error:
                stats["skipped"] += 1
                continue

            stats["processed"] += 1
            text = _extract_first(row, TEXT_COLUMNS)
            label = _normalize_label(row)
            if not text or label is None:
                stats["skipped"] += 1
                continue

            rows.append({"text": text, "label": int(label)})
            stats["kept"] += 1

    return pd.DataFrame(rows, columns=["text", "label"]), stats


def merge_email_datasets(data_dir: Path, output_csv: Path) -> dict[str, Any]:
    _set_max_csv_field_size()

    files = [
        data_dir / "cleaned_ceas.csv",
        data_dir / "ling.csv",
        data_dir / "nazario.csv",
        data_dir / "nigerian_fraud.csv",
        data_dir / "spamassasin.csv",
    ]

    merged_parts: list[pd.DataFrame] = []
    file_stats: dict[str, dict[str, int]] = {}

    for file_path in files:
        if not file_path.exists():
            file_stats[file_path.name] = {"processed": 0, "kept": 0, "skipped": 0}
            continue

        df, stats = _read_dataset(file_path)
        file_stats[file_path.name] = stats
        if not df.empty:
            merged_parts.append(df)

    if not merged_parts:
        merged_df = pd.DataFrame(columns=["text", "label"])
    else:
        merged_df = pd.concat(merged_parts, ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_csv, index=False)

    summary = {
        "output_csv": str(output_csv),
        "rows": int(len(merged_df)),
        "label_distribution": {int(k): int(v) for k, v in merged_df["label"].value_counts().to_dict().items()}
        if not merged_df.empty
        else {},
        "files": file_stats,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge email scam datasets into text,label CSV.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("app/fraud_memory/data/data_email_scam"),
        help="Directory containing email datasets.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("app/text_analysis/email_analyzer/merged_email_dataset.csv"),
        help="Output merged CSV path.",
    )
    args = parser.parse_args()

    summary = merge_email_datasets(args.data_dir, args.output_csv)
    print("Merge complete")
    print(summary)


if __name__ == "__main__":
    main()
