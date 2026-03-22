"""
prepare_dataset.py — Build training data from EMBER pre-vectorized parquet.

Maps EMBER's 2381 vectorized features to the 23 FEATURE_COLS defined in
classifier.py using fully vectorized pandas/numpy operations.

EMBER 2018 v2 vector layout (F1..F2381):
  F1-F256     : Byte histogram (256 bins)
  F257-F512   : Byte-entropy histogram (256 bins)
  F513        : numstrings
  F514        : avlength (average string length)
  F515        : printables
  F612        : string entropy
  F613        : paths count
  F614        : urls count
  F615        : registry key count
  F616        : MZ header count
  F617        : file size
  F618        : virtual size
  F619-F626   : general flags (has_debug, exports, imports, etc.)
  F689-F943   : Section features (names, sizes, entropies)
  F944-F2223  : Import hash bins
  Label       : 0.0=benign, 1.0=malicious
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SANDBOX_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _SANDBOX_ROOT not in sys.path:
    sys.path.insert(0, _SANDBOX_ROOT)

from app.static_analysis.classifier import FEATURE_COLS

logger = logging.getLogger(__name__)

# _SANDBOX_ROOT = .../attachment-sandbox/  (package root for imports)
# Data lives at attachment-sandbox/data/
_DATA_ROOT = Path(_SANDBOX_ROOT) / "data"

_EMBER_DIR = os.environ.get(
    "EMBER_DATA_DIR",
    str(_DATA_ROOT / "emberdataset"),
)
_PROCESSED_DIR = os.environ.get(
    "PROCESSED_DATA_DIR",
    str(_DATA_ROOT / "processed"),
)
_FEATURES_PATH = os.environ.get(
    "STATIC_FEATURES_PATH",
    os.path.join(_PROCESSED_DIR, "static_features.parquet"),
)

# Byte histogram column names (F1..F256)
_HIST_COLS = [f"F{i}" for i in range(1, 257)]
# Section entropy histogram bins (F739..F788, 50 bins)
_SECT_ENTROPY_COLS = [f"F{i}" for i in range(739, 789)]
# Import hash bins to check for suspicious APIs (F1200..F1280)
_IMPORT_PROBE_COLS = [f"F{i}" for i in range(1200, 1280)]
# Section name hash bins (F689..F738, 50 bins)
_SECT_NAME_COLS = [f"F{i}" for i in range(689, 739)]


def _vectorized_entropy(df: pd.DataFrame) -> np.ndarray:
    """Compute Shannon entropy from byte histograms for all rows at once."""
    hist = df[_HIST_COLS].values.astype(np.float64)
    totals = hist.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1, totals)  # avoid div-by-zero
    probs = hist / totals
    # -sum(p * log2(p)) with log2(0) handled
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.where(probs > 0, np.log2(probs), 0.0)
    entropy = -np.sum(probs * log_probs, axis=1)
    return np.round(entropy, 4)


def map_ember_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Map an entire EMBER DataFrame to our 23 FEATURE_COLS (vectorized)."""
    out = pd.DataFrame(index=df.index)

    # ── Base features ───────────────────────────────────────────────────
    out["file_size"] = df["F617"].astype(np.float32)
    out["entropy"] = _vectorized_entropy(df).astype(np.float32)
    out["strings_count"] = df["F513"].astype(np.float32)
    out["has_ip_pattern"] = (df["F614"] > 0).astype(np.float32)
    out["has_registry_keys"] = (df["F615"] > 0).astype(np.float32)
    out["has_powershell"] = np.float32(0.0)  # not in EMBER
    out["has_base64_blob"] = (
        (df["F514"] > 40) & (df["F515"] > 100)
    ).astype(np.float32)

    # ── PE features ─────────────────────────────────────────────────────
    out["section_count"] = df["F689"].astype(np.float32)

    # max section entropy proxy: highest non-zero bin in entropy histogram
    sect_ent = df[_SECT_ENTROPY_COLS].values
    highest_bin = np.zeros(len(df), dtype=np.float32)
    for bin_idx in range(sect_ent.shape[1] - 1, -1, -1):
        mask = (sect_ent[:, bin_idx] > 0) & (highest_bin == 0)
        highest_bin[mask] = bin_idx
    out["max_section_entropy"] = np.round(highest_bin / 49.0 * 8.0, 4)

    # suspicious section names
    sect_names = df[_SECT_NAME_COLS].values
    out["has_suspicious_section"] = (sect_names.sum(axis=1) > 10).astype(np.float32)

    out["import_count"] = df["F621"].astype(np.float32)

    # suspicious API proxy from import hash region
    imp_probes = df[_IMPORT_PROBE_COLS].values
    out["suspicious_api_count"] = (imp_probes > 0).sum(axis=1).astype(np.float32)

    # overlay heuristic
    fsize = df["F617"].values.astype(np.float64)
    vsize = df["F618"].values.astype(np.float64)
    out["has_overlay"] = (
        (fsize > 0) & (vsize > 0) & (fsize > vsize * 1.2)
    ).astype(np.float32)

    # ── PDF / Office (all zero for EMBER PE-only dataset) ──────────────
    for col in [
        "page_count", "has_javascript", "has_embedded_files",
        "has_launch_action", "has_suspicious_urls",
        "has_macros", "has_auto_open", "has_external_links",
        "has_dde", "has_obfuscated_strings",
    ]:
        out[col] = np.float32(0.0)

    # Reorder to match FEATURE_COLS exactly
    out = out[FEATURE_COLS]
    return out


import pyarrow.parquet as pq

def load_ember_parquet(
    data_dir: str | None = None,
    max_samples: int = 0,
) -> pd.DataFrame:
    """Load EMBER parquet files in batches and map to our FEATURE_COLS (vectorized)."""
    data_dir = data_dir or _EMBER_DIR
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"EMBER data directory not found: {data_dir}")

    parquet_files = sorted(data_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files in {data_dir}")

    frames: list[pd.DataFrame] = []
    total_loaded = 0
    BATCH_SIZE = 50_000

    for pq_file in parquet_files:
        print(f"\n  Processing {pq_file.name} in batches of {BATCH_SIZE} ...")
        pf = pq.ParquetFile(pq_file)
        
        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
            raw_df = batch.to_pandas()
            
            # Normalise label column name
            if "label" in raw_df.columns and "Label" not in raw_df.columns:
                raw_df = raw_df.rename(columns={"label": "Label"})

            # Drop unlabeled (-1)
            raw_df = raw_df[raw_df["Label"].isin([0.0, 1.0])].copy()
            if len(raw_df) == 0:
                continue

            if max_samples > 0:
                remaining = max_samples - total_loaded
                if remaining <= 0:
                    break
                raw_df = raw_df.head(remaining)

            # Vectorised mapping
            mapped = map_ember_vectorized(raw_df)
            mapped["label"] = raw_df["Label"].values.astype(int)
            frames.append(mapped)
            total_loaded += len(mapped)
            
            print(f"    Batch {batch_idx+1}: processed {len(mapped)} samples (Total: {total_loaded})")
            
            if max_samples > 0 and total_loaded >= max_samples:
                print("    Reached maximum samples count. Stopping.")
                break
                
        if max_samples > 0 and total_loaded >= max_samples:
            break

    if not frames:
        raise RuntimeError("No labeled data found in parquet files.")

    combined = pd.concat(frames, ignore_index=True)

    counts = combined["label"].value_counts().sort_index()
    print(f"\nTotal Dataset: {len(combined)} labeled samples")
    print(f"  Benign  (0): {counts.get(0, 0)}")
    print(f"  Malicious(1): {counts.get(1, 0)}")

    return combined


def save_processed_dataset(df: pd.DataFrame, out_path: str | None = None) -> str:
    """Save DataFrame to parquet."""
    out_path = out_path or _FEATURES_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {len(df)} rows -> {out_path} ({size_kb:.1f} KB)")
    return out_path


def build_dataset(
    ember_dir: str | None = None,
    max_ember_samples: int = 0,
) -> pd.DataFrame:
    """Build the combined training dataset from EMBER parquet files."""
    return load_ember_parquet(ember_dir, max_samples=max_ember_samples)


if __name__ == "__main__":
    print("=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)
    df = build_dataset()
    save_processed_dataset(df)
    print("\nDone.")
