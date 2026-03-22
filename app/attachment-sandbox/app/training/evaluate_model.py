"""
evaluate_model.py — Evaluate a saved static classifier on a directory of files.

Usage:
    python -m app.training.evaluate_model --test-dir /path/to/files
    python -m app.training.evaluate_model --test-dir C:\\Windows\\System32 --max 50
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Make imports work ───────────────────────────────────────────────────────
_SANDBOX_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _SANDBOX_ROOT not in sys.path:
    sys.path.insert(0, _SANDBOX_ROOT)

from app.static_analysis.classifier import load_model, predict, FEATURE_COLS


def evaluate_directory(test_dir: str, max_files: int = 0) -> None:
    """Run predict() on every file in test_dir and print summary."""
    model = load_model()
    if model is None:
        print("ERROR: No model found. Train the model first.")
        sys.exit(1)

    # Force the singleton to use our loaded model
    import app.static_analysis.classifier as clf
    clf._MODEL = model
    clf._MODEL_LOADED = True

    target = Path(test_dir)
    if not target.exists():
        print(f"ERROR: Directory not found: {test_dir}")
        sys.exit(1)

    files = [f for f in target.iterdir() if f.is_file()]
    if max_files > 0:
        files = files[:max_files]

    print(f"Evaluating {len(files)} files from {test_dir}")
    print("-" * 60)

    results: list[tuple[str, float]] = []
    errors = 0

    for fpath in files:
        try:
            prob, feats = predict(str(fpath))
            results.append((fpath.name, prob))
        except Exception:
            errors += 1

    if not results:
        print("No files could be processed.")
        return

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    scores = [r[1] for r in results]

    flagged = sum(1 for s in scores if s >= 0.5)
    avg_score = sum(scores) / len(scores)

    print(f"\nSUMMARY")
    print(f"  Total files processed: {len(results)}")
    print(f"  Processing errors:     {errors}")
    print(f"  Flagged (≥ 0.5):       {flagged}")
    print(f"  Average confidence:    {avg_score:.6f}")

    print(f"\nTOP 10 HIGHEST-RISK FILES:")
    print(f"  {'File':<45s} {'Score':>8s}")
    print(f"  {'─' * 45} {'─' * 8}")
    for name, score in results[:10]:
        flag = " ⚠" if score >= 0.5 else ""
        print(f"  {name:<45s} {score:>8.4f}{flag}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate static classifier on files")
    parser.add_argument("--test-dir", required=True, help="Directory of files to scan")
    parser.add_argument("--max", type=int, default=0, help="Max files to process (0=all)")
    args = parser.parse_args()

    evaluate_directory(args.test_dir, args.max)


if __name__ == "__main__":
    main()
