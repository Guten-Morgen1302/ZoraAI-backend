"""Static analysis module — public interface.

Usage:
    from app.static_analysis import predict, FEATURE_COLS
    probability, features = predict("/path/to/file")
"""

from app.static_analysis.classifier import predict, FEATURE_COLS  # noqa: F401

__all__ = ["predict", "FEATURE_COLS"]
