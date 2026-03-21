"""Backward-compatible exports for URL analysis helpers."""

from app.url_analysis.cookie_analyzer import (
	analyze_cookie_attributes,
	analyze_cookies,
	compute_cookie_score,
	detect_cookie_issues,
	detect_session_fixation,
)
from app.url_analysis.domain_intelligence import extract_domain_features
from app.url_analysis.feature_extractor import extract_url_features
from app.url_analysis.homoglyph_detector import extract_homoglyph_features
from app.url_analysis.sandbox_analyzer import analyze_url, analyze_url_sync
from app.url_analysis.tls_intelligence import extract_tls_features
from app.url_analysis.url_analysis import (
	extract_all_features,
	extract_all_features_async,
	extract_phase_1_features,
	extract_phase_2_features,
	extract_phase_3_features,
	extract_phase_4_features_async,
)

__all__ = [
	"analyze_cookie_attributes",
	"detect_cookie_issues",
	"compute_cookie_score",
	"detect_session_fixation",
	"analyze_cookies",
	"extract_url_features",
	"extract_domain_features",
	"extract_tls_features",
	"extract_homoglyph_features",
	"analyze_url",
	"analyze_url_sync",
	"extract_phase_1_features",
	"extract_phase_2_features",
	"extract_phase_3_features",
	"extract_phase_4_features_async",
	"extract_all_features",
	"extract_all_features_async",
]
