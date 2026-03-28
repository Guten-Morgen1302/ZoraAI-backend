import logging
from typing import Any, Dict

from app.static_analysis.yara_scanner import scan_yara
from app.static_analysis.clamav_scanner import scan_clamav
from app.static_analysis.classifier import predict

logger = logging.getLogger(__name__)

def run_static_pipeline(file_path: str, progress_callback=None) -> Dict[str, Any]:
    """
    Executes the comprehensive 3-stage Static Analysis Pipeline:
    Stage 1: YARA Indicators (Rules)
    Stage 2: ClamAV Signatures (AV)
    Stage 3: EMBER Thrember (Machine Learning)
    """
    
    logger.info(f"Initiating full 3-Stage Static Pipeline against {file_path}")
    if callable(progress_callback):
        progress_callback("processing_yara")
    
    # ── Stage 1: YARA Scan ───────────────────────────────────────────────────
    yara_hits = scan_yara(file_path)
    
    # ── Stage 2: ClamAV Scan ─────────────────────────────────────────────────
    logger.info("Stage 2/3: Calling ClamAV scan for %s", file_path)
    if callable(progress_callback):
        progress_callback("processing_clamav")
    clamav_is_malicious, clamav_signature = scan_clamav(file_path)
    logger.info(
        "Stage 2/3: ClamAV completed for %s | flagged=%s | signature=%s",
        file_path,
        clamav_is_malicious,
        clamav_signature,
    )
    
    # ── Stage 3: EMBER Machine Learning ──────────────────────────────────────
    if callable(progress_callback):
        progress_callback("processing_ember")
    ml_score, extracted_features = predict(file_path)
    
    # ── Synthesize Output Report ─────────────────────────────────────────────
    # We define it as suspicious if AV flags it, or if ML is highly confident (e.g. > 0.75),
    # or if there are multiple YARA hits (heuristic).
    ml_flag = ml_score >= 0.75
    yara_flag = len(yara_hits) > 0
    clam_flag = clamav_is_malicious
    
    is_suspicious = ml_flag or yara_flag or clam_flag
    
    report = {
        "final_verdict": "suspicious" if is_suspicious else "clean",
        "engines": {
            "yara": {
                "hits": yara_hits,
                "is_flagged": yara_flag
            },
            "clamav": {
                "is_flagged": clam_flag,
                "signature": clamav_signature
            },
            "ember_ml": {
                "is_flagged": ml_flag,
                "score": ml_score
            }
        },
        "features": extracted_features
    }
    
    return report
