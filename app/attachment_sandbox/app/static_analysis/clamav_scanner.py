import os
import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def _scan_with_clamscan_binary(file_path: str) -> tuple[bool, Optional[str]]:
    """Fallback scanner using local clamscan executable (no daemon required)."""
    configured_path = os.environ.get("CLAMSCAN_PATH", "").strip()
    clamscan_bin = configured_path or shutil.which("clamscan") or shutil.which("clamscan.exe")

    if not clamscan_bin:
        logger.warning(
            "ClamAV daemon unavailable and clamscan binary not found. "
            "Set CLAMSCAN_PATH or install ClamAV locally."
        )
        return False, None

    cmd = [clamscan_bin, "--no-summary", file_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to execute clamscan binary at %s: %s", clamscan_bin, e)
        return False, None

    # clamscan exit codes: 0 = clean, 1 = infected, 2 = error
    output = (result.stdout or "").strip()
    if result.returncode == 1:
        # Typical output: <path>: <signature> FOUND
        signature = None
        if ":" in output and output.endswith(" FOUND"):
            try:
                right = output.split(":", 1)[1].strip()
                signature = right[:-6].strip() if right.endswith(" FOUND") else right
            except Exception:  # noqa: BLE001
                signature = None
        return True, signature or "ClamAV-Signature"

    if result.returncode == 0:
        return False, None

    stderr = (result.stderr or "").strip()
    logger.warning("clamscan returned error code %s. stdout=%s stderr=%s", result.returncode, output, stderr)
    return False, None

def scan_clamav(file_path: str) -> tuple[bool, Optional[str]]:
    """
    Connects to the ClamAV daemon and streams the file for signature scanning.
    Returns a tuple: (is_malicious: bool, signature_name: str | None).
    """
    try:
        import clamd
    except ImportError:
        logger.warning("clamd module not installed. Proceeding without ClamAV.")
        return False, None

    # Connect to ClamAV. We use the host from the environment, defaulting to localhost since docker isn't ready.
    host = os.environ.get("CLAMAV_HOST", "localhost")
    port = int(os.environ.get("CLAMAV_PORT", "3310"))
    
    try:
        cd = clamd.ClamdNetworkSocket(host=host, port=port)
    except Exception as e:
        logger.warning("Could not initialize ClamAV client at %s:%s -> %s", host, port, e)
        return False, None

    # Fast health check so connection failures are logged once as warning, not traceback.
    try:
        cd.ping()
    except Exception as e:
        logger.warning("ClamAV daemon unreachable at %s:%s -> %s", host, port, e)
        return _scan_with_clamscan_binary(file_path)

    try:
        # Stream file bytes directly to ClamAV to avoid Shared Volume issues in Docker
        with open(file_path, "rb") as f:
            result = cd.instream(f)
            
        if not result:
            return False, None
            
        # result format from clamd looks like: {'stream': ('FOUND', 'Eicar-Test-Signature')}
        status, signature = list(result.values())[0]
        
        if status == "FOUND":
            return True, signature
            
        return False, None
        
    except Exception as e:
        logger.warning("ClamAV scan failed for %s: %s", file_path, e)
        return _scan_with_clamscan_binary(file_path)
