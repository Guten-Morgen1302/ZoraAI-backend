import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

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
        logger.warning(f"Could not connect to ClamAV socket at {host}:{port} -> {e}")
        return False, None

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
        logger.exception(f"ClamAV scan failed for {file_path}")
        return False, None
