import os
import logging
from typing import List

logger = logging.getLogger(__name__)

# Pre-compiled YARA rules cached in memory for extreme pipeline speed
_YARA_RULES = None
_YARA_LOADED = False

def _get_yara_rules():
    """Lazily load and compile the standard YARA ruleset on first invocation."""
    global _YARA_RULES, _YARA_LOADED
    
    if not _YARA_LOADED:
        _YARA_LOADED = True
        try:
            import yara
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models", "yara_rules.yar"
            )
            rules_path = os.environ.get("YARA_RULES_PATH", os.path.abspath(default_path))
            
            if os.path.isfile(rules_path):
                _YARA_RULES = yara.compile(filepath=rules_path)
                logger.info(f"Loaded YARA rules successfully from {rules_path}")
            else:
                logger.warning(f"YARA rules file not found at {rules_path}")
        except ImportError:
            logger.warning("yara-python package not installed. Skipping YARA analysis.")
        except yara.SyntaxError as e:
            logger.error(f"Syntax error compiling YARA rules: {e}")
        except Exception as e:
            logger.exception(f"Failed to load YARA rules: {e}")
            
    return _YARA_RULES

def scan_yara(file_path: str) -> List[str]:
    """
    Scans the given file against the compiled YARA ruleset.
    Returns a list of matched rule names.
    """
    rules = _get_yara_rules()
    if rules is None:
        return []

    try:
        matches = rules.match(file_path)
        return [match.rule for match in matches]
    except Exception as e:
        logger.exception(f"YARA scan failed for {file_path}")
        return []
