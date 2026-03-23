import os
import sys
import json
from pathlib import Path

# Fix python path
_SANDBOX_ROOT = str(Path(__file__).resolve().parent.parent)
if _SANDBOX_ROOT not in sys.path:
    sys.path.insert(0, _SANDBOX_ROOT)

from app.static_analysis.pipeline import run_static_pipeline

def test_pipeline():
    print("="*60)
    print("TESTING 3-STAGE STATIC ANALYSIS PIPELINE")
    print("="*60)
    
    # 1. Test EICAR standard string
    eicar_path = "eicar_test.txt"
    with open(eicar_path, "w") as f:
        f.write(r"X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*")
        
    print("\nRunning EICAR test file...")
    try:
        report = run_static_pipeline(eicar_path)
        print(json.dumps(report, indent=2))
    finally:
        os.remove(eicar_path)
        
    # 2. Test benign Python installer
    benign_path = "python-3.12.5-amd64.exe"
    if os.path.exists(benign_path):
        print("\nRunning benign Python installer...")
        report = run_static_pipeline(benign_path)
        print(json.dumps(report, indent=2))
        
if __name__ == "__main__":
    test_pipeline()
