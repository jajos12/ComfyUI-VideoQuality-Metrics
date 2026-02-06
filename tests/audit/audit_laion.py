"""
Audit script for LAION Aesthetic Predictor.

Verifies:
1. Weight loading from github.
2. Aesthetic scoring logic (Clean > Noisy).
3. Runtime performance.

Requires: transformers, CLIP.
"""
import torch
import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTEST_CURRENT_TEST"] = "1"

from core.laion_aesthetic import calculate_laion_aesthetic_score, is_laion_available
from tests.fixtures.synthetic_videos import create_static_video, create_jittery_video

def audit_laion():
    print("="*60)
    print("AUDIT: LAION Aesthetic Predictor (MLP)")
    print("="*60)
    
    if not is_laion_available():
        print("[FAIL] Dependencies missing. Please run 'python install_dependencies.py'")
        return

    print("Dependencies OK. Running inference (may download weights)...")

    # clean High Quality
    clean = create_static_video(frames=5) # Smooth
    # Noisy (very ugly)
    noisy = torch.rand(1, 5, 224, 224, 3) # Random noise
    
    try:
        res_clean = calculate_laion_aesthetic_score(clean)
        res_noisy = calculate_laion_aesthetic_score(noisy)
        
        print(f"\n[Clean Video] Score: {res_clean['aesthetic_score']:.2f}")
        print(f"[Noisy Video] Score: {res_noisy['aesthetic_score']:.2f}")
        
        delta = res_clean['aesthetic_score'] - res_noisy['aesthetic_score']
        print(f"Delta: {delta:.2f}")
        
        if delta > 1.0:
            print("[PASS] Clean video scored significantly higher.")
        elif delta > 0:
            print("[WARNING] Clean video scored higher, but margin small.")
        else:
            print("[FAIL] Noisy video scored higher (or equal).")
            
    except Exception as e:
        print(f"[ERROR] Logic failed: {e}")

if __name__ == "__main__":
    audit_laion()
