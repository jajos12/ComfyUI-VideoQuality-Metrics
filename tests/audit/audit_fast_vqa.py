"""
Audit script for FAST-VQA Metric.

Verifies:
1. Weight loading compatibility.
2. Differentiation between Clean and Noise.
"""
import torch
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTEST_CURRENT_TEST"] = "1"

from core.fast_vqa import calculate_fastvqa_quality, is_fastvqa_available
from tests.fixtures.synthetic_videos import create_static_video

def create_noisy_video(frames=16, h=224, w=224):
    return torch.rand(1, frames, h, w, 3)

def audit_fast_vqa():
    print("="*60)
    print("AUDIT: FAST-VQA ")
    print("="*60)
    
    clean = create_static_video(frames=8, h=224, w=224)
    noisy = create_noisy_video(frames=8, h=224, w=224)
    
    # Run
    # This triggers weight download attempt
    print("Calculating scores...")
    try:
        res_clean = calculate_fastvqa_quality(clean)
        print(f"\n[Clean] Score: {res_clean['quality_score']:.4f}")
        
        res_noisy = calculate_fastvqa_quality(noisy)
        print(f"\n[Noisy] Score: {res_noisy['quality_score']:.4f}")
        
        diff = res_clean['quality_score'] - res_noisy['quality_score']
        print(f"Delta: {diff:.4f}")
        
        if diff > 0.1:
            print("[PASS] Clean > Noisy")
        elif diff < -0.1:
            print("[FAIL] Noisy > Clean")
        else:
            print("[WARNING] Scores indistinguishable (Weights not loaded?)")
            
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    audit_fast_vqa()
