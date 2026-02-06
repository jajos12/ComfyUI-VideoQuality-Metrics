"""
Audit script for DOVER Metric.

Verifies:
1. Model loading and weight downloading (if possible).
2. Input shape handling.
3. Quality scoring differentiation (Clean vs Noisy).

Note: Without pretrained weights, scores will be random (untrained/ImageNet).
"""
import torch
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bypass package imports
os.environ["PYTEST_CURRENT_TEST"] = "1"

from core.dover import calculate_dover_quality, is_dover_available
from tests.fixtures.synthetic_videos import create_static_video, create_jittery_video

def create_noisy_video(frames=16, h=224, w=224):
    """Create high noise video."""
    return torch.rand(1, frames, h, w, 3)

def audit_dover():
    print("="*60)
    print("AUDIT: DOVER Video Quality Metric")
    print("="*60)
    
    if not is_dover_available():
        print("[FAIL] DOVER dependencies not met.")
        return

    print("dependencies available.")

    # 1. Clean Video (Static white square)
    # Using 224x224 as Swin-T expects larger inputs usually, though code handles resize
    msg = "Creating videos..."
    print(msg)
    
    clean = create_static_video(frames=16, h=224, w=224)
    # 2. Noisy Video
    noisy = create_noisy_video(frames=16, h=224, w=224)
    
    print("Calculating scores (this may download weights)...")
    
    try:
        res_clean = calculate_dover_quality(clean)
        print(f"\n[Clean Video]")
        print(f"  Aesthetic: {res_clean['aesthetic_score']:.4f}")
        print(f"  Technical: {res_clean['technical_score']:.4f}")
        
        res_noisy = calculate_dover_quality(noisy)
        print(f"\n[Noisy Video]")
        print(f"  Aesthetic: {res_noisy['aesthetic_score']:.4f}")
        print(f"  Technical: {res_noisy['technical_score']:.4f}")
        
        # Check alignment
        # Technical score should favor clean video
        tech_diff = res_clean['technical_score'] - res_noisy['technical_score']
        print(f"\nTechnical Delta (Clean - Noisy): {tech_diff:.4f}")
        
        if tech_diff > 0.1:
            print("[PASS] Technical score correctly favors clean video.")
        elif tech_diff < -0.1:
            print("[FAIL] Technical score favors noisy video (Inverted?).")
        else:
            print("[WARNING] Scores are too similar. Model might be untrained/random.")
            
    except Exception as e:
        print(f"[ERROR] Calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    audit_dover()
