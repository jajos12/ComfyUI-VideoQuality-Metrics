"""
Audit script for Temporal Flickering metric.

We verify if the metric correctly detects:
1. Global flickering (whole frame brightness changes).
2. Local flickering (small region brightness changes).
3. Static video (baseline).

Current implementation averages brightness globally first, which may dilute local flickering.
"""
import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bypass __init__.py package loading for tests
import os
os.environ["PYTEST_CURRENT_TEST"] = "1"

from core.temporal import calculate_temporal_flickering
from tests.fixtures.synthetic_videos import create_static_video

def create_global_flickering_video(frames=16, h=128, w=128):
    """Whole frame flashes."""
    video = torch.zeros(1, frames, h, w, 3)
    for t in range(frames):
        val = 1.0 if t % 2 == 0 else 0.5
        video[:, t, :, :, :] = val
    return video

def create_local_flickering_video(frames=16, h=128, w=128):
    """Small square flashes (approx 5% of area)."""
    video = torch.zeros(1, frames, h, w, 3)
    # Background 0.2
    video[:] = 0.2
    
    # Flashing square
    square = 28 # 28x28 / 128x128 ~ 4.7%
    for t in range(frames):
        val = 1.0 if t % 2 == 0 else 0.0
        video[:, t, 50:50+square, 50:50+square, :] = val
    return video

def run_audit():
    print("="*60)
    print("AUDIT: Temporal Flickering Metric")
    print("="*60)
    
    # 1. Static
    static = create_static_video()
    res_static = calculate_temporal_flickering(static)
    print(f"\n[Static Video]")
    print(f"  Score: {res_static['flickering_score'].item():.4f}")
    print(f"  Var:   {res_static['brightness_variance'].item():.6f}")

    # 2. Global Flickering
    global_flick = create_global_flickering_video()
    res_global = calculate_temporal_flickering(global_flick)
    print(f"\n[Global Flickering] (Should be High)")
    print(f"  Score: {res_global['flickering_score'].item():.4f}")
    print(f"  Var:   {res_global['brightness_variance'].item():.6f}")

    # 3. Local Flickering
    local_flick = create_local_flickering_video()
    res_local = calculate_temporal_flickering(local_flick)
    print(f"\n[Local Flickering] (Should be Moderate/High)")
    print(f"  Score: {res_local['flickering_score'].item():.4f}")
    print(f"  Var:   {res_local['brightness_variance'].item():.6f}")
    
    # Check sensitivity
    if res_local['flickering_score'] < 0.1:
        print("\n[WARNING] Local flickering score is very low! Metric might be under-sensitive to local artifacts.")
    else:
        print("\n[PASS] Local flickering detected.")

if __name__ == "__main__":
    run_audit()
