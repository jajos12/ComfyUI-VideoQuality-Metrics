"""Quick smoke test for production NR-VQA models."""
import torch
import sys
sys.path.insert(0, '.')

from core.dover import calculate_dover_quality
from core.fast_vqa import calculate_fastvqa_quality

print("=" * 50)
print("Production NR-VQA Smoke Test")
print("=" * 50)

# Create test video
video = torch.rand(1, 8, 128, 128, 3)

print("\n[1] Testing DOVER...")
dover_result = calculate_dover_quality(video)
print(f"    Aesthetic: {dover_result['aesthetic_score']:.3f}")
print(f"    Technical: {dover_result['technical_score']:.3f}")
print(f"    Overall:   {dover_result['overall_score']:.3f}")

print("\n[2] Testing FAST-VQA...")
fastvqa_result = calculate_fastvqa_quality(video)
print(f"    Quality: {fastvqa_result['quality_score']:.3f}")
print(f"    Fragments: {fastvqa_result['num_fragments']}")

print("\n" + "=" * 50)
print("All production tests PASSED!")
print("=" * 50)
