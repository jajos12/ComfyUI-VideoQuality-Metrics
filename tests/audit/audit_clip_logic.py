"""
Audit script for CLIP Aesthetic Score logic (Mocked).

Since CLIP/Transformers might not be installed, we mock the model to verify:
1. Input shape handling (BHWC vs BCHW validation).
2. Scoring logic (Positive vs Negative averaging).
3. Normalization and Sigmoid application.
"""
import torch
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bypass package loading
import os
os.environ["PYTEST_CURRENT_TEST"] = "1"

# Import the module to audit
# We need to mock the imports inside core.clip_iqa if they fail
sys.modules["transformers"] = MagicMock()
sys.modules["clip"] = MagicMock()

from core.clip_iqa import calculate_clip_aesthetic_score, _get_clip_model

def audit_clip_logic():
    print("="*60)
    print("AUDIT: CLIP Aesthetic Score Logic (Mocked)")
    print("="*60)
    
    # Mock the _get_clip_model function to return our mock
    with patch("core.clip_iqa._get_clip_model") as mock_get_model:
        # Setup Mock Model
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        # Setup Mock Output
        # We simulate 8 prompts (4 pos, 4 neg)
        # Logits shape: [1, 8] for 1 image
        # Let's say Positive prompts get score 5.0, Negative get 0.0
        # Mean Pos = 5.0, Mean Neg = 0.0. Diff = 5.0. Sigmoid(5.0) ~ 0.993
        
        # We need to control the output behaviors
        # The code implementation:
        # outputs = model(**inputs)
        # logits = outputs.logits_per_image[0]
        
        mock_output = MagicMock()
        # [1, 8] tensor
        mock_output.logits_per_image = torch.tensor([[5.0]*4 + [0.0]*4])
        mock_model.return_value = mock_output
        
        # Determine strict dependency path
        # The module tries to import. If we forced sys.modules, it might have picked 'transformers' source
        # Let's verify which source it thinks it has
        import core.clip_iqa
        core.clip_iqa._CLIP_AVAILABLE = True
        core.clip_iqa._CLIP_SOURCE = "transformers"
        
        # Create fake image
        img = torch.rand(1, 224, 224, 3) # BHWC
        
        print("\n[Test 1] High Aesthetic Score Simulation")
        result = calculate_clip_aesthetic_score(img)
        score = result['aesthetic_score']
        print(f"  Pos Logits: 5.0, Neg Logits: 0.0")
        print(f"  Result Score: {score:.4f}")
        
        if score > 0.9:
            print("  [PASS] Logic correctly yields high score for positive dominance.")
        else:
            print("  [FAIL] Score unexpectedly low.")

        # [Test 2] Low Aesthetic Score
        # Pos = 0.0, Neg = 5.0. Diff = -5.0. Sigmoid(-5.0) ~ 0.006
        mock_output.logits_per_image = torch.tensor([[0.0]*4 + [5.0]*4])
        
        print("\n[Test 2] Low Aesthetic Score Simulation")
        result = calculate_clip_aesthetic_score(img)
        score = result['aesthetic_score']
        print(f"  Pos Logits: 0.0, Neg Logits: 5.0")
        print(f"  Result Score: {score:.4f}")
        
        if score < 0.1:
            print("  [PASS] Logic correctly yields low score for negative dominance.")
        else:
            print("  [FAIL] Score unexpectedly high.")
            
        # [Test 3] BCHW Input Handling
        print("\n[Test 3] BCHW Input Automatic Permutation")
        img_bchw = torch.rand(1, 3, 224, 224)
        mock_output.logits_per_image = torch.tensor([[2.0]*8]) # Neutralish
        
        try:
            calculate_clip_aesthetic_score(img_bchw)
            print("  [PASS] Function accepted BCHW input without error.")
        except Exception as e:
            print(f"  [FAIL] BCHW input raised error: {e}")

if __name__ == "__main__":
    audit_clip_logic()
