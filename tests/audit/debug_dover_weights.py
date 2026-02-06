"""
Debug script to inspect DOVER weight compatibility.

Downloads weights (if missing), loads them, and prints key differences 
between the state_dict and our model.
"""
import torch
import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTEST_CURRENT_TEST"] = "1"

from core.dover import _get_dover_model, _download_weights, DOVER_WEIGHTS_URL

def debug_dover_weights():
    print("="*60)
    print("DEBUG: DOVER Weight Compatibility")
    print("="*60)
    
    # Ensure weights exist
    weight_path = Path.home() / ".cache" / "video_quality_metrics" / "dover" / "DOVER.pth"
    if not weight_path.exists():
        print("Downloading weights...")
        _download_weights(DOVER_WEIGHTS_URL, "DOVER.pth")
    
    if not weight_path.exists():
        print("[FAIL] Weights not found.")
        return

    # Load state dict
    try:
        sd = torch.load(weight_path, map_location='cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
    except Exception as e:
        print(f"[FAIL] Could not load weights: {e}")
        return

    print(f"Loaded {len(sd)} keys from weights.")
    
    # Create model
    model = _get_dover_model(device=torch.device('cpu'))
    my_keys = set(model.state_dict().keys())
    weight_keys = set(sd.keys())
    
    intersection = my_keys.intersection(weight_keys)
    missing = my_keys - weight_keys
    unexpected = weight_keys - my_keys
    
    print(f"\nMatched Keys: {len(intersection)}")
    print(f"Missing Keys (In model, not in weights): {len(missing)}")
    print(f"Unexpected Keys (In weights, not in model): {len(unexpected)}")
    
    # Print sample diffs to identify pattern
    print("\n--- Sample Missing Keys (My code expects) ---")
    for k in list(missing)[:5]:
        print(f"  {k}")
        
    print("\n--- Sample Unexpected Keys (Weights have) ---")
    for k in list(unexpected)[:5]:
        print(f"  {k}")

if __name__ == "__main__":
    debug_dover_weights()
