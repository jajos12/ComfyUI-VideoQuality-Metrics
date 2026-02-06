"""
CLIP-based Image/Video Quality Assessment.

Provides:
- Aesthetic quality scoring for individual frames
- Text-video alignment scoring (prompt adherence)

Uses OpenAI's CLIP model or HuggingFace Transformers for inference.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any

# Try to import CLIP from different sources
_CLIP_AVAILABLE = False
_clip_model = None
_clip_processor = None

try:
    from transformers import CLIPModel, CLIPProcessor
    _CLIP_AVAILABLE = True
    _CLIP_SOURCE = "transformers"
except ImportError:
    try:
        import clip
        _CLIP_AVAILABLE = True
        _CLIP_SOURCE = "openai"
    except ImportError:
        _CLIP_SOURCE = None


def _get_clip_model(device: Optional[torch.device] = None):
    """
    Load CLIP model lazily.
    
    Returns:
        tuple: (model, processor/preprocess)
    """
    global _clip_model, _clip_processor
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if _clip_model is not None:
        return _clip_model, _clip_processor
    
    if not _CLIP_AVAILABLE:
        raise ImportError(
            "CLIP not available. Install via: pip install transformers "
            "or pip install git+https://github.com/openai/CLIP.git"
        )
    
    if _CLIP_SOURCE == "transformers":
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    else:
        _clip_model, _clip_processor = clip.load("ViT-B/32", device=device)
        _clip_model.eval()
    
    return _clip_model, _clip_processor


def calculate_clip_aesthetic_score(
    images: torch.Tensor,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Calculate aesthetic quality score for images using CLIP.
    
    Uses the insight that CLIP's image embeddings can be projected
    onto learned aesthetic axes (high quality vs low quality).
    
    Args:
        images: Tensor [B, H, W, C] or [B, C, H, W] in range [0, 1]
        device: Target device
        
    Returns:
        dict with:
            - aesthetic_score: Mean score (0-1, higher = more aesthetic)
            - per_image_scores: List of scores per image
    """
    if device is None:
        device = images.device
    
    model, processor = _get_clip_model(device)
    
    # Ensure BHWC format
    if images.dim() == 4 and images.shape[1] in [1, 3, 4]:
        images = images.permute(0, 2, 3, 1)
    
    # Define quality prompts
    positive_prompts = [
        "a high quality photo",
        "a professional photo",
        "a beautiful image",
        "an aesthetically pleasing image",
    ]
    negative_prompts = [
        "a low quality photo",
        "a blurry photo",
        "a ugly image",
        "a poorly composed image",
    ]
    
    scores = []
    
    with torch.no_grad():
        for i in range(images.shape[0]):
            img = images[i]
            
            # Convert to PIL-like format for processor
            if img.max() <= 1.0:
                img = (img * 255).clamp(0, 255).byte()
            
            img_np = img.cpu().numpy()
            
            if _CLIP_SOURCE == "transformers":
                # Use transformers processor
                inputs = processor(
                    text=positive_prompts + negative_prompts,
                    images=[img_np],
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                outputs = model(**inputs)
                logits = outputs.logits_per_image[0]
                
                # Average positive vs negative
                pos_score = logits[:len(positive_prompts)].mean()
                neg_score = logits[len(positive_prompts):].mean()
                
            else:
                # Use OpenAI CLIP
                from PIL import Image
                pil_img = Image.fromarray(img_np)
                img_input = processor(pil_img).unsqueeze(0).to(device)
                
                text_inputs = clip.tokenize(positive_prompts + negative_prompts).to(device)
                
                img_features = model.encode_image(img_input)
                text_features = model.encode_text(text_inputs)
                
                img_features = F.normalize(img_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                similarity = (img_features @ text_features.T)[0]
                
                pos_score = similarity[:len(positive_prompts)].mean()
                neg_score = similarity[len(positive_prompts):].mean()
            
            # Convert to 0-1 score using softmax-like approach
            score = torch.sigmoid(pos_score - neg_score).item()
            scores.append(score)
    
    return {
        "aesthetic_score": sum(scores) / len(scores),
        "per_image_scores": scores
    }


def calculate_text_video_alignment(
    video: torch.Tensor,
    prompt: str,
    sample_frames: int = 8,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Calculate how well a video aligns with a text prompt.
    
    Samples frames from the video and measures CLIP similarity
    between each frame and the prompt.
    
    Args:
        video: Tensor [B, T, H, W, C] or [T, H, W, C] in range [0, 1]
        prompt: Text description to check alignment against
        sample_frames: Number of frames to sample (default 8)
        device: Target device
        
    Returns:
        dict with:
            - alignment_score: Mean alignment (0-1, higher = better match)
            - per_frame_scores: List of scores per sampled frame
            - frame_indices: Which frames were sampled
    """
    if device is None:
        device = video.device
    
    # Handle input shapes
    if video.dim() == 4:
        video = video.unsqueeze(0)
    
    B, T, H, W, C = video.shape
    
    # Sample frames uniformly
    if T <= sample_frames:
        frame_indices = list(range(T))
    else:
        step = T / sample_frames
        frame_indices = [int(i * step) for i in range(sample_frames)]
    
    model, processor = _get_clip_model(device)
    
    scores = []
    
    with torch.no_grad():
        for idx in frame_indices:
            frame = video[0, idx]  # [H, W, C]
            
            if frame.max() <= 1.0:
                frame = (frame * 255).clamp(0, 255).byte()
            
            frame_np = frame.cpu().numpy()
            
            if _CLIP_SOURCE == "transformers":
                inputs = processor(
                    text=[prompt],
                    images=[frame_np],
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                outputs = model(**inputs)
                # Normalize to 0-1 range
                similarity = outputs.logits_per_image[0, 0].item()
                # CLIP logits are scaled dot products (can be large, e.g., 20.0)
                # Use sigmoid to map to [0, 1] probability-like score
                score = torch.sigmoid(torch.tensor(similarity)).item()
                
            else:
                from PIL import Image
                pil_img = Image.fromarray(frame_np)
                img_input = processor(pil_img).unsqueeze(0).to(device)
                text_input = clip.tokenize([prompt]).to(device)
                
                img_features = model.encode_image(img_input)
                text_features = model.encode_text(text_input)
                
                img_features = F.normalize(img_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                similarity = (img_features @ text_features.T)[0, 0].item()
                score = (similarity + 1) / 2  # Map to [0, 1]
            
            scores.append(score)
    
    return {
        "alignment_score": sum(scores) / len(scores),
        "per_frame_scores": scores,
        "frame_indices": frame_indices
    }


def is_clip_available() -> bool:
    """Check if CLIP is available for use."""
    return _CLIP_AVAILABLE
