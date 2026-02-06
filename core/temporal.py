"""
Temporal consistency metrics for video quality assessment.

Includes:
- Optical flow estimation (RAFT pretrained or Lucas-Kanade fallback)
- Warping error calculation
- Temporal flickering detection
- Motion smoothness analysis
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_optical_flow_raft(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    """
    Compute optical flow using pretrained RAFT model.
    
    Args:
        frame1: Tensor [B, H, W, C] - first frame
        frame2: Tensor [B, H, W, C] - second frame
        
    Returns:
        flow: Tensor [B, 2, H, W] - optical flow (u, v components)
    """
    try:
        from ..models.downloader import get_raft_model
    except (ImportError, ValueError):
        from models.downloader import get_raft_model
    
    device = frame1.device
    raft = get_raft_model(device)
    
    # Convert to NCHW format
    if frame1.dim() == 4 and frame1.shape[-1] in [1, 3, 4]:
        frame1 = frame1.permute(0, 3, 1, 2)
        frame2 = frame2.permute(0, 3, 1, 2)
    
    # RAFT expects [B, C, H, W] with values in [0, 1] or [0, 255]
    # Normalize to expected range if needed
    if frame1.max() <= 1.0:
        frame1 = frame1 * 255
        frame2 = frame2 * 255
    
    with torch.no_grad():
        # RAFT returns list of flow predictions (multi-scale)
        flow_predictions = raft(frame1, frame2)
        flow = flow_predictions[-1]  # Use final prediction
    
    return flow


def compute_optical_flow_lk(frame1: torch.Tensor, frame2: torch.Tensor, 
                            window_size: int = 15) -> torch.Tensor:
    """
    Compute optical flow using Lucas-Kanade method (PyTorch implementation).
    
    Lightweight fallback when RAFT is not available.
    
    Args:
        frame1: Tensor [B, H, W, C] or [B, C, H, W] - first frame
        frame2: Tensor [B, H, W, C] or [B, C, H, W] - second frame
        window_size: Size of the local window for flow estimation
        
    Returns:
        flow: Tensor [B, 2, H, W] - optical flow (u, v components)
    """
    # Ensure NCHW format
    if frame1.dim() == 4 and frame1.shape[-1] in [1, 3, 4]:
        frame1 = frame1.permute(0, 3, 1, 2)
        frame2 = frame2.permute(0, 3, 1, 2)
    
    # Convert to grayscale if RGB
    if frame1.shape[1] == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=frame1.device)
        frame1 = (frame1 * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        frame2 = (frame2 * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    
    B, C, H, W = frame1.shape
    
    # Compute image gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=frame1.dtype, device=frame1.device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=frame1.dtype, device=frame1.device).view(1, 1, 3, 3) / 8.0
    
    Ix = F.conv2d(frame1, sobel_x, padding=1)
    Iy = F.conv2d(frame1, sobel_y, padding=1)
    It = frame2 - frame1
    
    # Create averaging kernel for local window
    kernel = torch.ones(1, 1, window_size, window_size, 
                       dtype=frame1.dtype, device=frame1.device) / (window_size ** 2)
    pad = window_size // 2
    
    # Compute structure tensor components
    Ix2 = F.conv2d(Ix * Ix, kernel, padding=pad)
    Iy2 = F.conv2d(Iy * Iy, kernel, padding=pad)
    Ixy = F.conv2d(Ix * Iy, kernel, padding=pad)
    Ixt = F.conv2d(Ix * It, kernel, padding=pad)
    Iyt = F.conv2d(Iy * It, kernel, padding=pad)
    
    # Solve 2x2 linear system for each pixel
    det = Ix2 * Iy2 - Ixy * Ixy + 1e-8
    u = -(Iy2 * Ixt - Ixy * Iyt) / det
    v = -(Ix2 * Iyt - Ixy * Ixt) / det
    
    # Clamp extreme values
    max_flow = min(H, W) / 4
    u = torch.clamp(u, -max_flow, max_flow)
    v = torch.clamp(v, -max_flow, max_flow)
    
    flow = torch.cat([u, v], dim=1)
    return flow


def compute_optical_flow(frame1: torch.Tensor, frame2: torch.Tensor,
                         use_raft: bool = True) -> torch.Tensor:
    """
    Compute optical flow using best available method.
    
    Args:
        frame1: Tensor [B, H, W, C] - first frame
        frame2: Tensor [B, H, W, C] - second frame
        use_raft: Whether to try RAFT first (recommended)
        
    Returns:
        flow: Tensor [B, 2, H, W]
    """
    if use_raft:
        try:
            return compute_optical_flow_raft(frame1, frame2)
        except (ImportError, RuntimeError) as e:
            print(f"[VQ Metrics] RAFT unavailable, falling back to Lucas-Kanade: {e}")
    
    return compute_optical_flow_lk(frame1, frame2)


def warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp a frame using optical flow (backward warping).
    
    Args:
        frame: Tensor [B, C, H, W] - frame to warp
        flow: Tensor [B, 2, H, W] - optical flow
        
    Returns:
        warped: Tensor [B, C, H, W] - warped frame
    """
    B, C, H, W = frame.shape
    
    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.arange(H, device=frame.device, dtype=frame.dtype),
        torch.arange(W, device=frame.device, dtype=frame.dtype),
        indexing='ij'
    )
    
    # Add flow to coordinates
    x_new = x.unsqueeze(0) + flow[:, 0]
    y_new = y.unsqueeze(0) + flow[:, 1]
    
    # Normalize to [-1, 1] for grid_sample
    x_norm = 2.0 * x_new / (W - 1) - 1.0
    y_norm = 2.0 * y_new / (H - 1) - 1.0
    
    grid = torch.stack([x_norm, y_norm], dim=-1)
    
    warped = F.grid_sample(frame, grid, mode='bilinear', 
                           padding_mode='border', align_corners=True)
    return warped


def compute_occlusion_mask(flow_forward: torch.Tensor, 
                           flow_backward: torch.Tensor,
                           threshold: float = 1.0) -> torch.Tensor:
    """
    Compute occlusion mask using forward-backward consistency check.
    
    Args:
        flow_forward: Tensor [B, 2, H, W] - flow from frame1 to frame2
        flow_backward: Tensor [B, 2, H, W] - flow from frame2 to frame1
        threshold: Consistency threshold in pixels
        
    Returns:
        mask: Tensor [B, 1, H, W] - 1 for valid, 0 for occluded
    """
    # Warp backward flow using forward flow
    warped_backward = warp_frame(flow_backward, flow_forward)
    
    # Check consistency
    flow_diff = flow_forward + warped_backward
    flow_magnitude = torch.sqrt((flow_diff ** 2).sum(dim=1, keepdim=True))
    
    mask = (flow_magnitude < threshold).float()
    return mask


def calculate_warping_error(video: torch.Tensor, 
                            bidirectional: bool = True,
                            use_raft: bool = True) -> dict:
    """
    Calculate temporal warping error for a video sequence.
    
    Args:
        video: Tensor [B, T, H, W, C] or [T, H, W, C] - video frames
        bidirectional: Whether to use bidirectional flow for occlusion masking
        use_raft: Whether to use RAFT for optical flow (recommended)
        
    Returns:
        dict with:
            - warping_error: Mean warping error (lower = more consistent)
            - per_frame_error: Error per frame transition
    """
    # Handle input shapes
    if video.dim() == 4:
        video = video.unsqueeze(0)
    
    B, T, H, W, C = video.shape
    
    if T < 2:
        return {
            'warping_error': torch.tensor(0.0, device=video.device),
            'per_frame_error': []
        }
    
    # Convert to NCHW for processing
    video_nchw = video.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
    
    per_frame_errors = []
    
    for t in range(T - 1):
        frame1 = video_nchw[:, t]
        frame2 = video_nchw[:, t + 1]
        
        # Compute forward flow
        flow_forward = compute_optical_flow(
            frame1.permute(0, 2, 3, 1), 
            frame2.permute(0, 2, 3, 1),
            use_raft=use_raft
        )
        
        # Warp frame2 back to frame1
        warped = warp_frame(frame2, -flow_forward)
        
        if bidirectional:
            flow_backward = compute_optical_flow(
                frame2.permute(0, 2, 3, 1), 
                frame1.permute(0, 2, 3, 1),
                use_raft=use_raft
            )
            mask = compute_occlusion_mask(flow_forward, flow_backward)
        else:
            mask = torch.ones(B, 1, H, W, device=video.device)
        
        # Compute masked error
        error = torch.abs(frame1 - warped)
        masked_error = (error * mask).sum() / (mask.sum() * C + 1e-8)
        per_frame_errors.append(masked_error)
    
    mean_error = torch.stack(per_frame_errors).mean()
    
    return {
        'warping_error': mean_error,
        'per_frame_error': [e.item() for e in per_frame_errors]
    }


def calculate_temporal_flickering(video: torch.Tensor, 
                                  window_size: int = 5) -> dict:
    """
    Detect temporal flickering in video by analyzing brightness variance.
    
    Args:
        video: Tensor [B, T, H, W, C] or [T, H, W, C] - video frames
        window_size: Temporal window for variance calculation
        
    Returns:
        dict with:
            - flickering_score: Overall flickering intensity (0-1)
            - brightness_variance: Mean brightness variance per frame
    """
    if video.dim() == 4:
        video = video.unsqueeze(0)
    
    B, T, H, W, C = video.shape
    
    if T < window_size:
        return {
            'flickering_score': torch.tensor(0.0, device=video.device),
            'brightness_variance': torch.tensor(0.0, device=video.device)
        }
    
    # Compute per-frame mean brightness
    brightness = video.mean(dim=(3, 4))  # [B, T, C]
    brightness_gray = brightness.mean(dim=-1)  # [B, T]
    
    # Compute rolling variance
    variances = []
    for t in range(T - window_size + 1):
        window = brightness_gray[:, t:t + window_size]
        var = window.var(dim=1)
        variances.append(var)
    
    variance_tensor = torch.stack(variances, dim=1)
    mean_variance = variance_tensor.mean()
    
    # Normalize to 0-1 score (empirical scaling)
    flickering_score = torch.tanh(mean_variance * 50)
    
    return {
        'flickering_score': flickering_score,
        'brightness_variance': mean_variance
    }


def calculate_motion_smoothness(video: torch.Tensor,
                                use_raft: bool = True) -> dict:
    """
    Evaluate motion smoothness by analyzing flow acceleration (jerk).
    
    Lower jerk = smoother, more natural motion.
    Uses normalized jerk (relative to image diagonal) for scale invariance.
    
    Args:
        video: Tensor [B, T, H, W, C] or [T, H, W, C] - video frames
        use_raft: Whether to use RAFT for optical flow
        
    Returns:
        dict with:
            - smoothness_score: Motion smoothness (0-1, higher = smoother)
            - mean_jerk: Mean normalized jerk magnitude
    """
    if video.dim() == 4:
        video = video.unsqueeze(0)
    
    B, T, H, W, C = video.shape
    
    # Need at least 3 frames to compute acceleration
    if T < 3:
        return {
            'smoothness_score': torch.tensor(1.0, device=video.device),
            'mean_jerk': torch.tensor(0.0, device=video.device)
        }
    
    # Normalization factor: image diagonal (for scale invariance)
    diagonal = (H ** 2 + W ** 2) ** 0.5
    
    video_nchw = video.permute(0, 1, 4, 2, 3)
    
    # Compute flows for consecutive frame pairs
    flows = []
    for t in range(T - 1):
        frame1 = video_nchw[:, t]
        frame2 = video_nchw[:, t + 1]
        flow = compute_optical_flow(
            frame1.permute(0, 2, 3, 1),
            frame2.permute(0, 2, 3, 1),
            use_raft=use_raft
        )
        flows.append(flow)
    
    flows = torch.stack(flows, dim=1)  # [B, T-1, 2, H, W]
    
    # Normalize flows by diagonal for scale invariance
    flows_normalized = flows / diagonal
    
    # Compute velocity change (first derivative of flow)
    if flows_normalized.shape[1] < 2:
        return {
            'smoothness_score': torch.tensor(1.0, device=video.device),
            'mean_jerk': torch.tensor(0.0, device=video.device)
        }
    
    velocity = flows_normalized[:, 1:] - flows_normalized[:, :-1]  # [B, T-2, 2, H, W]
    
    # Compute jerk (second derivative: change in velocity change)
    if velocity.shape[1] < 2:
        # Only one velocity difference - use velocity magnitude as proxy
        jerk_magnitude = torch.sqrt((velocity ** 2).sum(dim=2)).mean()
    else:
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, T-3, 2, H, W]
        jerk_magnitude = torch.sqrt((acceleration ** 2).sum(dim=2)).mean()
    
    # Convert to smoothness score using sigmoid-like function
    # jerk_magnitude is now in [0, ~0.5] range for typical videos
    # Map so that jerk=0 -> smoothness=1, jerk=0.1 -> smoothness~0.5
    # Using: smoothness = 1 / (1 + k * jerk) where k controls sensitivity
    k = 10.0  # Tuned: jerk=0.1 gives 0.5, jerk=0.5 gives ~0.17
    smoothness_score = 1.0 / (1.0 + k * jerk_magnitude)
    
    return {
        'smoothness_score': smoothness_score,
        'mean_jerk': jerk_magnitude
    }

