import torch
import torch.nn.functional as F
import numpy as np

def tensor_to_hchw(tensor):
    """
    Converts ComfyUI tensor [B, H, W, C] to [B, C, H, W]
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.permute(0, 3, 1, 2)

def tensor_to_bhwc(tensor):
    """
    Converts [B, C, H, W] to [B, H, W, C]
    """
    return tensor.permute(0, 2, 3, 1)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
