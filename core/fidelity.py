import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculates PSNR between two images.
    img1, img2: Tensors of shape [B, H, W, C] in range [0, 1]
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculates SSIM between two images.
    img1, img2: Tensors of shape [B, H, W, C] in range [0, 1]
    """
    try:
        from ..utils.tensor_ops import create_window, tensor_to_hchw
    except (ImportError, ValueError):
        from utils.tensor_ops import create_window, tensor_to_hchw

    # Convert to NCHW
    img1 = tensor_to_hchw(img1)
    img2 = tensor_to_hchw(img2)
    
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
