import torch

def rgb_to_lab(rgb):
    """
    Converts RGB tensor [B, H, W, 3] to LAB tensor [B, H, W, 3]
    Assumes RGB is in range [0, 1]
    """
    # 1. RGB to sRGB (assume it's already sRGB if in [0, 1])
    # 2. sRGB to linear RGB
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    # 3. Linear RGB to XYZ (D65)
    m = torch.tensor([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ], device=rgb.device)
    xyz = torch.matmul(rgb_linear, m.t())
    
    # 4. XYZ to CIELAB
    # Relative to D65 white point
    xyz_ref = torch.tensor([0.95047, 1.00000, 1.08883], device=rgb.device)
    xyz_normalized = xyz / xyz_ref
    
    mask = xyz_normalized > 0.008856
    f_xyz = torch.where(mask, xyz_normalized ** (1/3), 7.787 * xyz_normalized + 16/116)
    
    l = (116 * f_xyz[..., 1]) - 16
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
    
    return torch.stack([l, a, b], dim=-1)

def calculate_ciede2000(img1, img2):
    """
    Calculates CIEDE2000 color difference between two images.
    img1, img2: Tensors of shape [B, H, W, 3] in range [0, 1]
    """
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)
    
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    
    # Delta E 2000 algorithm
    # Reference: https://en.wikipedia.org/wiki/Color_difference#CIEDE2000
    
    kL, kC, kH = 1, 1, 1
    
    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - torch.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    
    a1_p = (1 + G) * a1
    a2_p = (1 + G) * a2
    
    C1_p = torch.sqrt(a1_p**2 + b1**2)
    C2_p = torch.sqrt(a2_p**2 + b2**2)
    
    h1_p = torch.atan2(b1, a1_p) * 180 / torch.pi
    h1_p = torch.where(h1_p < 0, h1_p + 360, h1_p)
    
    h2_p = torch.atan2(b2, a2_p) * 180 / torch.pi
    h2_p = torch.where(h2_p < 0, h2_p + 360, h2_p)
    
    dL_p = L2 - L1
    dC_p = C2_p - C1_p
    
    dh_p = h2_p - h1_p
    dh_p = torch.where(torch.abs(dh_p) > 180, torch.where(h2_p <= h1_p, dh_p + 360, dh_p - 360), dh_p)
    
    dH_p = 2 * torch.sqrt(C1_p * C2_p) * torch.sin(dh_p / 2 * torch.pi / 180)
    
    L_bar_p = (L1 + L2) / 2
    C_bar_p = (C1_p + C2_p) / 2
    
    h_bar_p = torch.where(torch.abs(h1_p - h2_p) > 180, (h1_p + h2_p + 360) / 2, (h1_p + h2_p) / 2)
    h_bar_p = torch.where(h_bar_p >= 360, h_bar_p - 360, h_bar_p)
    
    T = 1 - 0.17 * torch.cos((h_bar_p - 30) * torch.pi / 180) + \
        0.24 * torch.cos(2 * h_bar_p * torch.pi / 180) + \
        0.32 * torch.cos((3 * h_bar_p + 6) * torch.pi / 180) - \
        0.20 * torch.cos((4 * h_bar_p - 63) * torch.pi / 180)
    
    d_theta = 30 * torch.exp(-((h_bar_p - 275) / 25)**2)
    Rc = 2 * torch.sqrt(C_bar_p**7 / (C_bar_p**7 + 25**7))
    RT = -torch.sin(2 * d_theta * torch.pi / 180) * Rc
    
    SL = 1 + (0.015 * (L_bar_p - 50)**2) / torch.sqrt(20 + (L_bar_p - 50)**2)
    SC = 1 + 0.045 * C_bar_p
    SH = 1 + 0.015 * C_bar_p * T
    
    dE = torch.sqrt((dL_p / (kL * SL))**2 + (dC_p / (kC * SC))**2 + (dH_p / (kH * SH))**2 + RT * (dC_p / (kC * SC)) * (dH_p / (kH * SH)))
    
    return dE.mean()
