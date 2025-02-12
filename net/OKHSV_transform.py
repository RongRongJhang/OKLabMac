import torch
import torch.nn as nn
import torch.nn.functional as F
import math

PI = math.pi

class RGB_OKHSV(nn.Module):
    def __init__(self):
        super(RGB_OKHSV, self).__init__()
    
    def toe(self, x):
        k1, k2, k3 = 0.206, 0.03, (1 + 0.206) / (1 + 0.03)
        return 0.5 * (k3 * x - k1 + torch.sqrt((k3 * x - k1) ** 2 + 4 * k2 * k3 * x))

    def toe_inv(self, x):
        k1, k2, k3 = 0.206, 0.03, (1 + 0.206) / (1 + 0.03)
        return (x ** 2 + k1 * x) / (k3 * (x + k2))

    def to_ST(self, cusp):
        L, C = cusp
        return C / L, C / (1 - L)
    
    def srgb_transfer_function(self, x):
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1/2.4)) - 0.055)
    
    def srgb_transfer_function_inv(self, x):
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    
    def find_cusp(self, a, b):
        # 這裡應該實作 cusp 計算，根據色彩範圍調整
        return torch.stack([torch.full_like(a, 0.5), torch.full_like(b, 1.0)], dim=1)
    
    def linear_srgb_to_oklab(self, srgb):
        # 轉換 sRGB 到 Oklab（根據 Oklab 公式）
        M = torch.tensor([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005]
        ], device=srgb.device, dtype=srgb.dtype)
        return torch.einsum('ij,bjhw->bihw', M, srgb)
    
    def oklab_to_linear_srgb(self, oklab):
        # 轉換 Oklab 到 sRGB
        M_inv = torch.tensor([
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.707614701]
        ], device=oklab.device, dtype=oklab.dtype)
        return torch.einsum('ij,bjhw->bihw', M_inv, oklab)
    
    def OKHSVT(self, img):
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        r, g, b = self.srgb_transfer_function_inv(r), self.srgb_transfer_function_inv(g), self.srgb_transfer_function_inv(b)
        
        lab = self.linear_srgb_to_oklab(torch.stack([r, g, b], dim=1))
        L, a, b = lab[:, 0, :, :], lab[:, 1, :, :], lab[:, 2, :, :]
        C = torch.sqrt(a ** 2 + b ** 2)
        h = 0.5 + 0.5 * torch.atan2(-b, -a) / PI
        
        a_ = a / C
        b_ = b / C
        cusp = self.find_cusp(a_, b_)
        S_max, T_max = self.to_ST(cusp)
        S_0, k = 0.5, 1 - 0.5 / S_max
        
        t = T_max / (C + L * T_max)
        L_v, C_v = t * L, t * C
        
        L_vt, C_vt = self.toe_inv(L_v), C_v * self.toe_inv(L_v) / L_v
        
        rgb_scale = self.oklab_to_linear_srgb(torch.stack([L_vt, a_ * C_vt, b_ * C_vt], dim=1))
        scale_L = torch.cbrt(1 / torch.clamp(rgb_scale.max(dim=1)[0], min=1e-6))
        
        L, C = L / scale_L, C / scale_L
        C = C * self.toe(L) / L
        L = self.toe(L)
        
        v = L / L_v
        s = (S_0 + T_max) * C_v / ((T_max * S_0) + T_max * k * C_v)
        
        return torch.stack([h, s, v], dim=1)
    
    def POKHSVT(self, img):
        h, s, v = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        
        a_ = torch.cos(2 * PI * h)
        b_ = torch.sin(2 * PI * h)
        
        cusp = self.find_cusp(a_, b_)
        S_max, T_max = self.to_ST(cusp)
        S_0, k = 0.5, 1 - 0.5 / S_max
        
        L_v = 1 - s * S_0 / (S_0 + T_max - T_max * k * s)
        C_v = s * T_max * S_0 / (S_0 + T_max - T_max * k * s)
        
        L = v * L_v
        C = v * C_v
        
        L_vt, C_vt = self.toe_inv(L_v), C_v * self.toe_inv(L_v) / L_v
        
        rgb_scale = self.oklab_to_linear_srgb(torch.stack([L_vt, a_ * C_vt, b_ * C_vt], dim=1))
        scale_L = torch.pow(1 / torch.clamp(rgb_scale.max(dim=1)[0], min=1e-6), 1/3)
        
        L, C = L * scale_L, C * scale_L
        
        rgb = self.oklab_to_linear_srgb(torch.stack([L, C * a_, C * b_], dim=1))

        return self.srgb_transfer_function(rgb)
