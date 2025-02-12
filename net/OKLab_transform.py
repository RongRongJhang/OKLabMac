import torch
import torch.nn as nn

class RGB_OKLab(nn.Module):
    def __init__(self):
        super(RGB_OKLab, self).__init__()
        
    def OKLabT(self, img):
        """
        將線性 sRGB 轉換為 Oklab 空間。
        輸入:
            img: 張量，形狀為 (B, 3, H, W)，通道順序為 r, g, b
        輸出:
            oklab: 張量，形狀為 (B, 3, H, W)，通道順序為 L, a, b
        """
        # 分離 r, g, b 通道
        r = img[:, 0:1, :, :]
        g = img[:, 1:2, :, :]
        b = img[:, 2:3, :, :]
        
        # 將線性 sRGB 轉換至中間表徵 l, m, s
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        
        # 分別取立方根 (使用 torch.sign 來正確處理正負值)
        eps = 1e-6
        l_ = torch.sign(l) * (torch.abs(l) + eps).pow(1/3)
        m_ = torch.sign(m) * (torch.abs(m) + eps).pow(1/3)
        s_ = torch.sign(s) * (torch.abs(s) + eps).pow(1/3)
        
        # 計算 Oklab 各通道
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b_out = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        
        # 合併 L, a, b 三個通道
        oklab = torch.cat([L, a, b_out], dim=1)
        return oklab

    def POKLabT(self, img):
        """
        將 Oklab 空間轉換回線性 sRGB。
        輸入:
            img: 張量，形狀為 (B, 3, H, W)，通道順序為 L, a, b
        輸出:
            rgb: 張量，形狀為 (B, 3, H, W)，通道順序為 r, g, b
        """
        # 分離 L, a, b 通道
        L = img[:, 0:1, :, :]
        a = img[:, 1:2, :, :]
        b_in = img[:, 2:3, :, :]
        
        # 計算中間值 l_, m_, s_
        l_ = L + 0.3963377774 * a + 0.2158037573 * b_in
        m_ = L - 0.1055613458 * a - 0.0638541728 * b_in
        s_ = L - 0.0894841775 * a - 1.2914855480 * b_in
        
        # 將中間值還原 (立方)
        l = l_.pow(3)
        m = m_.pow(3)
        s = s_.pow(3)
        
        # 計算回線性 sRGB 各通道
        r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        
        # 合併 r, g, b 三個通道
        rgb = torch.cat([r, g, b], dim=1)
        return rgb