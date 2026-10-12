# -*- coding: utf-8 -*-
"""污染模拟器（合成数据仿真用）
说明：本模块用 RGB 线性域乘增益模拟生鲜灯污染，用于算法在受控合成
数据上的验证；不是真实灯具光谱与传感器成像模型。灯谱参数为设定的
仿真参数，非实测标定值。真实灯具与实机验证方案另见交底书实施例。
"""
import numpy as np

# ============ 生鲜灯光谱参数库 ============
LAMP_SPECTRA = {
    "red_meat": {
        "peaks": [(625.0, 18.0), (590.0, 25.0)],
        "gain_rgb": (1.55, 0.95, 0.75),
        "label": "红光肉类灯",
    },
    "green_veg": {
        "peaks": [(525.0, 22.0), (450.0, 20.0)],
        "gain_rgb": (0.80, 1.50, 0.95),
        "label": "绿光蔬菜灯",
    },
    "blue_sea": {
        "peaks": [(465.0, 20.0), (520.0, 18.0)],
        "gain_rgb": (0.75, 0.95, 1.60),
        "label": "蓝光水产灯",
    },
    "hidden_red": {
        "peaks": [(630.0, 20.0), (555.0, 30.0)],
        "gain_rgb": (1.35, 1.02, 0.88),
        "label": "隐形红(拍照显红)",
    },
}

def simulate_lamp_pollution(img_f32, lamp_type, strength=None):
    """
    模拟生鲜灯对图像的污染（合成仿真模型）：
    本色图(显示域) →逆gamma→ 物理线性域 →叠加灯增益→ gamma编码回显示域。
    超过1的线性值clip（模拟传感器过曝，不可逆）。
    注意：这是用于算法验证的简化仿真，非真实光谱成像模型。
    """
    cfg = LAMP_SPECTRA[lamp_type]
    s = 0.8 if strength is None else strength
    g = np.array(cfg["gain_rgb"], dtype=np.float32)

    lin = _srgb2lin(img_f32)
    gain = 1.0 + (g - 1.0) * s
    polluted_lin = lin * gain[None, None, :]
    polluted = _lin2srgb(polluted_lin)
    return polluted.astype(np.float32), s


def _srgb2lin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


if __name__ == "__main__":
    white = np.full((1, 1, 3), 0.92, dtype=np.float32)
    pol, _ = simulate_lamp_pollution(white, "red_meat", strength=0.8)
    print("白托盘(0.92)在red_meat s=0.8下:", pol[0, 0].round(4))
    gt = np.full((64, 64, 3), 0.5, dtype=np.float32)
    for lt in LAMP_SPECTRA:
        p, _ = simulate_lamp_pollution(gt, lt, strength=0.8)
        print(f"{lt}: 灰0.5→", p[0, 0].round(3))
