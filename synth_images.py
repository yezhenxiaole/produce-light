# -*- coding: utf-8 -*-
"""
测试图像生成器：替代真实生鲜照片
==================================
你无真实设备与生鲜食材，采用两种来源：
1. 程序化合成"准生鲜"测试图（肉/菜/水产纹理 + 白色托盘中性区）；
2. （可选）如本机存在任意自然图像，可加载作为素材。

合成图的构成：
- 背景：白色托盘/案板（提供中性区，供光源污染识别锚定）
- 前景：肉（肌红蛋白红+脂肪白纹理）/ 菜（叶绿素绿）/ 鱼（粉白+鳞光）
- 光照：均匀漫射 + 轻微高斯噪声，模拟手机直拍
"""
import numpy as np


def _noise_texture(h, w, rng, scale=8):
    """生成低频纹理（模拟肌肉纹理/叶脉）"""
    low = rng.random((h // scale, w // scale))
    from scipy.ndimage import zoom
    tex = zoom(low, scale * h / (h // scale) / scale)[:h, :w]
    return tex / (tex.max() + 1e-6)


def _tray_background(h, w, rng, base=0.85):
    """
    真实感托盘背景：径向亮度渐变（中心亮角落暗，模拟顶光照射）+
    低频反射不均 + 轻噪点。避免均匀托盘被生鲜灯整片推过曝的合成伪影。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h * 0.5, w * 0.5
    r = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)
    gradient = 1.0 - 0.35 * r          # 中心1.0 → 角屦0.65
    tex = _noise_texture(h, w, rng, scale=16)
    tex = 0.9 + 0.2 * tex               # 0.9-1.1 低频不均
    val = base * gradient * tex
    bgr = np.stack([val, val, val * 0.995], axis=-1)
    bgr += rng.normal(0, 0.006, bgr.shape).astype(np.float32)
    return np.clip(bgr, 0, 1)


def make_meat_image(h=256, w=256, seed=0):
    """合成鲜肉本色图（D65自然光下）：红肉+白色脂肪纹理+真实感托盘"""
    rng = np.random.default_rng(seed)
    img = _tray_background(h, w, rng)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    ry, rx = h * 0.32, w * 0.40
    meat_mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    tex = _noise_texture(h, w, rng)
    # 肌红蛋白本色：中等红、不鲜艳
    base = np.array([0.52, 0.23, 0.22], dtype=np.float32)
    variation = (tex - 0.5)[..., None] * 0.12
    layer = np.clip(base[None, None, :] + variation, 0, 1)
    img[meat_mask] = layer[meat_mask]
    # 白色脂肪纹理（大理石纹，随机细条）
    fat = rng.random((h, w)) > 0.985
    fat &= meat_mask
    img[fat] = [0.90, 0.87, 0.84]
    # 轻噪声
    img += rng.normal(0, 0.008, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def make_veg_image(h=256, w=256, seed=0):
    """合成蔬菜本色图：绿叶+真实感托盘"""
    rng = np.random.default_rng(seed)
    img = _tray_background(h, w, rng)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    ry, rx = h * 0.36, w * 0.44
    veg_mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    tex = _noise_texture(h, w, rng, scale=6)
    base = np.array([0.20, 0.42, 0.16], dtype=np.float32)
    variation = (tex - 0.5)[..., None] * 0.15
    layer = np.clip(base[None, None, :] + variation, 0, 1)
    img[veg_mask] = layer[veg_mask]
    # 叶脉（略亮线条）
    veins = (rng.random((h, w)) > 0.98) & veg_mask
    img[veins] = [0.35, 0.55, 0.28]
    img += rng.normal(0, 0.008, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def make_sea_image(h=256, w=256, seed=0):
    """合成水产本色图：鱼身粉白+鳞光+真实感托盘"""
    rng = np.random.default_rng(seed)
    img = _tray_background(h, w, rng)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h * 0.5, w * 0.5
    ry, rx = h * 0.28, w * 0.46
    sea_mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    tex = _noise_texture(h, w, rng, scale=4)
    base = np.array([0.72, 0.68, 0.66], dtype=np.float32)
    variation = (tex - 0.5)[..., None] * 0.10
    layer = np.clip(base[None, None, :] + variation, 0, 1)
    img[sea_mask] = layer[sea_mask]
    # 鳞光点
    scales = (rng.random((h, w)) > 0.99) & sea_mask
    img[scales] = [0.88, 0.86, 0.84]
    img += rng.normal(0, 0.008, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


if __name__ == "__main__":
    import cv2
    # 自检：生成三张本色图并保存
    for name, fn in [("meat", make_meat_image), ("veg", make_veg_image), ("sea", make_sea_image)]:
        img = fn(seed=42)
        cv2.imwrite(f"ground_{name}.png", (img * 255).astype(np.uint8)[:, :, ::-1])
        print(f"ground_{name}.png 已生成, shape={img.shape}")