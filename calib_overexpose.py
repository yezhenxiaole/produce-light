# -*- coding: utf-8 -*-
"""
标定过曝拒绝阈值：扫描背景亮度→过曝比例，观察还原dE退化拐点
方法：用 red_meat s=0.9 污染，通过提亮本色图控制过曝比例，
      输出（过曝比例, 识别情况, 还原dE）曲线。
"""
import numpy as np
from synth_images import make_meat_image
from pollution_sim import _srgb2lin, _lin2srgb
from restore_engine import estimate_pollution, restore_image
from color_metrics import rgb_to_lab, ciede2000_std

print(f"{'提亮':>6} {'过曝比例':>8} {'detected':>8} {'dE':>6}")
gt0 = make_meat_image(seed=1)
for boost in (1.0, 1.05, 1.1, 1.12, 1.15, 1.18, 1.2, 1.25, 1.3, 1.4):
    gt = np.clip(gt0 * boost, 0, 1).astype(np.float32)
    lin = _srgb2lin(gt) * np.array([1.0 + 0.55 * 0.9, 1.0 - 0.05 * 0.9, 1.0 - 0.25 * 0.9],
                                   dtype=np.float32)[None, None, :]
    pol = np.clip(_lin2srgb(np.clip(lin, 0, 1)), 0, 1).astype(np.float32)
    over = (pol >= 0.999).any(axis=2).mean()
    est = estimate_pollution(pol)
    rec, _, rep = restore_image(pol, est)
    if rec is not None:
        dE = float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean())
    else:
        dE = None
    print(f"{boost:>6.2f} {over:>8.3f} {str(est['detected']):>8} "
          f"{('%0.2f' % dE) if dE is not None else 'rejected':>8}")
