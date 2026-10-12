# -*- coding: utf-8 -*-
"""
exp: 过曝拒绝阈值多场景重标定（V0.5，响应第二轮审查意见7）
================================================================
原标定为单场景（red_meat s=0.9, seed=1）。本实验扩展为：
  3食品（meat/veg/sea）× 3灯型（red_meat/green_veg/blue_sea）× 5种子，
  每组合扫描提亮系数控制过曝比例，记录：
    - 过曝比例
    - 识别是否成功（detected）
    - 还原dE（未拒绝时）或rejected
输出每组合的"可用性分界"（dE≤2.5的最大过曝比例）与全组合汇总：
  - 各组合分界的分布（min/max/中位数）
  - 阈值敏感性：候选阈值0.20/0.25/0.30/0.35下的
    漏拒（超过阈值仍dE>3的比例）与误拒（低于阈值但dE≤2.5的比例）
结果写 results_calib_multi.json。
"""
import json
import numpy as np
from synth_images import make_meat_image, make_veg_image, make_sea_image
from pollution_sim import LAMP_SPECTRA, _srgb2lin, _lin2srgb
from restore_engine import estimate_pollution, restore_image, OVEREXPOSED_REJECT_TH
from color_metrics import rgb_to_lab, ciede2000_std

FOODS = {"meat": make_meat_image, "veg": make_veg_image, "sea": make_sea_image}
LAMPS = ["red_meat", "green_veg", "blue_sea"]
BOOSTS = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5)
STRENGTH = 0.9
DE_USABLE = 2.5   # 可用性判据：还原dE上限
DE_BAD = 3.0      # 漏拒判据：dE超过此值视为严重失真

def main():
    results = {}
    boundaries = []
    for food, mk in FOODS.items():
        for lamp in LAMPS:
            combo_key = f"{food}/{lamp}"
            rows = []
            for sd in (1, 2, 3, 4, 5):
                gt0 = mk(seed=sd)
                g = np.array(LAMP_SPECTRA[lamp]["gain_rgb"], dtype=np.float32)
                for boost in BOOSTS:
                    gt = np.clip(gt0 * boost, 0, 1).astype(np.float32)
                    lin = _srgb2lin(gt) * (1.0 + (g - 1.0) * STRENGTH)[None, None, :]
                    pol = np.clip(_lin2srgb(np.clip(lin, 0, 1)), 0, 1).astype(np.float32)
                    over = float((pol >= 0.999).any(axis=2).mean())
                    est = estimate_pollution(pol)
                    rec, _, rep = restore_image(pol, est, food_type=food)
                    if rec is not None and est["detected"]:
                        dE = float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean())
                        dE = round(dE, 2)
                    else:
                        dE = None  # 识别失败或被拒
                    rows.append(dict(seed=sd, boost=boost, over=round(over, 4),
                                     detected=bool(est["detected"]), dE=dE))
            # 每种子的可用性分界：dE≤DE_USABLE的最大过曝比例
            for sd in (1, 2, 3, 4, 5):
                ok_over = [r["over"] for r in rows
                           if r["seed"] == sd and r["dE"] is not None and r["dE"] <= DE_USABLE]
                if ok_over:
                    boundaries.append(max(ok_over))
            results[combo_key] = rows
            print(f"  {combo_key}: 完成 {len(rows)} 组")

    b = np.array(boundaries)
    summary = dict(
        n_boundary_points=len(boundaries),
        boundary_min=round(float(b.min()), 4),
        boundary_median=round(float(np.median(b)), 4),
        boundary_max=round(float(b.max()), 4),
    )
    # 阈值敏感性：候选阈值下的漏拒/误拒（逐点统计）
    sens = {}
    for th in (0.20, 0.25, 0.30, 0.35):
        pts = [r for rows in results.values() for r in rows if r["dE"] is not None]
        # 漏拒：过曝比例低于阈值但dE>DE_BAD（该还原不应发生）
        miss = sum(1 for r in pts if r["over"] < th and r["dE"] > DE_BAD)
        # 误拒：过曝比例高于阈值但dE≤DE_USABLE（该还原本可用却被拒）
        # 注意：被拒的点dE=None不计入；用全曲线点估计
        pts_all = [r for rows in results.values() for r in rows]
        usable_above = sum(1 for r in pts_all
                           if r["over"] >= th and r["dE"] is not None and r["dE"] <= DE_USABLE)
        sens[str(th)] = dict(miss_leak=miss, usable_rejected=usable_above)
        print(f"  阈值{th}: 漏拒{miss}例, 误拒可用例{usable_above}例")
    out = dict(summary=summary, sensitivity=sens, per_combo=results)
    from exp_meta import save_results
    save_results("results_calib_multi.json", out,
                 params=dict(foods=["meat", "veg", "sea"],
                             lamps=["red_meat", "green_veg", "blue_sea"],
                             strength_grid=[0.4, 0.6, 0.8, 0.95],
                             candidate_thresholds=[0.20, 0.25, 0.30, 0.35],
                             note="过曝比例-还原dE 退化曲线，多场景阈值敏感性"))

if __name__ == "__main__":
    main()
