# -*- coding: utf-8 -*-
"""
实验五：灯库参数失配鲁棒性（合成数据仿真，回应"模型内自测"质疑）
====================================================================
质疑：污染生成与识别用同一参数库，属于模型内自测。
本实验把生成污染的灯参数与识别灯库参数**故意失配**（±10%/±20%随机扰动、
旋转色偏方向），测试识别与还原是否仍有效——模拟真实灯具与预置库参数
存在偏差的情形。
失配设定：
  A) 增益扰动：生成用 g*(1+ε)，ε~U(-δ,δ)，δ=10%/20%，逐通道独立；
  B) 强度失配：生成用 s*=1.15s（灯离样品更近/功率更高）；
  C) 色偏方向旋转：生成用 g 绕亮度轴旋转 ±8°（模拟灯具光谱差异）。
输出：results_exp5.json
"""
import json
import numpy as np
from pollution_sim import LAMP_SPECTRA, _srgb2lin, _lin2srgb
from restore_engine import estimate_pollution, restore_image
from color_metrics import rgb_to_lab, ciede2000_std
from synth_images import make_meat_image, make_veg_image, make_sea_image

FOOD_LAMP = {"meat": "red_meat", "veg": "green_veg", "sea": "blue_sea"}
FOOD_MAKE = {"meat": make_meat_image, "veg": make_veg_image, "sea": make_sea_image}


def pollute_with_mismatch(gt, lamp, s, delta=0.0, s_scale=1.0, rot_deg=0.0, seed=0):
    """用失配参数生成污染图（识别端仍用原灯库）"""
    rng = np.random.default_rng(seed)
    g = np.array(LAMP_SPECTRA[lamp]["gain_rgb"], dtype=np.float64)
    # 色偏方向旋转（绕[1,1,1]轴）
    if rot_deg:
        axis = np.ones(3) / np.sqrt(3)
        th = np.radians(rot_deg)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        g = R @ g
    # 增益扰动
    if delta:
        g = g * (1 + rng.uniform(-delta, delta, 3))
    s_true = min(1.0, s * s_scale)
    gain = 1.0 + (g - 1.0) * s_true
    lin = _srgb2lin(gt)
    pol = _lin2srgb(lin * gain[None, None, :].astype(np.float32))
    return np.clip(pol, 0, 1).astype(np.float32)


def run():
    results = []
    scenarios = [
        dict(tag="match(基准)", delta=0.0, s_scale=1.0, rot_deg=0.0),
        dict(tag="增益±10%", delta=0.10, s_scale=1.0, rot_deg=0.0),
        dict(tag="增益±20%", delta=0.20, s_scale=1.0, rot_deg=0.0),
        dict(tag="强度+15%", delta=0.0, s_scale=1.15, rot_deg=0.0),
        dict(tag="方向旋转±8°", delta=0.0, s_scale=1.0, rot_deg=8.0),
        dict(tag="综合(10%+15%+8°)", delta=0.10, s_scale=1.15, rot_deg=8.0),
    ]
    for sc in scenarios:
        n_ok, n_id, n_total = 0, 0, 0
        dEs = []
        for food, lamp in FOOD_LAMP.items():
            for s in (0.4, 0.6, 0.8):
                for seed in (1, 2, 3):
                    gt = FOOD_MAKE[food](seed=seed)
                    pol = pollute_with_mismatch(gt, lamp, s, delta=sc["delta"],
                                                s_scale=sc["s_scale"], rot_deg=sc["rot_deg"], seed=seed)
                    est = estimate_pollution(pol)
                    rec, _, rep = restore_image(pol, est, food_type=food)
                    n_total += 1
                    if est["detected"] and est["lamp_type"] == lamp:
                        n_id += 1
                    if rec is not None:
                        dE = float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean())
                        dEs.append(dE)
                        if dE < 3.0:
                            n_ok += 1
        results.append(dict(tag=sc["tag"], n=n_total,
                            lamp_id_rate=n_id / n_total,
                            restore_dE_mean=float(np.mean(dEs)), restore_dE_std=float(np.std(dEs)),
                            dE_lt3_rate=n_ok / max(len(dEs), 1)))
        print(f"{sc['tag']:>18}: 灯型识别率={n_id}/{n_total}={n_id/n_total*100:.0f}%, "
              f"还原dE={np.mean(dEs):.2f}±{np.std(dEs):.2f}, dE<3比例={n_ok/len(dEs)*100:.0f}%")
    return results


if __name__ == "__main__":
    results = run()
    from exp_meta import save_results
    save_results("results_exp5.json", dict(mismatch=results),
                 params=dict(scenarios=["match", "gain±10%", "gain±20%",
                                        "strength+15%", "rotate±8°", "combined"],
                             n_per_scenario=27, n_seeds=3,
                             note="灯库失配鲁棒性（人为注入失配）"))
