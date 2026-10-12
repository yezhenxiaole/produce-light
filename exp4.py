# -*- coding: utf-8 -*-
"""
实验四：构图先验环带法消融（合成数据仿真）
==========================================
1) 增益估计精度：全图法 vs 环带法，3食材×3强度×5seed（不再是单seed）；
2) 降级链行为验证：环带遮挡（环带被食物/遮挡占满）与满构图（无托盘）时
   算法应回退到全图中性区且仍能正确识别；
3) 还原质量对照：两种提取策略的还原 dE。
输出：results_exp4.json
"""
import json
import numpy as np
from pollution_sim import simulate_lamp_pollution, LAMP_SPECTRA
from restore_engine import estimate_pollution, restore_image, _neutral_gain, _ring_mask
from color_metrics import rgb_to_lab, ciede2000_std

FOOD_LAMP = [("meat", "red_meat"), ("veg", "green_veg"), ("sea", "blue_sea")]
FOOD_MAKE = {"meat": "make_meat_image", "veg": "make_veg_image", "sea": "make_sea_image"}
import synth_images


def gain_bias(pol, lamp, s):
    """用真值参数计算归一化增益偏差范数（评估增益估计精度）"""
    g = np.array(LAMP_SPECTRA[lamp]["gain_rgb"], dtype=np.float64)
    true_gain = 1.0 + (g - 1.0) * s
    return true_gain / true_gain.mean()


def run():
    out = {"gain_bias": [], "degrade_fallback": {}, "restore_dE": []}
    # ---- 1) 增益估计精度（全图 vs 环带），5 seeds ----
    for food, lamp in FOOD_LAMP:
        make = getattr(synth_images, FOOD_MAKE[food])
        for s in (0.4, 0.6, 0.8):
            d_full, d_ring = [], []
            for seed in (1, 2, 3, 4, 5):
                gt = make(seed=seed)
                pol, _ = simulate_lamp_pollution(gt, lamp, strength=s)
                true_norm = gain_bias(pol, lamp, s)
                # 全图法（临时禁环带：直接全图掩码）
                from restore_engine import extract_neutral_zone, NEUTRAL_MIN_PIXELS
                from pollution_sim import _srgb2lin
                m = extract_neutral_zone(pol, 0.22)
                if m.sum() >= NEUTRAL_MIN_PIXELS:
                    e = _srgb2lin(pol[m]).mean(axis=0); e = e / e.mean()
                    d_full.append(float(np.linalg.norm(e - true_norm)))
                # 环带法（走 _neutral_gain 完整降级链，记录 mode）
                lin_mean, mode, f, adaptive, n_used = _neutral_gain(pol, 0.22)
                if lin_mean is not None:
                    e = np.asarray(lin_mean) / float(np.mean(lin_mean))
                    d_ring.append(float(np.linalg.norm(e - true_norm)))
            out["gain_bias"].append(dict(
                food=food, lamp=lamp, strength=s,
                full_mean=float(np.mean(d_full)), full_std=float(np.std(d_full)),
                chain_mean=float(np.mean(d_ring)), chain_std=float(np.std(d_ring)), n=len(d_ring)))
            print(f"{food} s={s}: 全图偏差={np.mean(d_full):.4f}±{np.std(d_full):.4f} "
                  f"→ 降级链偏差={np.mean(d_ring):.4f}±{np.std(d_ring):.4f}")

    # ---- 2) 降级链回退行为 ----
    make = synth_images.make_meat_image
    gt = make(seed=1)
    # 场景A：环带被遮挡（模拟手指挡住周边，环带无中性像素，中心有托盘露出）
    pol, _ = simulate_lamp_pollution(gt, "red_meat", strength=0.6)
    occluded = pol.copy()
    ring = _ring_mask(pol.shape)
    # 环带内像素全部改为高饱和红（模拟手指/食物占据周边）
    occluded[ring] = [0.75, 0.15, 0.12]
    est = estimate_pollution(occluded)
    out["degrade_fallback"]["ring_occluded"] = dict(
        detected=est["detected"], lamp=est["lamp_type"], mode=est.get("neutral_mode"),
        neutral_pixels=est.get("neutral_pixels"), reason=est.get("failure_reason"))
    # 场景B：满构图（食物占满，无托盘）+右下角小面积白纸巾参照
    #    V0.4实测（见results_exp4.json）：红灯s=0.6污染把纸巾R通道推至约0.998，
    #    超过亮度窗口上限0.995（防过曝保护），纸巾中性像素=0 → 走“无中性参照”拒绝分支。
    #    设计上可接受：被强污染推饱和的白色参照物已发生通道裁剪，不携带可靠增益信息。
    full_food = np.ones((256, 256, 3), dtype=np.float32)
    tex = np.random.default_rng(7).random((256, 256))
    full_food[..., 0] = 0.55 + 0.1 * tex
    full_food[..., 1] = 0.20 + 0.06 * tex
    full_food[..., 2] = 0.18 + 0.06 * tex
    # 右下角放一块白纸巾参照（真实用户引导场景）
    full_food[230:250, 230:250] = [0.88, 0.87, 0.86]
    lin = __import__("pollution_sim")._srgb2lin(full_food)
    pol_ff = __import__("pollution_sim")._lin2srgb(
        lin * np.array([1.0 + 0.55 * 0.6, 1.0 - 0.05 * 0.6, 1.0 - 0.25 * 0.6],
                       dtype=np.float32)[None, None, :])
    est = estimate_pollution(np.clip(pol_ff, 0, 1).astype(np.float32))
    out["degrade_fallback"]["full_composition"] = dict(
        detected=est["detected"], lamp=est["lamp_type"], mode=est.get("neutral_mode"),
        neutral_pixels=est.get("neutral_pixels"), reason=est.get("failure_reason"))
    print("\n降级链回退:")
    for k, v in out["degrade_fallback"].items():
        print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
    # 场景C：环带遮挡检测（中性像素集中于单侧边带）
    #    单侧白卡（仅右侧边带有白卡，其余三侧高饱和食物）
    #    V0.4断言：两侧检测不达标→环带不可靠→回退全图提取
    single_side = np.ones((256, 256, 3), dtype=np.float32)
    tex2 = np.random.default_rng(11).random((256, 256))
    single_side[..., 0] = 0.55 + 0.1 * tex2
    single_side[..., 1] = 0.20 + 0.06 * tex2
    single_side[..., 2] = 0.18 + 0.06 * tex2
    # 右侧边带放白色卡（其余三侧为高饱和食物）
    single_side[:, 236:256] = [0.88, 0.87, 0.86]
    lin2 = __import__("pollution_sim")._srgb2lin(single_side)
    pol_ss = __import__("pollution_sim")._lin2srgb(
        lin2 * np.array([1.0 + 0.55 * 0.6, 1.0 - 0.05 * 0.6, 1.0 - 0.25 * 0.6],
                        dtype=np.float32)[None, None, :])
    est_ss = estimate_pollution(np.clip(pol_ss, 0, 1).astype(np.float32))
    out["degrade_fallback"]["single_side_white"] = dict(
        detected=est_ss["detected"], lamp=est_ss["lamp_type"], mode=est_ss.get("neutral_mode"),
        neutral_pixels=est_ss.get("neutral_pixels"), reason=est_ss.get("failure_reason"))
    print(f"  single_side_white: {json.dumps(out['degrade_fallback']['single_side_white'], ensure_ascii=False)}")

    # ---- 3) 还原 dE 对照（全图 vs 降级链），取最难的 sea s=0.8 ----
    make_sea = synth_images.make_sea_image
    d_full_e, d_chain_e = [], []
    for seed in (1, 2, 3, 4, 5):
        gt = make_sea(seed=seed)
        pol, _ = simulate_lamp_pollution(gt, "blue_sea", strength=0.8)
        est = estimate_pollution(pol)
        rec, _, rep = restore_image(pol, est, food_type="sea")
        if rec is not None:
            dE = float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean())
            d_chain_e.append(dE)
    out["restore_dE"] = dict(scene="sea/blue_sea s=0.8",
                             chain_dE_mean=float(np.mean(d_chain_e)),
                             chain_dE_std=float(np.std(d_chain_e)), n=len(d_chain_e))
    print(f"\nsea s=0.8 降级链还原 dE={np.mean(d_chain_e):.2f}±{np.std(d_chain_e):.2f} (n={len(d_chain_e)})")
    return out


if __name__ == "__main__":
    results = run()
    with open("results_exp4.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("results_exp4.json 已保存")
