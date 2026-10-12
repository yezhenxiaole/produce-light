# -*- coding: utf-8 -*-
"""
实验一/二：三类场景污染还原对比（合成数据仿真）
====================================================
流程：合成本色图(GT) → 施加灯污染(仿真) → 本发明还原 vs 通用AWB → 量化对比
指标：标准 CIEDE2000（color_metrics，经 Sharma 34对 + colour-science 交叉验证）、PSNR、SSIM
报告：逐 seed 记录，输出均值 ± 标准差
输出：results_exp1_2.json + 对比图 exp1_2_comparison.png
"""
import json
import numpy as np
import cv2
from pollution_sim import simulate_lamp_pollution
from restore_engine import estimate_pollution, restore_image, generic_awb, psnr, ssim_rgb
from color_metrics import rgb_to_lab, ciede2000_std
from synth_images import make_meat_image, make_veg_image, make_sea_image

FOOD_LAMP = [("meat", "red_meat"), ("veg", "green_veg"), ("sea", "blue_sea")]
FOOD_MAKE = {"meat": make_meat_image, "veg": make_veg_image, "sea": make_sea_image}
FOOD_FOODTYPE = {"meat": "meat", "veg": "veg", "sea": "sea"}


def metrics_pair(gt, img):
    gt_lab = rgb_to_lab(gt)
    img_lab = rgb_to_lab(img)
    dE = float(ciede2000_std(gt_lab, img_lab).mean())
    return dE, psnr(gt, img), ssim_rgb(gt, img)


def run_pair(food, lamp, strengths=(0.4, 0.6, 0.8), seeds=(1, 2, 3, 4, 5)):
    results = []
    for s in strengths:
        dE_pol, dE_awb, dE_ours = [], [], []
        psnr_pol, psnr_awb, psnr_ours = [], [], []
        ssim_ours = []
        est_info, overexp = [], []
        for seed in seeds:
            gt = FOOD_MAKE[food](seed=seed)
            pol, _ = simulate_lamp_pollution(gt, lamp, strength=s)
            est = estimate_pollution(pol)
            rec, _M, report = restore_image(pol, est, food_type=FOOD_FOODTYPE[food])
            if rec is None:  # 未输出还原图（识别拒绝/过曝/自检门2），按契约用输入图顶替记录
                rec = pol  # 记录用
            awb = generic_awb(pol)
            for tag, img in [("pol", pol), ("awb", awb), ("ours", rec)]:
                dE, P, S = metrics_pair(gt, img)
                if tag == "pol":
                    dE_pol.append(dE); psnr_pol.append(P)
                elif tag == "awb":
                    dE_awb.append(dE); psnr_awb.append(P)
                else:
                    dE_ours.append(dE); psnr_ours.append(P); ssim_ours.append(S)
            est_info.append(dict(detected=est["detected"], lamp_type=est["lamp_type"],
                                 strength=est["strength"], confidence=est["confidence"],
                                 neutral_mode=est.get("neutral_mode"),
                                 neutral_pixels=est.get("neutral_pixels")))
            overexp.append(report["overexposed_ratio"])
        # ---- 强度估计误差分区间统计（V0.4：不再用单一≤±0.08概括）----
        detected_info = [e for e in est_info if e["detected"]]
        errs = [abs(e["strength"] - s) for e in detected_info]
        est_err_by_bucket = dict(
            n_detected=len(detected_info), n_total=len(est_info),
            mean=float(np.mean(errs)), max=float(np.max(errs)),
        ) if errs else dict(n_detected=len(detected_info), n_total=len(est_info))
        results.append(dict(
            food=food, lamp=lamp, strength=s, n=len(seeds),
            polluted=dict(dE_mean=float(np.mean(dE_pol)), dE_std=float(np.std(dE_pol)),
                          psnr_mean=float(np.mean(psnr_pol))),
            generic_awb=dict(dE_mean=float(np.mean(dE_awb)), dE_std=float(np.std(dE_awb)),
                             psnr_mean=float(np.mean(psnr_awb))),
            ours=dict(dE_mean=float(np.mean(dE_ours)), dE_std=float(np.std(dE_ours)),
                      psnr_mean=float(np.mean(psnr_ours)),
                      psnr_std=float(np.std(psnr_ours)),
                      ssim_mean=float(np.mean(ssim_ours)), ssim_std=float(np.std(ssim_ours))),
            dE_per_seed=dict(polluted=[round(x, 3) for x in dE_pol],
                             awb=[round(x, 3) for x in dE_awb],
                             ours=[round(x, 3) for x in dE_ours]),
            mean_overexposed_ratio=float(np.mean(overexp)),
            est_info=est_info,
            **({"strength_est_err": est_err_by_bucket} if est_err_by_bucket else {}),
        ))
    return results


def save_comparison_grid():
    """对比图：3食物 × (本色|污染|AWB|本发明)，strength=0.8"""
    rows = []
    for food, lamp in FOOD_LAMP:
        gt = FOOD_MAKE[food](seed=1)
        pol, _ = simulate_lamp_pollution(gt, lamp, strength=0.8)
        est = estimate_pollution(pol)
        rec, _, _ = restore_image(pol, est, food_type=FOOD_FOODTYPE[food])
        if rec is None:
            rec = pol  # rec is None 契约：识别拒绝时用输入图顶替做对比展示
        awb = generic_awb(pol)
        rows.append([gt, pol, awb, rec])
    h, w = rows[0][0].shape[:2]
    canvas = np.ones((h * 3 + 20 * 4, w * 4 + 20 * 5, 3), dtype=np.float32) * 0.15
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y = 20 + r * (h + 20); x = 20 + c * (w + 20)
            canvas[y:y+h, x:x+w] = img
    cv2.imwrite("exp1_2_comparison.png", (canvas * 255).astype(np.uint8)[:, :, ::-1])
    print("exp1_2_comparison.png 已保存")


if __name__ == "__main__":
    all_results = {}
    for food, lamp in FOOD_LAMP:
        print(f"=== {food} / {lamp} ===")
        res = run_pair(food, lamp)
        for r in res:
            print(f"  s={r['strength']:.1f} | 污染 dE={r['polluted']['dE_mean']:.2f} "
                  f"| AWB dE={r['generic_awb']['dE_mean']:.2f}±{r['generic_awb']['dE_std']:.2f} "
                  f"| 本发明 dE={r['ours']['dE_mean']:.2f}±{r['ours']['dE_std']:.2f} "
                  f"PSNR={r['ours']['psnr_mean']:.1f}±{r['ours']['psnr_std']:.1f}")
        all_results[f"{food}_{lamp}"] = res
    save_comparison_grid()
    from exp_meta import save_results
    save_results("results_exp1_2.json", all_results,
                 params=dict(food_lamp=[f"{a}/{b}" for a, b in FOOD_LAMP],
                             strengths=[0.4, 0.6, 0.8], n_seeds=5,
                             metric="CIEDE2000(std)+PSNR"))
