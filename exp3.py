# -*- coding: utf-8 -*-
"""
实验三：四灯型自动识别 + 五类失败分支 + 误报率（合成数据仿真）
================================================================
1) 识别：三食材×四灯型×四强度×3seed=144组混淆矩阵；
2) 失败分支：自然光/过暗/非库内黄光/无中性参照/严重过曝 五类；
3) 误报率：自然光+非生鲜灯偏色输入，统计错误检出率（应为0）。
输出：results_exp3.json
"""
import json
import numpy as np
from pollution_sim import simulate_lamp_pollution, LAMP_SPECTRA, _srgb2lin, _lin2srgb
from restore_engine import estimate_pollution, restore_image
from synth_images import make_meat_image, make_veg_image, make_sea_image

LAMPS = ["red_meat", "green_veg", "blue_sea", "hidden_red"]
FOOD_MAKE = {"meat": make_meat_image, "veg": make_veg_image, "sea": make_sea_image}


def run():
    n_correct, n_total = 0, 0
    confusion = {lt: {lt2: 0 for lt2 in LAMPS + ["none"]} for lt in LAMPS}
    strengths = [0.3, 0.5, 0.7, 0.9]
    for food, make in FOOD_MAKE.items():
        for lamp in LAMPS:
            for s in strengths:
                for seed in (1, 2, 3):
                    gt = make(seed=seed)
                    pol, _ = simulate_lamp_pollution(gt, lamp, strength=s)
                    est = estimate_pollution(pol)
                    pred = est["lamp_type"] if est["detected"] else "none"
                    confusion[lamp][pred] += 1
                    n_total += 1
                    if pred == lamp:
                        n_correct += 1
    acc = n_correct / n_total
    print(f"总体识别准确率: {acc*100:.1f}% ({n_correct}/{n_total})")
    header = "        " + "  ".join(f"{lt[:9]:>9}" for lt in LAMPS + ["none"])
    print(header)
    for lt in LAMPS:
        row = f"{lt[:9]:>9}"
        for pred in LAMPS + ["none"]:
            row += f"  {confusion[lt][pred]:>9}"
        print(row)
    return dict(accuracy=acc, confusion=confusion, n_total=n_total)


def run_failure_branches():
    """失败分支验证：五类应触发提示的场景"""
    results = {}
    # 1) 无污染（自然光）：三种食材本色图都不应检出
    reasons = []
    for make in FOOD_MAKE.values():
        est = estimate_pollution(make(seed=1))
        reasons.append(est["failure_reason"] or "(误检出)")
    results["natural_light"] = dict(detected=False, reasons=reasons)
    # 2) 过暗图像
    gt = make_meat_image(seed=1)
    dark = (gt * 0.05).astype(np.float32)
    est = estimate_pollution(dark)
    results["too_dark"] = dict(detected=est["detected"], reason=est["failure_reason"])
    # 3) 非库内光源（纯黄光偏移，不在灯库中）
    lin = _srgb2lin(gt)
    yellow_gain = np.array([1.3, 1.3, 0.7], dtype=np.float32)
    yellow = _lin2srgb(lin * yellow_gain[None, None, :])
    est = estimate_pollution(np.clip(yellow, 0, 1).astype(np.float32))
    results["unknown_source"] = dict(detected=est["detected"], reason=est["failure_reason"],
                                      fitted_lamp=est["lamp_type"], confidence=est["confidence"])
    # 4) 无中性参照（纯色食物本体占满构图：全图高饱和无中性区）
    #    V0.4断言：灰世界不再兑底，必须命中“无中性参照”独立失败分支
    food_only = np.ones((256, 256, 3), dtype=np.float32)
    tex = np.random.default_rng(3).random((256, 256))
    food_only[..., 0] = 0.55 + 0.1 * tex   # 深红肉本体
    food_only[..., 1] = 0.20 + 0.06 * tex
    food_only[..., 2] = 0.18 + 0.06 * tex
    est = estimate_pollution(food_only)
    reason_no_ref = est["failure_reason"] or ""
    # V0.8：结构化判定（禁止中文提示语匹配）——无有效参照候选失败分支
    code_no_ref = est.get("failure_code")
    hit_no_ref = code_no_ref in ("NO_VALID_REFERENCE", "SAME_MATERIAL",
                                 "INSUFFICIENT_EVIDENCE")
    results["no_neutral_ref"] = dict(detected=est["detected"], reason=reason_no_ref,
                                      failure_code=code_no_ref,
                                      neutral_mode=est.get("neutral_mode"),
                                      neutral_pixels=est.get("neutral_pixels"),
                                      branch_hit=hit_no_ref)
    # 5) 严重过曝（红灯高强度+提亮背景→过曝超阈值：识别成功时还原被拒、
    #    识别被干扰时报告过曝干扰）
    gt = make_meat_image(seed=1)
    lin = _srgb2lin(np.clip(gt * 1.3, 0, 1).astype(np.float32)) * \
        np.array([1.0 + 0.55 * 0.9, 1.0 - 0.05 * 0.9, 1.0 - 0.25 * 0.9],
                 dtype=np.float32)[None, None, :]
    burn = np.clip(_lin2srgb(np.clip(lin, 0, 1)), 0, 1).astype(np.float32)
    burn[burn >= 0.999] = 1.0
    est = estimate_pollution(burn)
    rec, _M, report = restore_image(burn, est)
    results["severe_overexposure"] = dict(detected=est["detected"],
                                          detect_reason=est["failure_reason"],
                                          restore_rejected=report["rejected"],
                                          overexposed_ratio=report["overexposed_ratio"],
                                          returned_image_is_none=rec is None,
                                          advice=report["advice"])
    print("\n失败分支验证:")
    for k, v in results.items():
        print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
    # 断言：无中性参照必须命中独立分支（不能靠灰世界兑底蒙混过去）
    assert results["no_neutral_ref"]["branch_hit"], \
        "无中性参照场景未命中独立失败分支（灰世界兑底泄漏？）"
    print("  [断言通过] 无中性参照命中独立分支，非灰世界兑底")
    return results


def run_false_positive():
    """误报率：无生鲜灯污染的输入不应报检出"""
    n_fp, n_total = 0, 0
    cases = []
    # 自然光三食材×5seed
    for food, make in FOOD_MAKE.items():
        for seed in (1, 2, 3, 4, 5):
            est = estimate_pollution(make(seed=seed))
            n_total += 1
            if est["detected"]:
                n_fp += 1
                cases.append(f"{food}_seed{seed}→误报{est['lamp_type']}")
    # 非库内黄光（三食材）
    for food, make in FOOD_MAKE.items():
        gt = make(seed=1)
        lin = _srgb2lin(gt)
        yellow = _lin2srgb(lin * np.array([1.3, 1.3, 0.7], dtype=np.float32)[None, None, :])
        est = estimate_pollution(np.clip(yellow, 0, 1).astype(np.float32))
        n_total += 1
        if est["detected"]:
            n_fp += 1
            cases.append(f"{food}_yellow→误报{est['lamp_type']}")
    # 弱暖光（家用灯泡常见轻微暖偏，g≈(1.08,1.02,0.94)，不应触发生鲜灯检出）
    for food, make in FOOD_MAKE.items():
        gt = make(seed=1)
        lin = _srgb2lin(gt)
        warm = _lin2srgb(lin * np.array([1.08, 1.02, 0.94], dtype=np.float32)[None, None, :])
        est = estimate_pollution(np.clip(warm, 0, 1).astype(np.float32))
        n_total += 1
        if est["detected"]:
            n_fp += 1
            cases.append(f"{food}_warmweak→误报{est['lamp_type']}")
    print(f"\n误报率: {n_fp}/{n_total} = {n_fp/n_total*100:.1f}%")
    if cases:
        print("  误报明细:", "; ".join(cases))
    return dict(n_fp=n_fp, n_total=n_total, rate=n_fp / n_total, cases=cases)


if __name__ == "__main__":
    ident = run()
    fail = run_failure_branches()
    fp = run_false_positive()
    from exp_meta import save_results
    save_results("results_exp3.json",
                 dict(identification=ident, failure_branches=fail, false_positive=fp),
                 params=dict(n_lamps=4, n_strengths=4, n_seeds=3,
                             strengths=[0.3, 0.5, 0.7, 0.9],
                             n_total=144, false_positive_n=21,
                             note="识别+五类失败分支+仿真负样本误报率"))
