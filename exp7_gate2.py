# -*- coding: utf-8 -*-
"""
exp7_gate2: 门2区分力实验与阈值标定（V0.7a，第六轮审查意见1）
====================================================================
命题（响应审查）：门2 以实际还原输出 lin_restored 计算 dev_post 后，
  它到底能区分什么、不能区分什么？阈值应取多少？

数学前提（诚实声明，先写清边界再实验）：
  还原算子为逐通道乘法 M=diag(v)，参照集均值的还原后归一化增益为
  normalize(mean(lin_ref))·diag(v)/mean(...)——其方向由"观测参照方向×v"
  决定。识别端 g_hat 本身来自同一参照集的观测方向，因此门2 在代数上是
  拟合闭合差（g_hat 与 G_lib(s_fit) 的方向差）叠加 scene_adjust 系统偏差
  的函数，**不是对目标食材颜色正确性的独立验证**。它能检验的只有：
  "实际交付的还原图在采信参照处是否满足中性回归契约"。

实验组（每组 3食材×3强度×3种子=27，合成数据）：
  G1 正确还原（灯库与生成一致）      → dev_post 底噪（含 scene_adjust 偏差）
  G2 错误灯库（生成 red、库用 blue） → 识别端多数被拒；放行样本 dev_post
  G3 混合光 70:30 / 10:90 / 90:10    → 误接受时的 dev_post（应显著高于 G1）
  G4 裁剪场景（污染推至接近饱和）    → clip 后参照信息损失对 dev_post 的影响
输出：results_exp7.json（dev_post 分布+建议阈值+隔离带结论）
"""
import numpy as np
from restore_engine import estimate_pollution, restore_image
from restore_engine import RESTORE_SELFCHECK_DEV_TH
from pollution_sim import LAMP_SPECTRA, _srgb2lin, _lin2srgb
from color_metrics import rgb_to_lab, ciede2000_std
from synth_images import make_meat_image, make_veg_image, make_sea_image

FOOD_MAKE = {"meat": make_meat_image, "veg": make_veg_image, "sea": make_sea_image}
FOOD_LAMP = {"meat": "red_meat", "veg": "green_veg", "sea": "blue_sea"}


def _pollute_with_gain(gt, gain_vec):
    lin = _srgb2lin(gt)
    return np.clip(_lin2srgb((lin * gain_vec[None, None, :]).astype(np.float32)),
                   0, 1).astype(np.float32)


def _lamp_gain(lamp, s):
    g = np.array(LAMP_SPECTRA[lamp]["gain_rgb"], dtype=np.float64)
    return 1.0 + (g - 1.0) * s


def _mix_gain(lamp_a, lamp_b, s, mix_a):
    ga = np.array(LAMP_SPECTRA[lamp_a]["gain_rgb"], dtype=np.float64)
    gb = np.array(LAMP_SPECTRA[lamp_b]["gain_rgb"], dtype=np.float64)
    return 1.0 + (mix_a * (ga - 1.0) + (1 - mix_a) * (gb - 1.0)) * s


def _dev_post_of(gt, pol, food):
    """识别+还原，返回 (est摘要, rep摘要, dE或None)"""
    est = estimate_pollution(pol)
    rec, _, rep = restore_image(pol, est, food_type=food)
    dE = None
    if rec is not None and est["detected"]:
        dE = round(float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean()), 2)
    sc = rep.get("selfcheck") or {}
    est_sum = dict(detected=bool(est["detected"]),
                   lamp=est["lamp_type"],
                   s=round(float(est.get("strength", 0) or 0), 3),
                   conf=round(float(est.get("confidence", 0) or 0), 3))
    rep_sum = dict(rejected=rep.get("rejected"),
                   gate=rep.get("reject_gate"),
                   dev_post=sc.get("dev_post"))
    return est_sum, rep_sum, dE


def _collect(group_fn, tag, note=""):
    devs_ok, devs_out, dEs, details = [], [], [], []
    n_id, n_gate2, n_ident_rej, n_total = 0, 0, 0, 0
    for food, lamp in FOOD_LAMP.items():
        for s in (0.4, 0.6, 0.8):
            for seed in (1, 2, 3):
                gt = FOOD_MAKE[food](seed=seed)
                pol = group_fn(gt, food, lamp, s, seed)
                est, rep, dE = _dev_post_of(gt, pol, food)
                n_total += 1
                if est["detected"]:
                    n_id += 1
                if rep["gate"] == "selfcheck":
                    n_gate2 += 1
                if not est["detected"]:
                    n_ident_rej += 1
                dev = rep["dev_post"]
                if dev is not None:
                    # 门2拒绝的样本：dev超阈（应拦截）；输出样本：底噪
                    (devs_out if (rep["rejected"] or (dE is not None and dE >= 3)) else devs_ok).append(dev)
                if dE is not None:
                    dEs.append(dE)
                details.append(dict(food=food, s=s, seed=seed, est=est, rep=rep, dE=dE))
    def _stat(a):
        return dict(n=len(a), mean=round(float(np.mean(a)), 4) if a else None,
                    std=round(float(np.std(a)), 4) if a else None,
                    min=round(float(np.min(a)), 4) if a else None,
                    max=round(float(np.max(a)), 4) if a else None)
    out = dict(tag=tag, note=note, n_total=n_total, n_identified=n_id,
               n_ident_reject=n_ident_rej, n_gate2_reject=n_gate2,
               dev_output=_stat(devs_ok), dev_rejected_or_bad=_stat(devs_out),
               dE=_stat([float(x) for x in dEs]))
    print(f"  {tag}: 识别{n_id}/{n_total}, 识别拒{n_ident_rej}, 门2拒{n_gate2}, "
          f"输出dev={out['dev_output']['mean']}, 拒/坏dev={out['dev_rejected_or_bad']['mean']}")
    return out, details


def run():
    out_all = {}

    # G1 正确还原：dev_post 底噪（实际输出含 scene_adjust）
    g1, _ = _collect(
        lambda gt, food, lamp, s, sd: _pollute_with_gain(gt, _lamp_gain(lamp, s)),
        "G1_correct", "灯库与生成一致（27组），dev_post 实际输出口径底噪")
    out_all["G1_correct"] = g1

    # G2 错误灯库：生成用 red_meat，把识别端灯库假设换成错灯（模拟库参数被误配）
    # 实现：生成 red 污染图，但让还原端用错误灯型/强度（通过直接构造 est）
    from restore_engine import restore_image as _ri
    devs_wrong, dEs_wrong, details_wrong = [], [], []
    for food in FOOD_MAKE:
        for s in (0.4, 0.6, 0.8):
            for seed in (1, 2, 3):
                gt = FOOD_MAKE[food](seed=seed)
                pol = _pollute_with_gain(gt, _lamp_gain("red_meat", s))
                est = estimate_pollution(pol)
                if not est["detected"]:
                    continue
                # 强制错误灯库：red 污染却按 blue_sea 还原（模拟库/场景错配）
                est_wrong = dict(est)
                est_wrong["lamp_type"] = "blue_sea"
                est_wrong["strength"] = min(est["strength"] + 0.15, 0.95)
                rec, _, rep = _ri(pol, est_wrong, food_type=food)
                sc = rep.get("selfcheck") or {}
                dev = sc.get("dev_post")
                dE = None
                if rec is not None:
                    dE = round(float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean()), 2)
                    dEs_wrong.append(dE)
                if dev is not None:
                    devs_wrong.append(dev)
                details_wrong.append(dict(food=food, s=s, seed=seed,
                                          lamp=est_wrong["lamp_type"],
                                          rejected=rep.get("rejected"),
                                          gate=rep.get("reject_gate"),
                                          dev_post=dev, dE=dE))
    out_all["G2_wrong_lamp"] = dict(
        tag="G2_wrong_lamp", note="red污染按blue还原（错误灯库），门2应拒绝",
        n_tested=len(devs_wrong),
        dev=dict(n=len(devs_wrong),
                 mean=round(float(np.mean(devs_wrong)), 4) if devs_wrong else None,
                 min=round(float(np.min(devs_wrong)), 4) if devs_wrong else None,
                 max=round(float(np.max(devs_wrong)), 4) if devs_wrong else None),
        dE=dict(n=len(dEs_wrong),
                mean=round(float(np.mean(dEs_wrong)), 2) if dEs_wrong else None,
                max=round(float(np.max(dEs_wrong)), 2) if dEs_wrong else None),
        detail=details_wrong)
    print(f"  G2_wrong_lamp: 测试{len(devs_wrong)}, dev均值={out_all['G2_wrong_lamp']['dev']['mean']}, "
          f"dE均值={out_all['G2_wrong_lamp']['dE']['mean']}")

    # G3 混合光：70:30 / 10:90 / 90:10（红+蓝）
    for mix_a, tag in ((0.7, "G3_mix70:30"), (0.1, "G3_mix10:90"), (0.9, "G3_mix90:10")):
        g3, _ = _collect(
            lambda gt, food, lamp, s, sd, ma=mix_a:
                _pollute_with_gain(gt, _mix_gain("red_meat", "blue_sea", s, ma)),
            tag, f"红蓝混合 mix_a={mix_a}，误接受时 dev_post（应被门2拦截）")
        out_all[tag] = g3

    # G4 裁剪场景：高强度污染把参照推向接近饱和（clip 后信息损失）
    g4, _ = _collect(
        lambda gt, food, lamp, s, sd:
            _pollute_with_gain(gt, _lamp_gain(lamp, 0.9)),
        "G4_clip", "s=0.9 高强度（参照部分被推近饱和/clip），dev_post 分布")
    out_all["G4_clip"] = g4

    # 阈值建议：G1 max（底噪上界）与全部拦截侧样本 dev 下界的隔离带
    # （V0.7a：纳入 G3 全部混合比例与 G4 裁剪场景的拦截侧，
    # 与 restore_engine.py 门2标定注释块口径一致）
    noise_max = g1["dev_output"]["max"]
    mix_devs = []
    for t in ("G3_mix70:30", "G3_mix10:90", "G3_mix90:10", "G4_clip"):
        st = out_all[t]["dev_rejected_or_bad"]
        if st["n"]:
            mix_devs.append(st["min"])
    th_low = noise_max
    th_high = min(mix_devs) if mix_devs else None
    out_all["threshold_analysis"] = dict(
        g1_output_dev_max=noise_max,
        mix_min_dev=(min(mix_devs) if mix_devs else None),
        current_th=RESTORE_SELFCHECK_DEV_TH,
        recommended_range=(th_low, th_high),
        note="建议阈值取 G1 输出底噪上界与混合光拦截侧下界之间；两界不可分时如实记录")
    print(f"  阈值分析: G1底噪max={noise_max}, 混合min={th_high}, "
          f"当前阈值={RESTORE_SELFCHECK_DEV_TH}")
    return out_all


if __name__ == "__main__":
    out_all = run()
    from exp_meta import save_results
    save_results("results_exp7.json", dict(gate2=out_all),
                 params=dict(groups=["G1_correct", "G2_wrong_lamp", "G3_mix70:30",
                                     "G3_mix10:90", "G3_mix90:10", "G4_clip"],
                             n_per_group=27,
                             dev_post_def="实际还原输出lin_restored采信参照集归一化增益偏离",
                             note="门2区分力实验：能拦混合光误接受与错误灯库，"
                                  "不能拦与纯单灯物理同构的90:10；非食材颜色独立验证"))
