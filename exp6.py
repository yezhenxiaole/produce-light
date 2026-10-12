# -*- coding: utf-8 -*-
"""
exp6: 假中性攻击与构图先验泛化实验（V0.6）
====================================================
A. 攻击场景（应全部拒绝或识别失败）：
   A1 浅色脂肪碎块满构图（碎片化）
   A2 满幅浅色鱼身（大面积真中性色食物，自然光）——V0.6 独立参照检验
      应结构性拦截（环带与中央候选同材质、方向差→0）
   A3 满幅鱼身+蓝灯污染（被中和场景）
   A4 满幅浅色食物+红灯污染
   A5 包装反光小亮斑（小块高光假中性）
B. 构图正向验证（应全部识别成功且灯型正确）：
   B1 标准构图（食物居中+四侧托盘，256×256，方图）
   B2 竖屏 9:16（144×256）
   B3 左右两侧托盘（上下为食物）
   B4 上下两侧托盘（左右为食物）
   B5 横屏 16:9（256×144）
C. 混合光可辨识性边界（V0.6 口径更新）：
   C1 红灯+蓝灯混合 50:50 → 单灯拟合
   C2 红灯+绿灯混合 50:50
   C3 混合比例扫描（红:蓝 = 9:1 ~ 1:9），记录误接受时的 conf/强度/
       还原色差——接近单一光源的混合与纯单灯物理同构，可能被接受，
       该边界如实量化写入交底书（不宣称已拦截）。
每场景5种子。结果写 results_exp6.json。
"""
import json
import numpy as np
from restore_engine import estimate_pollution
from pollution_sim import _srgb2lin, _lin2srgb, LAMP_SPECTRA


def _pollute(img, lamp, s):
    g = np.array(LAMP_SPECTRA[lamp]["gain_rgb"], dtype=np.float32)
    lin = _srgb2lin(img)
    return np.clip(_lin2srgb(lin * (1.0 + (g - 1.0) * s)[None, None, :]), 0, 1).astype(np.float32)


def _pollute_mix(img, lamp_a, lamp_b, s, mix_a):
    """两种灯混合：mix_a为a灯能量占比（0~1），总强度s保持"""
    ga = np.array(LAMP_SPECTRA[lamp_a]["gain_rgb"], dtype=np.float32)
    gb = np.array(LAMP_SPECTRA[lamp_b]["gain_rgb"], dtype=np.float32)
    gain_mix = 1.0 + (mix_a * (ga - 1.0) + (1 - mix_a) * (gb - 1.0)) * s
    lin = _srgb2lin(img)
    return np.clip(_lin2srgb(lin * gain_mix[None, None, :]), 0, 1).astype(np.float32)


def _meat_food(shape, seed):
    h, w = shape
    tex = np.random.default_rng(seed).random((h, w))
    img = np.ones((h, w, 3), dtype=np.float32)
    img[..., 0] = 0.55 + 0.1 * tex
    img[..., 1] = 0.20 + 0.06 * tex
    img[..., 2] = 0.18 + 0.06 * tex
    return img


def _fish_food(shape, seed):
    h, w = shape
    tex = np.random.default_rng(seed).random((h, w))
    img = np.ones((h, w, 3), dtype=np.float32)
    img[..., 0] = 0.72 + 0.08 * tex
    img[..., 1] = 0.66 + 0.08 * tex
    img[..., 2] = 0.62 + 0.08 * tex
    return img


def _fat_blob_food(shape, seed):
    """红肉底色+散布脂肪斑（浅色碎块假中性源）"""
    img = _meat_food(shape, seed)
    rng = np.random.default_rng(seed + 100)
    h, w = shape
    n_blob = int(h * w / 160)  # 斑块密度
    for _ in range(n_blob):
        y, x = rng.integers(0, h - 10), rng.integers(0, w - 10)
        img[y:y+10, x:x+10] = [0.82, 0.74, 0.66]
    return img


def _with_tray(img, sides=("all",)):
    """在指定边带放灰托盘（真实中性参照）"""
    h, w = img.shape[:2]
    m = int(min(h, w) * 0.12)
    tray = np.array([0.78, 0.76, 0.74], dtype=np.float32)
    out = img.copy()
    if "all" in sides or "top" in sides:
        out[:m, :] = tray
    if "all" in sides or "bottom" in sides:
        out[-m:, :] = tray
    if "all" in sides or "left" in sides:
        out[:, :m] = tray
    if "all" in sides or "right" in sides:
        out[:, -m:] = tray
    return out


def _specular_food(shape, seed):
    """深色食物+少量包装反光亮斑（小块高光假中性）"""
    img = _meat_food(shape, seed) * 0.7
    rng = np.random.default_rng(seed + 200)
    h, w = shape
    for _ in range(12):
        y, x = rng.integers(0, h - 6), rng.integers(0, w - 6)
        img[y:y+6, x:x+6] = [0.95, 0.94, 0.92]
    return img


def run_attack():
    """A组：攻击场景（应拒绝）；同时量化误还原伤害（还原输出 vs 未污染本色图）"""
    from restore_engine import restore_image
    from color_metrics import rgb_to_lab, ciede2000_std
    out = {}
    scenes = {
        "A1_fat_blob":        lambda sd: _fat_blob_food((256, 256), sd),
        "A2_fish_full":       lambda sd: _fish_food((256, 256), sd),
        "A3_fish_full_blue":  lambda sd: _pollute(_fish_food((256, 256), sd), "blue_sea", 0.6),
        "A4_fish_full_red":   lambda sd: _pollute(_fish_food((256, 256), sd), "red_meat", 0.6),
        "A5_specular_small":  lambda sd: _specular_food((256, 256), sd),
    }
    for name, fn in scenes.items():
        res = []
        dE_restore = []   # 误还原时的色差（相对本色图）
        for sd in (1, 2, 3, 4, 5):
            gt = _fish_food((256, 256), sd) if name != "A1_fat_blob" and name != "A5_specular_small" \
                else (_fat_blob_food((256, 256), sd) if name == "A1_fat_blob" else _specular_food((256, 256), sd))
            img = fn(sd)
            est = estimate_pollution(img)
            rec, _, rep = restore_image(img, est, food_type="sea" if "fish" in name else "meat")
            if rec is not None and est["detected"]:
                dE_restore.append(float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean()))
            res.append(dict(detected=bool(est["detected"]), lamp=est["lamp_type"],
                            reason=est["failure_reason"],
                            conf=round(float(est.get("confidence", 0)), 3)))
        rejected = sum(1 for r in res if not r["detected"])
        dm = float(np.mean(dE_restore)) if dE_restore else None
        out[name] = dict(rejected=f"{rejected}/5",
                         wrongly_restored=len(dE_restore),
                         wrong_restore_dE=(round(dm, 2) if dm is not None else None),
                         detail=res)
        print(f"  {name}: 拒绝 {rejected}/5" +
              (f", 误还原{n_d}例" if (n_d := len(dE_restore)) else ", 无还原输出") +
              (f", 误还原dE={dm:.2f}" if dm is not None else "") +
              (f"  首例原因: {res[0]['reason']}" if res[0]["reason"] else ""))
    return out


def run_positive():
    """B组：构图正向验证（应识别成功且灯型正确）"""
    out = {}
    scenes = [
        ("B1_std_square", (256, 256), ("all",)),
        ("B2_portrait_916", (144, 256), ("all",)),
        ("B3_left_right", (256, 256), ("left", "right")),
        ("B4_top_bottom", (256, 256), ("top", "bottom")),
        ("B5_landscape_169", (256, 144), ("all",)),
    ]
    for name, shape, sides in scenes:
        res = []
        for sd in (1, 2, 3, 4, 5):
            gt = _with_tray(_meat_food(shape, sd), sides)
            pol = _pollute(gt, "red_meat", 0.6)
            est = estimate_pollution(pol)
            res.append(dict(detected=bool(est["detected"]), lamp=est["lamp_type"],
                            strength=float(est["strength"]),
                            mode=est.get("neutral_mode"), reason=est["failure_reason"]))
        ok = sum(1 for r in res if r["detected"] and r["lamp"] == "red_meat")
        out[name] = dict(identified=f"{ok}/5", detail=res)
        print(f"  {name}: 正确识别 {ok}/5"
              + (f"  首例: lamp={res[0]['lamp']}, mode={res[0]['mode']}" if res[0]["detected"]
                 else f"  首例拒绝原因: {res[0]['reason']}"))
    return out


def run_mixed():
    """
    C组：混合光对抗（可辨识性边界量化，V0.6 更新）。
    诚实结论（写入交底书）：
      - 50:50 附近可拒绝；
      - 接近单一光源的混合比例（如 90:10、70:30）可能被接受——单张 RGB
        下其增益向量与纯单灯物理同构（90:10 红蓝混合与纯红灯 s≈0.55 的
        增益向量差异仅约 4.7%），无法仅凭方向拟合区分；
      - 因此对误接受样本补量化还原色差（dE vs 本色），作为可辨识性边界
        数据写入交底书，而不是宣称“混合光已全部拦截”。
    """
    from restore_engine import restore_image
    from color_metrics import rgb_to_lab, ciede2000_std
    out = {}
    combos = [
        ("red+blue", "red_meat", "blue_sea"),
        ("red+green", "red_meat", "green_veg"),
    ]
    for cname, la, lb in combos:
        res = []
        for sd in (1, 2, 3, 4, 5):
            gt = _with_tray(_meat_food((256, 256), sd), ("all",))
            pol = _pollute_mix(gt, la, lb, 0.6, 0.5)
            est = estimate_pollution(pol)
            rec, _, rep = restore_image(pol, est, food_type="meat")
            dE = None
            if rec is not None and est["detected"]:
                dE = round(float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean()), 2)
            res.append(dict(detected=bool(est["detected"]), lamp=est["lamp_type"],
                            strength=float(est.get("strength", 0)),
                            conf=float(est.get("confidence", 0)),
                            restore_dE_vs_ground=dE,
                            reason=est["failure_reason"]))
        accepted = sum(1 for r in res if r["detected"])
        out[cname] = dict(accepted=f"{accepted}/5", detail=res)
        print(f"  {cname} 50:50混合: 误接受 {accepted}/5"
              + (f"  首例: {res[0]['reason'][:40]}" if not res[0]["detected"]
                 else f"  首例误判为 {res[0]['lamp']}"))
    # 比例扫描（V0.6：记录 conf/strength/误接受还原dE，量化可辨识性边界）
    scan = []
    for mix_a in (0.9, 0.7, 0.5, 0.3, 0.1):
        acc = 0
        confs, strengths, dEs = [], [], []
        for sd in (1, 2, 3, 4, 5):
            gt = _with_tray(_meat_food((256, 256), sd), ("all",))
            pol = _pollute_mix(gt, "red_meat", "blue_sea", 0.6, mix_a)
            est = estimate_pollution(pol)
            rec, _, rep = restore_image(pol, est, food_type="meat")
            if est["detected"]:
                acc += 1
                confs.append(float(est.get("confidence", 0)))
                strengths.append(float(est.get("strength", 0)))
                if rec is not None:
                    dEs.append(float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean()))
        entry = dict(mix_red=mix_a, accepted=f"{acc}/5")
        if confs:
            entry["conf_mean"] = round(float(np.mean(confs)), 3)
            entry["strength_mean"] = round(float(np.mean(strengths)), 3)
        entry["restore_dE_mean"] = (round(float(np.mean(dEs)), 2) if dEs else None)
        scan.append(entry)
        print(f"  红灯占比{mix_a:.0%}: 误接受 {acc}/5"
              + (f", conf={np.mean(confs):.3f}, s={np.mean(strengths):.3f}, "
                 f"误接受还原dE={np.mean(dEs):.2f}" if dEs else ""))
    out["ratio_scan_red_blue"] = scan
    return out


if __name__ == "__main__":
    print("== A. 假中性攻击场景（应全部拒绝）==")
    attack = run_attack()
    print("\n== B. 构图正向验证（应全部正确识别）==")
    positive = run_positive()
    print("\n== C. 混合光可辨识性边界（V0.6 口径）==")
    mixed = run_mixed()
    results = dict(attack=attack, positive=positive, mixed=mixed)
    from exp_meta import save_results
    save_results("results_exp6.json", results,
                 params=dict(attack_scenes=["A1_fat_blob", "A2_fish_full",
                                            "A3_fish_full_blue", "A4_fish_full_red",
                                            "A5_specular_small"],
                             positive_scenes=["B1~B5构图"],
                             mixed_scan=[0.9, 0.7, 0.5, 0.3, 0.1],
                             n_seeds=5,
                             note="A2结构性拒绝（独立参照检验）；混合光边界量化"))
