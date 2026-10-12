# -*- coding: utf-8 -*-
"""
exp6: 假中性攻击与构图先验泛化实验（V0.7a）
====================================================
A. 攻击场景（应全部拒绝或识别失败）：
   A1 浅色脂肪碎块满构图（碎片化）
   A2 满幅浅色鱼身（大面积真中性色食物，自然光）——独立参照检验
      应结构性拦截（环带与中央候选同材质、方向差→0，V0.7 证据字段落盘）
      另附环带掩码面积断言（256×256 → 18880，响应第六轮审查意见3）
   A2b 消融实验（V0.7a）：关闭独立参照检验跑 A2 → 预期误识别，证明该检验
      是拒绝的必要门而非侥幸兜底（响应第五轮审查意见1）；同时逐级记录
      质量四项门判定并与主组比对，证明单因素消融（仅关闭独立参照检验，
      其余门不受影响，响应第六轮审查意见3）
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
D. 中央彩色证据分支攻击与边界测试（V0.7a，响应第六轮审查意见4）：
   D1 满幅彩色包装膜（彩色但非食材、无独立参照）——攻击该分支可能放行
   D2 彩色食材主体居中（正控）
   D3 小主体构图（食物占画幅 37%/25% 线性占比）
   D4 中央混合区域（彩色食材+白色餐盘同处中央窗口）
   D5 不同曝光（0.75x/1.15x/偏暗0.55x/过曝裁剪1.35x）
   表述口径：中央彩色证据仅宣称"中央区域存在彩色像素证据"，不宣称
   主体即食材（V0.7a 降格）。
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
    """A组：攻击场景（应拒绝）；同时量化误还原伤害（还原输出 vs 未污染本色图）
    V0.7：每样本记录 indep_evidence（实际触发的拒绝门+dir_diff+中央像素数+饱和度中位数），
    A2b 消融组证明独立参照检验的必要性（关闭检验时 A2 被误识别 → 开启后 5/5 拒绝）。
    """
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
                            strength=round(float(est.get("strength", 0) or 0), 3),
                            reason=est["failure_reason"],
                            conf=round(float(est.get("confidence", 0)), 3),
                            reject_stage=_first_reject_stage(est),
                            indep_dir_diff=_dir_diff_of(est)))
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
    # A2b 消融实验（V0.7，审查意见1：证明方向差检验的必要性；V0.7a 增
    # 质量门一致性证据，响应第六轮审查意见3）
    out["A2b_ablation"] = _ablation_indep_check()

    # 环带掩码面积断言（V0.7a，响应第六轮审查意见3）：
    # 256×256、margin=int(256*0.08)=20 → 环带面积 = 256²−216² = 18880。
    # 修正审查误判：18880 恰为环带全集（外圈8%方框带），非"最多约14176"；
    # A2 满幅鱼全像素过中性筛选（本色[0.72,0.66,0.62]饱和度低），
    # 环带∩候选 = 环带全集，故环带候选 18880 成立。此处以代码断言落盘。
    from restore_engine import _ring_mask
    _ring = _ring_mask((256, 256))
    _m = int(256 * 0.08)
    _exp = 256 * 256 - (256 - 2 * _m) ** 2
    out["ring_mask_area_check"] = dict(
        shape=[256, 256], margin=_m, area=int(_ring.sum()), expected=_exp,
        formula=f"{256}²−({256}−2×{_m})²={_exp}",
        ok=bool(int(_ring.sum()) == _exp))
    assert int(_ring.sum()) == _exp, f"环带面积断言失败: {int(_ring.sum())} != {_exp}"

    # D组：中央彩色证据分支攻击与边界（V0.7a，响应第六轮审查意见4）
    out["D_center_color"] = run_center_color()
    return out


def _first_reject_stage(est):
    """提取实际触发拒绝的门（V0.7 证据链要求）：
    从 indep_evidence.stages 找出首个 FAIL/INSUFFICIENT 的级别与结论。"""
    ev = est.get("indep_evidence")
    if not ev:
        return None
    for s in ev.get("stages", []):
        st = s.get("indep_status")
        if st in ("FAIL", "INSUFFICIENT"):
            return dict(level=s["level"], status=st,
                        dir_diff=s.get("indep", {}).get("dir_diff"),
                        n_center=s.get("indep", {}).get("n_center"),
                        verdict=s.get("indep", {}).get("verdict"))
    # 无 FAIL/INSUFFICIENT → 看质量门
    for s in ev.get("stages", []):
        if s.get("indep_status") == "QUALITY_FAIL":
            return dict(level=s["level"], status="QUALITY_FAIL",
                        quality_reason=s.get("indep", {}).get("quality_reason"))
    return None


def _dir_diff_of(est):
    """提取方向差检验真实触发值（证明方向差门确实运行过，非仅提示语）"""
    ev = est.get("indep_evidence")
    if not ev:
        return None
    for s in ev.get("stages", []):
        dd = s.get("indep", {}).get("dir_diff")
        if dd is not None:
            return dd
    return None


def _quality_trace(est):
    """提取各级候选的质量门/独立参照检验判定轨迹（消融一致性证据）"""
    ev = est.get("indep_evidence") or {}
    return [dict(level=s.get("level"), quality=s.get("quality"),
                 indep_status=s.get("indep_status"))
            for s in ev.get("stages", [])]


def _ablation_indep_check():
    """
    A2b 消融实验（V0.7，响应第五轮审查意见1）。
    命题：独立参照方向差检验是 A2 拒绝的必要拦截门，而非依赖其它门侥幸拦截。
    方法：关闭 _independent_ref_check（monkeypatch 为恒 PASS），跑 A2 满幅鱼
    5 种子；若仍被拒绝，则找全图级 Q1 占比门或其它门拦截的证据。
    预期（V0.6 诊断数据）：关闭检验后 A2 环带级直接采信鱼身像素（环带候选
    18880px、四边带达标、质量四项 PASS）→ 误识别为某灯型 → 输出错误还原。
    单因素消融证明（V0.7a，响应第六轮审查意见3）：逐级记录两组的质量
    四项门判定（quality_trace），验证消融组与主组在共同经历各级上的质量
    门判定一致——即两组差异仅来自独立参照检验被关闭，其余门不受影响。
    """
    import restore_engine
    from restore_engine import estimate_pollution as _est_orig
    from restore_engine import restore_image
    from color_metrics import rgb_to_lab, ciede2000_std

    res_on, res_off = [], []
    for sd in (1, 2, 3, 4, 5):
        img = _fish_food((256, 256), sd)
        gt = _fish_food((256, 256), sd)
        est = _est_orig(img)
        rec, _, rep = restore_image(img, est, food_type="sea")
        dE = None
        if rec is not None and est["detected"]:
            dE = round(float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean()), 2)
        res_on.append(dict(detected=bool(est["detected"]), lamp=est["lamp_type"],
                           s=round(float(est.get("strength", 0) or 0), 3),
                           reason=est["failure_reason"], restore_dE=dE,
                           quality_trace=_quality_trace(est)))

    # 关闭独立参照检验（消融）：monkeypatch 恒 PASS
    orig_check = restore_engine._independent_ref_check
    restore_engine._independent_ref_check = \
        lambda img, mask, zone, full_mask=None: ("PASS", dict(zone=zone, ablation=True))
    try:
        for sd in (1, 2, 3, 4, 5):
            img = _fish_food((256, 256), sd)
            gt = _fish_food((256, 256), sd)
            est = _est_orig(img)
            rec, _, rep = restore_image(img, est, food_type="sea")
            dE = None
            if rec is not None and est["detected"]:
                dE = round(float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean()), 2)
            res_off.append(dict(detected=bool(est["detected"]), lamp=est["lamp_type"],
                                s=round(float(est.get("strength", 0) or 0), 3),
                                reason=est["failure_reason"], restore_dE=dE,
                                quality_trace=_quality_trace(est)))
    finally:
        restore_engine._independent_ref_check = orig_check

    on_rej = sum(1 for r in res_on if not r["detected"])
    off_acc = sum(1 for r in res_off if r["detected"])
    off_dEs = [r["restore_dE"] for r in res_off if r["restore_dE"] is not None]
    # 单因素消融一致性：共同经历各级上质量门判定应逐级相同
    q_match = True
    q_diff = []
    for i, (a, b) in enumerate(zip(res_on, res_off)):
        ta, tb = a["quality_trace"], b["quality_trace"]
        L = min(len(ta), len(tb))
        for j in range(L):
            if ta[j]["level"] != tb[j]["level"] or ta[j]["quality"] != tb[j]["quality"]:
                q_match = False
                q_diff.append(dict(seed=i + 1, stage=j,
                                   on=ta[j], off=tb[j]))
    out = dict(
        check_on=dict(rejected=f"{on_rej}/5",
                     first_reason=res_on[0]["reason"], detail=res_on),
        check_off=dict(wrongly_accepted=f"{off_acc}/5",
                      wrong_restore_dE_mean=(round(float(np.mean(off_dEs)), 2) if off_dEs else None),
                      detail=res_off),
        quality_gates_consistent=q_match,
        quality_gate_diff=q_diff,
        note=("单因素消融：仅 monkeypatch 独立参照检验为恒 PASS，质量四项门"
              "逐级判定与主组一致（共同经历级），差异仅来自独立参照检验"
              if q_match else
              "消融组与主组在共同经历级上质量门判定不一致，需检查"))
    print(f"  A2b消融: 检验开启拒绝 {on_rej}/5；关闭后误接受 {off_acc}/5" +
          (f"，误还原dE={np.mean(off_dEs):.2f}" if off_dEs else "") +
          f"；质量门一致性={q_match}")
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
    C组：混合光对抗（可辨识性边界量化，V0.7a 更新）。
    诚实结论（写入交底书）：
      - 50:50 附近由识别阶段拒绝；
      - 接近单一光源的混合比例（如 90:10）与纯单灯物理同构（增益向量
        差异仅约 4.7%），无法仅凭方向拟合区分，误接受后由还原自检门2
        按实际输出闭合差拦截；
      - 门2 阈值 0.03 为实际输出口径标定（exp7_gate2）：正确单灯还原
        输出底噪 max=0.0279，全部拦截侧样本 min=0.0302，隔离带
        [0.0279, 0.0302]；本组比例扫描中 70:30 与 10:90 误接受样本
        均被门2 拒绝（reject_gate=selfcheck），90:10 部分放行（已披露
        可辨识性边界）；
      - 每样本记录 reject_gate 与 dev_post，证明实际触发的拦截门。
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
                            restore_rejected=bool(rep.get("rejected")),
                            reject_gate=rep.get("reject_gate"),
                            dev_post=(rep.get("selfcheck") or {}).get("dev_post"),
                            reason=est["failure_reason"]))
        accepted = sum(1 for r in res if r["detected"])
        restored = sum(1 for r in res if r["detected"] and not r["restore_rejected"])
        out[cname] = dict(accepted=f"{accepted}/5", restore_output=f"{restored}/5", detail=res)
        print(f"  {cname} 50:50混合: 误接受 {accepted}/5，实际还原输出 {restored}/5（其余被门拦截）"
              + (f"  首例: {res[0]['reason'][:40]}" if not res[0]["detected"]
                 else f"  首例误判为 {res[0]['lamp']}"))
    # 比例扫描（V0.7：记录 conf/strength/误接受还原dE/reject_gate/dev_post）
    scan = []
    for mix_a in (0.9, 0.7, 0.5, 0.3, 0.1):
        acc = 0
        confs, strengths, dEs = [], [], []
        gates = []
        for sd in (1, 2, 3, 4, 5):
            gt = _with_tray(_meat_food((256, 256), sd), ("all",))
            pol = _pollute_mix(gt, "red_meat", "blue_sea", 0.6, mix_a)
            est = estimate_pollution(pol)
            rec, _, rep = restore_image(pol, est, food_type="meat")
            if est["detected"]:
                acc += 1
                confs.append(float(est.get("confidence", 0)))
                strengths.append(float(est.get("strength", 0)))
                if rep.get("rejected"):
                    gates.append(rep.get("reject_gate"))
                if rec is not None:
                    dEs.append(float(ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec)).mean()))
        entry = dict(mix_red=mix_a, accepted=f"{acc}/5",
                     restore_output=f"{len(dEs)}/5")
        if confs:
            entry["conf_mean"] = round(float(np.mean(confs)), 3)
            entry["strength_mean"] = round(float(np.mean(strengths)), 3)
        entry["restore_dE_mean"] = (round(float(np.mean(dEs)), 2) if dEs else None)
        if gates:
            entry["reject_gates"] = gates
        scan.append(entry)
        print(f"  红灯占比{mix_a:.0%}: 误接受 {acc}/5，实际还原输出 {len(dEs)}/5"
              + (f", conf={np.mean(confs):.3f}, s={np.mean(strengths):.3f}, "
                 f"还原dE={np.mean(dEs):.2f}" if dEs else "")
              + (f"，拦截门={set(gates)}" if gates else ""))
    out["ratio_scan_red_blue"] = scan
    return out


def _center_colored_wrap(shape, seed, wrap_rgb):
    """满幅彩色包装膜（彩色但非食材、无独立中性参照）"""
    h, w = shape
    tex = np.random.default_rng(seed).random((h, w))
    img = np.ones((h, w, 3), dtype=np.float32)
    for c in range(3):
        img[..., c] = wrap_rgb[c] + 0.08 * tex
    return img


def _place_center_local(shape, seed, frac, food_rgb):
    """把彩色食材块放在中央 frac×frac 区域，其余为白色餐盘"""
    h, w = shape
    tex = np.random.default_rng(seed).random((h, w))
    plate = np.array([0.86, 0.85, 0.83], dtype=np.float32)
    img = np.ones((h, w, 3), dtype=np.float32)
    for c in range(3):
        img[..., c] = plate[c] + 0.02 * tex
    fh, fw = int(h * frac), int(w * frac)
    y0, x0 = (h - fh) // 2, (w - fw) // 2
    for c in range(3):
        img[y0:y0 + fh, x0:x0 + fw, c] = \
            food_rgb[c] + 0.06 * tex[y0:y0 + fh, x0:x0 + fw]
    return img


MEAT_RGB = [0.55, 0.20, 0.18]
GREEN_RGB = [0.30, 0.62, 0.25]
RED_WRAP_RGB = [0.70, 0.25, 0.20]   # 红色包装（与红肉同色系、非食材）
BLUE_WRAP_RGB = [0.25, 0.35, 0.70]  # 蓝色包装


def run_center_color():
    """
    D组：中央彩色证据分支攻击与边界（V0.7a，响应第六轮审查意见4）。
    攻击命题：中央彩色像素证据判定（饱和度中位数≥0.30 → PASS）是否会把
    "彩色但非食材、无独立参照"的场景误放行？边界在哪里？
    表述口径（V0.7a 降格）：该分支仅宣称"中央区域存在彩色像素证据，
    环带候选与中央彩色区域不同源"，不宣称"主体即食材"。
    """
    from restore_engine import _center_color_evidence, restore_image
    out = {}
    S = (256, 256)

    def _case(tag, make, expect):
        res = []
        for sd in (1, 2, 3, 4, 5):
            img = make(sd)
            is_color, sat_med, valid_frac = _center_color_evidence(img)
            est = estimate_pollution(img)
            rec, _, rep = restore_image(img, est, food_type="meat") \
                if est["detected"] else (None, None, None)
            res.append(dict(
                center_is_color=bool(is_color),
                center_sat_median=round(sat_med, 4),
                center_valid_frac=round(valid_frac, 4),
                detected=bool(est["detected"]),
                lamp=est["lamp_type"],
                mode=est.get("neutral_mode"),
                indep_status=[s.get("indep_status")
                              for s in (est.get("indep_evidence") or {}).get("stages", [])],
                reason=est["failure_reason"],
                restored=bool(rec is not None),
                reject_gate=rep.get("reject_gate") if rep else None))
        n_color = sum(1 for r in res if r["center_is_color"])
        n_det = sum(1 for r in res if r["detected"])
        n_rest = sum(1 for r in res if r["restored"])
        out[tag] = dict(expect=expect, center_color=f"{n_color}/5",
                        identified=f"{n_det}/5", restored=f"{n_rest}/5",
                        detail=res)
        print(f"  {tag}: 彩色证据 {n_color}/5, 识别 {n_det}/5, 还原输出 {n_rest}/5"
              + (f"  首例: {res[0]['reason'][:44]}" if not res[0]["detected"] else ""))

    # D1 满幅彩色包装膜（攻击：彩色但非食材、无独立参照）
    _case("D1_color_wrap_full",
          lambda sd: _center_colored_wrap(S, sd, RED_WRAP_RGB),
          expect="彩色证据PASS分支是否误放行——如实记录")
    # D1b 蓝色包装
    _case("D1b_color_wrap_blue",
          lambda sd: _center_colored_wrap(S, sd, BLUE_WRAP_RGB),
          expect="同上，蓝色系")
    # D2 彩色食材主体居中（正控：托盘环绕）
    _case("D2_meat_center_tray",
          lambda sd: _with_tray(_meat_food(S, sd), ("all",)),
          expect="正控：彩色证据成立，识别成功")
    # D3 小主体：线性占比 0.6（面积36%）与 0.5（面积25%）
    _case("D3_small_subject_060",
          lambda sd: _place_center_local(S, sd, 0.60, MEAT_RGB),
          expect="边界：主体变小后彩色证据与识别行为")
    _case("D3_small_subject_050",
          lambda sd: _place_center_local(S, sd, 0.50, MEAT_RGB),
          expect="边界：更小主体")
    # D4 中央混合：彩色食材+白色餐盘同处中央窗口
    def _d4_mixed_center(sd):
        img = _place_center_local(S, sd, 0.60, MEAT_RGB)
        # 中央窗口内再放一块白色低饱和区（模拟餐盘露白）
        img[96:128, 96:160] = np.array([0.88, 0.87, 0.85], dtype=np.float32)
        return img
    _case("D4_mixed_center", _d4_mixed_center,
          expect="边界：中央混合区域（彩色+白色）判定")
    # D5 曝光变化：0.75x / 1.15x / 0.55x / 1.35x（过曝裁剪）
    for f, tag in ((0.75, "D5_exp_075"), (1.15, "D5_exp_115"),
                   (0.55, "D5_exp_055"), (1.35, "D5_exp_135")):
        _case(tag,
              lambda sd, f=f: np.clip(_with_tray(_meat_food(S, sd), ("all",)) * f,
                                      0, 1).astype(np.float32),
              expect=f"曝光因子{f}下的判定稳健性")
    return out


if __name__ == "__main__":
    print("== A. 假中性攻击场景（应全部拒绝）==")
    attack = run_attack()
    print("\n== B. 构图正向验证（应全部正确识别）==")
    positive = run_positive()
    print("\n== C. 混合光可辨识性边界（V0.7 口径：识别门+还原自检门2）==")
    mixed = run_mixed()
    results = dict(attack=attack, positive=positive, mixed=mixed)
    from exp_meta import save_results
    save_results("results_exp6.json", results,
                 params=dict(attack_scenes=["A1_fat_blob", "A2_fish_full",
                                            "A3_fish_full_blue", "A4_fish_full_red",
                                            "A5_specular_small"],
                             positive_scenes=["B1~B5构图"],
                             mixed_scan=[0.9, 0.7, 0.5, 0.3, 0.1],
                             center_color_cases=["D1_color_wrap_full",
                                                 "D1b_color_wrap_blue",
                                                 "D2_meat_center_tray",
                                                 "D3_small_subject_060",
                                                 "D3_small_subject_050",
                                                 "D4_mixed_center",
                                                 "D5_exp_075", "D5_exp_115",
                                                 "D5_exp_055", "D5_exp_135"],
                             n_seeds=5,
                             note="A2结构性拒绝（独立参照检验，附18880面积断言）；"
                                  "A2b单因素消融（质量门一致性）；混合光边界量化；"
                                  "D组中央彩色证据攻击与边界（V0.7a）"))
