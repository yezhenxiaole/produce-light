# -*- coding: utf-8 -*-
"""
exp8: 强攻击仿真实验（V0.8，第十二轮A哥批准的设计稿V2.1执行）
====================================================================
核心命题：当独立场景证据分支确实放行一个非中性环带候选时，门2（还原后
参照一致性检查）能否稳定阻止错误还原？三种可能结论（实验前不预设）：
  A 门2稳定拦截 → 该分支可作"低置信度、受门2保护"工程路径保留
  B 门2不能拦截（放行且dE>3）→ 该分支必须降级为提示/辅助信息
  C 攻击进不了该分支 → 构造不足，"前置门拦截"不作为安全性结论

十个场景（设计稿V2.1 §3，六类拆分+一个对照）：
  G1  彩色包装+色偏灰边框（非纯中性，两变体偏红/偏蓝）      攻击
  G2a 真托盘在场+彩色标签污染（采信干净托盘=正确行为）       攻击
  G2b 无真托盘+色偏标签边框冒充参照                          攻击
  G3  色偏暖灰背景+彩色广告牌（ref_truth_mask=空集）          攻击
  G3b 真中性背景对照+彩色广告牌（采信真背景=正确）           对照
  G4  彩色塑料盒+木台面（已知可达性风险）                     攻击
  G5  彩色人工物+环带低饱和区（六项进入条件逐项核对）         攻击
  G6a 反光制造假中性候选（方向差≈灯方向）                     攻击
  G6b 阴影劈同材质（假方向差；误采信机理单列：dE_target>3）    攻击
  G6c 反光+阴影组合拉大方向差                                 攻击

与D组（exp6）的本质区别：本实验全部场景**施加灯污染**（本色图→污染→
识别/还原→dE 才有"还原伤害"含义，门2才可测）。

四步执行（不可跳序，设计稿§7）：
  1) precheck：每场景1种子，判据四条（至少3类进子模式/1类PASS/1类可
     计算误采信/1类可测门2兜底），不达标修构造≤2轮，仍不达标标
     "不可达场景"停止报A哥。→ results_exp8_precheck.json
  2) 正式：可达场景×5种子，逐样本全量落盘 → results_exp8_attack.json
  3) 文档后置（JSON冻结后才动report/交底书/README）。

统计口径（§2.b）：
  accepted_area/truth_area/intersection/ref_precision/ref_recall
  非中性误采信判定式：truth非空→precision<0.5；truth空集→accepted>0；
  accepted=0→未采信（单独记录，不算误采信）。
meta：code_version=V0.8（exp_meta.make_meta 沿用，git_commit 绑定
冻结提交 ef152d4）。旧8个结果JSON只读，本实验输出独立新文件。
"""
import json
import numpy as np
from pollution_sim import _srgb2lin, _lin2srgb, LAMP_SPECTRA
from restore_engine import (estimate_pollution, restore_image, _ring_mask,
                            extract_neutral_zone)
from color_metrics import rgb_to_lab, ciede2000_std

S = (256, 256)
SEEDS_FORMAL = (1, 2, 3, 4, 5)
SEED_PRECHECK = 1
LAMP, LAMP_S = "red_meat", 0.55   # 全场景统一施红肉灯污染（识别链主灯型）


# ============ 公共工具 ============

def _pollute(img, lamp=LAMP, s=LAMP_S):
    """施加灯污染（与 pollution_sim 同模型）"""
    g = np.array(LAMP_SPECTRA[lamp]["gain_rgb"], dtype=np.float32)
    lin = _srgb2lin(img)
    return np.clip(_lin2srgb(lin * (1.0 + (g - 1.0) * s)[None, None, :]),
                   0, 1).astype(np.float32)


def _lowfreq_tex(h, w, rng, scale=16):
    """低频纹理（模拟材质反射不均），返回0~1"""
    from scipy.ndimage import zoom
    low = rng.random((max(2, h // scale), max(2, w // scale)))
    tex = zoom(low, scale * h / (h // scale) / scale)[:h, :w]
    return tex / (tex.max() + 1e-6)


def _ring_frame(img, rgb, rng, tint=0.0, tint_rgb=None, frac=0.12):
    """
    环带边框：满幅区域为 rgb（可带色偏 tint×tint_rgb 与低频纹理+噪点），
    原图仅保留中央 (1-2*frac) 窗口。
    tint=0 → 纯色边框；tint>0 → 本色带色偏（非纯中性，G1/G2b/G3机理）。
    """
    h, w = img.shape[:2]
    m = int(min(h, w) * frac)
    frame = np.ones_like(img)
    base = np.array(rgb, dtype=np.float32)
    if tint > 0 and tint_rgb is not None:
        base = np.clip(base + tint * np.array(tint_rgb, dtype=np.float32), 0, 1)
    tex = _lowfreq_tex(h, w, rng, scale=16)[..., None]   # 低频反射纹理
    noise = rng.normal(0, 0.006, img.shape).astype(np.float32)
    layer = np.clip(base[None, None, :] * (0.92 + 0.16 * tex) + noise, 0, 1)
    out = img.copy()
    ring = _ring_mask(img.shape)
    out[ring] = layer[ring]
    return out, m


def _color_center(img, rgb, size_frac=0.50, seed=0, sat_var=0.08):
    """中央彩色主体（包装/广告牌/塑料盒/人工物），带纹理。返回新图。"""
    h, w = img.shape[:2]
    fh, fw = int(h * size_frac), int(w * size_frac)
    y0, x0 = (h - fh) // 2, (w - fw) // 2
    out = img.copy()
    tex = np.random.default_rng(seed).random((fh, fw))
    for c in range(3):
        out[y0:y0 + fh, x0:x0 + fw, c] = \
            np.clip(rgb[c] + sat_var * tex, 0, 1)
    return out, (y0, x0, y0 + fh, x0 + fw)


def _mask_box(shape, box):
    m = np.zeros(shape[:2], dtype=bool)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def _accept_mask(est, img):
    """
    重建算法采信掩码（与 estimate_pollution/_restore_selfcheck 同参数）：
    neutral_mode 含"宽"→sat_th=0.40 否则0.22；含"环带"→∩ring。
    未采信（neutral_mode=None）返回 None。
    """
    mode = est.get("neutral_mode")
    if not mode:
        return None
    sat_th = 0.40 if "宽" in mode else 0.22
    m = extract_neutral_zone(img, sat_th)
    if "环带" in mode:
        m = m & _ring_mask(img.shape)
    return m


def _stats(accept_mask, truth_mask):
    """§2.b 四量统计。accepted=0→(0,truth,0,None,None)单独记录。"""
    aa = int(accept_mask.sum()) if accept_mask is not None else 0
    ta = int(truth_mask.sum()) if truth_mask is not None else 0
    inter = int((accept_mask & truth_mask).sum()) if (accept_mask is not None and truth_mask is not None) else 0
    precision = inter / aa if aa > 0 else None
    recall = inter / ta if ta > 0 else None
    return dict(accepted_area=aa, truth_area=ta, intersection=inter,
                ref_precision=(round(precision, 4) if precision is not None else None),
                ref_recall=(round(recall, 4) if recall is not None else None))


def _mistrust(st, dE_target, scene_def):
    """
    非中性误采信判定（§2.b三情形+场景定义）。
    G6b 机理单列：采信的不是非中性材质而是被阴影破坏的观测——
    判据改为"还原输出 dE_target>3"。
    G2a/G3b 允许正确采信：truth非空且precision>=0.5 且 dE<=3 → 正确。
    """
    if scene_def.get("mistrust_rule") == "dE_only":
        return bool(dE_target is not None and dE_target > 3)
    aa = st["accepted_area"]
    ta = st["truth_area"]
    prec = st["ref_precision"]
    if aa == 0:
        return False           # 未采信，不构成误采信（单独记录）
    if ta == 0:
        return True            # 整图无真中性，任何采信都是误采信
    # truth 非空：precision<0.5 → 误采信
    mistrusted = bool(prec is not None and prec < 0.5)
    if scene_def.get("mistrust_rule") == "precision_or_dE":
        # G2a：采到标签/混合区（precision<0.5）且还原错才算完整误采信；
        # 但 precision<0.5 本身即采信非中性，dE 单独上报
        return mistrusted
    return mistrusted


def _dE_target(rec, gt, target_mask):
    """dE_target = CIEDE2000(还原输出, gt_scene) 仅在 target_mask 区域"""
    if rec is None:
        return None
    dE = ciede2000_std(rgb_to_lab(gt), rgb_to_lab(rec))
    return round(float(dE[target_mask].mean()), 3)


def _entered_submode(est):
    """
    §2.c 结构化判定"进入独立场景证据子模式"：
    存在一级 stages 满足——indep_status==PASS 且该级 indep 证据含
    center_sat_median（中央候选<100 触发彩色场景证据分支）。
    不做中文提示语匹配。
    """
    ev = est.get("indep_evidence") or {}
    for s in ev.get("stages", []):
        if s.get("indep_status") != "PASS":
            continue
        indep = s.get("indep") or {}
        if "center_sat_median" in indep and "dir_diff" not in indep:
            return True
    return False


def _branch_mode(est):
    """采信级的分支证据模式：center_color_scene / direction_diff / None"""
    ev = est.get("indep_evidence") or {}
    for s in ev.get("stages", []):
        if s.get("indep_status") != "PASS":
            continue
        indep = s.get("indep") or {}
        if "center_sat_median" in indep and "dir_diff" not in indep:
            return "center_color_scene"
        if "dir_diff" in indep:
            return "direction_diff"
    return None


def _deE_stats(est):
    return dict(
        indep_status_all=[s.get("indep_status") for s in (est.get("indep_evidence") or {}).get("stages", [])],
        accepted_level=(est.get("neutral_mode")),
    )


# ============ 十个场景构造（设计稿§3 六要素） ============

def sc_G1(seed, variant=0):
    """
    G1 彩色包装+色偏灰边框。攻击对象：环带"灰"边框本色带轻微未知色偏
    （两变体：偏红/偏蓝）+低频反射纹理+噪点，非纯中性。
    真值：ref_truth_mask=空集（整图无真中性）。target_mask=中央包装。
    """
    rng = np.random.default_rng(seed)
    h, w = S
    base = np.array([0.62, 0.60, 0.58], dtype=np.float32)     # 灰边框本色
    tint_rgb = [(0.15, -0.05, -0.05), (-0.05, -0.05, 0.15)][variant]
    plate = np.full((h, w, 3), 0.60, dtype=np.float32)         # 底板（会被框覆盖）
    img, _ = _ring_frame(plate, base, rng, tint=0.12, tint_rgb=tint_rgb)
    img, box = _color_center(img, [0.70, 0.25, 0.20], size_frac=0.50, seed=seed)
    ring = _ring_mask(S)
    return dict(gt=img.copy(), target=_mask_box(S, box),
                truth=np.zeros(S, dtype=bool), ring=ring)


def sc_G2a(seed):
    """
    G2a 真托盘在场+彩色标签污染。真灰托盘[0.78,0.76,0.74]四边带12%，
    其中一侧贴大块彩色标签（污染源）。采信干净托盘=正确；采信标签/
    混合区=误采信。truth=干净托盘区（三侧+被标签劈开一侧的剩余部分）。
    """
    rng = np.random.default_rng(seed)
    h, w = S
    m = int(min(h, w) * 0.12)
    tray = np.array([0.78, 0.76, 0.74], dtype=np.float32)
    tex = _lowfreq_tex(h, w, rng, 16)[..., None]
    img = np.full((h, w, 3), 0.45, dtype=np.float32)          # 案板底
    img[:, :] = tray * (0.94 + 0.12 * tex)
    img = img.astype(np.float32)
    clean = np.zeros(S, dtype=bool)
    clean[:m, :] = True; clean[-m:, :] = True
    clean[:, :m] = True; clean[:, -m:] = True
    # 顶部边带中央贴彩色标签（占顶边带60%宽）
    lab_w = int(w * 0.6)
    x0 = (w - lab_w) // 2
    label = np.zeros(S, dtype=bool)
    label[:m, x0:x0 + lab_w] = True
    ltex = np.random.default_rng(seed + 1).random((m, lab_w))
    for c, v in enumerate([0.85, 0.30, 0.25]):
        img[:m, x0:x0 + lab_w, c] = np.clip(v + 0.06 * ltex, 0, 1)
    truth = clean & ~label                                        # 干净托盘
    img, box = _color_center(img, [0.55, 0.20, 0.18], size_frac=0.52, seed=seed)  # 中央红肉，填满中央窗口
    return dict(gt=img.copy(), target=_mask_box(S, box), truth=truth,
                label=label, ring=_ring_mask(S))


def sc_G2b(seed, variant=0):
    """G2b 无真托盘：色偏标签边框冒充参照。机理同G1（truth=空集）。"""
    rng = np.random.default_rng(seed)
    h, w = S
    base = np.array([0.66, 0.64, 0.62], dtype=np.float32)
    tint_rgb = [(0.15, -0.05, -0.05), (-0.05, -0.05, 0.15)][variant]
    plate = np.full((h, w, 3), 0.62, dtype=np.float32)
    img, _ = _ring_frame(plate, base, rng, tint=0.12, tint_rgb=tint_rgb)
    img, box = _color_center(img, [0.25, 0.35, 0.70], size_frac=0.50, seed=seed)  # 蓝色商品
    return dict(gt=img.copy(), target=_mask_box(S, box),
                truth=np.zeros(S, dtype=bool), ring=_ring_mask(S))


def sc_G3(seed, variant=0):
    """
    G3 色偏灰背景+彩色广告牌。背景带轻微色偏（暖灰两变体），复用径向
    亮度渐变+低频纹理+噪点机理（synth_images._tray_background 同构）。
    truth=空集。target=广告牌。
    """
    rng = np.random.default_rng(seed)
    h, w = S
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)
    grad = 1.0 - 0.30 * r
    tex = _lowfreq_tex(h, w, rng, 16)
    tint = [(0.055, 0.020, 0.000), (0.000, 0.020, 0.055)][variant]  # 暖灰/冷灰（中度色偏）
    img = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        img[..., c] = np.clip((0.80 + tint[c]) * grad * (0.92 + 0.14 * tex), 0, 1)
    img += rng.normal(0, 0.005, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    img, box = _color_center(img, [0.20, 0.45, 0.85], size_frac=0.52, seed=seed)  # 蓝广告牌，填满中央窗口
    return dict(gt=img.copy(), target=_mask_box(S, box),
                truth=np.zeros(S, dtype=bool), ring=_ring_mask(S))


def sc_G3b(seed):
    """
    G3b 真中性背景对照（非攻击）：纯灰[0.80,0.80,0.80]无色偏+径向渐变+
    轻噪点，中央彩色广告牌。truth=背景全区（排除广告牌后的背景）。
    正确行为=采信真实背景（precision>=0.5）。
    """
    rng = np.random.default_rng(seed)
    h, w = S
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)
    grad = 1.0 - 0.30 * r
    tex = _lowfreq_tex(h, w, rng, 16)
    img = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        img[..., c] = np.clip(0.80 * grad * (0.92 + 0.14 * tex), 0, 1)
    img += rng.normal(0, 0.005, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    img, box = _color_center(img, [0.20, 0.45, 0.85], size_frac=0.52, seed=seed)
    truth = np.ones(S, dtype=bool) & ~_mask_box(S, box)      # 背景全区（除广告牌）
    return dict(gt=img.copy(), target=_mask_box(S, box), truth=truth,
                ring=_ring_mask(S))


def sc_G4(seed):
    """
    G4 彩色塑料盒+木台面。环带为木台面色（暖棕带色偏低饱和，非中性），
    覆盖环带≥80%面积。已知可达性风险：候选可能不足。truth=空集。
    """
    rng = np.random.default_rng(seed)
    h, w = S
    wood = np.array([0.62, 0.52, 0.40], dtype=np.float32)      # 木色（带色偏）
    tex = _lowfreq_tex(h, w, rng, 12)
    img = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        img[..., c] = np.clip(wood[c] * (0.85 + 0.30 * tex), 0, 1)
    img += rng.normal(0, 0.006, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    img, box = _color_center(img, [0.85, 0.30, 0.30], size_frac=0.46, seed=seed)  # 红色塑料盒
    return dict(gt=img.copy(), target=_mask_box(S, box),
                truth=np.zeros(S, dtype=bool), ring=_ring_mask(S))


def sc_G5(seed):
    """
    G5 彩色人工物+环带低饱和区（最强攻击，预检第一优先）。
    环带专门构造逐项过前置门：低饱和候选≥800、连通块≥64px、碎片化<30%、
    Q1≤40%（环带天然）、Q3不适用、Q4离散度≤0.15、中央候选<100、
    中央饱和度≥0.30。带轻微色偏（低饱和但非纯中性）。truth=空集。
    """
    rng = np.random.default_rng(seed)
    h, w = S
    base = np.array([0.64, 0.62, 0.60], dtype=np.float32)
    plate = np.full((h, w, 3), 0.62, dtype=np.float32)
    img, _ = _ring_frame(plate, base, rng, tint=0.12,
                         tint_rgb=(0.12, -0.03, -0.05))
    img, box = _color_center(img, [0.72, 0.26, 0.22], size_frac=0.52, seed=seed)  # 红人工物（R不过曝）
    return dict(gt=img.copy(), target=_mask_box(S, box),
                truth=np.zeros(S, dtype=bool), ring=_ring_mask(S))


def sc_G6a(seed):
    """
    G6a 反光制造假中性候选：真托盘在场+环带一侧放镜面反光斑（≈光源色）。
    反光候选方向≈灯方向（物理使然），试图骗过方向差检验。
    truth=干净托盘区；反光斑真值掩码单独标注（误采信判据）。
    """
    rng = np.random.default_rng(seed)
    h, w = S
    m = int(min(h, w) * 0.12)
    tray = np.array([0.78, 0.76, 0.74], dtype=np.float32)
    tex = _lowfreq_tex(h, w, rng, 16)[..., None]
    img = np.full((h, w, 3), 0.45, dtype=np.float32)
    img = (tray[None, None, :] * (0.94 + 0.12 * tex)).astype(np.float32)
    truth = np.zeros(S, dtype=bool)
    truth[:m, :] = True; truth[-m:, :] = True
    truth[:, :m] = True; truth[:, -m:] = True
    # 左侧边带中部放反光亮斑（16×80，显示域近光源色偏=近似中性高光）
    spec = np.zeros(S, dtype=bool)
    y0, y1 = int(h * 0.4), int(h * 0.4) + 80
    spec[y0:y1, :16] = True
    truth &= ~spec
    img[spec] = [0.95, 0.93, 0.91]
    img, box = _color_center(img, [0.55, 0.20, 0.18], size_frac=0.44, seed=seed)
    return dict(gt=img.copy(), target=_mask_box(S, box), truth=truth,
                specular=spec, ring=_ring_mask(S))


def sc_G6b(seed):
    """
    G6b 阴影劈同材质：真托盘被阴影劈成两半（阴影半增益偏移→环带vs中央
    方向差虚高→假"不同源"证据）。误采信机理单列：采信被阴影破坏的观测，
    判据=dE_target>3（truth=托盘全区，材质本中性，阴影不算非中性）。
    """
    rng = np.random.default_rng(seed)
    h, w = S
    m = int(min(h, w) * 0.12)
    tray = np.array([0.78, 0.76, 0.74], dtype=np.float32)
    tex = _lowfreq_tex(h, w, rng, 16)[..., None]
    img = (tray[None, None, :] * (0.94 + 0.12 * tex)).astype(np.float32)
    # 上半边带（亮侧）与下半边带（阴影侧×0.62线性域）
    shadow = np.zeros(S, dtype=bool)
    shadow[-m:, :] = True
    lin = _srgb2lin(img)
    lin[shadow] *= 0.62
    img = np.clip(_lin2srgb(lin), 0, 1).astype(np.float32)
    truth = np.zeros(S, dtype=bool)
    truth[:m, :] = True; truth[-m:, :] = True
    truth[:, :m] = True; truth[:, -m:] = True
    img, box = _color_center(img, [0.55, 0.20, 0.18], size_frac=0.44, seed=seed)
    return dict(gt=img.copy(), target=_mask_box(S, box), truth=truth,
                shadow=shadow, ring=_ring_mask(S),
                mistrust_rule="dE_only")


def sc_G6c(seed):
    """
    G6c 反光+阴影组合：反光斑在环带（近光源色）、阴影在中央区域，
    方向差被人为拉大→PASS假独立证据→增益估计用反光斑→还原错。
    truth=托盘干净区；反光斑掩码单独标注，采信掩码∩反光斑>50% 记误采信。
    """
    rng = np.random.default_rng(seed)
    h, w = S
    m = int(min(h, w) * 0.12)
    tray = np.array([0.78, 0.76, 0.74], dtype=np.float32)
    tex = _lowfreq_tex(h, w, rng, 16)[..., None]
    img = (tray[None, None, :] * (0.94 + 0.12 * tex)).astype(np.float32)
    spec = np.zeros(S, dtype=bool)
    y0, y1 = int(h * 0.4), int(h * 0.4) + 80
    spec[y0:y1, :16] = True
    img[spec] = [0.95, 0.93, 0.91]
    truth = np.zeros(S, dtype=bool)
    truth[:m, :] = True; truth[-m:, :] = True
    truth[:, :m] = True; truth[:, -m:] = True
    truth &= ~spec
    img, box = _color_center(img, [0.55, 0.20, 0.18], size_frac=0.44, seed=seed)
    # 中央区域内再压一块阴影（劈开中央主体与反光的方向）
    shadow = np.zeros(S, dtype=bool)
    ch0, cw0 = int(h * 0.25), int(w * 0.25)
    shadow[ch0:ch0 + 40, cw0:cw0 + 40] = True
    lin = _srgb2lin(img)
    lin[shadow] *= 0.55
    img = np.clip(_lin2srgb(lin), 0, 1).astype(np.float32)
    return dict(gt=img.copy(), target=_mask_box(S, box), truth=truth,
                specular=spec, shadow=shadow, ring=_ring_mask(S))


SCENES = {
    "G1_colorwrap_offset_frame":  dict(mk=lambda sd: sc_G1(sd, 0), attack=True,
        expect="色偏边框被误采信→识别/还原错→测门2", name_note="边框偏红变体"),
    "G1b_colorwrap_offset_blue":  dict(mk=lambda sd: sc_G1(sd, 1), attack=True,
        expect="同G1，偏蓝变体", name_note="边框偏蓝变体"),
    "G2a_tray_label_pollution":   dict(mk=sc_G2a, attack=True,
        expect="采信干净托盘=正确；采信标签/混合区=误采信",
        mistrust_rule="precision_or_dE"),
    "G2b_offset_label_frame":     dict(mk=lambda sd: sc_G2b(sd, 0), attack=True,
        expect="色偏标签边框冒充参照（无真托盘）"),
    "G3_offset_background":      dict(mk=lambda sd: sc_G3(sd, 0), attack=True,
        expect="色偏暖灰背景被误采信？", name_note="暖灰变体"),
    "G3b_neutral_background_ctrl": dict(mk=sc_G3b, attack=False,
        expect="对照：真中性背景应被正确采信（precision>=0.5）"),
    "G4_plastic_box_wood":        dict(mk=sc_G4, attack=True,
        expect="木台面低饱和带色偏，可达性风险场景"),
    "G5_crafted_lowsat_ring":     dict(mk=sc_G5, attack=True,
        expect="逐项过前置门的强攻击（预检第一优先）"),
    "G6a_specular_neutral":       dict(mk=sc_G6a, attack=True,
        expect="反光斑≈光源色骗方向差"),
    "G6b_shadow_split_material":  dict(mk=sc_G6b, attack=True,
        expect="阴影劈同材质造假方向差；误采信判据=dE_target>3"),
    "G6c_specular_shadow_combo":  dict(mk=sc_G6c, attack=True,
        expect="反光+阴影组合拉大方向差"),
}


# ============ 单样本执行（§4 逐级证据结构） ============

def run_one(scene_name, seed):
    """构造本色图→施灯污染→识别→还原→逐级证据+真值评价全量记录。"""
    sc = SCENES[scene_name]
    d = sc["mk"](seed)                       # 本色图+真值掩码
    gt, target_mask, truth_mask = d["gt"], d["target"], d["truth"]
    img = _pollute(gt)                       # 施灯污染（核心区别于D组）

    est = estimate_pollution(img)
    rec, _M, rep = (restore_image(img, est, food_type="meat")
                    if est["detected"] else (None, None, None))

    accept_mask = _accept_mask(est, img)
    st = _stats(accept_mask, truth_mask)
    dE_target = _dE_target(rec, gt, target_mask)
    mistrust = _mistrust(st, dE_target, sc)

    # G6a/G6c：反光斑掩码交叠判据（采信掩码∩specular>50%）
    spec_overlap = None
    if d.get("specular") is not None and accept_mask is not None:
        so = int((accept_mask & d["specular"]).sum())
        spec_overlap = dict(spec_area=int(d["specular"].sum()),
                            accept_on_spec=so,
                            frac=(round(so / max(1, st["accepted_area"]), 4)
                                  if st["accepted_area"] else None))

    sample = dict(
        scene=scene_name, seed=seed, attack=sc["attack"], expect=sc["expect"],
        # 前置门逐级轨迹
        stages=(est.get("indep_evidence") or {}).get("stages"),
        candidate_gate_summary=_deE_stats(est),
        # 分支证据（结构化，无中文匹配）
        independent_mode=_branch_mode(est),
        independent_status_final=(est.get("indep_evidence") or {}).get(
            "stages", [{}])[-1].get("indep_status") if (est.get("indep_evidence") or {}).get("stages") else None,
        entered_submode=_entered_submode(est),
        indep_evidence=est.get("indep_evidence"),
        # 识别与还原（防御式取值，四出口键不对称）
        estimate_detected=bool(est["detected"]),
        lamp=est.get("lamp_type"), strength=est.get("strength"),
        confidence=est.get("confidence"), neutral_mode=est.get("neutral_mode"),
        failure_code=est.get("failure_code"), failure_reason=est.get("failure_reason"),
        restore_rejected=(rep.get("rejected") if rep else None),
        reject_gate=(rep.get("reject_gate") if rep else None),
        dev_post=((rep.get("selfcheck") or {}).get("dev_post") if rep else None),
        rep_failure_code=(rep.get("failure_code") if rep else None),
        rec_is_none=(rec is None),
        # 真值评价（§2.b 四量）
        dE_target=dE_target,
        accepted_area=st["accepted_area"], truth_area=st["truth_area"],
        intersection=st["intersection"], ref_precision=st["ref_precision"],
        ref_recall=st["ref_recall"],
        mistrust=bool(mistrust),
        mistrust_rule=sc.get("mistrust_rule", "default"),
        # 双层口径（第十五轮）：nonneutral_accept=非中性采信（逐样本事实层，
        # 即旧 mistrust 原值）；severe_mistrust=严重误采信（采信非中性且还原
        # 输出dE>3，伤害层；dE_only场景两者同值）。旧字段保留兼容历史JSON。
        nonneutral_accept=bool(mistrust),
        severe_mistrust=bool(mistrust and dE_target is not None and dE_target > 3),
        spec_overlap=spec_overlap,
    )
    return sample


# ============ precheck（§5：1种子/场景，四判据） ============

def run_precheck():
    per_scene = {}
    for name in SCENES:
        sample = run_one(name, SEED_PRECHECK)
        stages = sample["stages"] or []
        per_scene[name] = dict(
            seed=SEED_PRECHECK,
            entered_candidate_gate=bool(stages),        # 是否进入候选质量门
            entered_submode=sample["entered_submode"],
            independent_status=sample["independent_status_final"],
            branch_mode=sample["independent_mode"],
            failure_code=sample["failure_code"],
            failure_reason=sample["failure_reason"],
            estimate_detected=sample["estimate_detected"],
            restore_rejected=sample["restore_rejected"],
            reject_gate=sample["reject_gate"],
            dev_post=sample["dev_post"],
            restore_output=(not sample["rec_is_none"]),
            dE_target=sample["dE_target"],
            accepted_area=sample["accepted_area"],
            truth_area=sample["truth_area"],
            ref_precision=sample["ref_precision"],
            mistrust=sample["mistrust"],
            stages=stages,
            stopped_tuning=False, tune_rounds=0,
        )
        s = per_scene[name]
        print(f"  {name}: 子模式={'Y' if s['entered_submode'] else 'N'} "
              f"分支={s['branch_mode']} 状态={s['independent_status']} "
              f"detected={s['estimate_detected']} code={s['failure_code']} "
              f"reject_gate={s['reject_gate']} dev_post={s['dev_post']} "
              f"dE={s['dE_target']} prec={s['ref_precision']}")
    # 四判据自审
    n_submode = sum(1 for s in per_scene.values() if s["entered_submode"])
    # 判据2：至少1类返回PASS（任一级PASS且最终采信，或识别成功即采信级PASS）
    n_pass = sum(1 for s in per_scene.values()
                 if any(st.get("indep_status") == "PASS" for st in (s["stages"] or [])))
    # 判据3：至少1类能计算非中性误采信（truth空集进入并采信，或precision<0.5可判）
    n_mistrust_calc = sum(1 for s in per_scene.values()
                          if (s["truth_area"] == 0 and s["accepted_area"] > 0)
                          or (s["ref_precision"] is not None))
    # 判据4：至少1类能测门2兜底（识别接受且还原阶段可观测）
    n_gate2_obs = sum(1 for s in per_scene.values()
                      if s["estimate_detected"] and s["reject_gate"] is not None)
    n_restore_output = sum(1 for s in per_scene.values()
                           if s["estimate_detected"] and s["restore_rejected"] is not None)
    criteria = dict(
        c1_at_least_3_submode=dict(need=3, got=n_submode,
                                    scenes=[k for k, s in per_scene.items() if s["entered_submode"]]),
        c2_at_least_1_pass=dict(need=1, got=n_pass,
                                scenes=[k for k, s in per_scene.items()
                                        if any(st.get("indep_status") == "PASS" for st in (s["stages"] or []))]),
        c3_mistrust_computable=dict(need=1, got=n_mistrust_calc,
                                     scenes=[k for k, s in per_scene.items()
                                             if (s["truth_area"] == 0 and s["accepted_area"] > 0)
                                             or (s["ref_precision"] is not None)]),
        c4_gate2_observable=dict(need=1, got=max(n_gate2_obs, n_restore_output),
                                 scenes=[k for k, s in per_scene.items()
                                         if s["estimate_detected"] and s["reject_gate"] is not None]),
    )
    all_pass = all(v["got"] >= v["need"] for v in criteria.values())
    print("\n预检四判据：")
    for k, v in criteria.items():
        print(f"  [{k}] {v['got']}/{v['need']} → {'PASS' if v['got'] >= v['need'] else 'FAIL'}")
    print(f"预检总判定: {'通过，可进入5种子正式实验' if all_pass else '不通过，需修构造（≤2轮）'}")
    return dict(per_scene=per_scene, criteria=criteria, all_pass=all_pass)


# ============ 正式实验（5种子全量落盘） ============

def run_formal():
    out = {}
    for name in SCENES:
        samples = [run_one(name, sd) for sd in SEEDS_FORMAL]
        n_total = len(samples)
        n_submode = sum(1 for s in samples if s["entered_submode"])
        # PASS样本 = 最终采信级PASS（识别成功=某级PASS且被采信）
        n_pass = sum(1 for s in samples if s["estimate_detected"])
        # 误采信（分子）：PASS且非中性误采信且还原输出dE>3
        n_mistrust = sum(1 for s in samples
                         if s["estimate_detected"] and s["mistrust"]
                         and (s["dE_target"] is not None and s["dE_target"] > 3))
        # 双层口径（第十五轮）：非中性采信数（逐样本事实层）/严重误采信数（伤害层）
        n_nonneutral = sum(1 for s in samples
                           if s["estimate_detected"] and s["nonneutral_accept"])
        n_severe = sum(1 for s in samples
                       if s["estimate_detected"] and s["severe_mistrust"])
        # 最终错误输出率：rec非None且dE>3
        n_wrong = sum(1 for s in samples
                      if (not s["rec_is_none"]) and s["dE_target"] is not None
                      and s["dE_target"] > 3)
        n_rejected_by_gate2 = sum(1 for s in samples if s["reject_gate"] == "selfcheck")
        out[name] = dict(
            expect=SCENES[name]["expect"], attack=SCENES[name]["attack"],
            n_total=n_total,
            entered_submode_rate=round(n_submode / n_total, 3),
            pass_rate_vs_candidate=None,   # 见下方补充
            pass_rate_vs_total=round(n_pass / n_total, 3),
            mistrust_rate=(round(n_mistrust / n_pass, 3) if n_pass else None),
            # 双层口径（第十五轮）：分母均为 n_pass（识别成功数）
            nonneutral_accept_count=n_nonneutral,
            severe_mistrust_count=n_severe,
            severe_mistrust_rate=(round(n_severe / n_pass, 3) if n_pass else None),
            final_wrong_output_rate=round(n_wrong / n_total, 3),
            gate2_intercept_rate=round(n_rejected_by_gate2 / n_total, 3),
            summary=dict(n_submode=n_submode, n_pass=n_pass,
                         n_mistrust=n_mistrust, n_wrong_output=n_wrong,
                         n_gate2_rejected=n_rejected_by_gate2,
                         n_nonneutral_accept=n_nonneutral,
                         n_severe_mistrust=n_severe),
            per_seed=samples,
        )
        # PASS率双分母之"进入候选验证级"分母
        n_entered_cand = sum(1 for s in samples if s["stages"])
        out[name]["pass_rate_vs_candidate"] = (round(n_pass / n_entered_cand, 3)
                                               if n_entered_cand else None)
        sm = out[name]["summary"]
        print(f"  {name}: 子模式{n_submode}/5 识别{n_pass}/5 "
              f"误采信{n_mistrust} 非中性采信{n_nonneutral} 严重误采信{n_severe} "
              f"错误输出{n_wrong} 门2拦截{n_rejected_by_gate2}")
    return out


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "precheck"
    from exp_meta import save_results
    if mode == "precheck":
        print("== V0.8 precheck：每场景1种子（设计稿§5）==")
        res = run_precheck()
        save_results("results_exp8_precheck.json",
                     dict(precheck=res),
                     params=dict(mode="precheck", n_seeds=1,
                                 seed=SEED_PRECHECK, lamp=LAMP,
                                 lamp_strength=LAMP_S,
                                 scenes=list(SCENES.keys()),
                                 note="V0.8强攻击可达性预检（第十二轮A哥批准，四判据硬性门槛）"))
    elif mode == "formal":
        print("== V0.8 正式实验：5种子全量落盘（设计稿§4/§7）==")
        res = run_formal()
        save_results("results_exp8_attack.json",
                     res,
                     params=dict(mode="formal", n_seeds=5,
                                 seeds=list(SEEDS_FORMAL), lamp=LAMP,
                                 lamp_strength=LAMP_S,
                                 scenes=list(SCENES.keys()),
                                 note="V0.8强攻击仿真（逐样本证据全量落盘，禁中文提示语匹配判定）",
                                 field_semantics=dict(
                                     nonneutral_accept="逐样本：采信了非中性参照（事实层，无伤害判定）",
                                     severe_mistrust="逐样本：采信非中性且还原输出dE>3（伤害层）",
                                     nonneutral_accept_count="汇总：识别成功样本中非中性采信数（分母n_pass）",
                                     severe_mistrust_count="汇总：识别成功样本中严重误采信数（分母n_pass）",
                                     severe_mistrust_rate="汇总：severe_mistrust_count/n_pass",
                                     mistrust_rate="旧字段（严格口径）：非中性采信且dE>3，分母n_pass；与severe_mistrust_rate等值，保留兼容历史JSON",
                                     mistrust="旧字段（逐样本）：=nonneutral_accept，保留兼容历史JSON")))
    else:
        print("用法: python exp8_attack.py [precheck|formal]")

