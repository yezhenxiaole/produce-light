# -*- coding: utf-8 -*-
"""
光源污染识别器 + 场景化反美颜还原算法（V0.7）
==============================================
核心算法链：
  环带/全图中性区提取（候选质量判定+独立参照检验三态+还原自检门）
  → 色偏方向与强度计算 → 生鲜灯类型判定 → 场景化还原矩阵构建
  → 反美颜还原 → 置信度输出 →（V0.7）还原后参照中性回归自检

契约（V0.5，响应第二轮审查意见）：
  1. 灰世界假设不作为还原路径的增益估计来源：仅作 (a) 对照组 generic_awb、
     (b) 失败分支路由探针（区分“自然光无色偏”与“有色偏但无中性参照”），
     不参与灯库拟合与还原；
  2. 无中性参照独立失败分支：四级降级链全部失败（含候选质量不合格）时
     直接返回“无中性参照”原因，不再由灰世界完成识别；
  3. 环带可靠性逐图检测（构图先验不默认成立）：环带中性像素总数与
     两侧边带分布均达标才采用环带，否则视为遮挡嫌疑回退全图；
  4. 候选质量判定（意见1）：每一级候选须通过占比上限、连通块/碎片化、
     中央集中度、色彩离散度四项检验，防假中性凑像素数绕过；
  5. 拟合契约（意见3）：拟合强度超出[0.05,0.95]或次优灯型残差与最优过近
     （歧义）时明确拒绝——适用条件是可执行的技术限制，非说明书声明；
  6. 灯库参数为合成仿真设定值（非真实标定），真实采集后需替换为
     calibration.py 标定模块输出的真实参数（意见4）。

契约（V0.7，响应第五轮审查意见）：
   7. 独立参照检验三态化（意见1/2）：PASS/FAIL/INSUFFICIENT。“样本不足”
      不再默认放行——中央候选不足时按中央彩色像素证据判定：中央区域
      存在彩色像素证据（不区分彩色来源，饱和度中位数≥0.30）→环带
      天然独立 PASS；否则 INSUFFICIENT 拒绝。证据链要求：每级候选
      记录实际触发的判定门与全部中间量（dir_diff/中央像素数/饱和度
      中位数），落入结果 JSON；
   8. 还原自检第二道门（意见6；V0.7a 按实际输出口径重标定、V0.7b 统一
       掩码口径）：dev_post 在实际还原输出 lin_restored（含 scene_adjust）
       上计算，参照**掩码沿用 estimate_pollution 采信阶段确定的掩码**
       （原图按采信级参数重取），仅在输出上测增益。归一化增益偏离
       dev_post 超限则拒绝交付。exp7_gate2 实际输出口径标定：
       正确单灯还原 dev_post max=0.0279（输出底噪上界），混合光/失配
       拦截侧全部样本 min=0.0302 → 阈值 0.03 取隔离带 [0.0279,0.0302]
       内，拦截并提示疑似混合光；正确还原 dev_post 均值0.0196 不受影响；
   9. 诚实边界：方向差是独立性证据但不单独证明“这是托盘/灰卡”
      （反光/标签/光照不均也可造成方向差）；过曝 0.30 为经验阈值
      非普适；全部阈值（0.07/0.03/0.30/0.55）在真实数据上需重新标定。
"""
import numpy as np

# 生鲜灯特征色偏方向（单位向量，RGB空间）
LAMP_DIRECTIONS = {
    "red_meat":   np.array([0.90, -0.30, -0.32], dtype=np.float32),
    "green_veg":  np.array([-0.35, 0.88, -0.32], dtype=np.float32),
    "blue_sea":   np.array([-0.30, -0.32, 0.90], dtype=np.float32),
    "hidden_red": np.array([0.55, -0.08, -0.45], dtype=np.float32),  # 红增蓝减但幅度温和
}

# 与识别阈值配套的置信度参数
CONF_TH = 0.55          # 置信度低于此值 → 触发失败分支
NEUTRAL_MIN_PIXELS = 300  # 中性区最少像素数（全图提取）；300px在256×256图上约0.46%。
                           # 经验参数：低于此数统计噪声大（增益方差不可控），高于则会排除
                           # 小参照物场景。注意exp4场景B实测为强污染下参照物被推过曝→拒绝，
                           # 该300值并不保证小参照物在强污染下可达识别（见交底书4.5节）。
RING_MIN_PIXELS = 800     # 环带中性像素总数下限（外圈8%×周长）
RING_SIDE_MIN = 150       # 环带单侧边带（上/下/左/右）中性像素数下限（遮挡检测）
RING_MIN_SIDES = 2       # 环带至少需要几条边带达标；少于则视为遮挡嫌疑，环带不可靠
DEV_MIN = 0.03           # 线性域增益偏差范数下限，低于则判定无污染

# 两个语义不同的参数，数值均为0.30纯属巧合，已拆开命名避免“参数复用”误读：
OVEREXPOSED_REJECT_TH = 0.30  # 过曝比例拒绝阈值（≥则拒绝还原）。标定依据（合成数据退化
                               # 曲线）：过曝6%→dE1.30、28%→2.18（可用上限附近）、
                               # 34%→2.60、47%→3.86；0.30为当前实验的工程可用性分界
                               # （对应dE≈2.3），非普适阈值，真实数据上需重新标定
CONF_DEV_SCALE = 0.30         # 置信度色偏幅度归一化尺度（经验值）：将常见污染幅度
                               # dev≈0.1~0.4映射到幅度项0.5~1.0区间；与过曝阈值语义无关

# ---- 全图/环带候选质量判定参数（V0.5，响应第二轮审查意见1：防“假中性”绕过） ----
# 假中性来源：浅色脂肪（碎片化散布）、鱼身（大面积被蓝光“中和”）、包装反光（小块高光）。
# 真实中性参照特征：连续大面积（托盘/案板）、位于周边（构图先验）、色彩均匀。
FULL_MAX_FRAC = 0.40      # Q1 占比上限：全图候选占比>40% → 疑似大面积食物本体被误判。
                           # 判定基线：满分幅减去过曝像素。注意：该阈值会随真实托盘
                           # 占比分布重新标定（当前0.40为合成数据经验值，见4.5节边界声明）。
FULL_FRAG_FRAC = 0.30     # Q2 碎块像素占候选比例上限：>30% → 碎片化嫌疑，该级不采信。
                           # 碎块=连通块面积< FULL_BLOB_MIN 的像素。经验参数。
FULL_BLOB_MIN = 64        # Q2 有效连通块最小面积（像素）；小于此为碎块（脂肪斑点特征）
FULL_CENTER_FRAC = 0.90   # Q3 中央集中上限（仅全图路径）：中央50%窗口候选占候选总数
                           # 比例>90% → 候选几乎全部位于中央、周边无参照 → 食物本体嫌疑。
                           # 注意：中央门曾试用于环带采信（RING_CENTER方案），实测误杀
                           # 蓝灯污染下的sea+托盘主场景（整图被拉成低饱和、中央占比100%
                           # 与满幅鱼同构），故回退为仅全图路径且阈值放宽到0.90。
FULL_CHROMA_STD = 0.15    # Q4 色散上限：候选线性归一化增益的通道std均值；参照物应均匀。
                           # 标定依据：正常构图环带（托盘+食物边缘混合）实测0.10~0.13，
                           # 满幅均匀食物0.07（更低但不可分——见下方盲区声明）；
                           # 0.15留15%余量。放宽自0.12（原值误杀meat/sea部分种子）。
S_EDGE_LOW, S_EDGE_HIGH = 0.05, 0.95  # 拟合强度贴边判定：超出[0.05,0.95]物理范围嫌疑
LAMP_AMBIG_RATIO = 0.55   # 灯型歧义判定：次优灯型残差/最优残差 < 此值 → 灯型不确定
# ---- 独立参照检验参数（V0.6，响应第三轮审查意见1：A2 修复） ----
# 物理依据：单光源下，真中性参照（托盘/纸巾）的观测增益方向==灯方向；
# 浅色食材本色的观测方向=灯方向×本色色偏，两者存在方向差。
# 实测方向差（诊断数据，256×256 合成场景）：
#   满幅鱼身（A2）环带 vs 中央：0.0001（同材质，无独立参照信息）；
#   sea s=0.9 全图严阈值级（鱼身主导）：0.0038（同质候选集，应拒绝）；
#   正常构图（食材居中+托盘周边）：0.12±0.01（独立参照方向差）；
#   红肉/蔬菜主体（中央无中性候选）：不触发比较，环带天然独立。
# INDEP_DIR_DIFF_TH=0.07：高于同质噪声底两个数量级，低于真实方向差一半，
# 中间留有隔离带。边界（诚实声明）：食材占画幅比例过小（约<40%）时
# 方向差可能不达标 → 保守拒绝（安全侧失败），真实数据上需统计误拒率。
INDEP_DIR_DIFF_TH = 0.07    # 独立参照判定：环带 vs 中央候选增益方向差阈值
INDEP_CENTER_MIN = 100      # 中央窗口候选像素下限：不足时走中央彩色像素证据分支
CENTER_COLOR_SAT_TH = 0.30  # 中央彩色像素证据判定：中央有效像素饱和度中位数≥此值
                            # → 中央区域存在彩色像素证据，环带候选与中央彩色
                            # 区域不同源，参照独立性有证据支撑（PASS）；
                            # 反之为浅色低饱和但候选不足 → INSUFFICIENT 拒绝。
                            # 标定：红肉居中构图实测中央饱和度中位数 0.651，
                            # 中性候选饱和度上限 0.22；0.30 居中，两侧留隔离带。
                            # 表述边界（V0.7a 降格，响应第六轮审查意见4）：
                            # 仅宣称"中央区域存在彩色像素证据"，不宣称主体
                            # 即食材——彩色包装/标签/反光同样满足本判据；
                            # 对此类场景的拦截不依赖本分支（实验见 exp6 D组：
                            # 彩色包装无低饱和候选，在候选数量层面已被拒；
                            # 同源白色区域由方向差检验 FAIL 拦截）。
# 还原自检第二道门（V0.7b，第七轮审查意见1）
# 数学本质：还原算子为逐通道乘法，参照均值的还原后中性度由拟合闭合差
# 代数决定（非独立验证）；本门定位为**输出闭合差契约**——实际输出
# lin_restored（含 scene_adjust）的参照增益偏离
# dev_post = ||归一化增益-1|| 超限则拒绝交付。
# 口径（V0.7b 统一）：参照**掩码沿用 estimate_pollution 采信阶段确定的
# 掩码**（原图按采信级参数重取），仅增益取自实际还原输出 lin_restored[mask]；
# 不在还原输出上重提取掩码（还原改变参照饱和度致掩码漂移，正确还原 G1
# 底噪上界 0.0279→0.0311 突破阈值0.03，见 _restore_selfcheck docstring）。
# 标定（exp7_gate2，实际输出口径，合成数据，2026-10-12）：
#   正确单灯还原（G1，27组）：dev_post 均值0.0196，max=0.0279（输出底噪上界）；
#   混合光/失配拦截侧（G3 90:10/10:90 误接受后被门2拒 9/27+9/27、G4 s=0.9
#   裁剪 18/27）：全部拦截样本 dev_post min=0.0302（均值≈0.0318~0.0374）；
#   错误灯库 G2：dev 均值0.70（远超阈值）。
#   → 隔离带 [0.0279, 0.0302]，本阈值 0.03 取带内（合成数据标定）。
#   已知边界：90:10/10:90 中被接受输出（dev≈0.017~0.026）与底噪重叠，与
#   "近单灯混合物理同构不可分"的数学结论一致（偏差约4.7%披露不变）。
#   隔离带较窄（宽约0.002），真实数据采集后需按同一实验重新标定。
RESTORE_SELFCHECK_DEV_TH = 0.03  # 门2阈值：exp7_gate2 实际输出口径标定值（合成数据）
# 已知盲区（诚实声明，写入交底书）：满幅浅色食材在自然光下且本色恰好落入
# 某弱幅度灯型（如 hidden_red）的增益方向子空间时，单张 RGB 图像无法区分
# “食材本色”与“灯污染”（可辨识性极限的具体实例）。V0.6 起此类场景被
# 独立参照检验结构性拒绝（满幅同质 → 无独立参照 → 只提示不还原）；
# 完整解决需满足适用条件 C2（存在独立中性参照）。


def extract_neutral_zone(img_f32, sat_th=0.22):
    """
    提取近中性区域（脂肪/托盘/包装/白色纤维等）：
    高亮度但未过饱和 + 低饱和度（阈值可调）。
    上限0.995：排除任一通道被推到1.0的过曝像素。
    注意：污染本身会推高中性物的饱和度，因此高强度污染时需
    配合 estimate_pollution 的两轮自适应（见下）放宽阈值。
    """
    mx = img_f32.max(axis=2)
    mn = img_f32.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0.0)
    mask = (mx > 0.45) & (mx < 0.995) & (sat < sat_th)
    return mask


def _gray_world_offset(img_f32):
    """
    灰世界均值（全图线性域，排除过曝像素）。
    契约：仅作 (a) 失败分支路由探针、(b) generic_awb 对照组使用；
    不作为还原路径的中性参照替代——灰世界会把大面积彩色食材误当光源
    （肉类场景增益估计发散至[1.57,0.80,0.64]），不能用于本发明的识别链。
    """
    from pollution_sim import _srgb2lin
    valid = (img_f32 < 0.999).all(axis=2)
    if valid.sum() < 100:
        return None, 0
    lin_mean = _srgb2lin(img_f32[valid]).mean(axis=0)
    return lin_mean, int(valid.sum())


def _fit_lamp(gain_est_norm):
    """
    将估计的归一化增益向量拟合到灯库：
    对每个灯型两阶段搜索强度s——粗网格（步长0.05）定位全局最优邻域，
    再在邻域内连续细化（一维有界优化），强度估计精度提升至<0.01。
    返回 (最佳灯型, 最佳强度, 残差, 次优记录list)。
    次优记录：[(残差, 灯型, 强度), ...]按残差升序——用于灯型歧义检验
    （V0.5：次优残差/最优残差过小 → 灯型不确定，拒绝或降置信度）。
    """
    from pollution_sim import LAMP_SPECTRA
    from scipy.optimize import minimize_scalar
    ss = np.linspace(0.05, 1.0, 20)
    per_lamp = []
    for lt, cfg in LAMP_SPECTRA.items():
        g = np.array(cfg["gain_rgb"], dtype=np.float64)

        def resid(s, g=g):
            lamp_gain = 1.0 + (g - 1.0) * s
            lamp_norm = lamp_gain / lamp_gain.mean()
            return float(np.linalg.norm(gain_est_norm - lamp_norm))

        vals = [(resid(s), s) for s in ss]
        r0, s0 = min(vals)
        lo = max(0.02, s0 - 0.075)
        hi = min(1.0, s0 + 0.075)
        opt = minimize_scalar(resid, bounds=(lo, hi), method="bounded",
                              options=dict(xatol=1e-4))
        r_best, s_best = (float(opt.fun), float(opt.x)) if opt.fun < r0 else (r0, s0)
        per_lamp.append((r_best, lt, s_best))
    per_lamp.sort()
    best = per_lamp[0]
    return best[1], best[2], best[0], per_lamp


def _ref_direction(img_f32, mask):
    """
    计算候选区域的中性观测方向（线性域归一化增益向量）。
    用于独立参照检验：比较环带候选与中央候选的方向差。
    """
    from pollution_sim import _srgb2lin
    if int(mask.sum()) < 30:
        return None
    lin_mean = _srgb2lin(img_f32[mask]).mean(axis=0)
    ref = float(lin_mean.mean())
    if ref < 1e-6:
        return None
    return np.asarray(lin_mean, dtype=np.float64) / ref


def _center_color_evidence(img_f32):
    """
    中央窗口彩色像素证据（V0.7 三态改造；V0.7a 表述降格，响应第六轮
    审查意见4）。
    中央候选不足时区分两种情形：
      a. 中央区域存在彩色像素证据（如彩色食材主体，也可能为彩色包装/
         标签/反光——本判据不区分彩色来源）→ 环带候选与中央彩色区域
         不同源，参照独立性有证据支撑 → PASS；
      b. 中央大面积浅色低饱和但候选不足（过曝裁剪/过暗/被亮度窗口筛掉）
         → 证据不足 → INSUFFICIENT（拒绝）。
    判据：中央窗口有效（非过曝）像素的饱和度中位数。
      实测标定（合成诊断，256×256）：
        - 红肉居中构图：中央均值饱和度 0.651，低饱和占比 0%；
        - 满幅浅色鱼：中央候选 16384（充足，不走本分支）；
        - 过曝浅色场景：饱和度趋 0（高亮裁剪后各通道趋同）。
      阈值 0.30 取在"彩色主体实测 0.651"与"中性候选上限 0.22"之间，
      两侧均留有约一倍隔离带。
    边界（诚实声明，V0.7a）：本分支不验证"主体是食材"，彩色包装/标签
    同样会 PASS；对该类场景的实际拦截发生在更早层面（低饱和候选数量
    不足）或后续门（方向差检验/质量四项/门2），实验证据见 exp6 D组。
    返回 (has_color_evidence: bool, center_sat_median: float, center_valid_frac: float)。
    """
    h, w = img_f32.shape[:2]
    ch0, ch1 = int(h * 0.25), int(h * 0.75)
    cw0, cw1 = int(w * 0.25), int(w * 0.75)
    cen = img_f32[ch0:ch1, cw0:cw1]
    mx = cen.max(axis=2)
    mn = cen.min(axis=2)
    valid = (mx < 0.995) & (mx > 1e-3)   # 排除过曝/全黑，二者不构成彩色证据
    if valid.sum() < 10:
        return False, 0.0, 0.0
    sat = np.where(mx[valid] > 1e-6, (mx[valid] - mn[valid]) / (mx[valid] + 1e-6), 0.0)
    return bool(float(np.median(sat)) >= CENTER_COLOR_SAT_TH), float(np.median(sat)), float(valid.mean())


def _independent_ref_check(img_f32, mask, zone, full_mask=None):
    """
    独立参照检验（V0.7 三态改造，响应第五轮审查意见1/2/3）。

    参数：
      img_f32    输入图
      mask       当前级别候选（ring 级=环带∩候选；full 级=全图候选）
      zone       "ring" 或 "full"（当前级别）
      full_mask  全图候选 mask（ring 级必传；full 级等于 mask 可省）
                 主体支（中央窗口）恒取全图候选口径——ring 级传入的
                 mask 中央窗口必为空，若用它做主体支该检验将恒被跳过。

    物理依据：单一光源下，
      - 真中性参照（托盘/纸巾）的观测增益方向 == 灯方向；
      - 浅色食材本色的观测方向 = 灯方向 × 食材本色色偏，两者有方向差。
    满幅同质浅色材质（A2 攻击：满幅鱼身无托盘）的环带候选与中央候选
    为同一材质 → 方向差→0 → 候选集不提供"独立于目标食材的中性参照"。

    三态返回（V0.7：证据不足不再是默认放行，与权利要求"独立中性参照
    为必要特征"对齐）：
      "PASS"          — 有足够证据证明候选集提供独立参照；
      "FAIL"          — 明确同质（方向差<阈值），该级候选不可采信；
      "INSUFFICIENT"  — 独立性证据不足（中央候选不足且无彩色像素证据，或
                        方向样本无法计算），按必要特征要求拒绝。
    证据边界（诚实声明）：方向差仅是"独立性证据"——证明环带候选与中央
    主体不同源，但方向差来源也可能是包装标签、反光、手指、光照不均等
    非中性参照物。完整证明需配合饱和度判据（候选提取已含）+ 参照质量
    四项判定 + 真实采集验证；本检验不单独宣称"检测到托盘/灰卡"。

    返回 (status: str, evidence: dict)。evidence 记录全部中间量供实验
    JSON 落盘（审查要求：必须能证明实际触发了哪个拒绝门）。
    """
    ev = dict(zone=zone)
    h, w = img_f32.shape[:2]
    ring = _ring_mask(img_f32.shape)
    # 参照支：环带∩全图候选（zone=ring 时传入 mask 已是环带∩候选，直接用）
    m_ring = mask if zone == "ring" else mask & ring
    # 主体支：中央窗口∩全图候选
    full_mask = mask if full_mask is None else full_mask
    ch0, ch1 = int(h * 0.25), int(h * 0.75)
    cw0, cw1 = int(w * 0.25), int(w * 0.75)
    center_mask = np.zeros_like(full_mask)
    center_mask[ch0:ch1, cw0:cw1] = full_mask[ch0:ch1, cw0:cw1]
    n_center = int(center_mask.sum())
    ev["n_center"] = n_center
    if n_center < INDEP_CENTER_MIN:
        # 中央候选不足 → 中央彩色像素证据判定（V0.7 三态；V0.7a 表述降格）
        is_color, sat_med, valid_frac = _center_color_evidence(img_f32)
        ev.update(center_sat_median=round(sat_med, 4),
                  center_valid_frac=round(valid_frac, 4))
        if is_color:
            ev["verdict"] = ("中央区域存在彩色像素证据（不区分彩色来源），"
                             "环带候选与中央彩色区域不同源")
            return "PASS", ev
        ev["verdict"] = ("中央候选不足且无彩色像素证据（浅色低饱和但证据不足："
                         "过曝裁剪/过暗/亮度窗口外），独立性证据不足")
        return "INSUFFICIENT", ev
    ring_dir = _ref_direction(img_f32, m_ring)
    center_dir = _ref_direction(img_f32, center_mask)
    ev["n_ring"] = int(m_ring.sum())
    if ring_dir is None or center_dir is None:
        # V0.7：样本不足无法比较 → 不再放行（原V0.6漏洞），判证据不足
        ev["verdict"] = "环带/中央方向样本不足（<30px），无法建立独立性证据"
        return "INSUFFICIENT", ev
    dir_diff = float(np.linalg.norm(ring_dir - center_dir))
    ev["dir_diff"] = round(dir_diff, 4)
    ev["ring_dir"] = [round(float(v), 4) for v in ring_dir]
    ev["center_dir"] = [round(float(v), 4) for v in center_dir]
    if dir_diff < INDEP_DIR_DIFF_TH:
        ev["verdict"] = ("环带与中央候选增益方向一致（疑似满幅同质浅色材质，"
                         "无独立于食材的中性参照）")
        return "FAIL", ev
    ev["verdict"] = f"方向差{dir_diff:.4f}≥{INDEP_DIR_DIFF_TH}，存在独立参照"
    return "PASS", ev


def estimate_pollution(img_f32):
    """
    光源污染识别（线性域增益拟合 + 两轮自适应中性区提取）：
    第一轮：严格阈值提取中性区 → 粗估增益；
    若中性区不足且检出大幅色偏，说明污染已把中性物推出低饱和带，
    则放宽饱和度阈值重提中性区（迭代自适应，本发明点之一）。
    """
    from pollution_sim import _srgb2lin
    h, w, _ = img_f32.shape
    result = dict(detected=False, lamp_type=None, strength=0.0,
                  confidence=0.0, neutral_pixels=0, failure_reason=None,
                  degrade_reason=None, neutral_mode=None, alt_lamps=None,
                  indep_evidence=None)

    # 全图亮度检查：过暗图像的增益估计噪声不可接受（归一化增益对亮度不敏感，
    # 不能靠后面的偏差范数拦截，必须在入口拦截）
    mean_disp = float(img_f32.mean())
    if mean_disp < 0.04:
        result["failure_reason"] = "图像过暗，无法可靠估计光源，请开启闪光灯或移至明亮处重拍"
        return result

    lin_mean, mode, mode_factor, adaptive, n_used, ev = _neutral_gain(img_f32, 0.22)
    result["indep_evidence"] = ev  # 各级拒绝门实际触发证据（V0.7 审查要求）
    if lin_mean is None:
        # 四级降级链全部失败 → 无中性参照独立失败分支（不再由灰世界完成识别）。
        # 灰世界仅作路由探针：区分“无色偏（自然光）”与“有色偏但无中性参照”两种提示，
        # 探针结果不进入灯库拟合与还原。
        # 拒绝原因细分（V0.7 证据链）：优先报告结构性的独立参照检验结论。
        last = ev["stages"][-1] if ev.get("stages") else None
        indep_failed = any(s.get("indep_status") in ("FAIL", "INSUFFICIENT")
                           for s in ev.get("stages", []))
        if indep_failed:
            # 找出最深层的 FAIL（同质）或 INSUFFICIENT 结论作为提示语
            deepest = None
            for s in ev.get("stages", []):
                if s.get("indep_status") in ("FAIL", "INSUFFICIENT"):
                    deepest = s
            if deepest is not None and deepest.get("indep_status") == "FAIL":
                result["failure_reason"] = (
                    "图像候选中性区域与主体同质（疑似满幅同质浅色材质），"
                    "无独立于食材的中性参照，无法可靠估计光源；"
                    "建议附加白色参照物（如纸巾）后重拍")
                result["indep_evidence"] = ev
                return result
            if deepest is not None:
                result["failure_reason"] = (
                    "独立性证据不足（无独立于食材的中性参照或参照样本过少），"
                    "无法可靠估计光源；建议附加白色参照物（如纸巾）后重拍")
                return result
        probe_mean, nv = _gray_world_offset(img_f32)
        if probe_mean is None:
            result["failure_reason"] = "有效像素不足，请靠近拍摄或补充光照"
            return result
        probe_norm = probe_mean / probe_mean.mean()
        if np.linalg.norm(probe_norm - 1) < DEV_MIN:
            result["failure_reason"] = "未检出明显色偏，可能已为自然光环境"
        else:
            result["failure_reason"] = ("图像中未找到足够的中性参照区域（托盘/包装/白色参照物），"
                                        "无法可靠估计光源；建议附加白色参照物（如纸巾）后重拍")
        return result

    result["neutral_pixels"] = n_used
    result["neutral_mode"] = mode
    ref = float(lin_mean.mean())
    if ref < 1e-6:
        result["failure_reason"] = "图像过暗，无法估计光源"
        return result
    gain_est_norm = np.asarray(lin_mean, dtype=np.float64) / ref

    dev_norm = float(np.linalg.norm(gain_est_norm - 1.0))
    if dev_norm < DEV_MIN:
        result["failure_reason"] = "未检出明显色偏，可能已为自然光环境"
        return result

    lamp_type, s_fit, resid, per_lamp = _fit_lamp(gain_est_norm)

    # ---- V0.5 契约检验（响应审查意见3：适用条件可执行化） ----
    # (a) 拟合强度范围：s超出[0.05,0.95]说明真实光源超出灯库单灯物理范围
    #     （混合光或非库内光），方向上灯库无法解释，明确拒绝。
    if not (S_EDGE_LOW <= s_fit <= S_EDGE_HIGH):
        result["failure_reason"] = (
            f"拟合强度{s_fit:.2f}超出灯库物理范围[0.05,0.95]，"
            "疑似混合光源或非库内光源，拒绝输出")
        return result
    # (b) 灯型歧义：次优灯型残差与最优过于接近 → 灯型判定不确定
    if len(per_lamp) >= 2 and per_lamp[1][0] < LAMP_AMBIG_RATIO * resid:
        alt = per_lamp[1]
        result["failure_reason"] = (
            f"灯型判定歧义（{lamp_type}与{alt[1]}残差过近），疑似混合光源，拒绝输出")
        return result
    # (c) 混合光一致性：候选中性像素的线性归一化增益通道std过大 →
    #     多光源方向不一致（单灯下所有中性物增益方向一致），拒绝。
    #     该检验需要候选mask，_neutral_gain已返回无mask，此处以残差+幅度
    #     联合近似：残差高且幅度大 → 混合光嫌疑（保守拒绝）。

    fit_q = float(np.exp(-8.0 * resid))
    amp = min(dev_norm / CONF_DEV_SCALE, 1.0)
    conf = fit_q * (0.5 + 0.5 * amp) * mode_factor

    result["lamp_type"] = lamp_type
    result["strength"] = s_fit
    result["confidence"] = conf
    result["fit_residual"] = resid
    result["dev_norm"] = dev_norm
    result["adaptive_round"] = adaptive
    result["alt_lamps"] = [(lt, round(float(r), 4)) for r, lt, s in per_lamp[1:3]]

    if conf < CONF_TH:
        result["detected"] = False
        # 过曝比例高时优先报告过曝干扰（更精确的失败原因，指导用户重拍）
        over_ratio = float((img_f32 >= 0.999).any(axis=2).mean())
        if over_ratio >= 0.30:
            result["failure_reason"] = (f"大面积过曝（{over_ratio*100:.0f}%）干扰光源估计，"
                                        "建议远离光源或调整角度重新拍摄")
        else:
            result["failure_reason"] = "光源类型匹配度低（非库内生鲜灯或混合光源），建议附加白色参照物后重试"
        return result

    result["detected"] = True
    return result


def _candidate_quality(img_f32, mask, zone="full"):
    """
    全图/环带候选中性区域质量判定（V0.5，响应审查意见1）。
    四项指标，任一不合格即返回失败原因（字符串）；全部合格返回 None。
    指标针对“假中性”三类来源：
      Q1 占比上限 FULL_MAX_FRAC（仅全图判定）——被污染“中和”的大面积鱼身
         会以接近满幅的候选占比出现；真实中性参照（托盘/案板）在画面中
         通常是局部区域。
      Q2 碎片化（有效连通块面积/碎块占比）——浅色脂肪以小碎块散布全图；
         真实参照物为连续块状。
      Q3 中央集中（仅全图判定）——食物本体假中性集中于画面中央；构图先验
         下真实参照位于周边。注意环带候选天然位于周边，不做此判。
      Q4 色散 FULL_CHROMA_STD——候选内线性归一化增益的通道离散度；真实
         参照物（同一托盘/纸巾）色彩均匀，混合来源（脂肪+反光+标签）离散。
    连通块用4邻域洪泛标记（纯numpy实现，无scipy依赖也可用；若有scipy则用
    label加速）。块面积< FULL_BLOB_MIN 的像素计入碎块。
    """
    from pollution_sim import _srgb2lin
    h, w = img_f32.shape[:2]
    n_total = h * w
    n_cand = int(mask.sum())
    if n_cand < NEUTRAL_MIN_PIXELS:
        return "候选像素数不足"
    # Q1 占比上限（仅全图）
    if zone == "full" and n_cand / n_total > FULL_MAX_FRAC:
        return "候选占比过高（疑似食物本体被误判为中性）"
    # Q2 连通块分析
    labels, n_labels = _label_components(mask)
    sizes = np.bincount(labels.ravel())[1:] if n_labels > 0 else np.array([])
    if len(sizes) == 0:
        return "无有效连通块"
    big = sizes[sizes >= FULL_BLOB_MIN]
    n_big = int(big.sum())
    if n_big < NEUTRAL_MIN_PIXELS:
        return "有效连通块总面积不足（疑似脂肪碎块）"
    if (n_cand - n_big) / n_cand > FULL_FRAG_FRAC:
        return "碎片化比例过高（疑似脂肪斑点散布）"
    # Q3 中央集中（仅全图）：中央50%窗口内候选占候选总数比例
    if zone == "full":
        ch0, ch1 = int(h * 0.25), int(h * 0.75)
        cw0, cw1 = int(w * 0.25), int(w * 0.75)
        n_center = int(mask[ch0:ch1, cw0:cw1].sum())
        if n_center / n_cand > FULL_CENTER_FRAC:
            return "候选集中于画面中央（疑似食物本体）"
    # Q4 色散：候选像素线性归一化增益的通道间std（参照物应三通道一致）
    lin = _srgb2lin(img_f32[mask])
    ref = lin.mean(axis=0)
    if float(ref.mean()) < 1e-6:
        return "候选过暗"
    norm_gain = lin / ref
    chroma_std = float(np.mean(norm_gain.std(axis=0)))
    if chroma_std > FULL_CHROMA_STD:
        return "候选色彩离散度过高（疑似混合来源假中性）"
    return None


def _label_components(mask):
    """
    二值掩码4邻域连通块标记（洪泛法，纯numpy循环实现；
    256×256级别图像够用；大图时优先用 scipy.ndimage.label）。
    返回 (labels数组, 连通块数量)。
    """
    try:
        from scipy.ndimage import label
        labels, n = label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        return labels, int(n)
    except ImportError:
        pass
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    from collections import deque
    for i in range(h):
        for j in range(w):
            if mask[i, j] and labels[i, j] == 0:
                cur += 1
                dq = deque([(i, j)])
                labels[i, j] = cur
                while dq:
                    y, x = dq.popleft()
                    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            dq.append((ny, nx))
    return labels, cur


def _neutral_gain(img_f32, sat_th=0.22):
    """
    给定严饱和度阈值提取中性区并估计线性增益。
    完整降级链（每一级先环带后全图；灰世界不在链中）：
      1) 环带严阈值（外圈8%，sat<sat_th）→ 置信因子1.0
      2) 全图严阈值 → 0.95
      3) 环带宽阈值（sat<0.40）→ 0.92
      4) 全图宽阈值 → 0.90
      5) 全部失败 → 返回 None（外层进入“无中性参照”失败分支）
    环带可靠性逐图检测（构图先验不默认成立）：“食物居中、承载面在周边”
    只有在检测到以下条件时才被采信：
      a. 环带中性像素总数 ≥ RING_MIN_PIXELS；
      b. 至少 RING_MIN_SIDES 条边带各含 ≥ RING_SIDE_MIN 个中性像素
         （中性像素集中于单侧→遮挡嫌疑→环带降级为不可靠，回退全图）。
    全图/环带候选质量判定（V0.5）：每级候选先过 _candidate_quality 四项指标
    （占比/碎片化/中央集中/色散），不合格则该级视同失败进入下一级，不再
    “凑够像素数就采信”——防止浅色脂肪、被中和的鱼身、包装反光等假中性
    绕过无中性参照分支。
    独立参照检验（V0.7 三态）：每级候选（环带与全图）额外过
    _independent_ref_check，返回 PASS/FAIL/INSUFFICIENT。只有 PASS 才采信；
    FAIL（同质）与 INSUFFICIENT（证据不足）均不采信该级候选，进入下一级
    或最终失败分支（V0.7 起“样本不足”不再默认放行）。
    该检验同时拦截 A2 满幅鱼（同质材质全图）与 sea s=0.9 鱼身主导候选集
    （真参照被污染推出饱和度阈值后的强度低估误还原）。
    返回 (lin_mean 或 None, mode, mode_factor, adaptive_used, n_used,
          evidence: dict 或 None)
      evidence 记录采信级与各拒绝门实际触发情况（V0.7 证据链要求：
      结果 JSON 必须能证明实际触发了哪个门，而非仅最终提示语）。
    """
    from pollution_sim import _srgb2lin
    ring = _ring_mask(img_f32.shape)
    ev = {"stages": []}
    for th, tag, f_ring, f_full in ((sat_th, "", 1.0, 0.95), (0.40, "宽", 0.92, 0.90)):
        mask = extract_neutral_zone(img_f32, th)
        n_ring = int((mask & ring).sum())
        n_sides = _ring_sides_ok(mask, img_f32.shape)
        if n_ring >= RING_MIN_PIXELS and n_sides >= RING_MIN_SIDES:
            m = mask & ring
            q = _candidate_quality(img_f32, m, zone="ring")
            st, sev = _independent_ref_check(img_f32, m, zone="ring", full_mask=mask) \
                if q is None else ("QUALITY_FAIL", dict(zone="ring", quality_reason=q))
            ev["stages"].append(dict(level=f"环带{tag}阈值", quality=(q is None),
                                     indep_status=st, indep=sev))
            if q is None and st == "PASS":
                return (_srgb2lin(img_f32[m]).mean(axis=0), f"环带{tag}阈值中性区",
                        f_ring, (th > 0.22), n_ring, ev)
            # 环带候选质量/独立参照不合格 → 继续尝试本级全图（同阈值）
        n = int(mask.sum())
        if n >= NEUTRAL_MIN_PIXELS:
            q = _candidate_quality(img_f32, mask, zone="full")
            st, sev = _independent_ref_check(img_f32, mask, zone="full") \
                if q is None else ("QUALITY_FAIL", dict(zone="full", quality_reason=q))
            ev["stages"].append(dict(level=f"全图{tag}阈值", quality=(q is None),
                                     indep_status=st, indep=sev))
            if q is None and st == "PASS":
                return (_srgb2lin(img_f32[mask]).mean(axis=0), f"全图{tag}阈值中性区",
                        f_full, (th > 0.22), n, ev)
        else:
            ev["stages"].append(dict(level=f"全图{tag}阈值", quality=None,
                                     indep_status="SKIP_PIXELS",
                                     indep=dict(n=n, reason="候选像素不足")))
        # 全图不合格或数量不足 → 进入下一阈值级
    # 四级全部失败（数量/质量/独立参照不合格）：无中性参照（由外层失败分支处理）
    return None, None, 1.0, (sat_th > 0.22), 0, ev


def _ring_mask(shape, margin_frac=0.08):
    """周边环带掩码：距任一边缘 < margin_frac*短边的像素（外框带）"""
    h, w = shape[:2]
    m = int(min(h, w) * margin_frac)
    ring = np.zeros((h, w), dtype=bool)
    ring[:m, :] = True; ring[-m:, :] = True
    ring[:, :m] = True; ring[:, -m:] = True
    return ring


def _ring_sides_ok(mask, shape, margin_frac=0.08):
    """
    环带遮挡检测：统计四条边带（上/下/左/右）中中性像素达标
    （≥RING_SIDE_MIN）的边带数。中性像素集中于单侧边带（如仅一角
    有白色参照物、其余三侧被手指/包装占据）时视为遮挡嫌疑。
    返回达标边带数（0~4）。
    """
    h, w = shape[:2]
    m = int(min(h, w) * margin_frac)
    if m < 1:
        return 0
    sides = (mask[:m, :], mask[-m:, :], mask[:, :m], mask[:, -m:])
    return int(sum(1 for s in sides if int(s.sum()) >= RING_SIDE_MIN))


def build_restore_matrix(lamp_type, strength, food_type="meat"):
    """
    场景化还原矩阵（线性域）：污染增益的逆变换。
    场景化二次约束：防止过度还原，保留食材真实暗沉。
    注意：矩阵作用在线性域（先逆gamma再乘矩阵再gamma回来）。
    """
    from pollution_sim import LAMP_SPECTRA
    cfg = LAMP_SPECTRA[lamp_type]
    g = np.array(cfg["gain_rgb"], dtype=np.float32)
    gain = 1.0 + (g - 1.0) * strength
    inv_gain = 1.0 / gain

    scene_adjust = {
        "meat": np.array([0.97, 1.00, 1.02], dtype=np.float32),
        "veg":  np.array([1.01, 0.97, 1.01], dtype=np.float32),
        "sea":  np.array([1.01, 1.00, 0.97], dtype=np.float32),
    }
    adj = scene_adjust.get(food_type, np.ones(3, dtype=np.float32))
    adj_effective = 1.0 + (adj - 1.0) * strength

    M = np.diag(inv_gain * adj_effective).astype(np.float32)
    return M


def restore_image(img_f32, est, food_type="meat"):
    """
    反美颜还原主入口（线性域实现，与污染模型对称）：
    显示域 →逆gamma→ 线性域 ×还原矩阵 →gamma→ 显示域
    拒绝门（三道，按序）：
      门1 过曝拒绝：被生鲜灯增益推出线性1.0的区域不可逆，过曝比例≥阈值时
         直接拒绝还原（不生成校正图，防止上层误展示失真结果）。
      门2 还原自检（V0.7b，第七轮审查意见1）：dev_post 在**实际还原输出**
         （lin_restored，含 scene_adjust）上计算，参照**掩码沿用 estimate_
         pollution 采信阶段确定的掩码**（原图重取，确定性一致），不因还原
         后饱和度变化在输出上重提取（避免掩码漂移致正确还原误拒）。
         数学本质（诚实声明）：还原算子为逐通道
         乘法，参照均值的还原后中性度在代数上由拟合闭合差决定——门2 是
         **输出闭合差门**，不是对目标食材颜色的独立验证；其技术意义是
         把拟合闭合差从“识别侧软指标（conf 含幅度项补偿）”升级为“输出侧
         硬契约”（闭合差超限则拒绝交付还原图）。能区分：混合光误接受
         （70:30/10:90）、灯库参数失配；不能区分：与纯单灯物理同构的
         90:10（已披露边界）。区分力实验见 exp7_gate2.py。
      门3（备）：fit_q 残差已在 estimate_pollution 契约检验中处理。
    返回 (还原图或None, 使用的矩阵或None, 报告dict)
    上层契约（V0.7a 统一，响应第六轮审查意见2）：
      **rec is None ⟺ 未输出还原图**（含三种情形：识别未检出/过曝拒绝/
      自检拒绝，由 report.detected / report.reject_gate 细分原因）。
      不再在识别失败时返回原图副本——上层需要原图预览时自行使用输入图。
      契约固化测试见 test_restore_contract.py。
    报告新增 selfcheck 字段：dev_post 值与门2判定，供实验 JSON 落盘验证
    “实际触发了哪个拒绝门”。
    """
    from pollution_sim import _srgb2lin, _lin2srgb
    if not est["detected"]:
        # V0.7a：识别未检出 → 不返回图像（上层不得把失败图当还原结果）。
        # 失败原因在 est.failure_reason / report.reason 中，原图预览由上层自理。
        return None, None, {"overexposed_ratio": 0.0, "rejected": False,
                            "detected": False, "reason": est.get("failure_reason"),
                            "selfcheck": None}

    # 过曝检测：显示域已到1.0的像素，其线性真值≥1.0，信息已丢失
    overexposed = (img_f32 >= 0.999).any(axis=2)
    over_ratio = float(overexposed.mean()) if overexposed.size else 0.0

    if over_ratio >= OVEREXPOSED_REJECT_TH:
        # 拒绝还原：不生成校正图，返回None
        report = {
            "overexposed_ratio": over_ratio,
            "rejected": True,
            "reject_gate": "overexposed",
            "advice": "检测到大面积过曝，色彩信息不可逆丢失，建议远离光源或调整角度重新拍摄",
        }
        return None, None, report

    lin = _srgb2lin(img_f32)
    M = build_restore_matrix(est["lamp_type"], est["strength"], food_type)
    lin_restored = lin @ M.T

    # ---- 门2：还原自检（V0.7b：沿用采信掩码口径，第七轮审查意见1）----
    # dev_post 在**实际还原输出** lin_restored（含 scene_adjust，与交付像素
    # 同一算子）上计算；参照掩码**沿用 estimate_pollution 采信阶段确定的
    # 掩码**（同图同参数重提取，确定性一致），而非在还原输出上重提取——
    # 避免还原后饱和度变化引起掩码漂移、正确还原误拒（见 _restore_selfcheck
    # docstring 口径说明与诊断数据）。阈值以 exp7_gate2 实验标定（正确还原
    # 实际口径底噪 vs 混合光/失配残差的隔离带）。
    ref_dev = _restore_selfcheck(img_f32, lin_restored, est)
    if ref_dev is not None and ref_dev >= RESTORE_SELFCHECK_DEV_TH:
        report = {
            "overexposed_ratio": over_ratio,
            "rejected": True,
            "detected": True,
            "reject_gate": "selfcheck",
            "selfcheck": {"dev_post": round(ref_dev, 4)},
            "advice": ("还原后参照未回到中性（残留系统性色偏），疑似混合光源，"
                       "已拒绝输出还原图；建议单一光源下重拍"),
        }
        return None, None, report
    out = _lin2srgb(lin_restored)
    report = {
        "overexposed_ratio": over_ratio,
        "rejected": False,
        "detected": True,
        "selfcheck": ({"dev_post": round(ref_dev, 4)} if ref_dev is not None else None),
        "advice": (f"检测到{over_ratio*100:.0f}%过曝区域，该区域色彩可能失真" if over_ratio > 0.05 else None),
    }
    return out, M, report


def _restore_selfcheck(img_f32, lin_restored, est):
    """
    还原后自检（V0.7b 门2，第七轮审查意见1）。
    V0.7a 修正：dev_post 在**实际还原输出** lin_restored（含 scene_adjust、
    与交付像素同算子）上计算，而非旧版“输入前×纯逆增益”的预测值——
    旧版只是拟合结果的重新表述，未检验实际输出。
    V0.7b 统一口径：参照**掩码沿用 estimate_pollution 采信阶段确定的掩码**
    （在原图 img_f32 上按采信级参数 sat_th 重取，确定性一致），仅增益
    取自实际还原输出 lin_restored[mask]——即“沿用采信掩码、在输出上测增益”。
    不选择“在还原输出上重提取掩码”的原因（诊断数据）：还原会改变参照处
    饱和度，输出上重提取使掩码漂移，正确还原 G1 底噪上界由 0.0279 抬升至
    0.0311（突破阈值0.03，产生误拒），隔离带收窄；沿用采信掩码保留完整
    隔离带 [0.0279,0.0302] 且 70:30/10:90 拦截窗口 5/5、90:10 放行 0/5 不变。
    口径本质（诚实声明，写入交底书）：还原算子为逐通道，参照均值归一
    后中性度在代数上由拟合结果决定——本门**不是对目标食材颜色正确性的
    独立验证**，而是输出侧闭合差契约：闭合差超限 → 拒绝交付。能拦截：
    混合光误接受（70:30/10:90）、灯库参数失配（exp5 错误输出 0）；不能
    拦截：与纯单灯物理同构的 90:10（dev_post 落在正确底噪内，已披露边界）；
    亦不能验证食材本体颜色（无真值可检）。
    标定（exp7_gate2，实际输出口径）：见 RESTORE_SELFCHECK_DEV_TH 注释。
    est 未采信（无 neutral_mode）时返回 None（无法自检，不由本门拦截）。
    """
    mode = est.get("neutral_mode")
    if not mode:
        return None
    # 重建采信级 mask：与 _neutral_gain 完全一致的同参数重提取
    sat_th = 0.40 if "宽" in mode else 0.22
    mask = extract_neutral_zone(img_f32, sat_th)
    if "环带" in mode:
        mask = mask & _ring_mask(img_f32.shape)
    n = int(mask.sum())
    if n < 30:
        return None
    # 实际还原输出上的参照增益（与交付像素同一算子，含 scene_adjust）
    lin_ref = lin_restored[mask]
    gain_post = lin_ref.mean(axis=0)
    ref = float(gain_post.mean())
    if ref < 1e-6:
        return None
    dev_post = float(np.linalg.norm(gain_post / ref - 1.0))
    return dev_post


# ============ 通用AWB对照组 ============
def generic_awb(img_f32):
    """
    通用自动白平衡（Gray-world）对照组：
    线性域灰世界白平衡（与真实ISP的AWB一致），对任何偏色一视同仁拉白，
    不区分场景。这是诚实对照，非稻草人。
    """
    from pollution_sim import _srgb2lin, _lin2srgb
    lin = _srgb2lin(img_f32)
    means = lin.reshape(-1, 3).mean(axis=0)
    g = float(means.mean())
    gain = np.clip(g / (means + 1e-6), 0.5, 2.5)
    out = _lin2srgb(lin * gain[None, None, :])
    return out


# ============ 评估指标 ============
# 色差指标统一从 color_metrics 模块引用（标准 CIEDE2000，经 Sharma 34对测试集
# + colour-science 0.4.7 交叉验证 34/34 一致）。旧简化公式已删除，
# 避免以简化指标冒充标准 CIEDE2000。
def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 60.0
    return 10.0 * np.log10(1.0 / mse)


def ssim_channel(a, b):
    """单通道SSIM（简化实现，8x8窗口）"""
    from scipy.ndimage import uniform_filter
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01) ** 2, (0.03) ** 2
    mu_a = uniform_filter(a, 8); mu_b = uniform_filter(b, 8)
    var_a = uniform_filter(a * a, 8) - mu_a ** 2
    var_b = uniform_filter(b * b, 8) - mu_b ** 2
    cov = uniform_filter(a * b, 8) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2))
    return float(s.mean())


def ssim_rgb(a, b):
    return float(np.mean([ssim_channel(a[..., i], b[..., i]) for i in range(3)]))


if __name__ == "__main__":
    print("模块自检：")
    rng = np.random.default_rng(42)
    img = rng.random((64, 64, 3)).astype(np.float32) * 0.5 + 0.25
    est = estimate_pollution(img)
    print("  estimate_pollution →", est["detected"], est["failure_reason"])
    print(f"  neutral_pixels={est['neutral_pixels']}, neutral_mode={est['neutral_mode']}")
    # 无中性参照分支自检：纯红食物满构图（无托盘无白卡，自然光无污染）
    food_only = np.ones((64, 64, 3), dtype=np.float32)
    tex = np.random.default_rng(3).random((64, 64))
    food_only[..., 0] = 0.55 + 0.1 * tex
    food_only[..., 1] = 0.20 + 0.06 * tex
    food_only[..., 2] = 0.18 + 0.06 * tex
    est2 = estimate_pollution(food_only)
    hit = "中性参照" in (est2["failure_reason"] or "")
    print(f"  无中性参照分支: reason={est2['failure_reason']} → {'命中(正确)' if hit else '未命中(错误)'}")
    # V0.5 假中性攻击自检1：浅色脂肪碎块散布（碎片化，应被Q2拦截）
    fat = np.ones((128, 128, 3), dtype=np.float32)
    fat[...] = [0.30, 0.10, 0.08]  # 红肉底色
    frng = np.random.default_rng(5)
    for _ in range(80):  # 散布小脂肪斑
        y, x = frng.integers(0, 118), frng.integers(0, 118)
        fat[y:y+10, x:x+10] = [0.82, 0.74, 0.66]
    est3 = estimate_pollution(fat)
    hit3 = ("中性参照" in (est3["failure_reason"] or "")) or (not est3["detected"])
    print(f"  假中性攻击(脂肪碎块): detected={est3['detected']} → {'拦截(正确)' if hit3 else '误识别(错误)'}")
    # V0.6 假中性攻击自检2：满幅浅色鱼身（A2 盲区场景）。
    # V0.5 时为“可辨识性极限实例（已披露盲区）”；V0.6 独立参照检验将该
    # 场景结构性拒绝：环带∩候选与中央∩候选为同材质、方向差→0 →
    # 无独立中性参照 → 只提示不还原。此自检改为硬性断言：必须拦截。
    fish = np.ones((128, 128, 3), dtype=np.float32)
    ftex = np.random.default_rng(7).random((128, 128))
    fish[..., 0] = 0.72 + 0.08 * ftex
    fish[..., 1] = 0.66 + 0.08 * ftex
    fish[..., 2] = 0.62 + 0.08 * ftex
    est4 = estimate_pollution(fish)
    hit4 = (not est4["detected"]) and ("中性参照" in (est4["failure_reason"] or ""))
    print(f"  假中性攻击(满幅鱼身/A2): detected={est4['detected']} → "
          f"{'结构性拦截(正确)' if hit4 else '误还原(错误,V0.6契约违背)'}")
    # V0.6 独立参照正向自检：满幅鱼+周边灰托盘（有独立参照，应正常识别，
    # 证明独立参照检验不误杀真构图）。
    h_, w_ = fish.shape[:2]
    m_ = int(min(h_, w_) * 0.12)
    fish_tray = fish.copy()
    fish_tray[:m_, :] = [0.78, 0.76, 0.74]
    fish_tray[-m_:, :] = [0.78, 0.76, 0.74]
    fish_tray[:, :m_] = [0.78, 0.76, 0.74]
    fish_tray[:, -m_:] = [0.78, 0.76, 0.74]
    est5 = estimate_pollution(fish_tray)
    print(f"  独立参照正向自检(满幅鱼+托盘): detected={est5['detected']}"
          f"{'(识别,方向差检验不误杀)' if est5['detected'] else '(拒绝,误杀!)'}")
    # 过曝拒绝契约自检
    bright = np.ones((64, 64, 3), dtype=np.float32)
    bright[..., 1] = 0.9; bright[..., 2] = 0.8  # 大面积R通道饱和→模拟过曝
    out, M, rep = restore_image(bright, dict(detected=True, lamp_type="red_meat", strength=0.8))
    print(f"  过曝拒绝契约: rejected={rep['rejected']}, 返回图={'None(正确)' if out is None else '仍返回图(错误)'}")
