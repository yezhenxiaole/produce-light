# -*- coding: utf-8 -*-
"""
calibration: 真实灯库标定模块（V0.5，响应第二轮审查意见4）
================================================================
本模块定义【真实采集标定】的数据结构、增益计算公式、重复测量误差统计，
并提供一个仿真管道自检（用 pollution_sim 的合成“真值”代替真实采集数据，
验证标定公式与误差统计代码正确性）。

== 标定的物理设定 ==
真实标定输入（采集现场获得）：
  1. 参照物：标准灰卡（18%灰）/白卡（如 X-Rite ColorChecker 的灰白patch），
     反射率光谱在可见光内近似平坦（中性），sRGB显示值三通道近似相等；
  2. 两张图像：同一参照物、同一相机、同一曝光与白平衡设置：
     - img_ref：参照光源（D65标准光源或日光，已知无污染）下拍摄；
     - img_lamp：被标定生鲜灯下拍摄（灯-参照物几何与参照拍摄一致）；
  3. 元数据：相机型号/RAW管线版本、曝光参数、白平衡参数、灯距、灯角度、
     采集日期、操作者、环境光遮蔽情况。

== 增益计算公式（与识别器的乘性模型一致）==
设参照物在参照光下的线性域均值 m_ref∈R^3，灯下线性域均值 m_lamp∈R^3
（线性域=RAW demosaic后未经色彩矩阵的RGB，或已逆sRGB的显示RGB——
两种口径的增益不同，标定记录须注明口径；推荐RAW）。
则该灯在该相机上的满强度增益：
    g = m_lamp / m_ref        （三通道逐通道相除）
强度参数化：G(s) = 1 + (g - 1) * s，s∈[0,1]为灯驱动占空比/亮度档。
可选：多档位采集（s=0.25/0.5/0.75/1.0）验证线性度（H1检验），
若非线性则记录每档实测增益（查表模型，放弃线性参数化）。

== 重复测量误差 ==
同一配置重复N次（≥5），计算 g 的每通道相对标准差：
    rel_std_c = std(g[:, c]) / mean(g[:, c])
合格判据（工程默认）：每通道 rel_std ≤ 5%，否则标定无效需重采。

== 跨设备差异 ==
同一灯具在每款手机上分别标定（设备专属灯库）；库条目含设备标识。
未标定设备的处理：拒绝识别（适用条件C4），不做跨设备迁移默认。

== 数据结构（lamp library entry）==
{
  "lamp_id": "red_meat_brandX",
  "device_id": "phone_model_A",
  "calib_date": "2026-10-12",
  "calib_operator": "...",
  "domain": "raw" | "srgb_inverse",
  "gain_rgb": [g_r, g_g, g_b],
  "gain_rel_std": [r_r, r_g, r_b],
  "n_repeats": 5,
  "levels": {"0.25": [...], "0.5": [...], ...},   # 可选多档
  "meta": {...exposure/wb/distance/angle...}
}
"""
import numpy as np


def compute_gain(m_ref, m_lamp):
    """
    标定增益：两线性域均值逐通道相除。
    m_ref, m_lamp: shape (N,3) 的重复测量（每行一次测量）或 (3,)。
    返回 (gain(3,), rel_std(3,))——rel_std为跨重复测量的相对标准差。
    """
    m_ref = np.atleast_2d(np.asarray(m_ref, dtype=np.float64))
    m_lamp = np.atleast_2d(np.asarray(m_lamp, dtype=np.float64))
    if m_ref.shape != m_lamp.shape:
        raise ValueError("m_ref 与 m_lamp 形状不一致（重复测量次数须相同）")
    if m_ref.shape[0] < 1:
        raise ValueError("至少一次测量")
    gain = m_lamp.mean(axis=0) / m_ref.mean(axis=0)
    # 误差传播：相对std按增益比的组合传播（一阶近似）
    rel_std_ref = m_ref.std(axis=0) / m_ref.mean(axis=0)
    rel_std_lamp = m_lamp.std(axis=0) / m_lamp.mean(axis=0)
    rel_std = np.sqrt(rel_std_lamp ** 2 + rel_std_ref ** 2)
    return gain, rel_std


def validate_gain(gain, rel_std, rel_std_th=0.05):
    """标定合格判定：增益为正且每通道重复测量相对标准差≤阈值。"""
    ok_pos = bool(np.all(gain > 0))
    ok_std = bool(np.all(rel_std <= rel_std_th))
    return dict(gain_positive=ok_pos, rel_std_ok=ok_std,
                valid=ok_pos and ok_std,
                reason=None if (ok_pos and ok_std)
                else ("增益非正值" if not ok_pos else "重复测量离散度超限"))


def fit_strength_linearity(levels):
    """
    H1检验：多档位增益的线性度拟合。
    levels: {s: g(3,)}——档位→实测满强度归一增益（已减1、除以满档）。
    拟合 G(s) = 1 + (g_full - 1)*s 模型下各档偏差，返回最大相对偏差。
    偏差>10% → 建议查表模型（放弃线性参数化）。
    """
    ss = sorted(levels.keys())
    if len(ss) < 3:
        return dict(n_levels=len(ss), max_dev_pct=None,
                    advice="档位数<3，无法检验线性度")
    g_full = levels[max(ss)]
    devs = []
    for s in ss:
        pred = 1.0 + (g_full - 1.0) * s
        meas = levels[s]
        devs.append(float(np.max(np.abs(meas - pred) / np.maximum(pred, 1e-6))))
    md = float(np.max(devs))
    return dict(n_levels=len(ss), max_dev_pct=round(md * 100, 2),
                advice="线性模型可用" if md <= 0.10 else "偏差>10%，建议查表模型")


def make_library_entry(lamp_id, device_id, calib_date, m_ref, m_lamp,
                       domain="raw", operator=None, meta=None):
    """标定入口：从重复测量数据生成灯库条目（含合格判定）。"""
    gain, rel_std = compute_gain(m_ref, m_lamp)
    val = validate_gain(gain, rel_std)
    return dict(lamp_id=lamp_id, device_id=device_id, calib_date=calib_date,
                domain=domain, operator=operator,
                gain_rgb=[round(float(x), 4) for x in gain],
                gain_rel_std=[round(float(x), 4) for x in rel_std],
                n_repeats=int(np.atleast_2d(m_lamp).shape[0]),
                valid=val["valid"], valid_detail=val,
                meta=meta or {})


# ============ 仿真管道自检 ============
def _selftest():
    """
    用污染仿真器作为“真值发生器”验证标定公式：
      合成灰卡在D65下的线性值 + 已知增益 → 灯下线性值（加测量噪声），
      标定模块应恢复出增益（误差<1%）且误差统计正确。
    """
    from pollution_sim import LAMP_SPECTRA, _srgb2lin
    rng = np.random.default_rng(42)
    print("标定模块仿真自检（合成真值+测量噪声）：")
    all_ok = True
    for lt, cfg in LAMP_SPECTRA.items():
        g_true = np.array(cfg["gain_rgb"], dtype=np.float64)
        # 灰卡D65显示值0.75 → 线性真值
        card_disp = np.array([0.75, 0.75, 0.75])
        lin_ref = _srgb2lin(card_disp.astype(np.float32)).astype(np.float64)
        n = 5  # 重复测量
        noise = 0.005
        m_ref = lin_ref[None, :] * (1 + rng.normal(0, noise, (n, 3)))
        m_lamp = (lin_ref * g_true)[None, :] * (1 + rng.normal(0, noise, (n, 3)))
        entry = make_library_entry(f"{lt}_test", "sim_device", "2026-10-12",
                                   m_ref, m_lamp, domain="srgb_inverse",
                                   operator="selftest")
        err = np.abs(np.array(entry["gain_rgb"]) - g_true)
        ok = bool(np.all(err < 0.01)) and entry["valid"]
        all_ok &= ok
        print(f"  {lt}: 标定增益={entry['gain_rgb']} 真值={list(g_true)} "
              f"误差={np.round(err,4)} rel_std={entry['gain_rel_std']} "
              f"{'PASS' if ok else 'FAIL'}")
    # 线性度检验自检：合成多档
    levels = {}
    g_full = np.array([1.5, 0.95, 0.75])
    for s in (0.25, 0.5, 0.75, 1.0):
        levels[s] = 1.0 + (g_full - 1.0) * s
    lin_check = fit_strength_linearity(levels)
    print(f"  线性度检验（无噪声合成）: {lin_check}")
    # 非线性案例
    levels_nl = dict(levels)
    levels_nl[0.5] = levels_nl[0.5] * 1.15  # 人为15%偏离
    nl_check = fit_strength_linearity(levels_nl)
    print(f"  非线性检测（15%偏离注入）: {nl_check}")
    print(f"总体: {'PASS' if all_ok and lin_check['max_dev_pct'] == 0 else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    _selftest()
