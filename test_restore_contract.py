# -*- coding: utf-8 -*-
"""
test_restore_contract: restore_image 输出契约固化测试（V0.7b，第七轮审查意见2/4）
契约：** rec is None ⟺ 未输出还原图 **
  三种未输出情形（由 report 细分）：
    1. 识别未检出（report.detected=False, report.reason=失败原因）；
    2. 门1 过曝拒绝（report.reject_gate="overexposed", rejected=True）；
    3. 门2 自检拒绝（report.reject_gate="selfcheck", rejected=True）。
  rec is not None ⟺ 已交付还原图（rejected=False 且 detected=True）。
  识别失败时**不得**返回原图副本——上层需要预览时自行用输入图。
  用例4（V0.7b 强断言）：70:30/10:90 混合光必须被门2 selfcheck 拦截且
  dev_post>=0.03（不允许识别拒/放行并存）；90:10 如实记录为不可分边界。
  输出 UTF-8（第七轮审查意见4）：Windows 控制台 GBK 不再报错。
运行：python test_restore_contract.py（全部用例断言，exit code 0=PASS）
结果写 results_contract.json（供证据链落盘）。
"""
import numpy as np
from restore_engine import estimate_pollution, restore_image, _ring_mask
from restore_engine import extract_neutral_zone
from restore_engine import RESTORE_SELFCHECK_DEV_TH
from pollution_sim import _srgb2lin, _lin2srgb, LAMP_SPECTRA
from synth_images import make_meat_image
from exp6 import _with_tray, _meat_food, _pollute_mix
import json
import sys


# Windows 控制台 UTF-8 输出（第七轮审查意见4）：避免 GBK 编码报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _pollute(img, lamp, s):
    g = np.array(LAMP_SPECTRA[lamp]["gain_rgb"], dtype=np.float32)
    lin = _srgb2lin(img)
    return np.clip(_lin2srgb(lin * (1.0 + (g - 1.0) * s)[None, None, :]), 0, 1).astype(np.float32)


def _ring_area(shape=(256, 256), margin_frac=0.08):
    """环带掩码面积（供问题3数值核验：256×256、外圈8% → 65536-216²=18880）"""
    h, w = shape
    m = int(min(h, w) * margin_frac)
    ring = np.zeros((h, w), dtype=bool)
    ring[:m, :] = True; ring[-m:, :] = True
    ring[:, :m] = True; ring[:, -m:] = True
    return int(ring.sum())


def main():
    cases = []

    # 用例1：识别成功+门2通过 → rec 非 None、rejected=False、detected=True
    img = _pollute(make_meat_image(seed=1), "red_meat", 0.6)
    est = estimate_pollution(img)
    rec, M, rep = restore_image(img, est, food_type="meat")
    ok = (est["detected"] and rec is not None and not rep["rejected"]
          and rep.get("detected", False))
    cases.append(dict(case="1_正常输出", passed=bool(ok),
                      detected=est["detected"], rec_none=rec is None,
                      rejected=rep.get("rejected"), rep_detected=rep.get("detected")))

    # 用例2：识别未检出（自然光本色图）→ rec 必须为 None（不得返回原图）
    nat = make_meat_image(seed=2)
    est2 = estimate_pollution(nat)
    rec2, _, rep2 = restore_image(nat, est2, food_type="meat")
    ok2 = (not est2["detected"] and rec2 is None and not rep2.get("rejected", False)
           and rep2.get("detected") is False and rep2.get("reason") is not None)
    # 补充断言：返回值与输入图不是同一数组内容（防"原图副本冒充还原结果"）
    not_passthrough = True
    if rec2 is not None:
        not_passthrough = not np.array_equal(rec2, nat)
    cases.append(dict(case="2_识别未检出→rec必须None", passed=bool(ok2 and not_passthrough),
                      detected=est2["detected"], rec_none=rec2 is None,
                      reason=rep2.get("reason")))

    # 用例3：门1过曝拒绝 → rec=None、reject_gate="overexposed"
    burn = _pollute(make_meat_image(seed=3), "red_meat", 0.6)
    burn = np.clip(burn.astype(np.float32) * 1.8, 0, 1)  # 大面积推过曝
    est3 = estimate_pollution(burn)
    rec3, _, rep3 = restore_image(burn, est3, food_type="meat")
    if est3["detected"]:
        ok3 = (rec3 is None and rep3["rejected"] and rep3["reject_gate"] == "overexposed")
    else:
        # 识别被过曝干扰拒绝（conf不足）→ 同样 rec 必须 None
        ok3 = (rec3 is None)
    cases.append(dict(case="3_过曝拒绝→rec必须None", passed=bool(ok3),
                      detected=est3["detected"], rec_none=rec3 is None,
                      reject_gate=rep3.get("reject_gate")))

    # 用例4（V0.7b 强断言）：门2 自检拒绝（70:30 / 10:90 混合光）
    # 构造：exp6 _with_tray 图源 + 红蓝混合（识别端物理同构误接受窗口，
    # 见 exp6 run_mixed 比例扫描）。第七轮审查意见2：固定仿真条件下强断言
    # 门2 必须触发，不允许三种结果并存（门2拒/识别拒/输出）。
    def _gate_assert(mix_a, expect_selfcheck, note):
        gt = _with_tray(_meat_food((256, 256), 4), ("all",))
        img4 = _pollute_mix(gt, "red_meat", "blue_sea", 0.6, mix_a)
        est4 = estimate_pollution(img4)
        rec4, _, rep4 = restore_image(img4, est4, food_type="meat")
        sc = rep4.get("selfcheck") or {}
        dev = sc.get("dev_post")
        if expect_selfcheck:
            # 强断言：必须识别接受且门2 拒绝（reject_gate=="selfcheck"、dev_post>=0.03）
            ok = (est4["detected"] and rec4 is None
                  and rep4.get("rejected") and rep4.get("reject_gate") == "selfcheck"
                  and dev is not None and dev >= RESTORE_SELFCHECK_DEV_TH)
        else:
            # 90:10 近单灯：识别接受但门2 不拦（不可分边界样本，dev 落在底噪内）
            # 或识别拒绝——均如实记录为边界，不做放行/拦截强断言
            ok = True
        return ok, dict(note=note, est_detected=bool(est4["detected"]),
                        rec_none=rec4 is None, rejected=rep4.get("rejected"),
                        reject_gate=rep4.get("reject_gate"), dev_post=dev)

    ok_7030, d7030 = _gate_assert(0.7, True, "70:30 混合光，门2 必须拦截")
    ok_1090, d1090 = _gate_assert(0.1, True, "10:90 混合光，门2 必须拦截")
    ok_9010, d9010 = _gate_assert(0.9, False, "90:10 近单灯，不可分边界，如实记录")
    cases.append(dict(case="4a_70:30强断言门2拒", passed=bool(ok_7030),
                      threshold=RESTORE_SELFCHECK_DEV_TH, **d7030))
    cases.append(dict(case="4b_10:90强断言门2拒", passed=bool(ok_1090),
                      threshold=RESTORE_SELFCHECK_DEV_TH, **d1090))
    cases.append(dict(case="4c_90:10边界放行", passed=bool(ok_9010),
                      threshold=RESTORE_SELFCHECK_DEV_TH, **d9010))

    # 用例5：环带掩码面积数值核验（第六轮审查意见3：18880 是否可能）
    area = _ring_area()
    ok5 = (area == 18880)
    cases.append(dict(case="5_环带面积=18880核验", passed=bool(ok5),
                      ring_area=area, formula="256²−216²=65536−46656=18880"))

    # 用例6：A2 满幅鱼环带∩候选面积核验（自然光下全像素过中性筛选）
    tex = np.random.default_rng(1).random((256, 256))
    fish = np.ones((256, 256, 3), dtype=np.float32)
    fish[..., 0] = 0.72 + 0.08 * tex
    fish[..., 1] = 0.66 + 0.08 * tex
    fish[..., 2] = 0.62 + 0.08 * tex
    ring = _ring_mask(fish.shape)
    cand = extract_neutral_zone(fish, 0.22)
    n_ring_cand = int((ring & cand).sum())
    ok6 = (n_ring_cand == 18880)  # 满幅浅色鱼：环带∩候选=环带全集
    cases.append(dict(case="6_A2环带∩候选面积核验", passed=bool(ok6),
                      n_ring_cand=n_ring_cand, expected=18880))

    all_pass = all(c["passed"] for c in cases)
    for c in cases:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['case']}: " +
              ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("case", "passed")))
    print(f"契约测试总体: {'PASS' if all_pass else 'FAIL'}")

    from exp_meta import save_results
    save_results("results_contract.json",
                 dict(contract="rec is None ⟺ 未输出还原图", cases=cases,
                      all_pass=all_pass),
                 params=dict(gates=["detect_fail", "overexposed", "selfcheck"],
                             note="restore_image 输出契约固化测试（第六轮审查意见2/3）"))
    return all_pass


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
