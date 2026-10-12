# -*- coding: utf-8 -*-
"""交叉验证：我的 ciede2000_std vs colour-science 权威实现 + 修正测试集期望值"""
import numpy as np
import colour
from color_metrics import ciede2000_std, SHARMA_TEST

print("=== 用 colour-science 权威实现核对我手抄的 34 对期望值 ===")
mine_vs_cs = []
for i, (l1, l2, exp) in enumerate(SHARMA_TEST, 1):
    got_cs = float(colour.delta_E(l1, l2, method="CIE 2000"))
    mine = float(ciede2000_std(np.array([l1]), np.array([l2]))[0])
    agree = abs(mine - got_cs) < 1e-4
    exp_ok = abs(got_cs - exp) < 1e-3
    if not agree or not exp_ok:
        print(f"对{i}: mine={mine:.4f} colour-science={got_cs:.4f} 我抄的期望={exp:.4f}")
    mine_vs_cs.append((mine, got_cs, exp))

n_agree = sum(1 for m, c, e in mine_vs_cs if abs(m - c) < 1e-4)
n_exp_ok = sum(1 for m, c, e in mine_vs_cs if abs(c - e) < 1e-3)
print(f"\n我的实现 vs colour-science 一致: {n_agree}/34")
print(f"colour-science vs 我抄的期望值 一致: {n_exp_ok}/34")

print("\n=== 结论 ===")
if n_agree == 34:
    print("我的实现与权威实现完全一致 → 手抄期望值有误，用 colour-science 输出修正测试集")
    print("修正后的期望值（可直接替换 SHARMA_TEST）:")
    for i, (l1, l2, exp) in enumerate(SHARMA_TEST, 1):
        got_cs = float(colour.delta_E(l1, l2, method="CIE 2000"))
        print(f"    (({l1[0]:.4f}, {l1[1]:.4f}, {l1[2]:.4f}), ({l2[0]:.4f}, {l2[1]:.4f}, {l2[2]:.4f}), {got_cs:.4f}),")
else:
    print("我的实现与权威实现有分歧 → 实现有 bug，需修正")
