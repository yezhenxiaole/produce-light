# 仿真工程证据链说明（供代理机构/审查复核）

**当前版本：V0.7（2026-10-12）。本目录（`produce-light/`）为唯一正式仿真源码。**
**旧工作目录 `.temp/sim/` 已归档（V0.1~V0.4 历史版本，含 20+ 临时诊断脚本，
ARCHIVED_NOT_AUTHORITATIVE，不再维护），所有正式实验自 V0.5 起以本目录为准。
复现请勿运行旧目录。**

## 版本与元数据约定

- 源码、结果 JSON、交底书/调查报告使用同一版本号（当前 V0.7）；
- 每个结果 JSON 顶层含 `meta` 字段：源码版本、生成时间、Python 版本、git 提交号、
  参数摘要（见各 results_*.json 的 `meta` 键）；
- **同一版本的全部结果 JSON 由同一次冻结重跑生成，meta.git_commit 一致**
  （V0.7 起实验纪律：先提交冻结源码，再统一重跑，结果与提交号绑定）；
- 重跑任一实验前先 `git log --oneline -1` 确认版本，实验写入的 meta 会自动记录。

## 目录与用途

| 文件 | 用途 |
|---|---|
| `color_metrics.py` | 标准 CIEDE2000 实现 + rgb_to_lab；内嵌 Sharma 2005 官方 34 对测试数据 |
| `cross_check.py` | 与 colour-science 0.4.7（pip 权威实现）交叉验证；自动打印修正后期望值 |
| `selftest_output.txt` | `color_metrics.py` 自验证的落盘输出（34/34 通过） |
| `crosscheck_output.txt` | `cross_check.py` 的落盘输出（与 colour-science 一致性） |
| `synth_images.py` | 程序化合成三食材测试图（含托盘径向亮度渐变、噪声、纹理） |
| `pollution_sim.py` | 合成污染模拟器（线性域增益+clip；模型为设定参数，非真实光谱积分） |
| `restore_engine.py` | 被测算法（环带提取/候选质量判定/独立参照检验/灯库拟合/还原/失败分支/过曝拒绝） |
| `calibration.py` | 灯库标定流程软件模块（双拍增益计算+重复测量统计+线性度检验+灯库条目生成；
  含合成数据自检。**注意：当前仅完成"标定流程代码自洽"，未做真实灯具/真实手机/真实灰卡标定**） |
| `exp1_2.py` | 实验一/二：三场景还原质量（CIEDE2000/PSNR/SSIM，5种子±std） |
| `exp3.py` | 实验三：四灯型识别144组 + 五类失败分支 + 误报率 |
| `exp4.py` | 实验四：环带消融（全图 vs 降级链）+ 降级链回退行为（含单侧白卡遮挡检测） |
| `exp5.py` | 实验五：灯库参数失配鲁棒性（±10%/±20%/旋转/综合） |
| `exp6.py` | 实验六：假中性攻击（A1~A5，应全部拒绝）+ 构图正向验证（B1~B5）+ 混合光可辨识性边界（C组） |
| `calib_overexpose.py` | 过曝阈值标定实验（退化曲线，单场景） |
| `calib_overexpose_multi.py` | 过曝阈值多场景标定（9 食材×灯型组合退化曲线汇总） |
| `results_*.json` | 各实验结果（UTF-8，含 meta 元数据） |
| `exp1_2_comparison.png` | 实验一/二 3×4 对比图 |

## 运行命令（Windows PowerShell，需 UTF-8 环境）

```powershell
cd E:\明宇博士资料\专利\生鲜灯\produce-light   # 唯一运行入口（在本目录下运行）
$env:PYTHONIOENCODING='utf-8'; python -X utf8 color_metrics.py         # CIEDE2000自验证
$env:PYTHONIOENCODING='utf-8'; python -X utf8 cross_check.py           # 权威库交叉验证（需colour-science）
$env:PYTHONIOENCODING='utf-8'; python -X utf8 restore_engine.py        # 算法模块自检（含独立参照/过曝契约）
$env:PYTHONIOENCODING='utf-8'; python -X utf8 exp1_2.py                # 实验一/二
$env:PYTHONIOENCODING='utf-8'; python -X utf8 exp3.py                  # 实验三
$env:PYTHONIOENCODING='utf-8'; python -X utf8 exp4.py                  # 实验四
$env:PYTHONIOENCODING='utf-8'; python -X utf8 exp5.py                  # 实验五
$env:PYTHONIOENCODING='utf-8'; python -X utf8 exp6.py                  # 实验六（攻击/构图/混合光）
$env:PYTHONIOENCODING='utf-8'; python -X utf8 calibration.py           # 标定模块合成自检
$env:PYTHONIOENCODING='utf-8'; python -X utf8 calib_overexpose_multi.py # 过曝阈值多场景标定
```

依赖：numpy、scipy、opencv-python、colour-science≥0.4.7（仅交叉验证需要）。
**必须在 `produce-light/` 目录内运行**（脚本同目录相对导入）。

## 证据口径声明

1. 全部实验为**合成数据仿真**：污染由 `pollution_sim.py` 的线性域增益模型生成，灯库参数
   与生成参数同源（除 exp5 人为失配）；基准结果反映模型内一致性，不构成真实灯具/手机
   下的效果承诺；
2. CIEDE2000 实现经双重验证：Sharma 34 对官方测试集 + colour-science 0.4.7 交叉一致
   （34/34，最大偏差<1e-4）；
3. 所有 JSON/日志均为 UTF-8 编码；
4. 混合光边界（V0.7 口径）：50:50 附近由识别门拒绝；70:30、10:90 误接受后由
   **还原自检门2**（还原后参照中性回归检验，dev_post≥0.03）拦截——不再输出
   错误还原图；90:10 与纯单灯物理同构（增益向量差异约 4.7%，单张 RGB 不可分），
   放行为已披露的可辨识性边界，误接受还原 dE 量化见 `results_exp6.json` 的
   `mixed.ratio_scan_red_blue`（每样本含 reject_gate/dev_post 证据字段）；
5. 质量门参数（`FULL_*`、`INDEP_*`、`RING_*`、`RESTORE_SELFCHECK_DEV_TH` 系列）
   为合成数据经验值，真实数据上的误拒率/误采信率统计属真实采集阶段任务；
6. 独立参照检验证据（V0.7）：A1~A4 攻击场景的实际拦截门为方向差检验
   （每样本落盘 dir_diff/n_center/reject_stage）；A2b 消融实验证明该检验的
   必要性——关闭后 A2 满幅鱼误接受 5/5（误还原 dE=7.88），开启后拒绝 5/5；
7. 过曝阈值 0.30 为合成数据退化曲线的经验工程分界（非普适物理阈值），
   真实数据需重新标定（见 `calib_overexpose_multi.py` 输出与交底书声明）。
