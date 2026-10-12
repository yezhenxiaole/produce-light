# 仿真工程证据链说明（供代理机构/审查复核）

## 目录与用途

| 文件 | 用途 |
|---|---|
| `color_metrics.py` | 标准 CIEDE2000 实现 + rgb_to_lab；内嵌 Sharma 2005 官方 34 对测试数据 |
| `cross_check.py` | 与 colour-science 0.4.7（pip 权威实现）交叉验证；自动打印修正后期望值 |
| `selftest_output.txt` | `color_metrics.py` 自验证的落盘输出（34/34 通过） |
| `crosscheck_output.txt` | `cross_check.py` 的落盘输出（与 colour-science 一致性） |
| `synth_images.py` | 程序化合成三食材测试图（含托盘径向亮度渐变、噪声、纹理） |
| `pollution_sim.py` | 合成污染模拟器（线性域增益+clip；模型为设定参数，非真实光谱积分） |
| `restore_engine.py` | 被测算法（环带提取/灯库拟合/还原/失败分支/过曝拒绝） |
| `exp1_2.py` | 实验一/二：三场景还原质量（CIEDE2000/PSNR/SSIM，5种子±std） |
| `exp3.py` | 实验三：四灯型识别144组 + 五类失败分支 + 误报率 |
| `exp4.py` | 实验四：环带消融（全图 vs 降级链）+ 降级链回退行为（含单侧白卡遮挡检测） |
| `exp5.py` | 实验五：灯库参数失配鲁棒性（±10%/±20%/旋转/综合） |
| `calib_overexpose.py` | 过曝阈值标定实验（退化曲线） |
| `results_exp*.json` | 各实验结果（UTF-8） |
| `exp1_2_comparison.png` | 实验一/二 3×4 对比图 |

## 运行命令（Windows PowerShell，需 UTF-8 环境）

```powershell
$env:PYTHONIOENCODING='utf-8'; python -X utf8 .temp\sim\color_metrics.py    # CIEDE2000自验证
$env:PYTHONIOENCODING='utf-8'; python -X utf8 .temp\sim\cross_check.py      # 权威库交叉验证（需colour-science）
$env:PYTHONIOENCODING='utf-8'; python -X utf8 .temp\sim\restore_engine.py   # 算法模块自检（含无中性参照/过曝契约）
$env:PYTHONIOENCODING='utf-8'; python -X utf8 .temp\sim\exp1_2.py           # 实验一/二
$env:PYTHONIOENCODING='utf-8'; python -X utf8 .temp\sim\exp3.py             # 实验三
$env:PYTHONIOENCODING='utf-8'; python -X utf8 .temp\sim\exp4.py             # 实验四
$env:PYTHONIOENCODING='utf-8'; python -X utf8 .temp\sim\exp5.py             # 实验五
$env:PYTHONIOENCODING='utf-8'; python -X utf8 .temp\sim\calib_overexpose.py # 过曝阈值标定
```

依赖：numpy、scipy、opencv-python、colour-science≥0.4.7（仅交叉验证需要）。运行目录须为
`E:\明宇博士资料\专利\生鲜灯`（脚本内部相对导入）。

## 证据口径声明

1. 全部实验为**合成数据仿真**：污染由 `pollution_sim.py` 的线性域增益模型生成，灯库参数
   与生成参数同源（除 exp5 人为失配）；基准结果反映模型内一致性，不构成真实灯具/手机
   下的效果承诺；
2. CIEDE2000 实现经双重验证：Sharma 34 对官方测试集 + colour-science 0.4.7 交叉一致
   （34/34，最大偏差<1e-4）；
3. 所有 JSON/日志均为 UTF-8 编码。
