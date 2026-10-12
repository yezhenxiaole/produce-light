# -*- coding: utf-8 -*-
"""实验元数据工具（V0.7a）：结果 JSON 统一写入 meta 字段。

用途（响应第三轮审查意见3：复现风险）：
  每个实验结果 JSON 顶层写入：
    - code_version：源码版本（手动维护，与交底书/README 同步）
    - generated_at：运行时间（ISO 8601）
    - python_version：解释器版本
    - git_commit：运行时 HEAD 短哈希（无 git 环境时为 None）
    - params：实验关键参数摘要（各实验自定义 dict）
约定：meta 字段由 _save_results() 统一注入，禁止手工编辑结果文件。
"""
import subprocess
import sys
from datetime import datetime

CODE_VERSION = "V0.7a"


def _git_head():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def make_meta(params=None):
    """生成标准 meta dict。params 为该实验的关键参数摘要。"""
    return {
        "code_version": CODE_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "git_commit": _git_head(),
        "params": params or {},
    }


def save_results(path, results, params=None):
    """带 meta 保存结果 JSON（UTF-8，禁止手工编辑）。"""
    import json
    payload = {"meta": make_meta(params)}
    payload.update(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"{path} 已保存（meta: {payload['meta']['code_version']}, "
          f"{payload['meta']['generated_at']}, commit={payload['meta']['git_commit']}）")
