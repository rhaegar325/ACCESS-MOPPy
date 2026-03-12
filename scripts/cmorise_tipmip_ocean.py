#!/usr/bin/env python3
"""
批量 CMORisation 脚本 —— tipmipGr3b ACCESS-ESM1.6 海洋月均变量

用法:
    python cmorise_tipmip_ocean.py

或作为 PBS 任务运行（见同目录下的 submit_cmorise_ocean.pbs）。

文件名格式: tipmipGr3b-ffdbf476.ocean_month.nc-YYYYMMDD
    其中 YYYY = 年份（4 位），MM = 月份，DD = 日
    例: ocean_month.nc-01110101 → 第 111 年 1 月
    （注意：请先用 ls 确认实际文件名，再调整 FILE_PATTERN 和正则）
"""

import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path


# =============================================================================
#  用户配置区 — 请根据实际情况修改
# =============================================================================

# 输入数据目录（包含海洋月均输出文件）
# ⚠ 请先运行 ls 确认该路径存在且含有 .nc 文件
INPUT_DIR = Path(
    "/g/data/y99/lcm561/access-esm/archive"
    "/tipmipGr3b-ffdbf476/history/ocn"
)

# CMORisation 结果输出目录（建议放在 /scratch 下）
OUTPUT_DIR = Path("/scratch/tm70/yz9299/moppy_output_tipmip/tipmipGr3b_ocean")

# 需要处理的年份范围（含两端）
YEAR_START = 111
YEAR_END   = 210

# ── 文件匹配模式 ──────────────────────────────────────────────────────────────
# ACCESS-ESM1.6 (MOM5) 海洋月均文件常见命名格式：
#   格式 A: tipmipGr3b-ffdbf476.ocean_month.nc-YYYYMMDD  （年末日期后缀）
#   格式 B: tipmipGr3b-ffdbf476.ocean-YYYYMM.nc           （与大气同格式）
# ⚠ 请根据实际文件名选择并修改以下两项
FILE_PATTERN = "*.ocean_month.nc-*"          # glob 通配符
YEAR_REGEX   = r"\.nc-(\d{4})\d{4}$"        # 从文件名提取年份（格式 A）
# 格式 B 对应的正则: r"\.ocean-(\d{4})\d{2}\.nc$"

# CMIP6 实验元数据 — 请根据你的实验配置核实以下字段
EXPERIMENT_ID = "tipmip"
SOURCE_ID     = "ACCESS-ESM1-6"
VARIANT_LABEL = "r1i1p1f1"
GRID_LABEL    = "gn"
ACTIVITY_ID   = "TIPMIP"

# 要处理的变量列表（格式：table.variable）
VARIABLES = [
    "Omon.thetao",   # 位温（3D）
    "Omon.tos",      # 海表温度
    "Omon.so",       # 盐度（3D）
    "Omon.sos",      # 海表盐度
    "Omon.uo",       # 纬向流速（3D）
    "Omon.vo",       # 经向流速（3D）
    "Omon.zos",      # 海面高度
    "Omon.mlotst",   # 混合层深度
    "Omon.msftyz",   # 经向翻转流函数（y-z 空间）
    "Omon.msftmz",   # 经向翻转流函数（密度空间）
]

# =============================================================================


def parse_year_from_filename(filename: str) -> int | None:
    """
    从文件名中提取年份。

    格式 A（默认）: *.ocean_month.nc-YYYYMMDD
        例: "tipmipGr3b-ffdbf476.ocean_month.nc-01110101" → 111

    格式 B: *.ocean-YYYYMM.nc
        例: "tipmipGr3b-ffdbf476.ocean-011101.nc" → 111
    """
    match = re.search(YEAR_REGEX, filename)
    if match:
        return int(match.group(1))
    return None


def get_input_files(year_start: int, year_end: int) -> list[Path]:
    """
    返回 INPUT_DIR 中属于 [year_start, year_end] 年份范围的文件列表，
    按文件名（即时间顺序）排序。
    """
    all_files = sorted(INPUT_DIR.glob(FILE_PATTERN))
    filtered = []
    for f in all_files:
        year = parse_year_from_filename(f.name)
        if year is not None and year_start <= year <= year_end:
            filtered.append(f)
    return filtered


def cmorise_variable(
    variable: str,
    input_files: list[Path],
    output_dir: Path,
) -> dict:
    """
    对单个变量执行 CMORisation，返回结果字典。
    """
    from access_moppy import ACCESS_ESM_CMORiser

    result = {
        "variable":   variable,
        "status":     None,
        "error":      None,
        "n_files":    len(input_files),
        "start_time": datetime.now().isoformat(),
        "end_time":   None,
    }

    divider = "─" * 60
    print(f"\n{divider}")
    print(f"  变量: {variable}  |  文件数: {len(input_files)}")
    print(divider)

    try:
        cmoriser = ACCESS_ESM_CMORiser(
            input_data=[str(f) for f in input_files],
            compound_name=variable,
            experiment_id=EXPERIMENT_ID,
            source_id=SOURCE_ID,
            variant_label=VARIANT_LABEL,
            grid_label=GRID_LABEL,
            activity_id=ACTIVITY_ID,
            output_path=str(output_dir),
            model_id="ACCESS-ESM1.6",
        )

        cmoriser.run(write_output=True)

        result["status"] = "SUCCESS"
        print(f"  [SUCCESS] {variable} — CMORisation 完成，结果已写入 {output_dir}")

    except Exception as exc:
        result["status"] = "FAILED"
        result["error"]  = f"{type(exc).__name__}: {exc}"
        print(f"  [FAILED]  {variable}", file=sys.stderr)
        print(f"            原因: {result['error']}", file=sys.stderr)
        traceback.print_exc()

    finally:
        result["end_time"] = datetime.now().isoformat()

    return result


def print_summary(results: list[dict]) -> None:
    """打印格式化的结果汇总，并将详细报告写入 JSON 文件。"""
    successes = [r for r in results if r["status"] == "SUCCESS"]
    failures  = [r for r in results if r["status"] == "FAILED"]

    bar = "=" * 60
    print(f"\n{bar}")
    print("  CMORisation 结果汇总")
    print(bar)
    print(f"  总计: {len(results)} 个变量  |  成功: {len(successes)}  |  失败: {len(failures)}")
    print()

    if successes:
        print(f"  成功 ({len(successes)}):")
        for r in successes:
            print(f"    ✓  {r['variable']}")

    if failures:
        print(f"\n  失败 ({len(failures)}):")
        for r in failures:
            print(f"    ✗  {r['variable']}")
            print(f"       错误: {r['error']}")

    report_path = OUTPUT_DIR / "cmorise_ocean_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n  详细报告已保存至: {report_path}")
    print(bar)


def main() -> None:
    print("=" * 60)
    print("  MOPPy 批量 CMORisation — tipmipGr3b 海洋变量")
    print("=" * 60)
    print(f"  输入目录 : {INPUT_DIR}")
    print(f"  文件模式 : {FILE_PATTERN}")
    print(f"  输出目录 : {OUTPUT_DIR}")
    print(f"  年份范围 : 第 {YEAR_START} 年 — 第 {YEAR_END} 年")
    print(f"  变量列表 : {', '.join(VARIABLES)}")
    print()

    # 1. 收集输入文件
    input_files = get_input_files(YEAR_START, YEAR_END)
    if not input_files:
        print(
            f"错误：在 {INPUT_DIR} 中未找到年份 {YEAR_START}–{YEAR_END} 的文件！\n"
            f"  glob 模式 : {FILE_PATTERN}\n"
            f"  年份正则  : {YEAR_REGEX}\n"
            f"请先运行 ls {INPUT_DIR} | head -5 确认文件名格式，\n"
            f"再调整脚本顶部的 FILE_PATTERN 和 YEAR_REGEX。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  找到 {len(input_files)} 个文件（年份 {YEAR_START}–{YEAR_END}）")
    print(f"  首个文件: {input_files[0].name}")
    print(f"  末个文件: {input_files[-1].name}")

    # 2. 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 逐变量执行 CMORisation
    results = []
    for variable in VARIABLES:
        result = cmorise_variable(variable, input_files, OUTPUT_DIR)
        results.append(result)

    # 4. 打印汇总并写入报告
    print_summary(results)

    if any(r["status"] == "FAILED" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
