#!/usr/bin/env python3
"""
批量 CMORisation 脚本 —— tipmipGr3b ACCESS-ESM1.6 大气月均变量

用法:
    python cmorise_tipmip.py

或作为 PBS 任务运行（见同目录下的 submit_cmorise.pbs）。

文件名格式: tipmipGr3b-ffdbf476.pa-YYYMM_mon.nc
    其中 YYY = 年份（3 位），MM = 月份（2 位）
    例: pa-015104_mon.nc → 第 151 年 4 月
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

# 输入数据目录（包含 .pa-YYYMM_mon.nc 文件）
INPUT_DIR = Path(
    "/g/data/y99/lcm561/access-esm/archive"
    "/tipmipGr3b-ffdbf476/history/atm/netCDF"
)

# CMORisation 结果输出目录（建议放在 /scratch 下）
OUTPUT_DIR = Path("/scratch/y99/lcm561/moppy_output/tipmipGr3b")

# 需要处理的年份范围（含两端）
YEAR_START = 111
YEAR_END   = 210

# CMIP6 实验元数据 — 请根据你的实验配置核实以下字段
EXPERIMENT_ID = "tipmip"          # 实验 ID
SOURCE_ID     = "ACCESS-ESM1-6"   # 模式 ID（CMIP6 注册名）
VARIANT_LABEL = "r1i1p1f1"        # 成员标签
GRID_LABEL    = "gn"              # 网格标签（gn = native grid）
ACTIVITY_ID   = "TIPMIP"         # MIP 活动名称

# 要处理的变量列表（格式：table.variable）
# 注意：co2mass 目前不在 ACCESS-ESM1.6 的变量映射中，预期会失败
VARIABLES = [
    "Amon.tas",
    "Amon.pr",
    "Amon.psl",
    "Amon.evspsbl",
    "Amon.ua",
    "Amon.va",
    "Amon.co2mass",   # 警告：ACCESS-ESM1.6 映射中暂无此变量，预期失败
]

# =============================================================================


def parse_year_from_filename(filename: str) -> int | None:
    """
    从文件名中提取年份。

    文件名格式: tipmipGr3b-ffdbf476.pa-YYYMM_mon.nc
    其中 YYY（前3位）为年份，MM（后2位）为月份。

    示例:
        "tipmipGr3b-ffdbf476.pa-015104_mon.nc" → 151
        "tipmipGr3b-ffdbf476.pa-210012_mon.nc" → 210
    """
    match = re.search(r"\.pa-(\d{3})\d{2}_mon\.nc$", filename)
    if match:
        return int(match.group(1))
    return None


def get_input_files(year_start: int, year_end: int) -> list[Path]:
    """
    返回 INPUT_DIR 中属于 [year_start, year_end] 年份范围的文件列表，
    按文件名（即时间顺序）排序。
    """
    all_files = sorted(INPUT_DIR.glob("*.pa-*_mon.nc"))
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

    返回字典包含:
        variable   : 变量名（如 "Amon.tas"）
        status     : "SUCCESS" 或 "FAILED"
        error      : 失败时的错误信息（成功时为 None）
        n_files    : 输入文件数量
        start_time : 开始时间（ISO 格式）
        end_time   : 结束时间（ISO 格式）
    """
    # 延迟导入，仅在实际处理时加载 MOPPy
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
            model_id="ACCESS-ESM1.6",   # 指定映射文件
        )

        cmoriser.run(write_output=True)

        result["status"] = "SUCCESS"
        print(f"  [SUCCESS] {variable} — CMORisation 完成，结果已写入 {output_dir}")

    except Exception as exc:
        result["status"] = "FAILED"
        result["error"]  = f"{type(exc).__name__}: {exc}"
        print(f"  [FAILED]  {variable}", file=sys.stderr)
        print(f"            原因: {result['error']}", file=sys.stderr)
        # 打印完整 traceback 供调试
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

    # 保存 JSON 报告
    report_path = OUTPUT_DIR / "cmorise_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n  详细报告已保存至: {report_path}")
    print(bar)


def main() -> None:
    print("=" * 60)
    print("  MOPPy 批量 CMORisation — tipmipGr3b")
    print("=" * 60)
    print(f"  输入目录 : {INPUT_DIR}")
    print(f"  输出目录 : {OUTPUT_DIR}")
    print(f"  年份范围 : 第 {YEAR_START} 年 — 第 {YEAR_END} 年")
    print(f"  变量列表 : {', '.join(VARIABLES)}")
    print()

    # 1. 收集输入文件
    input_files = get_input_files(YEAR_START, YEAR_END)
    if not input_files:
        print(
            f"错误：在 {INPUT_DIR} 中未找到年份 {YEAR_START}–{YEAR_END} 的文件！\n"
            f"请检查路径和文件命名格式。",
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

    # 有失败则以非零状态码退出（方便 PBS 判断作业状态）
    if any(r["status"] == "FAILED" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
