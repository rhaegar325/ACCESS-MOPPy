#!/usr/bin/env python3
"""
批量 CMORisation 脚本 —— ACCESS-ESM1.5 ILAMB 变量（大气 + 陆面月均）

用法:
    python cmorise_ilamb.py

或作为 PBS 任务运行（见同目录下的 submit_cmorise.pbs）。
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path


# =============================================================================
#  用户配置区 — 请根据实际情况修改
# =============================================================================

# 输入数据目录（包含模式输出 netCDF 文件的根目录）
INPUT_DIR = Path(
    "/g/data/p73/archive/CMIP7/ACCESS-ESM1-6/spinup/Feb26-PI-CNP-concentrations"
)

# CMORisation 结果输出目录
OUTPUT_DIR = Path(
    "/scratch/tm70/yz9299/ilamb_variables"
)

# 输入文件 glob 模式（相对于 INPUT_DIR）
# Amon/Lmon/Emon 变量共用同一批大气输出文件
FILE_PATTERN = "output00[0-9]/atmosphere/netCDF/*mon.nc"

# CMIP6 实验元数据
EXPERIMENT_ID  = "piControl"
SOURCE_ID      = "ACCESS-ESM1-5"
VARIANT_LABEL  = "r1i1p1f1"
GRID_LABEL     = "gn"
ACTIVITY_ID    = "CMIP"

# 要处理的变量列表（格式：table.variable）
VARIABLES = [
    # --- Emon ---
    "Emon.cSoil",
    # --- Lmon ---
    "Lmon.cVeg",
    "Lmon.gpp",
    "Lmon.lai",
    "Lmon.nbp",
    "Lmon.ra",
    "Lmon.rh",
    "Lmon.tsl",
    "Lmon.mrro",
    # --- Amon ---
    "Amon.evspsbl",
    "Amon.hfls",
    "Amon.hfss",
    "Amon.hurs",
    "Amon.pr",
    "Amon.rlds",
    "Amon.rlus",
    "Amon.rsds",
    "Amon.rsus",
    "Amon.tasmax",
    "Amon.tasmin",
    "Amon.tas",
]

# =============================================================================


def get_input_files() -> list[Path]:
    """
    返回 INPUT_DIR 中匹配 FILE_PATTERN 的文件列表，按文件名排序。
    """
    files = sorted(INPUT_DIR.glob(FILE_PATTERN))
    return files


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
    from access_moppy import ACCESS_ESM_CMORiser

    result = {
        "variable": variable,
        "status": None,
        "error": None,
        "n_files": len(input_files),
        "start_time": datetime.now().isoformat(),
        "end_time": None,
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
        )

        cmoriser.run(write_output=True)

        result["status"] = "SUCCESS"
        print(f"  [SUCCESS] {variable} — CMORisation 完成，结果已写入 {output_dir}")

    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
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
    print(
        f"  总计: {len(results)} 个变量  |  "
        f"成功: {len(successes)}  |  失败: {len(failures)}"
    )

    if successes:
        print(f"\n  成功 ({len(successes)}):")
        for r in successes:
            print(f"    ✓  {r['variable']}")

    if failures:
        print(f"\n  失败 ({len(failures)}):")
        for r in failures:
            print(f"    ✗  {r['variable']}")
            print(f"       错误: {r['error']}")

    report_path = OUTPUT_DIR / "cmorise_ilamb_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n  详细报告已保存至: {report_path}")
    print(bar)


def main() -> None:
    print("=" * 60)
    print("  MOPPy 批量 CMORisation — ILAMB 变量")
    print("=" * 60)
    print(f"  输入目录 : {INPUT_DIR}")
    print(f"  输出目录 : {OUTPUT_DIR}")
    print(f"  变量列表 : {', '.join(VARIABLES)}")
    print()

    # 单节点 Dask，避免 processes=False + n_workers>1 导致的内存 pause 问题
    from dask.distributed import Client
    client = Client(n_workers=1, threads_per_worker=4)
    print(f"  Dask dashboard : {client.dashboard_link}")

    # 1. 收集输入文件
    input_files = get_input_files()
    if not input_files:
        print(
            f"错误：在 {INPUT_DIR} 中未找到匹配 '{FILE_PATTERN}' 的文件！\n"
            f"请检查 INPUT_DIR 和 FILE_PATTERN 配置。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  找到 {len(input_files)} 个输入文件")
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

    client.close()

    if any(r["status"] == "FAILED" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
