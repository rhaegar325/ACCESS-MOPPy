#!/usr/bin/env python3
"""
批量 CMORisation 脚本 —— ACCESS-ESM1.6 海洋 BGC 年均变量 (Oyr)

用法:
    python cmorise_oceanbgc.py

或作为 PBS 任务运行。

输入文件为月均 oceanbgc 文件，CMORiser 会自动聚合为年均 Oyr 输出。

文件命名格式: {prefix}_YYYY_MM.nc（直接位于 INPUT_DIR 下）
    YYYY = 4 位年份（含前导零，如 0111 表示第 111 年）
    MM   = 2 位月份
    例: oceanbgc-3d-no3-1monthly-mean-ym_0111_01.nc

已实现变量（mappings 中已有条目）:
    Oyr.no3    ← oceanbgc-3d-no3-1monthly-mean-ym
    Oyr.o2     ← oceanbgc-3d-o2-1monthly-mean-ym
    Oyr.talk   ← oceanbgc-3d-alk-1monthly-mean-ym
    Oyr.detoc  ← oceanbgc-3d-det-1monthly-mean-ym
    Oyr.dissic ← oceanbgc-3d-dic-1monthly-mean-ym
    Oyr.dfe    ← oceanbgc-3d-fe-1monthly-mean-ym
    Oyr.zooc   ← oceanbgc-3d-zoo-1monthly-mean-ym
    Omon.intpp ← oceanbgc-2d-pprod_gross_2d-1monthly-mean-ym

待补充（mappings 中暂无条目，已注释）:
    Oyr.adic   ← oceanbgc-3d-adic-1monthly-mean-ym  # dissicabio 待讨论
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

# 输入数据根目录（包含各 oceanbgc 文件）
INPUT_DIR = Path(
    "/g/data/y99/lcm561/access-esm/archive"
    "/piControl/history/ocn"
)

# CMORisation 结果输出目录
OUTPUT_DIR = Path("/scratch/tm70/yz9299/moppy_output/oceanbgc_Oyr")

# 需要处理的年份范围（含两端）
YEAR_START = 111
YEAR_END   = 210

# CMIP6 实验元数据
EXPERIMENT_ID = "piControl"
SOURCE_ID     = "ACCESS-OM2"   # 暂无 CM3 元数据，以 CM2 代替
VARIANT_LABEL = "r1i1p1f1"
GRID_LABEL    = "gn"
ACTIVITY_ID   = "CMIP"

# ── 变量 → 文件前缀映射 ──────────────────────────────────────────────────────
#
# key   : CMIP6 compound name（table.variable）
# value : 所需文件前缀列表（文件命名 {prefix}_YYYY_MM.nc）
#
# NOTE: Oyr.adic 暂无 mapping 条目，待确认后取消注释
#
VARIABLE_FILE_DIRS: dict[str, list[str]] = {
    "Oyr.no3": [
        "oceanbgc-3d-no3-1monthly-mean-ym",
    ],
    "Oyr.o2": [
        "oceanbgc-3d-o2-1monthly-mean-ym",
    ],
    "Oyr.talk": [
        "oceanbgc-3d-alk-1monthly-mean-ym",
    ],
    "Oyr.detoc": [
        "oceanbgc-3d-det-1monthly-mean-ym",
    ],
    "Oyr.dissic": [
        "oceanbgc-3d-dic-1monthly-mean-ym",
    ],
    "Oyr.dfe": [
        "oceanbgc-3d-fe-1monthly-mean-ym",
    ],
    "Oyr.zooc": [
        "oceanbgc-3d-zoo-1monthly-mean-ym",
    ],
    "Omon.intpp": [
        "oceanbgc-2d-pprod_gross_2d-1monthly-mean-ym",
    ],
    # "Oyr.adic": [
    #     "oceanbgc-3d-adic-1monthly-mean-ym",
    # ],
}

VARIABLES = list(VARIABLE_FILE_DIRS.keys())

# =============================================================================


def parse_year_from_filename(filename: str) -> int | None:
    """从文件名末尾的 _YYYY_MM.nc 提取 4 位年份，返回整数。"""
    match = re.search(r"_(\d{4})_\d{2}\.nc$", filename)
    if match:
        return int(match.group(1))
    return None


def get_files_by_prefix(
    prefix: str,
    year_start: int,
    year_end: int,
) -> list[Path]:
    """
    在 INPUT_DIR 下查找以 prefix 开头的月均文件，
    过滤至 [year_start, year_end] 年份范围，按文件名排序返回。
    """
    all_files = sorted(INPUT_DIR.glob(f"{prefix}_????_??.nc"))
    if not all_files:
        raise FileNotFoundError(
            f"未找到任何文件，glob 模式: {INPUT_DIR}/{prefix}_????_??.nc\n"
            f"请检查 INPUT_DIR 和 VARIABLE_FILE_DIRS 中的前缀名。"
        )
    return [
        f for f in all_files
        if (y := parse_year_from_filename(f.name)) is not None
        and year_start <= y <= year_end
    ]


def build_input_for_variable(
    compound_name: str,
    year_start: int,
    year_end: int,
) -> tuple[list[Path] | object, int]:
    """
    根据 compound_name 构建传给 ACCESS_ESM_CMORiser 的 input_data。

    单前缀 → 返回 (list[Path], n_files)
    多前缀 → 各前缀文件按 (year, month) 对齐后 xr.merge，
             返回 (xr.Dataset, n_timesteps)
    """
    prefixes = VARIABLE_FILE_DIRS[compound_name]

    if len(prefixes) == 1:
        files = get_files_by_prefix(prefixes[0], year_start, year_end)
        return files, len(files)

    import xarray as xr

    def timestamp_map(files: list[Path]) -> dict[tuple[int, int], Path]:
        mapping = {}
        for f in files:
            m = re.search(r"_(\d{4})_(\d{2})\.nc$", f.name)
            if m:
                mapping[(int(m.group(1)), int(m.group(2)))] = f
        return mapping

    maps = {
        p: timestamp_map(get_files_by_prefix(p, year_start, year_end))
        for p in prefixes
    }

    common_timestamps = sorted(
        set.intersection(*(set(m.keys()) for m in maps.values()))
    )
    if not common_timestamps:
        raise FileNotFoundError(
            f"{compound_name} 所需的各文件序列在年份 {year_start}–{year_end} "
            f"内无公共时间步，请检查文件是否完整。\n"
            f"  前缀: {prefixes}"
        )

    per_step = []
    for ym in common_timestamps:
        step_ds = xr.merge([
            xr.open_dataset(str(maps[p][ym]), decode_cf=False)
            for p in prefixes
        ])
        per_step.append(step_ds)

    merged_ds = xr.concat(per_step, dim="time")
    return merged_ds, len(common_timestamps)


def cmorise_variable(compound_name: str, output_dir: Path) -> dict:
    """对单个变量执行 CMORisation，返回结果字典。"""
    from access_moppy import ACCESS_ESM_CMORiser

    result = {
        "variable":   compound_name,
        "status":     None,
        "error":      None,
        "n_files":    None,
        "start_time": datetime.now().isoformat(),
        "end_time":   None,
    }

    divider = "─" * 60
    print(f"\n{divider}")
    print(f"  变量: {compound_name}")
    print(divider)

    try:
        input_data, n = build_input_for_variable(compound_name, YEAR_START, YEAR_END)
        result["n_files"] = n

        if not n:
            raise FileNotFoundError(
                f"未找到 {compound_name} 的任何输入文件，"
                f"请检查 INPUT_DIR 和 VARIABLE_FILE_DIRS 配置。"
            )

        label = "个文件" if isinstance(input_data, list) else "个时间步（合并数据集）"
        print(f"  输入: {n} {label}")
        print(f"  前缀: {', '.join(VARIABLE_FILE_DIRS[compound_name])}")

        cmoriser_input = (
            [str(f) for f in input_data]
            if isinstance(input_data, list)
            else input_data
        )

        cmoriser = ACCESS_ESM_CMORiser(
            input_data=cmoriser_input,
            compound_name=compound_name,
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
        print(f"  [SUCCESS] {compound_name} — CMORisation 完成")

    except Exception as exc:
        result["status"] = "FAILED"
        result["error"]  = f"{type(exc).__name__}: {exc}"
        print(f"  [FAILED]  {compound_name}", file=sys.stderr)
        print(f"            原因: {result['error']}", file=sys.stderr)
        traceback.print_exc()

    finally:
        result["end_time"] = datetime.now().isoformat()

    return result


def print_summary(results: list[dict]) -> None:
    """打印结果汇总并写入 JSON 报告。"""
    successes = [r for r in results if r["status"] == "SUCCESS"]
    failures  = [r for r in results if r["status"] == "FAILED"]

    bar = "=" * 60
    print(f"\n{bar}")
    print("  CMORisation 结果汇总")
    print(bar)
    print(f"  总计: {len(results)} 个变量  |  成功: {len(successes)}  |  失败: {len(failures)}")

    if successes:
        print(f"\n  成功 ({len(successes)}):")
        for r in successes:
            print(f"    ✓  {r['variable']}")

    if failures:
        print(f"\n  失败 ({len(failures)}):")
        for r in failures:
            print(f"    ✗  {r['variable']}")
            print(f"       错误: {r['error']}")

    report_path = OUTPUT_DIR / "cmorise_oceanbgc_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n  详细报告已保存至: {report_path}")
    print(bar)


def main() -> None:
    print("=" * 60)
    print("  MOPPy 批量 CMORisation — oceanbgc Oyr 变量")
    print("=" * 60)
    print(f"  输入目录   : {INPUT_DIR}")
    print(f"  输出目录   : {OUTPUT_DIR}")
    print(f"  年份范围   : 第 {YEAR_START} 年 — 第 {YEAR_END} 年")
    print(f"  变量列表   : {', '.join(VARIABLES)}")
    print()

    # 快速验证：检查第一个变量的文件前缀
    first_var    = VARIABLES[0]
    first_prefix = VARIABLE_FILE_DIRS[first_var][0]
    try:
        probe_files = get_files_by_prefix(first_prefix, YEAR_START, YEAR_END)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    if not probe_files:
        print(
            f"错误：前缀 {first_prefix} 在年份 {YEAR_START}–{YEAR_END} 内未找到文件！\n"
            f"  预期文件名格式: {first_prefix}_YYYY_MM.nc\n"
            f"请检查 YEAR_START / YEAR_END 配置是否正确。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  文件探测通过（{first_prefix}: {len(probe_files)} 个文件）")
    print(f"  首个文件: {probe_files[0].name}")
    print(f"  末个文件: {probe_files[-1].name}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 不在外层创建 Dask Client —— ACCESS_ESM_CMORiser 内部会自行管理 Dask，
    # 若此处同时持有全局 Client，moppy 关闭其内部 client 时会连带关闭全局
    # client，导致后续变量处理时出现 ClosedClientError。
    results = []
    for variable in VARIABLES:
        result = cmorise_variable(variable, OUTPUT_DIR)
        results.append(result)

    print_summary(results)

    if any(r["status"] == "FAILED" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
