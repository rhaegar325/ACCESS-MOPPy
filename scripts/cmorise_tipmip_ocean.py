#!/usr/bin/env python3
"""
批量 CMORisation 脚本 —— tipmipGr3b ACCESS-ESM1.6 海洋月均变量

用法:
    python cmorise_tipmip_ocean.py

或作为 PBS 任务运行（见同目录下的 submit_cmorise_ocean.pbs）。

目录结构:
    INPUT_DIR/
        ocean-3d-pot_temp-1monthly-mean-ym/
            ocean-3d-pot_temp-1monthly-mean-ym_YYYY_MM.nc
            ...
        ocean-3d-salt-1monthly-mean-ym/
            ocean-3d-salt-1monthly-mean-ym_YYYY_MM.nc
            ...
        ...

文件命名格式: {subdir_name}_YYYY_MM.nc
    YYYY = 4 位年份（含前导零，如 0111 表示第 111 年）
    MM   = 2 位月份
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

# 输入数据根目录（包含各变量子目录）
INPUT_DIR = Path(
    "/g/data/y99/lcm561/access-esm/archive"
    "/tipmipGr3b-ffdbf476/history/ocn"
)

# CMORisation 结果输出目录
OUTPUT_DIR = Path("/scratch/tm70/yz9299/moppy_output_tipmip/tipmipGr3b_ocean")

# 需要处理的年份范围（含两端）
YEAR_START = 111
YEAR_END   = 210

# CMIP6 实验元数据
EXPERIMENT_ID = "tipmip"
SOURCE_ID     = "ACCESS-ESM1-6"
VARIANT_LABEL = "r1i1p1f1"
GRID_LABEL    = "gn"
ACTIVITY_ID   = "TIPMIP"

# ── 变量 → 输入子目录映射 ─────────────────────────────────────────────────────
#
# 每个 CMIP6 变量所需的子目录名列表（相对于 INPUT_DIR）。
# 子目录内的文件命名格式: {subdir_name}_YYYY_MM.nc
#
# 单个子目录：直接将文件列表传给 CMORiser（list[Path]）。
# 多个子目录：按 (year, month) 对齐后 xr.merge 合并为 Dataset 再传入。
#
# 注意:
#   sos      → surface_salt 目录（xarray 内部变量名仍为 sss，需与 MOPPy
#              mapping 中 "sos": {"model_variables": ["sss"]} 一致；
#              若实际变量名为 surface_salt，请将 MOPPy mapping 也对应修改）
#   msftyz / msftmz → ty_trans_submeso 仅有年均输出，月均处理时跳过，
#              只使用 ty_trans + ty_trans_gm
VARIABLE_FILE_DIRS: dict[str, list[str]] = {
    "Omon.thetao": [
        "ocean-3d-pot_temp-1monthly-mean-ym",
    ],
    "Omon.tos": [
        "ocean-2d-surface_temp-1monthly-mean-ym",
    ],
    "Omon.so": [
        "ocean-3d-salt-1monthly-mean-ym",
    ],
    # "Omon.sos": ["ocean-2d-surface_salt-1monthly-mean-ym"],  # 暂搁置：内部变量名待确认
    "Omon.uo": [
        "ocean-3d-u-1monthly-mean-ym",
    ],
    "Omon.vo": [
        "ocean-3d-v-1monthly-mean-ym",
    ],
    "Omon.zos": [
        "ocean-2d-sea_level-1monthly-mean-ym",
    ],
    "Omon.mlotst": [
        "ocean-2d-mld-1monthly-mean-ym",
    ],
    "Omon.msftyz": [
        "ocean-3d-ty_trans-1monthly-mean-ym",
        "ocean-3d-ty_trans_gm-1monthly-mean-ym",
        # ty_trans_submeso 仅有年均，不纳入月均处理
    ],
    "Omon.msftmz": [
        "ocean-3d-ty_trans-1monthly-mean-ym",
        "ocean-3d-ty_trans_gm-1monthly-mean-ym",
        # ty_trans_submeso 仅有年均，不纳入月均处理
    ],
}

# 要处理的变量列表（顺序即处理顺序）
VARIABLES = list(VARIABLE_FILE_DIRS.keys())

# =============================================================================


def parse_year_from_filename(filename: str) -> int | None:
    """从文件名末尾的 _YYYY_MM.nc 提取 4 位年份，返回整数。"""
    match = re.search(r"_(\d{4})_\d{2}\.nc$", filename)
    if match:
        return int(match.group(1))
    return None


def get_files_in_subdir(
    subdir_name: str,
    year_start: int,
    year_end: int,
) -> list[Path]:
    """
    返回 INPUT_DIR/{subdir_name}/ 下属于 [year_start, year_end] 的文件列表，
    按文件名（时间）顺序排序。

    文件命名格式: {subdir_name}_YYYY_MM.nc
    """
    subdir = INPUT_DIR / subdir_name
    if not subdir.is_dir():
        raise FileNotFoundError(
            f"子目录不存在: {subdir}\n"
            f"请检查 INPUT_DIR 和 VARIABLE_FILE_DIRS 中的目录名。"
        )
    all_files = sorted(subdir.glob(f"{subdir_name}_????_??.nc"))
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

    单子目录 → 返回 (list[Path], n_files)
    多子目录 → 各子目录文件按 (year, month) 对齐后 xr.merge，
               返回 (xr.Dataset, n_timesteps)
    """
    subdirs = VARIABLE_FILE_DIRS[compound_name]

    # ── 单子目录：直接返回文件列表 ────────────────────────────────────────────
    if len(subdirs) == 1:
        files = get_files_in_subdir(subdirs[0], year_start, year_end)
        return files, len(files)

    # ── 多子目录：按 (year, month) 对齐后合并 ────────────────────────────────
    import xarray as xr

    def timestamp_map(files: list[Path]) -> dict[tuple[int, int], Path]:
        mapping = {}
        for f in files:
            m = re.search(r"_(\d{4})_(\d{2})\.nc$", f.name)
            if m:
                mapping[(int(m.group(1)), int(m.group(2)))] = f
        return mapping

    maps = {
        sd: timestamp_map(get_files_in_subdir(sd, year_start, year_end))
        for sd in subdirs
    }

    common_timestamps = sorted(
        set.intersection(*(set(m.keys()) for m in maps.values()))
    )
    if not common_timestamps:
        raise FileNotFoundError(
            f"{compound_name} 所需的各子目录在年份 {year_start}–{year_end} "
            f"内无公共时间步，请检查文件是否完整。\n"
            f"  子目录: {subdirs}"
        )

    per_step = []
    for ym in common_timestamps:
        step_ds = xr.merge([
            xr.open_dataset(str(maps[sd][ym]), decode_cf=False)
            for sd in subdirs
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
        if isinstance(input_data, list):
            print(f"  目录: {VARIABLE_FILE_DIRS[compound_name][0]}")
        else:
            print(f"  目录: {', '.join(VARIABLE_FILE_DIRS[compound_name])}")

        # list[Path] → list[str]；xr.Dataset → 直接传入
        cmoriser_input = [str(f) for f in input_data] if isinstance(input_data, list) \
                         else input_data

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

    report_path = OUTPUT_DIR / "cmorise_ocean_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n  详细报告已保存至: {report_path}")
    print(bar)


def main() -> None:
    print("=" * 60)
    print("  MOPPy 批量 CMORisation — tipmipGr3b 海洋变量")
    print("=" * 60)
    print(f"  输入根目录 : {INPUT_DIR}")
    print(f"  输出目录   : {OUTPUT_DIR}")
    print(f"  年份范围   : 第 {YEAR_START} 年 — 第 {YEAR_END} 年")
    print(f"  变量列表   : {', '.join(VARIABLES)}")
    print()

    # 快速验证：检查第一个变量的子目录和文件
    first_var   = VARIABLES[0]
    first_subdir = VARIABLE_FILE_DIRS[first_var][0]
    try:
        probe_files = get_files_in_subdir(first_subdir, YEAR_START, YEAR_END)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    if not probe_files:
        print(
            f"错误：子目录 {first_subdir} 中未找到年份 {YEAR_START}–{YEAR_END} 的文件！\n"
            f"  预期文件名格式: {first_subdir}_YYYY_MM.nc\n"
            f"请检查 YEAR_START / YEAR_END 配置是否正确。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  文件探测通过（{first_subdir}: {len(probe_files)} 个文件）")
    print(f"  首个文件: {probe_files[0].name}")
    print(f"  末个文件: {probe_files[-1].name}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for variable in VARIABLES:
        result = cmorise_variable(variable, OUTPUT_DIR)
        results.append(result)

    print_summary(results)

    if any(r["status"] == "FAILED" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
