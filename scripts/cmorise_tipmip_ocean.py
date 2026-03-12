#!/usr/bin/env python3
"""
批量 CMORisation 脚本 —— tipmipGr3b ACCESS-ESM1.6 海洋月均变量

用法:
    python cmorise_tipmip_ocean.py

或作为 PBS 任务运行（见同目录下的 submit_cmorise_ocean.pbs）。

文件命名格式: {category}-{dims}-{varname}-1monthly-mean-ym_YYYY_MM.nc
    例: ocean-3d-salt-1monthly-mean-ym_0326_01.nc  → salt，第 326 年 1 月
        ocean-2d-mld-1monthly-mean-ym_0111_06.nc   → mld，第 111 年 6 月
    YYYY = 4 位年份（含前导零），MM = 2 位月份

与大气脚本的关键区别:
    海洋输出每个 model variable 独立存放在各自的文件序列中。
    因此对每个 CMIP6 变量，本脚本会按其所需的 model variable 名
    分别定位文件，再合并后传入 ACCESS_ESM_CMORiser。
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

# 输入数据目录（包含海洋月均 .nc 文件）
INPUT_DIR = Path(
    "/g/data/y99/lcm561/access-esm/archive"
    "/tipmipGr3b-ffdbf476/history/ocn"
)

# CMORisation 结果输出目录
OUTPUT_DIR = Path("/scratch/tm70/yz9299/moppy_output_tipmip/tipmipGr3b_ocean")

# 需要处理的年份范围（含两端）
YEAR_START = 111
YEAR_END   = 210

# 从文件名中提取年份的正则
# 匹配结尾 _YYYY_MM.nc，其中 YYYY 为 4 位年份
YEAR_REGEX = r"_(\d{4})_\d{2}\.nc$"

# CMIP6 实验元数据
EXPERIMENT_ID = "tipmip"
SOURCE_ID     = "ACCESS-ESM1-6"
VARIANT_LABEL = "r1i1p1f1"
GRID_LABEL    = "gn"
ACTIVITY_ID   = "TIPMIP"

# ── 变量 → model variable 文件名关键词映射 ───────────────────────────────────
#
# 每个条目对应一个 CMIP6 变量所需的 model variable 名列表。
# 脚本会用这些名称在 INPUT_DIR 中 glob 文件（匹配 *-{name}-* 片段）。
#
# 需要多个 model variable 的变量（msftyz / msftmz）：
#   文件会按 (year, month) 对齐后合并为单个 xr.Dataset 再传给 CMORiser。
#
# ⚠ 如果实际文件名中的变量名片段与此不同，请在此处修改对应的列表。
VARIABLE_MODEL_VARS: dict[str, list[str]] = {
    "Omon.thetao":  ["pot_temp"],
    "Omon.tos":     ["surface_temp"],
    "Omon.so":      ["salt"],
    "Omon.sos":     ["sss"],
    "Omon.uo":      ["u"],
    "Omon.vo":      ["v"],
    "Omon.zos":     ["sea_level"],
    "Omon.mlotst":  ["mld"],
    "Omon.msftyz":  ["ty_trans", "ty_trans_gm", "ty_trans_submeso"],
    "Omon.msftmz":  ["ty_trans", "ty_trans_gm", "ty_trans_submeso"],
}

# 要处理的变量列表（顺序即处理顺序）
VARIABLES = list(VARIABLE_MODEL_VARS.keys())

# =============================================================================


def parse_year_from_filename(filename: str) -> int | None:
    """从文件名末尾的 _YYYY_MM.nc 提取 4 位年份并返回整数。"""
    match = re.search(YEAR_REGEX, filename)
    if match:
        return int(match.group(1))
    return None


def get_files_for_model_var(
    model_var: str,
    year_start: int,
    year_end: int,
) -> list[Path]:
    """
    在 INPUT_DIR 中查找包含指定 model variable 的文件，
    过滤至 [year_start, year_end] 年份范围，按文件名排序返回。

    glob 模式: *-{model_var}-*_????_??.nc
    """
    pattern = f"*-{model_var}-*_????_??.nc"
    all_files = sorted(INPUT_DIR.glob(pattern))
    filtered = [
        f for f in all_files
        if (y := parse_year_from_filename(f.name)) is not None
        and year_start <= y <= year_end
    ]
    return filtered


def build_input_for_variable(
    compound_name: str,
    year_start: int,
    year_end: int,
) -> tuple[list[Path] | object, int]:
    """
    根据 compound_name 构建传给 ACCESS_ESM_CMORiser 的 input_data。

    单 model variable  → 返回 (list[Path], n_files)
    多 model variables → 按 (year, month) 对齐各文件序列，
                         合并为 xr.Dataset，返回 (dataset, n_timesteps)

    返回值为 (input_data, file_count)，其中 file_count 仅供日志展示。
    """
    model_vars = VARIABLE_MODEL_VARS[compound_name]

    # ── 单变量：直接返回文件列表 ──────────────────────────────────────────────
    if len(model_vars) == 1:
        files = get_files_for_model_var(model_vars[0], year_start, year_end)
        return files, len(files)

    # ── 多变量：按时间戳对齐后合并为 xr.Dataset ───────────────────────────────
    import xarray as xr

    # 提取 (year, month) → Path 映射，每个 model variable 独立
    def timestamp_map(files: list[Path]) -> dict[tuple[int, int], Path]:
        mapping = {}
        for f in files:
            m = re.search(r"_(\d{4})_(\d{2})\.nc$", f.name)
            if m:
                mapping[(int(m.group(1)), int(m.group(2)))] = f
        return mapping

    maps = {mv: timestamp_map(get_files_for_model_var(mv, year_start, year_end))
            for mv in model_vars}

    # 取各变量时间步的交集（按排序顺序）
    common_timestamps = sorted(
        set.intersection(*(set(m.keys()) for m in maps.values()))
    )
    if not common_timestamps:
        raise FileNotFoundError(
            f"变量 {compound_name} 所需的 model variables {model_vars} "
            f"在年份 {year_start}–{year_end} 内无公共时间步，请检查文件是否完整。"
        )

    # 逐时间步打开各 model variable 文件并合并
    per_step = []
    for ym in common_timestamps:
        step_datasets = [
            xr.open_dataset(str(maps[mv][ym]), decode_cf=False)
            for mv in model_vars
        ]
        per_step.append(xr.merge(step_datasets))

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
                f"未找到 {compound_name} 所需的任何输入文件，"
                f"请检查 INPUT_DIR 和 VARIABLE_MODEL_VARS 配置。"
            )

        label = "个文件" if isinstance(input_data, list) else "个时间步（合并数据集）"
        print(f"  输入: {n} {label}")

        # list[Path] → list[str]；xr.Dataset → 直接传入
        if isinstance(input_data, list):
            cmoriser_input = [str(f) for f in input_data]
        else:
            cmoriser_input = input_data  # xr.Dataset

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
    """打印格式化的结果汇总，并将详细报告写入 JSON 文件。"""
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
    print(f"  输入目录 : {INPUT_DIR}")
    print(f"  年份正则 : {YEAR_REGEX}")
    print(f"  输出目录 : {OUTPUT_DIR}")
    print(f"  年份范围 : 第 {YEAR_START} 年 — 第 {YEAR_END} 年")
    print(f"  变量列表 : {', '.join(VARIABLES)}")
    print()

    # 快速验证：检查至少有一个变量能找到文件
    first_var  = VARIABLES[0]
    first_mvars = VARIABLE_MODEL_VARS[first_var]
    probe_files = get_files_for_model_var(first_mvars[0], YEAR_START, YEAR_END)
    if not probe_files:
        print(
            f"错误：在 {INPUT_DIR} 中未找到 '{first_mvars[0]}' 相关文件！\n"
            f"  glob 模式: *-{first_mvars[0]}-*_????_??.nc\n"
            f"  年份正则 : {YEAR_REGEX}\n"
            f"请先运行:\n"
            f"  ls {INPUT_DIR} | head -10\n"
            f"确认文件名格式，再调整 YEAR_REGEX 和 VARIABLE_MODEL_VARS。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  文件探测通过（{first_mvars[0]}: {len(probe_files)} 个文件）")
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
