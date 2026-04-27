# Bug: `load_dataset` silently skips all files when required variables are absent, and surfaces a cryptic error at rename

## Summary

Two related bugs in `base.py` and `atmosphere.py` cause `CMORiser.run()` to fail with a misleading `ValueError: cannot rename 'xxx'` instead of a clear diagnostic when a required model variable is not present in the input files.

---

## Bug 1 — `_has_time` probe logic falls back to single-file mode when target variables are missing (`base.py`)

### Location

`src/access_moppy/base.py`, `load_dataset()` method (~line 392)

### Description

Before opening files with `xr.open_mfdataset`, `load_dataset` probes the first file to decide whether to concatenate along time (time-series) or open a single file (time-independent / `fx`):

```python
_probe_target_vars = (
    [v for v in required_vars if v in _probe.data_vars]
    if required_vars
    else list(_probe.data_vars)
)
_has_time = any("time" in _probe[v].dims for v in _probe_target_vars)
```

When **none** of the required variables exist in the probe file, `_probe_target_vars` is an empty list.  
`any([])` evaluates to `False`, so `_has_time = False`.

The code then falls into the `else` branch intended for time-independent (`fx`) files, which opens **only the first input file** via `xr.open_dataset` instead of concatenating all files with `xr.open_mfdataset`. As a result:

- Only one file is loaded, silently ignoring the rest.
- The required data variable is still missing from `self.ds` (since it was absent from the probe file and presumably from all files).
- No warning or error is raised at this point — the failure surfaces much later.

### Reproducer

Any variable whose model STASH code is not present in the input files triggers this path. For example, CMORising daily `zg` with a mapping that references a variable only available in monthly files.

### Fix (applied in branch `claude/fix-cmoriser-zg-error-xPowD`)

When `_probe_target_vars` is empty, fall back to checking whether **any** data variable in the probe file has a time dimension, rather than assuming the dataset is time-independent:

```python
if _probe_target_vars:
    _has_time = any("time" in _probe[v].dims for v in _probe_target_vars)
else:
    # None of the required variables are in the probe file; fall back to
    # checking whether the file itself is time-dependent so all files are
    # still concatenated (the missing-variable error surfaces later).
    _has_time = "time" in _probe.dims and any(
        "time" in _probe[v].dims for v in _probe.data_vars
    )
```

---

## Bug 2 — Missing validation after `load_dataset` produces a cryptic error (`atmosphere.py`)

### Location

`src/access_moppy/atmosphere.py`, `select_and_process_variables()` method (~line 159)

### Description

After `self.load_dataset(required_vars=required)` returns, there is no check that the required model variables were actually loaded into `self.ds`. If a variable is absent from all input files, `self.ds` will contain only coordinate/bounds variables.

The failure is only detected much later when the code attempts to rename the variable:

```python
# atmosphere.py line 170 (direct calculation path)
self.ds = self.ds.rename({required_vars[0]: self.cmor_name})
```

This raises:

```
ValueError: cannot rename 'fld_s16i201' because it is not a variable or dimension in this dataset
```

This error message does not tell the user:
- **Which** required variable is missing.
- **What** variables are actually available in the loaded dataset.
- **Where** to look (the mapping's `model_variables` entry).

The same silent failure also affects the `formula` calculation path (`atmosphere.py` ~line 172), which raises a `KeyError` on `self.ds[var]` with similarly little context.

### Fix (applied in branch `claude/fix-cmoriser-zg-error-xPowD`)

Add an explicit check immediately after `load_dataset`, before any processing:

```python
missing_model_vars = [v for v in required_vars if v not in self.ds]
if missing_model_vars:
    available = sorted(self.ds.data_vars)
    raise KeyError(
        f"Required model variable(s) {missing_model_vars} not found in the "
        f"input files for '{self.cmor_name}'. "
        f"Available data variables: {available}. "
        f"Check the 'model_variables' entry in the mapping."
    )
```

This surfaces the problem immediately with actionable information, for example:

```
KeyError: Required model variable(s) ['fld_s16i201'] not found in the input files
for 'zg'. Available data variables: ['lat_bnds', 'lon_bnds', 'time_bnds'].
Check the 'model_variables' entry in the mapping.
```

---

## Impact

Both bugs compound each other:

1. Bug 1 causes only the first file to be opened (wrong number of files loaded).
2. Bug 2 causes the error to surface at a confusing location (`rename`) rather than immediately after loading, making root-cause diagnosis difficult.

Together they make it very hard to diagnose mapping configuration errors where the specified model variable does not exist in the input files at the requested frequency.

---

## Related

- The immediate trigger was CMORising daily `zg` with `fld_s16i201` (model theta levels, STASH section 16) — which is only present in monthly output. The mapping has since been corrected to use `fld_s30i297` (pressure-level geopotential, STASH section 30), consistent with all other daily pressure-level variables (`hur`, `ua`, `wap`, etc.).
