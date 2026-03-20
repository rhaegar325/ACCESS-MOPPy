# Bug: Scalar height coordinate from raw data not renamed to CMIP6-standard `height` during CMORisation

## Summary

Near-surface variables (e.g. `tas`) in the raw ACCESS model output carry a scalar
coordinate named `height_0` (written by iris). During CMORisation this coordinate is
never renamed to the CMIP6-standard name `height`, causing the output file to reference
a non-existent variable and `height` to appear as a plain variable rather than a
coordinate.

## Background: What the raw data contains

The iris-written NetCDF file stores `height_0` as a scalar coordinate on `fld_s03i236`:

```
float fld_s03i236(...) ;
    fld_s03i236:coordinates = "height_0" ;   ← raw CF attribute

double height_0 ;                            ← scalar coordinate variable
    height_0:standard_name = "height" ;
    height_0:units = "m" ;
    height_0:value = 1.5 ;
```

When opened with `decode_cf=True` (xarray default), xarray **consumes** the
`coordinates` attribute and promotes `height_0` into a proper coordinate object, so
`.attrs` no longer contains it. When opened with `decode_cf=False` (as `base.py` does),
the raw attribute `coordinates = "height_0"` is preserved in `.attrs` unchanged.

## Root Cause

### Step 1 — `height_0` is never included in `required` variables

In `select_and_process_variables()` (`atmosphere.py`), the set of variables to load is:

```python
required = set(required_vars + list(axes_rename_map.keys()) + list(bounds_rename_map.keys()))
```

`height_0` does not appear in any of these lists because the mapping's `dimensions`
dict has no `height` key that maps back to `height_0`. So `height_0` is never loaded
into `self.ds`.

### Step 2 — The stale `coordinates = "height_0"` attr is silently inherited

Because `decode_cf=False` preserves the raw attribute, after renaming:

```python
self.ds = self.ds.rename({"fld_s03i236": "tas"})
self.ds["tas"].attrs["coordinates"]  # → "height_0"
```

`tas` now carries a `coordinates` attribute that points to `height_0`, but `height_0`
was never loaded — it does not exist in `self.ds`.

### Step 3 — Substring match prevents the correct `height` coordinate from being added

Later in `base.py`, when appending auxiliary coordinates to the `coordinates` attribute:

```python
existing_coords = vdat.attrs.get("coordinates", "")          # → "height_0"
coords_to_add = [c for c in aux_coords if c not in existing_coords]  # substring match!
```

`"height" in "height_0"` is `True`, so the CMIP6-standard `height` scalar coordinate
is silently skipped. The output file ends up with:

```
tas:coordinates = "height_0" ;   ← dangling reference, height_0 does not exist
```

## Full Bug Chain

```
iris-written NetCDF
 └─ fld_s03i236:coordinates = "height_0"  (height_0 scalar coord exists in file)
         ↓  open_mfdataset(decode_cf=False)
            height_0 not in `required` → never loaded into self.ds
 └─ ds["fld_s03i236"].attrs["coordinates"] = "height_0"  (stale, dangling)
         ↓  ds.rename({"fld_s03i236": "tas"})
 └─ ds["tas"].attrs["coordinates"] = "height_0"
         ↓  "height" in "height_0" → True  → height skipped by substring check
 └─ output: tas:coordinates = "height_0"   ← variable doesn't exist → invalid CMIP6
```

## Expected Behaviour

The CMORisation pipeline should recognise `height_0` as the raw-data representation of
the CMIP6 `height` coordinate and rename it accordingly. The output should be:

```
double height ;
    height:standard_name = "height" ;
    height:long_name = "height" ;
    height:units = "m" ;
    height:positive = "up" ;
    height:axis = "Z" ;

tas:coordinates = "height" ;   ← valid CMIP6 scalar coordinate
```

## Suggested Fix

Three changes are needed together:

**Fix 1** — Include scalar coordinates listed in `coordinates` attr in `required`,
and add them to the rename map so `height_0` → `height`:

```python
# In select_and_process_variables(), after building `required`:
raw_coord_attr = self.ds[self.cmor_name].attrs.get("coordinates", "")
for raw_coord_name in raw_coord_attr.split():
    # find the CMIP6 standard name for this coordinate and add to rename map
    required.add(raw_coord_name)
    axes_rename_map[raw_coord_name] = "height"  # resolve via vocab lookup
```

**Fix 2** — Clear the stale `coordinates` attr after loading so it does not
propagate to the output:

```python
if "coordinates" in self.ds[self.cmor_name].attrs:
    del self.ds[self.cmor_name].attrs["coordinates"]
```

**Fix 3** — Use whole-word matching instead of substring check when building
the output `coordinates` attribute:

```python
existing = set(existing_coords.split())
coords_to_add = [c for c in aux_coords if c not in existing]
```

## Affected Variables

Any near-surface variable whose iris-written source file carries a scalar height
coordinate under a name other than `height` (e.g. `height_0`, `height_1`), including
but not limited to: `tas`, `huss`, `uas`, `vas`, `sfcWind`.

## Environment

- Input format: ACCESS model output written by iris → NetCDF
- Affected files: `atmosphere.py` (`select_and_process_variables`), `base.py` (load + coordinates attribute writing)
