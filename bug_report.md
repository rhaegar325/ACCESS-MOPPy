# Bug: `coordinates` attribute containing stale scalar coordinate reference persists in CMORised output

## Summary

When CMORising near-surface variables (e.g. `tas`), the output NetCDF file contains a
`coordinates = "height_0"` attribute on the data variable, but the `height_0` variable
itself does not exist in the file. This produces a broken/invalid CMIP6 output.

## Root Cause (Two Compounding Issues)

### Issue 1: `decode_cf=False` preserves raw `coordinates` attribute

`base.py` opens input files with `decode_cf=False`:

```python
xr.open_mfdataset(..., decode_cf=False)  # base.py:384
```

The original model output (written by iris) stores a raw NetCDF attribute on
`fld_s03i236`:

```
fld_s03i236:coordinates = "height_0" ;
```

With `decode_cf=True` (xarray default), this attribute is **consumed**: xarray promotes
`height_0` into a proper coordinate object and removes it from `.attrs`. With
`decode_cf=False`, the attribute is **preserved as-is** in `.attrs`. After renaming
`fld_s03i236` → `tas`, the stale attribute is inherited:

```python
self.ds["tas"].attrs["coordinates"]  # → "height_0"
```

This can be confirmed by opening the same file two ways:

```python
ds_cf  = xr.open_dataset(file)                   # decode_cf=True  → no "coordinates" in attrs
ds_raw = xr.open_dataset(file, decode_cf=False)  # → attrs["coordinates"] = "height_0"
```

### Issue 2: Substring match prevents correct `height` coordinate from being added

In `base.py`, when building the `coordinates` attribute for the output file:

```python
existing_coords = vdat.attrs.get("coordinates", "")
coords_to_add = [c for c in aux_coords if c not in existing_coords]  # ← substring match!
```

Because `"height" in "height_0"` evaluates to `True`, the correct CMIP6 `height`
scalar coordinate (value = 2.0 m) is silently skipped and never written. The output
file ends up with:

```
tas:coordinates = "height_0" ;   ← references a variable that doesn't exist
```

instead of:

```
tas:coordinates = "height" ;     ← correct CMIP6 scalar coordinate
```

## Full Bug Chain

```
Original NetCDF (iris-written)
 └─ fld_s03i236:coordinates = "height_0"
         ↓  open_mfdataset(decode_cf=False)   raw attr preserved
 └─ ds["fld_s03i236"].attrs["coordinates"] = "height_0"
         ↓  ds.rename({"fld_s03i236": "tas"}) attrs inherited
 └─ ds["tas"].attrs["coordinates"] = "height_0"
         ↓  "height" in "height_0" → True     height skipped
 └─ coords_to_add = []                         height never written
         ↓  setncattr("coordinates", "height_0")
 └─ Output file: tas:coordinates = "height_0"  ← dangling reference, invalid output
```

## Expected Behaviour

Output file should contain a valid `height` scalar coordinate at 2.0 m:

```
double height ;
    height:long_name = "height" ;
    height:units = "m" ;
    height:positive = "up" ;
    height:axis = "Z" ;
    height:standard_name = "height" ;
tas:coordinates = "height" ;
```

## Suggested Fix

**Fix 1** — Strip the stale `coordinates` attr after loading (or before renaming):

```python
# After load_dataset / before or after rename
if "coordinates" in self.ds[self.cmor_name].attrs:
    del self.ds[self.cmor_name].attrs["coordinates"]
```

**Fix 2** — Use whole-word matching instead of substring check:

```python
existing = set(existing_coords.split())
coords_to_add = [c for c in aux_coords if c not in existing]
```

Both fixes should be applied together.

## Environment

- Variable: `tas` (and likely any near-surface variable with a scalar height coordinate)
- Input format: ACCESS model output converted via iris → NetCDF
- Affected file: `base.py` (load_dataset + coordinates attribute writing logic)
