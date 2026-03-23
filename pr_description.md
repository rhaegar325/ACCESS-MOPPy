## Fix scalar height coordinate missing from CMORised output (`tas` and similar)

### Problem

Near-surface variables (e.g. `tas`, `huss`, `uas`, `vas`) in ACCESS model output are written by iris with a raw NetCDF attribute `coordinates = "height_0"` on the data variable. Because `base.py` opens files with `decode_cf=False`, this raw attribute is preserved in `.attrs` unchanged (unlike `decode_cf=True` which would consume it). Two issues then compound:

1. The stale `coordinates = "height_0"` attribute is inherited by `tas` after renaming, but `height_0` is never loaded into the dataset — creating a dangling reference in the output file.
2. When the code later tried to append the CMIP6-standard `height` scalar coordinate to the `coordinates` attribute, the check `c not in existing_coords` used substring matching, causing `"height" in "height_0"` to evaluate `True` — silently skipping `height` entirely.

The net result was an invalid output file:
```
tas:coordinates = "height_0" ;   ← variable does not exist in file
```

### Changes

**`base.py` — collect non-string scalar coordinates into `aux_coords`**

Added a second loop after the existing `string_coords_info` loop to capture scalar (0-dimensional) and non-dimension coordinates that are not of string type (e.g. the float `height` coordinate created from the CMOR table `value` field). Without this, `height` was written as a plain variable rather than being declared in the `coordinates` attribute.

```python
# Also include non-string scalar coordinates (e.g. float 'height')
for coord_name in self.ds.coords:
    coord = self.ds[coord_name]
    is_scalar = coord.ndim == 0
    is_non_dim = coord_name not in self.ds.dims
    if (is_scalar or is_non_dim) and coord_name not in aux_coords:
        aux_coords.append(coord_name)
```

**`base.py` — fix substring match when deduplicating `coordinates` attribute**

Changed the guard that prevents duplicate entries in the `coordinates` attribute from a substring check to an equality check, so that `"height"` is no longer incorrectly treated as already present when `existing_coords = "height_0"`.

```python
# Before (substring match — buggy)
coords_to_add = [c for c in aux_coords if c not in existing_coords]

# After
coords_to_add = [c for c in aux_coords if c != existing_coords]
```

### Result

Output file now correctly contains:
```
double height ;
    height:standard_name = "height" ;
    height:units = "m" ;
    height:positive = "up" ;
    height:axis = "Z" ;

tas:coordinates = "height" ;   ← valid CMIP6 scalar coordinate
```

### Affected variables

Any near-surface variable whose iris-written source carries a scalar height coordinate under a name other than `height` (e.g. `height_0`): `tas`, `huss`, `uas`, `vas`, `sfcWind`.
