"""
Tests for to_iris() conversion from xarray Dataset to Iris Cube.

Verifies that ocean data with curvilinear grids is correctly converted:
- Single cube returned (not 3 separate cubes for data, lat, lon)
- latitude/longitude are auxiliary coordinates on the cube
- Fill values (1e20) are properly masked
- Coordinate bounds are preserved
"""

import numpy as np
import pytest
import xarray as xr

iris = pytest.importorskip("iris")
ncdata = pytest.importorskip("ncdata")


def _make_ocean_dataset(cmor_name="tos", nt=3, nj=4, ni=5, with_fill=True, with_bounds=True):
    """
    Build a minimal xarray Dataset that mimics the state after
    CMIP6_Ocean_CMORiser.run() for a curvilinear ocean variable.

    The dataset has:
    - cmor_name as data var on (time, j, i)
    - latitude/longitude as 2D data vars on (j, i)  [how ocean.py sets them]
    - vertices_latitude/vertices_longitude as 3D data vars on (j, i, vertices)
    - time_bnds as a data var
    - i, j, vertices, time as coordinates
    """
    np.random.seed(42)

    time_vals = np.arange(nt, dtype=np.float64)
    i_vals = np.arange(ni, dtype=np.float64)
    j_vals = np.arange(nj, dtype=np.float64)
    vertices_vals = np.arange(4, dtype=np.float64)

    # 2D lat/lon (curvilinear grid)
    lon_2d, lat_2d = np.meshgrid(
        np.linspace(0, 360, ni), np.linspace(-90, 90, nj)
    )

    # Main variable data
    data = np.random.uniform(270, 310, (nt, nj, ni)).astype(np.float64)
    if with_fill:
        # Set some cells to the CMIP6 fill value
        data[0, 0, :2] = 1e20
        data[1, -1, -1] = 1e20

    # Bounds (4 vertices per grid cell)
    lat_bnds = np.zeros((nj, ni, 4))
    lon_bnds = np.zeros((nj, ni, 4))
    dlat = 180.0 / nj / 2
    dlon = 360.0 / ni / 2
    for v, (dy, dx) in enumerate([(-dlat, -dlon), (-dlat, dlon), (dlat, dlon), (dlat, -dlon)]):
        lat_bnds[:, :, v] = lat_2d + dy
        lon_bnds[:, :, v] = lon_2d + dx

    # Time bounds
    time_bnds = np.column_stack([time_vals - 0.5, time_vals + 0.5])

    ds = xr.Dataset(
        data_vars={
            cmor_name: (
                ["time", "j", "i"],
                data,
                {
                    "standard_name": "sea_surface_temperature",
                    "units": "K",
                    "long_name": "Sea Surface Temperature",
                    "_FillValue": 1e20,
                    "missing_value": 1e20,
                },
            ),
            "time_bnds": (["time", "bnds"], time_bnds),
        },
        coords={
            "time": (
                "time",
                time_vals,
                {
                    "units": "days since 1850-01-01",
                    "calendar": "proleptic_gregorian",
                    "bounds": "time_bnds",
                },
            ),
            "i": ("i", i_vals),
            "j": ("j", j_vals),
            "vertices": ("vertices", vertices_vals),
            "bnds": ("bnds", np.arange(2, dtype=np.float64)),
        },
    )

    # Add latitude/longitude as DATA VARIABLES (this is how ocean.py does it)
    ds["latitude"] = xr.DataArray(
        lat_2d,
        dims=("j", "i"),
        attrs={
            "standard_name": "latitude",
            "units": "degrees_north",
            "bounds": "vertices_latitude",
        },
    )
    ds["longitude"] = xr.DataArray(
        lon_2d,
        dims=("j", "i"),
        attrs={
            "standard_name": "longitude",
            "units": "degrees_east",
            "bounds": "vertices_longitude",
        },
    )

    if with_bounds:
        ds["vertices_latitude"] = xr.DataArray(
            lat_bnds,
            dims=("j", "i", "vertices"),
            attrs={"standard_name": "latitude", "units": "degrees_north"},
        )
        ds["vertices_longitude"] = xr.DataArray(
            lon_bnds,
            dims=("j", "i", "vertices"),
            attrs={"standard_name": "longitude", "units": "degrees_east"},
        )

    # Global attrs (minimal CMIP6 set)
    ds.attrs = {
        "variable_id": cmor_name,
        "table_id": "Omon",
        "source_id": "ACCESS-OM2",
        "experiment_id": "historical",
        "variant_label": "r1i1p1f1",
        "grid_label": "gn",
    }

    return ds


def _make_atmos_dataset(cmor_name="tas", nt=3, nlat=5, nlon=10):
    """
    Build a minimal atmosphere dataset (regular lat/lon grid).
    lat/lon are dimension coordinates, not 2D data vars.
    """
    np.random.seed(42)

    time_vals = np.arange(nt, dtype=np.float64)
    lat_vals = np.linspace(-90, 90, nlat)
    lon_vals = np.linspace(0, 360, nlon, endpoint=False)
    data = np.random.uniform(250, 310, (nt, nlat, nlon)).astype(np.float64)

    ds = xr.Dataset(
        data_vars={
            cmor_name: (
                ["time", "lat", "lon"],
                data,
                {
                    "standard_name": "air_temperature",
                    "units": "K",
                    "long_name": "Near-Surface Air Temperature",
                },
            ),
        },
        coords={
            "time": (
                "time",
                time_vals,
                {
                    "units": "days since 1850-01-01",
                    "calendar": "proleptic_gregorian",
                },
            ),
            "lat": (
                "lat",
                lat_vals,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                "lon",
                lon_vals,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        },
        attrs={
            "variable_id": cmor_name,
            "table_id": "Amon",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "grid_label": "gn",
        },
    )

    return ds


class _FakeCMORiser:
    """Minimal stand-in for the cmoriser attribute of ACCESS_ESM_CMORiser."""

    def __init__(self, ds, cmor_name):
        self.ds = ds
        self.cmor_name = cmor_name


class TestToIrisOcean:
    """Tests for to_iris() with curvilinear ocean data."""

    def _make_driver(self, ds, cmor_name="tos"):
        """Create a minimal driver-like object with to_iris bound to it."""
        from access_moppy.driver import ACCESS_ESM_CMORiser

        # Bypass __init__ and wire up only what to_iris needs
        obj = object.__new__(ACCESS_ESM_CMORiser)
        obj.cmoriser = _FakeCMORiser(ds, cmor_name)
        return obj

    def test_returns_single_cube(self):
        """to_iris() should return one Cube, not a CubeList with lat/lon cubes."""
        ds = _make_ocean_dataset()
        driver = self._make_driver(ds)
        result = driver.to_iris()

        assert isinstance(result, iris.cube.Cube)
        assert result.var_name == "tos"

    def test_lat_lon_are_aux_coords(self):
        """latitude and longitude should be AuxCoords on the cube."""
        ds = _make_ocean_dataset()
        driver = self._make_driver(ds)
        cube = driver.to_iris()

        coord_names = [c.standard_name for c in cube.aux_coords]
        assert "latitude" in coord_names
        assert "longitude" in coord_names

    def test_lat_lon_units(self):
        """Aux coord units should be degree-based (cf-units normalises to 'degrees')."""
        ds = _make_ocean_dataset()
        driver = self._make_driver(ds)
        cube = driver.to_iris()

        lat = cube.coord("latitude")
        lon = cube.coord("longitude")
        # cf-units normalises degrees_north/degrees_east to "degrees"
        assert "degree" in str(lat.units).lower()
        assert "degree" in str(lon.units).lower()

    def test_lat_lon_are_2d(self):
        """Auxiliary lat/lon coords should be 2-dimensional (j, i)."""
        ds = _make_ocean_dataset(nj=4, ni=5)
        driver = self._make_driver(ds)
        cube = driver.to_iris()

        lat = cube.coord("latitude")
        lon = cube.coord("longitude")
        assert lat.ndim == 2
        assert lon.ndim == 2
        assert lat.shape == (4, 5)
        assert lon.shape == (4, 5)

    def test_fill_values_masked(self):
        """Cells with 1e20 fill values should be masked, not present as data."""
        ds = _make_ocean_dataset(with_fill=True)
        driver = self._make_driver(ds)
        cube = driver.to_iris()

        data = cube.data
        assert np.ma.is_masked(data), "Data should be a masked array"
        # The original had 1e20 in data[0,0,:2] and data[1,-1,-1]
        assert data.mask[0, 0, 0], "Cell [0,0,0] should be masked"
        assert data.mask[0, 0, 1], "Cell [0,0,1] should be masked"
        assert data.mask[1, -1, -1], "Cell [1,-1,-1] should be masked"
        # A normal cell should not be masked
        assert not data.mask[0, 1, 2], "Normal cell should not be masked"

    def test_no_1e20_in_data(self):
        """There should be no 1e20 values in the unmasked data."""
        ds = _make_ocean_dataset(with_fill=True)
        driver = self._make_driver(ds)
        cube = driver.to_iris()

        data = cube.data
        # compressed() gives only the non-masked values
        assert not np.any(
            np.isclose(data.compressed(), 1e20)
        ), "No unmasked cells should contain 1e20"

    def test_data_without_fill_values(self):
        """When no fill values present, data should still be valid."""
        ds = _make_ocean_dataset(with_fill=False)
        driver = self._make_driver(ds)
        cube = driver.to_iris()

        data = cube.data
        # Should have real temperature values
        assert np.all((data > 200) & (data < 400))

    def test_bounds_present(self):
        """Lat/lon aux coords should have bounds when vertices are available."""
        ds = _make_ocean_dataset(with_bounds=True)
        driver = self._make_driver(ds)
        cube = driver.to_iris()

        lat = cube.coord("latitude")
        lon = cube.coord("longitude")
        assert lat.has_bounds(), "Latitude should have bounds"
        assert lon.has_bounds(), "Longitude should have bounds"
        # Bounds shape: (nj, ni, 4) for 4-vertex ocean cells
        assert lat.bounds.shape == (4, 5, 4)
        assert lon.bounds.shape == (4, 5, 4)

    def test_no_bounds_when_vertices_missing(self):
        """When vertices are not in the dataset, coords should still work without bounds."""
        ds = _make_ocean_dataset(with_bounds=False)
        driver = self._make_driver(ds)
        cube = driver.to_iris()

        lat = cube.coord("latitude")
        lon = cube.coord("longitude")
        assert not lat.has_bounds()
        assert not lon.has_bounds()

    def test_time_bnds_not_separate_cube(self):
        """time_bnds should not appear as a separate cube."""
        ds = _make_ocean_dataset()
        driver = self._make_driver(ds)
        result = driver.to_iris()
        # Result is a single Cube, not a CubeList -- time_bnds isn't a cube
        assert isinstance(result, iris.cube.Cube)


class TestToIrisAtmosphere:
    """Tests for to_iris() with regular-grid atmosphere data."""

    def _make_driver(self, ds, cmor_name="tas"):
        from access_moppy.driver import ACCESS_ESM_CMORiser

        obj = object.__new__(ACCESS_ESM_CMORiser)
        obj.cmoriser = _FakeCMORiser(ds, cmor_name)
        return obj

    def test_atmos_returns_single_cube(self):
        """Atmosphere data should also return a single Cube."""
        ds = _make_atmos_dataset()
        driver = self._make_driver(ds)
        result = driver.to_iris()

        assert isinstance(result, iris.cube.Cube)
        assert result.var_name == "tas"

    def test_atmos_has_dim_coords(self):
        """Atmosphere cube should have lat/lon as dimension coordinates."""
        ds = _make_atmos_dataset()
        driver = self._make_driver(ds)
        cube = driver.to_iris()

        dim_coord_names = [c.standard_name for c in cube.dim_coords]
        assert "latitude" in dim_coord_names
        assert "longitude" in dim_coord_names


class TestToIrisEdgeCases:
    """Edge case tests for to_iris()."""

    def _make_driver(self, ds, cmor_name):
        from access_moppy.driver import ACCESS_ESM_CMORiser

        obj = object.__new__(ACCESS_ESM_CMORiser)
        obj.cmoriser = _FakeCMORiser(ds, cmor_name)
        return obj

    def test_raises_on_missing_variable(self):
        """Should raise ValueError when cmor_name not found in converted cubes."""
        ds = _make_ocean_dataset(cmor_name="tos")
        driver = self._make_driver(ds, cmor_name="nonexistent_var")
        with pytest.raises(ValueError, match="Could not find cube"):
            driver.to_iris()

    def test_original_dataset_unmodified(self):
        """to_iris() should not modify the original dataset on the cmoriser."""
        ds = _make_ocean_dataset(with_fill=True)
        driver = self._make_driver(ds, cmor_name="tos")

        # Snapshot original state
        orig_data_vars = set(ds.data_vars)
        orig_fill = ds["tos"].attrs.get("_FillValue")

        driver.to_iris()

        # Dataset should be unchanged
        assert set(driver.cmoriser.ds.data_vars) == orig_data_vars
        assert driver.cmoriser.ds["tos"].attrs.get("_FillValue") == orig_fill
        assert "latitude" in driver.cmoriser.ds.data_vars  # still a data var
