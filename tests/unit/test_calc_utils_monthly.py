"""
Unit tests for calculate_monthly_minimum and calculate_monthly_maximum
in access_moppy.derivations.calc_utils.

Focus on covering:
- Numeric float64 time decoding branch (lines 399-401, 479-480)
- cftime (dtype=object) time -- should skip decode branch
- datetime64 time -- should skip decode branch
- cell_methods appending behaviour
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from access_moppy.derivations.calc_utils import (
    calculate_monthly_maximum,
    calculate_monthly_minimum,
)


def _daily_da_numeric_time():
    """DataArray with raw numeric float64 time (as if loaded with decode_cf=False)."""
    time_values = np.arange(31, dtype=np.float64)  # Jan 2020: days 0–30
    return xr.DataArray(
        np.random.default_rng(0).normal(300, 5, 31),
        dims=["time"],
        coords={
            "time": xr.Variable(
                "time",
                time_values,
                attrs={"units": "days since 2020-01-01", "calendar": "standard"},
            )
        },
        attrs={"standard_name": "air_temperature", "units": "K"},
    )


def _daily_da_datetime64_time():
    """DataArray with decoded numpy datetime64 time."""
    time = pd.date_range("2020-01-01", periods=31, freq="D")
    return xr.DataArray(
        np.random.default_rng(1).normal(300, 5, 31),
        dims=["time"],
        coords={"time": time},
        attrs={"standard_name": "air_temperature", "units": "K"},
    )


def _daily_da_cftime_time():
    """DataArray with cftime objects (dtype=object, no decode needed)."""
    time = xr.cftime_range("2020-01-01", periods=31, freq="D", calendar="noleap")
    return xr.DataArray(
        np.random.default_rng(2).normal(300, 5, 31),
        dims=["time"],
        coords={"time": time},
        attrs={"standard_name": "air_temperature", "units": "K"},
    )


class TestCalculateMonthlyMinimumDecodeBranch:
    """Tests that numeric time triggers xr.decode_cf before resampling."""

    def test_numeric_time_decoded_and_resampled_to_monthly(self):
        da = _daily_da_numeric_time()
        assert da["time"].dtype == np.float64  # Confirm precondition

        result = calculate_monthly_minimum(da)

        assert len(result) == 1  # 31 days → 1 month
        assert "time: minimum" in result.attrs.get("cell_methods", "")

    def test_datetime64_time_skips_decode(self):
        da = _daily_da_datetime64_time()
        assert np.issubdtype(da["time"].dtype, np.datetime64)  # Confirm precondition

        result = calculate_monthly_minimum(da)

        assert len(result) == 1

    def test_cftime_time_skips_decode(self):
        da = _daily_da_cftime_time()
        assert da["time"].dtype == object  # cftime → dtype=object

        result = calculate_monthly_minimum(da)

        assert len(result) == 1

    def test_existing_cell_methods_are_appended(self):
        da = _daily_da_datetime64_time()
        da.attrs["cell_methods"] = "area: mean"

        result = calculate_monthly_minimum(da)

        assert "area: mean" in result.attrs["cell_methods"]
        assert "time: minimum" in result.attrs["cell_methods"]

    def test_preserve_attrs_false_omits_cell_methods(self):
        da = _daily_da_datetime64_time()

        result = calculate_monthly_minimum(da, preserve_attrs=False)

        assert "cell_methods" not in result.attrs


class TestCalculateMonthlyMaximumDecodeBranch:
    """Tests that numeric time triggers xr.decode_cf before resampling."""

    def test_numeric_time_decoded_and_resampled_to_monthly(self):
        da = _daily_da_numeric_time()
        assert da["time"].dtype == np.float64  # Confirm precondition

        result = calculate_monthly_maximum(da)

        assert len(result) == 1
        assert "time: maximum" in result.attrs.get("cell_methods", "")

    def test_datetime64_time_skips_decode(self):
        da = _daily_da_datetime64_time()

        result = calculate_monthly_maximum(da)

        assert len(result) == 1

    def test_cftime_time_skips_decode(self):
        da = _daily_da_cftime_time()

        result = calculate_monthly_maximum(da)

        assert len(result) == 1

    def test_existing_cell_methods_are_appended(self):
        da = _daily_da_datetime64_time()
        da.attrs["cell_methods"] = "area: mean"

        result = calculate_monthly_maximum(da)

        assert "area: mean" in result.attrs["cell_methods"]
        assert "time: maximum" in result.attrs["cell_methods"]

    def test_preserve_attrs_false_omits_cell_methods(self):
        da = _daily_da_datetime64_time()

        result = calculate_monthly_maximum(da, preserve_attrs=False)

        assert "cell_methods" not in result.attrs

    def test_missing_time_dim_raises_value_error(self):
        da = xr.DataArray([1.0, 2.0, 3.0], dims=["lat"])

        with pytest.raises(ValueError, match="Time dimension 'time' not found"):
            calculate_monthly_maximum(da)

    def test_missing_time_coord_raises_value_error(self):
        # Dimension named 'time' but not as a coordinate
        da = xr.DataArray(np.ones(5), dims=["time"])
        da = da.drop_vars("time", errors="ignore")

        with pytest.raises(ValueError, match="Time coordinate 'time' not found"):
            calculate_monthly_maximum(da)
