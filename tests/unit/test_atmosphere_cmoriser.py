"""
Unit tests for Atmosphere_CMORiser.

Focus on covering:
- select_and_process_variables: formula path that changes time resolution
  (lines 184-207 in atmosphere.py)
- update_attributes: skip astype() for decoded (cftime / datetime64) time coords
  (lines 344-350 in atmosphere.py)
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch

from access_moppy.atmosphere import Atmosphere_CMORiser
from access_moppy.base import CMORiser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cmoriser_for_update_attributes(time_values, time_attrs=None):
    """
    Build a minimal Atmosphere_CMORiser to exercise update_attributes().

    Only the attributes accessed inside update_attributes are populated.
    """
    cmoriser = object.__new__(Atmosphere_CMORiser)
    cmoriser.cmor_name = "tasmax"
    cmoriser.type_mapping = CMORiser.type_mapping

    if time_attrs is None:
        time_attrs = {"units": "days since 1850-01-01", "calendar": "standard"}

    ds = xr.Dataset(
        {
            "tasmax": xr.DataArray(
                np.array([310.0]),
                dims=["time"],
                coords={"time": (["time"], time_values, time_attrs)},
                attrs={"standard_name": "air_temperature", "units": "K"},
            )
        }
    )
    cmoriser.ds = ds

    vocab = MagicMock()
    vocab.get_required_global_attributes.return_value = {
        "source_id": "TEST",
        "experiment_id": "historical",
    }
    vocab.axes = {
        "time": {
            "out_name": "time",
            "standard_name": "time",
            "units": "days since 1850-01-01",
            "type": "double",
            "axis": "T",
        }
    }
    vocab.variable = {
        "units": "K",
        "standard_name": "air_temperature",
        "type": "real",
    }
    cmoriser.vocab = vocab

    # Stub out methods that are not under test here
    cmoriser._check_units = MagicMock()
    cmoriser._check_calendar = MagicMock()
    cmoriser._check_range = MagicMock()

    return cmoriser


def _make_cmoriser_for_formula(daily_ds):
    """
    Build a minimal Atmosphere_CMORiser to exercise the formula path
    inside select_and_process_variables().
    """
    cmoriser = object.__new__(Atmosphere_CMORiser)
    cmoriser.cmor_name = "tasmax"
    cmoriser.type_mapping = CMORiser.type_mapping
    cmoriser.ds = daily_ds

    cmoriser.mapping = {
        "tasmax": {
            "calculation": {
                "type": "formula",
                "formula": "calculate_monthly_maximum(tasmax)",
            },
            "model_variables": ["tasmax"],
        }
    }

    vocab = MagicMock()
    vocab._get_axes.return_value = ({}, {})
    vocab._get_required_bounds_variables.return_value = ({}, {})
    vocab.variable = {"dimensions": "time"}
    vocab.axes = {"time": {"out_name": "time"}}
    cmoriser.vocab = vocab

    cmoriser.load_dataset = MagicMock()
    cmoriser.sort_time_dimension = MagicMock()
    cmoriser.remove_spurious_time_dimensions = MagicMock()

    return cmoriser


# ---------------------------------------------------------------------------
# Tests: update_attributes – decoded-time astype skip
# ---------------------------------------------------------------------------

class TestUpdateAttributesDecodedTime:
    """
    Cover the branch at atmosphere.py lines 344-350 that skips astype()
    when the time coordinate is already decoded (cftime or datetime64).
    """

    def test_cftime_time_not_cast_to_float(self):
        """cftime (dtype=object) time must not be cast to float64."""
        cf_time = xr.cftime_range(
            "2020-01-31", periods=1, freq="ME", calendar="gregorian"
        )
        cmoriser = _make_cmoriser_for_update_attributes(cf_time)

        assert cmoriser.ds["time"].dtype == object

        cmoriser.update_attributes()

        # dtype must remain object (cftime), not float64
        assert cmoriser.ds["time"].dtype == object

    def test_datetime64_time_not_cast_to_float(self):
        """numpy datetime64 time must not be cast to float64."""
        dt_time = pd.date_range("2020-01-31", periods=1, freq="ME")
        cmoriser = _make_cmoriser_for_update_attributes(dt_time)

        assert np.issubdtype(cmoriser.ds["time"].dtype, np.datetime64)

        cmoriser.update_attributes()

        # dtype must remain datetime64
        assert np.issubdtype(cmoriser.ds["time"].dtype, np.datetime64)

    def test_numeric_time_is_cast_to_float(self):
        """Numeric (float64) time IS cast according to the type mapping."""
        num_time = np.array([0.0, 31.0], dtype=np.float64)

        # Build dataset with 2 time steps so the DataArray has numeric time
        ds = xr.Dataset(
            {
                "tasmax": xr.DataArray(
                    np.array([310.0, 311.0]),
                    dims=["time"],
                    coords={
                        "time": xr.Variable(
                            "time",
                            num_time,
                            attrs={
                                "units": "days since 1850-01-01",
                                "calendar": "standard",
                            },
                        )
                    },
                    attrs={"units": "K"},
                )
            }
        )
        cmoriser = object.__new__(Atmosphere_CMORiser)
        cmoriser.cmor_name = "tasmax"
        cmoriser.type_mapping = CMORiser.type_mapping
        cmoriser.ds = ds

        vocab = MagicMock()
        vocab.get_required_global_attributes.return_value = {}
        vocab.axes = {
            "time": {
                "out_name": "time",
                "standard_name": "time",
                "units": "days since 1850-01-01",
                "type": "double",
                "axis": "T",
            }
        }
        vocab.variable = {"units": "K", "type": "real"}
        cmoriser.vocab = vocab
        cmoriser._check_units = MagicMock()
        cmoriser._check_calendar = MagicMock()
        cmoriser._check_range = MagicMock()

        cmoriser.update_attributes()

        # Numeric time keeps float64 (the type_mapping for "double" is float64)
        assert np.issubdtype(cmoriser.ds["time"].dtype, np.floating)


# ---------------------------------------------------------------------------
# Tests: select_and_process_variables – time resolution change path
# ---------------------------------------------------------------------------

class TestSelectAndProcessVariablesTimeResolutionChange:
    """
    Cover atmosphere.py lines 184-207: when a formula reduces the number of
    time steps (e.g. daily → monthly) the dataset must be rebuilt from the
    result, not merged via __setitem__ (which would silently reindex).
    """

    def _make_daily_ds(self):
        daily_time = pd.date_range("2020-01-01", periods=31, freq="D")
        ds = xr.Dataset(
            {
                "tasmax": xr.DataArray(
                    np.random.default_rng(0).normal(305, 5, 31),
                    dims=["time"],
                    coords={"time": daily_time},
                    attrs={"units": "K"},
                ),
                # time-independent auxiliary variable
                "lat_bnds": xr.DataArray(
                    np.array([[0.0, 1.0]]), dims=["lat", "bnds"]
                ),
            }
        )
        ds["time"].attrs = {
            "units": "days since 1850-01-01",
            "calendar": "standard",
        }
        return ds

    def _monthly_result(self):
        monthly_time = pd.date_range("2020-01-31", periods=1, freq="ME")
        return xr.DataArray(
            np.array([315.0]),
            dims=["time"],
            coords={"time": monthly_time},
        )

    def test_formula_rebuilds_dataset_when_time_shrinks(self):
        """Output has monthly time length (1), not daily (31)."""
        daily_ds = self._make_daily_ds()
        monthly_result = self._monthly_result()

        cmoriser = _make_cmoriser_for_formula(daily_ds)

        with patch(
            "access_moppy.atmosphere.evaluate_expression",
            return_value=monthly_result,
        ):
            cmoriser.select_and_process_variables()

        assert "tasmax" in cmoriser.ds
        assert cmoriser.ds["tasmax"].sizes["time"] == 1

    def test_formula_preserves_time_independent_vars(self):
        """Time-independent variables (lat_bnds) survive the dataset rebuild."""
        daily_ds = self._make_daily_ds()
        monthly_result = self._monthly_result()

        cmoriser = _make_cmoriser_for_formula(daily_ds)

        with patch(
            "access_moppy.atmosphere.evaluate_expression",
            return_value=monthly_result,
        ):
            cmoriser.select_and_process_variables()

        assert "lat_bnds" in cmoriser.ds

    def test_formula_restores_original_time_attrs(self):
        """Original time attrs (units/calendar) are restored on the new time coord."""
        daily_ds = self._make_daily_ds()
        monthly_result = self._monthly_result()

        cmoriser = _make_cmoriser_for_formula(daily_ds)

        with patch(
            "access_moppy.atmosphere.evaluate_expression",
            return_value=monthly_result,
        ):
            cmoriser.select_and_process_variables()

        assert cmoriser.ds["time"].attrs.get("units") == "days since 1850-01-01"
        assert cmoriser.ds["time"].attrs.get("calendar") == "standard"

    def test_formula_same_time_length_uses_setitem(self):
        """When formula returns same number of time steps, __setitem__ path is used."""
        monthly_time = pd.date_range("2020-01-01", periods=12, freq="MS")
        monthly_ds = xr.Dataset(
            {
                "tasmax": xr.DataArray(
                    np.random.default_rng(3).normal(305, 5, 12),
                    dims=["time"],
                    coords={"time": monthly_time},
                    attrs={"units": "K"},
                )
            }
        )
        monthly_ds["time"].attrs = {"units": "days since 1850-01-01"}

        same_result = xr.DataArray(
            np.random.default_rng(4).normal(305, 5, 12),
            dims=["time"],
            coords={"time": monthly_time},
        )

        cmoriser = _make_cmoriser_for_formula(monthly_ds)

        with patch(
            "access_moppy.atmosphere.evaluate_expression",
            return_value=same_result,
        ):
            cmoriser.select_and_process_variables()

        # Same 12 time steps preserved
        assert cmoriser.ds["tasmax"].sizes["time"] == 12
