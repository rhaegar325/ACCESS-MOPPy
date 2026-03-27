"""Unit tests for batch CMORiser functionality."""

# Security: All subprocess usage in this file is for mocking in unit tests
# ruff: noqa: S603, S607
# bandit: skip
# semgrep: skip

from unittest.mock import Mock, mock_open, patch

import pytest

from access_moppy.batch_cmoriser import (
    create_job_script,
    get_variables_for_table,
    resolve_variables,
    submit_job,
)
from tests.mocks.mock_pbs import MockPBSManager, mock_qsub_success


class TestBatchCmoriser:
    """Unit tests for batch processing functions."""

    @patch("jinja2.Template")
    @patch("access_moppy.batch_cmoriser.files")
    @patch("os.chmod")
    @pytest.mark.unit
    def test_create_job_script(self, mock_chmod, mock_files, mock_template, temp_dir):
        """Test job script creation."""
        # Mock template files
        mock_file_obj = Mock()
        mock_file_obj.read.return_value = "mock template"
        mock_files.return_value.joinpath.return_value.open.return_value.__enter__.return_value = mock_file_obj

        # Mock template rendering
        mock_template_instance = Mock()
        mock_template_instance.render.return_value = "rendered script"
        mock_template.return_value = mock_template_instance

        config = {
            "cpus_per_node": 4,
            "mem": "16GB",
            "walltime": "01:00:00",
            "experiment_id": "historical",
        }

        with patch("builtins.open", mock_open()) as mock_file:
            result = create_job_script("Amon.tas", config, "/db/path", temp_dir)

        # Verify script was created
        expected_path = temp_dir / "cmor_Amon_tas.sh"
        assert result == expected_path
        mock_file.assert_called()
        mock_chmod.assert_called()

    @patch("subprocess.run")
    @pytest.mark.unit
    def test_submit_job_success(self, mock_run):
        """Test successful job submission."""
        mock_run.return_value = mock_qsub_success()

        job_id = submit_job("/path/to/script.sh")

        assert job_id is not None
        assert len(job_id) > 0
        mock_run.assert_called_once()

    @patch("subprocess.run")
    @pytest.mark.unit
    def test_submit_job_failure(self, mock_run):
        """Test failed job submission."""
        import subprocess  # nosec  # Only used for mocking CalledProcessError in tests

        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["qsub", "/path/to/script.sh"],
            stderr="qsub: job rejected by server",
        )

        job_id = submit_job("/path/to/script.sh")

        assert job_id is None

    @pytest.mark.unit
    def test_mock_pbs_manager(self):
        """Test the MockPBSManager functionality."""
        with MockPBSManager() as pbs:
            # Submit a mock job
            job_id = submit_job("/mock/script.sh")

            assert job_id is not None

            # Extract the numeric part of the job ID (remove .gadi-pbs suffix)
            job_id_key = job_id.split(".")[0] if "." in job_id else job_id

            # Test job state changes
            pbs.mark_job_running(job_id_key)
            pbs.mark_job_completed(job_id_key)

            # Verify job is tracked
            assert job_id_key in pbs.submitted_jobs
            assert pbs.submitted_jobs[job_id_key]["status"] == "C"


class TestTableExpansion:
    """Unit tests for table-based variable expansion."""

    @pytest.mark.unit
    def test_get_variables_for_table_amon(self):
        """Test expanding the Amon table returns known atmosphere variables."""
        result = get_variables_for_table("Amon", model_id="ACCESS-ESM1.6")
        # Should return compound names in the form Amon.<var>
        assert len(result) > 0
        assert all(v.startswith("Amon.") for v in result)
        # Known atmosphere variables that should be in both Amon table and mappings
        assert "Amon.tas" in result
        assert "Amon.pr" in result
        assert "Amon.hfss" in result

    @pytest.mark.unit
    def test_get_variables_for_table_omon(self):
        """Test expanding the Omon table returns known ocean variables."""
        result = get_variables_for_table("Omon", model_id="ACCESS-ESM1.6")
        assert len(result) > 0
        assert all(v.startswith("Omon.") for v in result)
        assert "Omon.tos" in result
        assert "Omon.so" in result

    @pytest.mark.unit
    def test_get_variables_for_table_filters_unsupported(self):
        """Test that variables without model mappings are excluded."""
        result = get_variables_for_table("Amon", model_id="ACCESS-ESM1.6")
        var_names = [v.split(".")[1] for v in result]
        # co2mass is in CMIP6_Amon.json but not in ACCESS-ESM1.6 mappings
        assert "co2mass" not in var_names

    @pytest.mark.unit
    def test_get_variables_for_table_sorted(self):
        """Test that results are sorted alphabetically."""
        result = get_variables_for_table("Amon", model_id="ACCESS-ESM1.6")
        assert result == sorted(result)

    @pytest.mark.unit
    def test_get_variables_for_table_invalid_model(self):
        """Test that a missing model mapping file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            get_variables_for_table("Amon", model_id="NONEXISTENT-MODEL")

    @pytest.mark.unit
    def test_resolve_variables_explicit_only(self):
        """Test resolve_variables with only explicit variables."""
        config = {"variables": ["Amon.pr", "Omon.tos"]}
        variables, model_map = resolve_variables(config)
        assert variables == ["Amon.pr", "Omon.tos"]
        assert model_map == {}

    @pytest.mark.unit
    def test_resolve_variables_tables_list(self):
        """Test resolve_variables with tables as a list (backward compatible)."""
        config = {"tables": ["Amon"], "model_id": "ACCESS-ESM1.6"}
        variables, model_map = resolve_variables(config)
        assert len(variables) > 0
        assert all(v.startswith("Amon.") for v in variables)
        assert "Amon.tas" in variables
        # List form uses global model_id, so no per-variable overrides
        assert model_map == {}

    @pytest.mark.unit
    def test_resolve_variables_tables_dict(self):
        """Test resolve_variables with tables as a dict (per-table model_id)."""
        config = {"tables": {"Amon": "ACCESS-ESM1.6"}}
        variables, model_map = resolve_variables(config)
        assert len(variables) > 0
        assert "Amon.tas" in variables
        # Dict form records per-variable model_id overrides
        assert model_map.get("Amon.tas") == "ACCESS-ESM1.6"
        assert all(model_map[v] == "ACCESS-ESM1.6" for v in variables)

    @pytest.mark.unit
    def test_resolve_variables_tables_dict_multiple(self):
        """Test resolve_variables with multiple tables using different models."""
        config = {
            "tables": {
                "Amon": "ACCESS-ESM1.6",
                "Omon": "ACCESS-ESM1.6",
            }
        }
        variables, model_map = resolve_variables(config)
        amon_vars = [v for v in variables if v.startswith("Amon.")]
        omon_vars = [v for v in variables if v.startswith("Omon.")]
        assert len(amon_vars) > 0
        assert len(omon_vars) > 0
        # Each group uses its specified model_id
        for v in amon_vars:
            assert model_map[v] == "ACCESS-ESM1.6"
        for v in omon_vars:
            assert model_map[v] == "ACCESS-ESM1.6"

    @pytest.mark.unit
    def test_resolve_variables_combined(self):
        """Test resolve_variables with both variables and tables."""
        config = {
            "variables": ["Omon.tos"],
            "tables": ["Amon"],
            "model_id": "ACCESS-ESM1.6",
        }
        variables, model_map = resolve_variables(config)
        assert "Omon.tos" in variables
        assert "Amon.tas" in variables

    @pytest.mark.unit
    def test_resolve_variables_deduplicates(self):
        """Test that duplicates from tables and explicit variables are removed."""
        config = {
            "variables": ["Amon.tas", "Amon.pr"],
            "tables": ["Amon"],
            "model_id": "ACCESS-ESM1.6",
        }
        variables, _ = resolve_variables(config)
        # No duplicates
        assert len(variables) == len(set(variables))
        # Both should still appear
        assert "Amon.tas" in variables
        assert "Amon.pr" in variables

    @pytest.mark.unit
    def test_resolve_variables_neither_specified(self):
        """Test that missing both variables and tables raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            resolve_variables({})

    @pytest.mark.unit
    def test_resolve_variables_empty_lists(self):
        """Test that empty variables and tables raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            resolve_variables({"variables": [], "tables": []})

    @pytest.mark.unit
    def test_resolve_variables_variable_resources_model_id(self):
        """Test that variable_resources can override model_id per variable."""
        config = {
            "variables": ["Omon.tos", "Amon.pr"],
            "variable_resources": {
                "Omon.tos": {"model_id": "ACCESS-OM3", "mem": "64GB"},
            },
        }
        variables, model_map = resolve_variables(config)
        assert model_map.get("Omon.tos") == "ACCESS-OM3"
        # Amon.pr has no override
        assert "Amon.pr" not in model_map

    @pytest.mark.unit
    def test_resolve_variables_variable_resources_overrides_table(self):
        """Test that variable_resources model_id takes precedence over table."""
        config = {
            "tables": {"Amon": "ACCESS-ESM1.6"},
            "variable_resources": {
                "Amon.tas": {"model_id": "ACCESS-CM3"},
            },
        }
        variables, model_map = resolve_variables(config)
        # variable_resources override should win for Amon.tas
        assert model_map["Amon.tas"] == "ACCESS-CM3"
        # Other Amon vars keep the table-level model_id
        other_amon = [v for v in variables if v.startswith("Amon.") and v != "Amon.tas"]
        assert len(other_amon) > 0
        for v in other_amon:
            assert model_map[v] == "ACCESS-ESM1.6"
