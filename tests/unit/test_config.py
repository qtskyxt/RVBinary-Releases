from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rvbinary.config import load_config

MINIMAL_YAML = """\
project:
  name: test-project
  output: results
  random_seed: 42
observations:
  mock_epochs: data/epochs.fits
  subsamples:
    - name: sample
      members: data/members.fits
"""


def write_config(tmp_path: Path, period_limits_days: list[float]) -> Path:
    path = tmp_path / "rvbinary.yaml"
    path.write_text(
        MINIMAL_YAML
        + "model:\n"
        + "  period_limits_days: "
        + repr(period_limits_days)
        + "\n"
    )
    return path


def test_defaults_and_grid_validation(tmp_path: Path) -> None:
    path = tmp_path / "rvbinary.yaml"
    path.write_text(MINIMAL_YAML)

    cfg = load_config(path)

    assert cfg.model.log10_period_mean == 4.8
    assert cfg.statistics.mode == "consistent"
    assert cfg.mock.error_floor.values_kms == [0.0]


def test_period_limits_must_be_positive(tmp_path: Path) -> None:
    path = write_config(tmp_path, period_limits_days=[100, -1])

    with pytest.raises(ValidationError):
        load_config(path)


def test_grids_are_sorted_and_unique(tmp_path: Path) -> None:
    raw = yaml.safe_load(MINIMAL_YAML)
    raw["model"] = {"period_limits_days": [1000, 10, 1000, 100]}
    raw["mock"] = {"error_floor": {"values_kms": [0.4, 0.0, 0.2, 0.2]}}
    raw["mcmc"] = {"credible_intervals": [0.95, 0.68, 0.95]}
    path = tmp_path / "rvbinary.yaml"
    path.write_text(yaml.safe_dump(raw))

    cfg = load_config(path)

    assert cfg.model.period_limits_days == [10.0, 100.0, 1000.0]
    assert cfg.mock.error_floor.values_kms == [0.0, 0.2, 0.4]
    assert cfg.mcmc.credible_intervals == [0.68, 0.95]


def test_paths_resolve_relative_to_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "rvbinary.yaml"
    path.write_text(
        MINIMAL_YAML.replace(
            "      members: data/members.fits",
            "      members: data/members.fits\n      epochs: data/sample-epochs.fits",
        )
    )

    cfg = load_config(path)

    assert cfg.project.output == tmp_path / "results"
    assert cfg.observations.mock_epochs == tmp_path / "data/epochs.fits"
    assert cfg.observations.subsamples[0].members == tmp_path / "data/members.fits"
    assert cfg.observations.subsamples[0].epochs == tmp_path / "data/sample-epochs.fits"


def test_unknown_nested_keys_are_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(MINIMAL_YAML)
    raw["model"] = {"unknown_setting": True}
    path = tmp_path / "rvbinary.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError, match="unknown_setting"):
        load_config(path)


def test_hdf5_mode_is_restricted(tmp_path: Path) -> None:
    raw = yaml.safe_load(MINIMAL_YAML)
    raw["execution"] = {"hdf5_mode": "serial"}
    path = tmp_path / "rvbinary.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError, match="hdf5_mode"):
        load_config(path)


def test_discrete_counts_reject_fractional_values(tmp_path: Path) -> None:
    raw = yaml.safe_load(MINIMAL_YAML)
    raw["mock"] = {"samples": 10.5}
    path = tmp_path / "rvbinary.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError, match="samples"):
        load_config(path)


def test_checked_in_example_uses_repository_synthetic_data() -> None:
    repository = Path(__file__).resolve().parents[2]
    synthetic_data = repository / "examples/synthetic_data"

    cfg = load_config(repository / "examples/rvbinary.yaml")

    observation_paths = [cfg.observations.mock_epochs]
    for subsample in cfg.observations.subsamples:
        observation_paths.append(subsample.members)
        if subsample.epochs is not None:
            observation_paths.append(subsample.epochs)
    assert all(path.is_relative_to(synthetic_data) for path in observation_paths)


def test_precomputed_binary2_mock_directory_resolves_relative_to_config(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(MINIMAL_YAML)
    raw["mock"] = {
        "samples": 10000,
        "precomputed": {
            "format": "binary2",
            "directory": "legacy-mocks",
            "error_floor_mode": "per_star_baked_in",
        },
    }
    path = tmp_path / "rvbinary.yaml"
    path.write_text(yaml.safe_dump(raw))

    cfg = load_config(path)

    assert cfg.mock.precomputed is not None
    assert cfg.mock.precomputed.directory == tmp_path / "legacy-mocks"
    assert cfg.mock.precomputed.error_floor_mode == "per_star_baked_in"


def test_precomputed_binary2_mock_rejects_period_and_errorfloor_grids(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(MINIMAL_YAML)
    raw["model"] = {"include_unrestricted": True, "period_limits_days": [100.0]}
    raw["mock"] = {
        "samples": 10000,
        "error_floor": {"values_kms": [0.0, 0.4]},
        "precomputed": {
            "format": "binary2",
            "directory": "legacy-mocks",
        },
    }
    path = tmp_path / "rvbinary.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError, match="precomputed Binary2"):
        load_config(path)
