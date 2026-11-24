from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.image as mpimg
import numpy as np
import pytest
from astropy.table import Table

from rvbinary.config import RVBinaryConfig
from rvbinary.workflow.runner import run_pipeline
from tests.integration.test_pdf_grid import make_precomputed_config


def make_config(tmp_path: Path) -> RVBinaryConfig:
    epochs = tmp_path / "epochs.fits"
    members = tmp_path / "members.fits"
    ids = np.repeat(np.array([11, 22], dtype=np.int64), 3)
    Table(
        {
            "TARGETID": ids,
            "MJD": np.tile([59000.0, 59002.0, 59007.0], 2),
            "VRAD": np.array([0.0, 0.2, -0.1, 0.0, 0.3, -0.2]),
            "VRAD_ERR": np.tile([0.5, 0.7, 0.9], 2),
            "Mass": np.repeat([0.9, 1.1], 3),
            "meanLOGG": np.repeat([4.4, 4.2], 3),
        }
    ).write(epochs)
    Table({"TARGETID": np.array([11, 22], dtype=np.int64)}).write(members)
    return RVBinaryConfig.model_validate(
        {
            "project": {"name": "workflow", "output": tmp_path / "out", "random_seed": 7},
            "observations": {
                "mock_epochs": epochs,
                "subsamples": [{"name": "all", "members": members}],
            },
            "model": {"include_unrestricted": True, "period_limits_days": [1e-12]},
            "mock": {"samples": 4, "error_floor": {"values_kms": [0.0, 0.4]}},
            "pdf": {"bins": 12},
            "mcmc": {
                "walkers": 8,
                "steps": 30,
                "burn_in": 5,
                "credible_intervals": [0.68],
                "allow_unconverged": True,
            },
        }
    )


def artifact_mtimes(result: object) -> dict[Path, int]:
    paths = [
        result.mock_artifacts.table_index,
        *result.mock_artifacts.binary_beta_indexes.values(),
        *result.mock_artifacts.nonbinary_beta_indexes.values(),
        *(artifact.path for artifact in result.pdf_artifacts),
        *result.posterior_paths,
        result.summary_csv,
        *result.plot_paths,
    ]
    return {path: path.stat().st_mtime_ns for path in paths}


def test_small_pipeline_outputs_every_grid_point_and_nonblank_plots(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result = run_pipeline(config)

    expected = len(result.period_limits) * len(result.error_floors)
    assert len(result.posterior_paths) == expected
    assert result.summary_csv.is_file()
    assert result.exit_status == 0
    assert result.converged is False
    assert len(set(result.plot_paths)) == len(result.plot_paths)
    for path in result.plot_paths:
        image = mpimg.imread(path)
        assert image.shape[0] >= 200
        assert image.shape[1] >= 300
        assert float(np.std(image)) > 0.01


def test_resume_preserves_every_digest_matching_artifact(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    first = run_pipeline(config)
    before = artifact_mtimes(first)

    second = run_pipeline(config)

    assert artifact_mtimes(second) == before


def test_precomputed_binary2_pipeline_skips_mock_and_runs_unrestricted_mcmc(
    tmp_path: Path,
) -> None:
    config = make_precomputed_config(tmp_path)
    raw = config.model_dump(mode="python")
    raw["mcmc"] = {
        "walkers": 8,
        "steps": 30,
        "burn_in": 5,
        "credible_intervals": [0.68],
        "allow_unconverged": True,
    }
    config = RVBinaryConfig.model_validate(raw)

    result = run_pipeline(config)

    assert result.mock_artifacts is None
    assert len(result.pdf_artifacts) == 1
    assert len(result.posterior_paths) == 1
    assert result.rows[0]["error_floor_kms"] is None
    assert result.rows[0]["error_floor_mode"] == "per_star_baked_in"
    assert result.rows[0]["error_floor_probability"] == 1.0
    assert result.summary_csv.is_file()
    assert all("errorfloor" not in path.name for path in result.plot_paths)


def test_precomputed_source_change_rebuilds_pdf_and_posterior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_precomputed_config(tmp_path)
    raw = config.model_dump(mode="python")
    raw["mcmc"] = {
        "walkers": 8,
        "steps": 30,
        "burn_in": 5,
        "credible_intervals": [0.68],
        "allow_unconverged": True,
    }
    config = RVBinaryConfig.model_validate(raw)
    first = run_pipeline(config)
    posterior_mtime = first.posterior_paths[0].stat().st_mtime_ns
    assert config.mock.precomputed is not None
    beta_path = config.mock.precomputed.directory / "mcs_beta_combined.h5"
    with h5py.File(beta_path, "r+") as handle:
        handle["list_1/arrays"][1] = np.array([0.8, 1.4])

    import rvbinary.workflow.runner as runner

    original = runner.run_binary_fraction_mcmc
    calls = 0

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "run_binary_fraction_mcmc", counted)

    second = run_pipeline(config)

    assert calls == 1
    assert second.posterior_paths[0].stat().st_mtime_ns != posterior_mtime


def test_old_noise_manifest_cannot_resume_as_epoch_noise(tmp_path: Path) -> None:
    import hashlib
    import json

    from rvbinary.workflow.manifest import ArtifactManifest

    config = make_config(tmp_path)
    output = Path(config.project.output)
    old_digest = hashlib.sha256(
        json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = ArtifactManifest(output / "manifest.json", old_digest)
    manifest.state["complete"] = True
    manifest.write()
    before = manifest.path.read_bytes()

    with pytest.raises(ValueError, match="manifest"):
        run_pipeline(config, stages=("mock",))

    assert manifest.path.read_bytes() == before
    assert not (output / "mock").exists()
