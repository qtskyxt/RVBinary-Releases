from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table
from typer.testing import CliRunner

from rvbinary.cli import app


def write_config(tmp_path: Path) -> Path:
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
    path = tmp_path / "rvbinary.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "cli", "output": "out", "random_seed": 7},
                "observations": {
                    "mock_epochs": "epochs.fits",
                    "subsamples": [{"name": "all", "members": "members.fits"}],
                },
                "model": {"period_limits_days": [1e-12]},
                "mock": {"samples": 4, "error_floor": {"values_kms": [0.0]}},
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
    )
    return path


def test_run_inspect_compact_and_submit_commands(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    runner = CliRunner()

    run = runner.invoke(app, ["run", str(config)])
    assert run.exit_code == 0, run.output
    index = tmp_path / "out" / "mock" / "indexes" / "tables.h5"
    inspected = runner.invoke(app, ["inspect", str(index)])
    assert inspected.exit_code == 0
    assert "rvbinary.mock/v1" in inspected.output
    compacted = tmp_path / "standalone.h5"
    compact = runner.invoke(app, ["compact", str(index), str(compacted)])
    assert compact.exit_code == 0
    assert compacted.is_file()
    script = tmp_path / "job.pbs"
    submitted = runner.invoke(
        app,
        ["submit", str(config), "--scheduler", "pbs", "--output", str(script)],
    )
    assert submitted.exit_code == 0
    assert script.is_file() and "qsub " not in script.read_text()


def test_stage_dry_runs_do_not_create_output(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    runner = CliRunner()

    for command in ("mock", "pdf", "mcmc", "run"):
        result = runner.invoke(app, [command, str(config), "--dry-run"])
        assert result.exit_code == 0, result.output

    assert not (tmp_path / "out").exists()


def test_run_adopts_precomputed_pdf_grid_without_rewriting(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["mock", str(config)]).exit_code == 0
    assert runner.invoke(app, ["pdf", str(config)]).exit_code == 0
    pdfs = list((tmp_path / "out" / "pdf").glob("**/pdf.npz"))
    before = {path: path.stat().st_mtime_ns for path in pdfs}
    (tmp_path / "out" / "manifest.json").unlink()

    result = runner.invoke(app, ["run", str(config)])

    assert result.exit_code == 0, result.output
    assert {path: path.stat().st_mtime_ns for path in pdfs} == before


def test_mock_command_is_a_noop_for_precomputed_binary2_data(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw["model"] = {"include_unrestricted": True, "period_limits_days": []}
    raw["mock"] = {
        "samples": 4,
        "precomputed": {"format": "binary2", "directory": "legacy"},
    }
    config.write_text(yaml.safe_dump(raw))

    result = CliRunner().invoke(app, ["mock", str(config)])

    assert result.exit_code == 0, result.output
    assert "Using precomputed mocks" in result.output
    assert not (tmp_path / "out").exists()
