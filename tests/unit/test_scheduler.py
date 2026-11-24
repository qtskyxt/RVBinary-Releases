from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rvbinary.config import RVBinaryConfig
from rvbinary.workflow.scheduler import render_scheduler, render_to_file


def config(tmp_path: Path) -> RVBinaryConfig:
    return RVBinaryConfig.model_validate(
        {
            "project": {"name": "cluster run", "output": tmp_path / "out", "random_seed": 1},
            "observations": {
                "mock_epochs": tmp_path / "epochs.fits",
                "subsamples": [{"name": "all", "members": tmp_path / "members.fits"}],
            },
            "execution": {"mpi_ranks": 8},
            "scheduler": {
                "walltime": "02:30:00",
                "memory": "24G",
                "queue": "science",
                "account": "astro",
                "conda_environment": "myenvpy312",
            },
        }
    )


@pytest.mark.parametrize("kind,launcher", [("slurm", "srun"), ("pbs", "mpiexec")])
def test_scheduler_script_is_rendered_not_submitted(
    kind: str, launcher: str, tmp_path: Path
) -> None:
    output = tmp_path / f"job-{kind}.sh"

    path = render_to_file(config(tmp_path), kind, output, Path("rvbinary.yaml"))
    text = path.read_text()

    assert launcher in text
    assert "rvbinary run" in text
    assert "conda activate myenvpy312" in text
    assert "OMP_NUM_THREADS=1" in text
    assert "MKL_NUM_THREADS=1" in text
    assert "8" in text and "02:30:00" in text and "24G" in text
    assert "sbatch " not in text and "qsub " not in text
    assert path.stat().st_mode & 0o111


def test_scheduler_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scheduler"):
        render_scheduler(config(tmp_path), "lsf")


def test_precomputed_scheduler_skips_mock_and_runs_mcmc(tmp_path: Path) -> None:
    raw = config(tmp_path).model_dump(mode="python")
    raw["mock"]["precomputed"] = {
        "format": "binary2",
        "directory": tmp_path / "legacy",
        "error_floor_mode": "per_star_baked_in",
    }
    resolved = RVBinaryConfig.model_validate(raw)

    text = render_scheduler(resolved, "pbs", Path("rvbinary.yaml"))

    assert "rvbinary mock" not in text
    assert "mpiexec -n 8 rvbinary pdf rvbinary.yaml" in text
    assert "rvbinary mcmc rvbinary.yaml" in text
    assert "rvbinary run" not in text


def test_scheduler_uses_explicit_conda_init_for_noninteractive_pbs(
    tmp_path: Path,
) -> None:
    raw = config(tmp_path).model_dump(mode="python")
    raw["scheduler"]["conda_init"] = "/opt/anaconda3/etc/profile.d/conda.sh"
    raw["scheduler"]["conda_environment"] = (
        "/home/user/.conda/envs/rvbinary"
    )
    resolved = RVBinaryConfig.model_validate(raw)

    text = render_scheduler(resolved, "pbs", Path("rvbinary.yaml"))

    assert "source /opt/anaconda3/etc/profile.d/conda.sh" in text
    assert "conda activate /home/user/.conda/envs/rvbinary" in text
    assert "conda info --base" not in text


def test_conda_environment_files_cover_serial_and_mpi_variants() -> None:
    root = Path(__file__).parents[2]
    serial = yaml.safe_load((root / "environment.yml").read_text())
    mpich = yaml.safe_load((root / "environment-mpich.yml").read_text())
    openmpi = yaml.safe_load((root / "environment-openmpi.yml").read_text())

    assert serial["name"] == "rvbinary"
    assert any("mpi_mpich" in str(item) for item in mpich["dependencies"])
    assert "mpich" in mpich["dependencies"]
    assert any("mpi_openmpi" in str(item) for item in openmpi["dependencies"])
    assert "openmpi" in openmpi["dependencies"]
