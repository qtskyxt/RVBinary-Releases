from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.table import Table

from rvbinary.config import RVBinaryConfig
from rvbinary.mock.writer import generate_mock
from rvbinary.parallel import generate_mock_mpi
from rvbinary.pdf import builder
from rvbinary.pdf.builder import build_pdf_grid
from rvbinary.pdf.histogram import beta_edges, histogram_counts


def make_config(tmp_path: Path, *, legacy_export: bool = False) -> RVBinaryConfig:
    epochs = tmp_path / "epochs.fits"
    members = tmp_path / "members.fits"
    ids = np.repeat(np.array([11, 22], dtype=np.int64), 3)
    Table(
        {
            "TARGETID": ids,
            "MJD": np.tile([59000.0, 59002.0, 59007.0], 2),
            "VRAD": np.zeros(6),
            "VRAD_ERR": np.tile([0.5, 0.7, 0.9], 2),
            "Mass": np.repeat([0.9, 1.1], 3),
            "meanLOGG": np.repeat([4.4, 4.2], 3),
        }
    ).write(epochs)
    Table({"TARGETID": np.array([22], dtype=np.int64)}).write(members)
    return RVBinaryConfig.model_validate(
        {
            "project": {"name": "pdf", "output": tmp_path / "out", "random_seed": 91},
            "observations": {
                "mock_epochs": epochs,
                "subsamples": [{"name": "FEH0", "members": members}],
            },
            "model": {
                "include_unrestricted": True,
                "period_limits_days": [1e-12],
            },
            "mock": {"samples": 3, "error_floor": {"values_kms": [0.0, 0.4]}},
            "pdf": {"bins": 12, "export_legacy_npy": legacy_export},
        }
    )


def test_pdf_grid_selects_subsample_and_applies_all_period_floor_products(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    mocks = generate_mock(config)

    artifacts = build_pdf_grid(config, mocks)

    assert len(artifacts) == 4
    limited = [artifact for artifact in artifacts if artifact.period_limit_days is not None]
    assert len(limited) == 2
    for artifact in limited:
        with np.load(artifact.path) as stored:
            np.testing.assert_array_equal(
                stored["binary_density"], stored["nonbinary_density"]
            )
            np.testing.assert_array_equal(stored["binary_counts"], stored["nonbinary_counts"])
            assert int(stored["sample_count"]) == 3
            assert int(stored["draw_count"]) == 3
            assert str(stored["statistic_mode"]) == "consistent"
            assert float(stored["error_floor_kms"]) in {0.0, 0.4}
            assert json.loads(str(stored["source_digests"]))


def test_pdf_grid_can_export_legacy_density_arrays(tmp_path: Path) -> None:
    config = make_config(tmp_path, legacy_export=True)
    mocks = generate_mock(config)

    artifacts = build_pdf_grid(config, mocks)

    for artifact in artifacts:
        assert (artifact.path.parent / "log_fprob_DM91_0.npy").is_file()
        assert (artifact.path.parent / "log_fprob_DM91_1.npy").is_file()


def test_pdf_grid_rejects_statistic_mode_mismatch_before_writing(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    mocks = generate_mock(config)
    with h5py.File(mocks.binary_beta_indexes[0.0], "r+") as handle:
        handle.attrs["statistic_mode"] = "legacy"

    with pytest.raises(ValueError, match="statistic mode"):
        build_pdf_grid(config, mocks)

    assert not (Path(config.project.output) / "pdf").exists()


def _write_legacy_lists(path: Path, samples: list[list[np.ndarray]]) -> None:
    with h5py.File(path, "w") as handle:
        for sample_id, groups in enumerate(samples):
            dataset = handle.create_group(f"list_{sample_id}").create_dataset(
                "arrays", (len(groups),), dtype=h5py.vlen_dtype(np.float64)
            )
            for index, values in enumerate(groups):
                dataset[index] = values


def make_precomputed_config(tmp_path: Path) -> RVBinaryConfig:
    config = make_config(tmp_path)
    raw = config.model_dump(mode="python")
    raw["model"] = {"include_unrestricted": True, "period_limits_days": []}
    raw["mock"] = {
        "samples": 2,
        "precomputed": {
            "format": "binary2",
            "directory": tmp_path / "legacy",
        },
    }
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    with h5py.File(legacy / "mcs_table_combined.h5", "w") as handle:
        table = handle.create_group("table_0")
        table.create_dataset("ID", data=np.array([11, 11, 22, 22]))
    _write_legacy_lists(
        legacy / "mcs_beta_combined.h5",
        [
            [np.array([0.1]), np.array([3.0, 400.0])],
            [np.array([0.2]), np.array([1.0, 2.0])],
        ],
    )
    _write_legacy_lists(
        legacy / "mcs_nberr_combined.h5",
        [
            [np.array([0.2]), np.array([0.5, 0.5])],
            [np.array([0.3]), np.array([0.25, 0.75])],
        ],
    )
    return RVBinaryConfig.model_validate(raw)


def test_precomputed_binary2_pdf_streams_lists_without_scanning_period_tables(
    tmp_path: Path,
) -> None:
    config = make_precomputed_config(tmp_path)

    artifacts = builder.build_precomputed_pdf_grid(config)

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.error_floor_kms is None
    assert artifact.period_limit_days is None
    expected_binary = histogram_counts(
        np.array([1.0 + np.sqrt(2.5)]), beta_edges(12)
    )
    expected_nonbinary = histogram_counts(
        np.array([1.5, 1.0 + np.sqrt(0.3125)]), beta_edges(12)
    )
    with np.load(artifact.path, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["binary_counts"], expected_binary)
        np.testing.assert_array_equal(stored["nonbinary_counts"], expected_nonbinary)
        assert str(stored["error_floor_mode"]) == "per_star_baked_in"
        assert int(stored["binary_out_of_range_count"]) == 1
        assert int(stored["nonbinary_out_of_range_count"]) == 0
        assert int(stored["sample_count"]) == 2
        assert int(stored["draw_count"]) == 2


def test_legacy_star_order_supports_compound_table_dataset(tmp_path: Path) -> None:
    path = tmp_path / "compound.h5"
    rows = np.array([(22,), (22,), (11,)], dtype=[("ID", "<i8")])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("table_0", data=rows)

    np.testing.assert_array_equal(builder._legacy_star_order(path), [22, 11])


def test_precomputed_binary2_requires_exact_sample_coverage(tmp_path: Path) -> None:
    config = make_precomputed_config(tmp_path)
    raw = config.model_dump(mode="python")
    raw["mock"]["samples"] = 1

    with pytest.raises(ValueError, match="sample coverage"):
        builder.build_precomputed_pdf_grid(RVBinaryConfig.model_validate(raw))


def test_precomputed_binary2_reads_each_sample_once_for_multiple_subsamples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_precomputed_config(tmp_path)
    second_members = tmp_path / "second-members.fits"
    Table({"TARGETID": np.array([11], dtype=np.int64)}).write(second_members)
    raw = config.model_dump(mode="python")
    raw["observations"]["subsamples"].append(
        {"name": "FEH1", "members": second_members, "epochs": None}
    )
    config = RVBinaryConfig.model_validate(raw)
    original = builder._legacy_sample_statistics
    calls = 0

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builder, "_legacy_sample_statistics", counted)

    artifacts = builder.build_precomputed_pdf_grid(config)

    assert len(artifacts) == 2
    assert calls == 4


@pytest.mark.mpi
def test_pdf_grid_mpi_count_reduction_matches_serial() -> None:
    mpi = pytest.importorskip("mpi4py.MPI")
    comm = mpi.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run with at least two MPI ranks")
    shared = comm.bcast(
        tempfile.mkdtemp(prefix="rvbinary-pdf-mpi-") if comm.Get_rank() == 0 else None,
        root=0,
    )
    if comm.Get_rank() == 0:
        config = make_config(Path(shared))
        raw = config.model_dump(mode="python")
    else:
        raw = None
    config = RVBinaryConfig.model_validate(comm.bcast(raw, root=0))
    comm.Barrier()
    mocks = generate_mock_mpi(config, comm)

    parallel_artifacts = build_pdf_grid(config, mocks, comm)
    if comm.Get_rank() == 0:
        parallel_counts = {
            (artifact.error_floor_kms, artifact.period_limit_days): (
                np.load(artifact.path)["binary_counts"],
                np.load(artifact.path)["nonbinary_counts"],
            )
            for artifact in parallel_artifacts
        }
        serial_artifacts = build_pdf_grid(config, mocks)
        for artifact in serial_artifacts:
            key = (artifact.error_floor_kms, artifact.period_limit_days)
            with np.load(artifact.path) as stored:
                np.testing.assert_array_equal(stored["binary_counts"], parallel_counts[key][0])
                np.testing.assert_array_equal(
                    stored["nonbinary_counts"], parallel_counts[key][1]
                )
    else:
        assert parallel_artifacts == []
    comm.Barrier()


@pytest.mark.mpi
def test_precomputed_binary2_mpi_count_reduction_matches_serial() -> None:
    mpi = pytest.importorskip("mpi4py.MPI")
    comm = mpi.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run with at least two MPI ranks")
    shared = comm.bcast(
        tempfile.mkdtemp(prefix="rvbinary-precomputed-pdf-mpi-")
        if comm.Get_rank() == 0
        else None,
        root=0,
    )
    if comm.Get_rank() == 0:
        config = make_precomputed_config(Path(shared))
        raw = config.model_dump(mode="python")
    else:
        raw = None
    config = RVBinaryConfig.model_validate(comm.bcast(raw, root=0))
    comm.Barrier()

    parallel_artifacts = builder.build_precomputed_pdf_grid(config, comm)
    if comm.Get_rank() == 0:
        assert len(parallel_artifacts) == 1
        with np.load(parallel_artifacts[0].path) as stored:
            parallel_binary = stored["binary_counts"].copy()
            parallel_nonbinary = stored["nonbinary_counts"].copy()
        serial_artifacts = builder.build_precomputed_pdf_grid(config)
        with np.load(serial_artifacts[0].path) as stored:
            np.testing.assert_array_equal(stored["binary_counts"], parallel_binary)
            np.testing.assert_array_equal(stored["nonbinary_counts"], parallel_nonbinary)
    else:
        assert parallel_artifacts == []
    comm.Barrier()


@pytest.mark.mpi
def test_precomputed_binary2_mpi_propagates_rank_local_read_failure() -> None:
    mpi = pytest.importorskip("mpi4py.MPI")
    comm = mpi.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run with at least two MPI ranks")
    shared = comm.bcast(
        tempfile.mkdtemp(prefix="rvbinary-precomputed-error-mpi-")
        if comm.Get_rank() == 0
        else None,
        root=0,
    )
    if comm.Get_rank() == 0:
        config = make_precomputed_config(Path(shared))
        assert config.mock.precomputed is not None
        with h5py.File(
            config.mock.precomputed.directory / "mcs_beta_combined.h5", "r+"
        ) as handle:
            del handle["list_1/arrays"]
        raw = config.model_dump(mode="python")
    else:
        raw = None
    config = RVBinaryConfig.model_validate(comm.bcast(raw, root=0))
    comm.Barrier()

    with pytest.raises(RuntimeError, match="MPI legacy PDF generation failed"):
        builder.build_precomputed_pdf_grid(config, comm)
    comm.Barrier()
