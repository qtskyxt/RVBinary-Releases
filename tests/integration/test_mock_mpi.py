from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.table import Table

from rvbinary.config import RVBinaryConfig
from rvbinary.io.legacy import open_mock_store
from rvbinary.io.observations import EpochData
from rvbinary.mock.writer import _artifact_set, generate_mock
from rvbinary.parallel import (
    extend_mock_stars_mpi,
    generate_mock_mpi,
    generate_mock_mpio,
    partition_indices,
)


class SerialComm:
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def bcast(self, value: object, root: int = 0) -> object:
        return value

    def Barrier(self) -> None:
        return None

    def allgather(self, value: object) -> list[object]:
        return [value]

    def Allreduce(self, send: object, receive: object) -> None:
        np.copyto(receive, send)


def make_config(tmp_path: Path, name: str, *, mode: str = "shards") -> RVBinaryConfig:
    root = tmp_path / name
    root.mkdir()
    epochs = root / "epochs.fits"
    members = root / "members.fits"
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
    Table({"TARGETID": np.array([11, 22], dtype=np.int64)}).write(members)
    return RVBinaryConfig.model_validate(
        {
            "project": {"name": name, "output": root / "out", "random_seed": 29},
            "observations": {
                "mock_epochs": epochs,
                "subsamples": [{"name": "all", "members": members}],
            },
            "mock": {"samples": 5, "error_floor": {"values_kms": [0.0, 0.4]}},
            "execution": {"hdf5_mode": mode},
        }
    )


def digest(index: Path) -> str:
    result = hashlib.sha256()
    with open_mock_store(index) as store:
        for sample_id in store.sample_ids:
            for name, values in sorted(store.sample(sample_id).arrays.items()):
                result.update(name.encode("ascii"))
                result.update(np.ascontiguousarray(values).tobytes())
    return result.hexdigest()


def one_star_epochs(targetid: int) -> EpochData:
    return EpochData(
        targetid=np.repeat(np.array([targetid], dtype=np.int64), 3),
        mjd=np.array([59000.0, 59002.0, 59007.0]),
        vrad_kms=np.zeros(3),
        vrad_err_kms=np.array([0.5, 0.7, 0.9]),
        mass_solar=np.full(3, 1.0),
        logg_cgs=np.full(3, 4.3),
    )


def test_partition_indices_is_balanced_and_complete() -> None:
    partitions = [partition_indices(range(7), rank, 3) for rank in range(3)]

    assert partitions == [range(0, 3), range(3, 5), range(5, 7)]
    assert [item for part in partitions for item in part] == list(range(7))


def test_partition_indices_supports_more_ranks_than_items() -> None:
    partitions = [partition_indices([10, 20], rank, 4) for rank in range(4)]

    assert partitions == [[10], [20], [], []]


def test_single_rank_mpi_generation_matches_serial(tmp_path: Path) -> None:
    mpi_config = make_config(tmp_path, "mpi")
    serial_config = make_config(tmp_path, "serial")

    mpi = generate_mock_mpi(mpi_config, SerialComm())
    serial = generate_mock(serial_config)

    assert digest(mpi.table_index) == digest(serial.table_index)
    for floor in (0.0, 0.4):
        assert digest(mpi.binary_beta_indexes[floor]) == digest(serial.binary_beta_indexes[floor])
        assert digest(mpi.nonbinary_beta_indexes[floor]) == digest(
            serial.nonbinary_beta_indexes[floor]
        )


def test_single_rank_mpi_star_extension_adds_new_target_without_rewriting_old_shards(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, "extension")
    initial = generate_mock(config)
    old_bytes = {path: path.read_bytes() for path in initial.shards}

    extended = extend_mock_stars_mpi(config, one_star_epochs(33), SerialComm())

    assert {path: path.read_bytes() for path in initial.shards} == old_bytes
    with open_mock_store(extended.table_index) as store:
        assert set(store.targetids) == {11, 22, 33}


@pytest.mark.skipif(h5py.get_config().mpi, reason="requires a serial h5py build")
def test_mpio_backend_fails_before_creating_outputs(tmp_path: Path) -> None:
    config = make_config(tmp_path, "no-mpio", mode="mpio")

    with pytest.raises(RuntimeError, match="MPI-enabled h5py"):
        generate_mock_mpio(config, SerialComm())

    assert not Path(config.project.output).exists()


@pytest.mark.mpi
@pytest.mark.parametrize("population", ["dm91", "ms17"])
def test_real_mpi_generation_matches_serial(population: str) -> None:
    mpi = pytest.importorskip("mpi4py.MPI")
    comm = mpi.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run with at least two MPI ranks")
    shared = comm.bcast(
        tempfile.mkdtemp(prefix="rvbinary-mpi-test-") if comm.Get_rank() == 0 else None,
        root=0,
    )
    root = Path(shared)
    if comm.Get_rank() == 0:
        distributed_config = make_config(root, "distributed")
        serial_config = make_config(root, "reference")
        if population == "ms17":
            from rvbinary.config import MS17Config
            from tests.unit.test_ms17 import table_file

            trial = MS17Config(period_table=table_file(root))
            serial_config.model = trial
            distributed_config.model = trial
        payload = (
            distributed_config.model_dump(mode="python"),
            serial_config.model_dump(mode="python"),
        )
    else:
        payload = None
    distributed_raw, serial_raw = comm.bcast(payload, root=0)
    distributed_config = RVBinaryConfig.model_validate(distributed_raw)
    serial_config = RVBinaryConfig.model_validate(serial_raw)
    comm.Barrier()

    distributed = generate_mock_mpi(distributed_config, comm)
    expected: dict[str, str] | None = None
    if comm.Get_rank() == 0:
        serial = generate_mock(serial_config)
        expected = {"table": digest(serial.table_index)}
        for floor in (0.0, 0.4):
            expected[f"binary:{floor}"] = digest(serial.binary_beta_indexes[floor])
            expected[f"nonbinary:{floor}"] = digest(serial.nonbinary_beta_indexes[floor])
    expected = comm.bcast(expected, root=0)

    assert digest(distributed.table_index) == expected["table"]
    for floor in (0.0, 0.4):
        assert digest(distributed.binary_beta_indexes[floor]) == expected[f"binary:{floor}"]
        assert digest(distributed.nonbinary_beta_indexes[floor]) == expected[f"nonbinary:{floor}"]


@pytest.mark.mpi
def test_real_mpi_star_extension_preserves_existing_shards() -> None:
    mpi = pytest.importorskip("mpi4py.MPI")
    comm = mpi.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run with at least two MPI ranks")
    shared = comm.bcast(
        tempfile.mkdtemp(prefix="rvbinary-extension-mpi-") if comm.Get_rank() == 0 else None,
        root=0,
    )
    root = Path(shared)
    if comm.Get_rank() == 0:
        config = make_config(root, "distributed")
        raw = config.model_dump(mode="python")
    else:
        raw = None
    config = RVBinaryConfig.model_validate(comm.bcast(raw, root=0))
    comm.Barrier()
    initial = generate_mock_mpi(config, comm)
    old_bytes = (
        {path: path.read_bytes() for path in initial.shards} if comm.Get_rank() == 0 else None
    )

    extended = extend_mock_stars_mpi(config, one_star_epochs(33), comm)

    if comm.Get_rank() == 0:
        assert {path: path.read_bytes() for path in initial.shards} == old_bytes
        with open_mock_store(extended.table_index) as store:
            assert set(store.targetids) == {11, 22, 33}
    comm.Barrier()


@pytest.mark.mpi_hdf5
@pytest.mark.parametrize("population", ["dm91", "ms17"])
def test_real_mpio_backend_matches_serial(population: str) -> None:
    if not h5py.get_config().mpi:
        pytest.skip("requires an MPI-enabled h5py build")
    mpi = pytest.importorskip("mpi4py.MPI")
    comm = mpi.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run with at least two MPI ranks")
    shared = comm.bcast(
        tempfile.mkdtemp(prefix="rvbinary-mpio-test-") if comm.Get_rank() == 0 else None,
        root=0,
    )
    root = Path(shared)
    if comm.Get_rank() == 0:
        mpio_config = make_config(root, "mpio", mode="mpio")
        serial_config = make_config(root, "reference")
        if population == "ms17":
            from rvbinary.config import MS17Config
            from tests.unit.test_ms17 import table_file

            trial = MS17Config(period_table=table_file(root))
            serial_config.model = trial
            mpio_config.model = trial
        payload = (
            mpio_config.model_dump(mode="python"),
            serial_config.model_dump(mode="python"),
        )
    else:
        payload = None
    mpio_raw, serial_raw = comm.bcast(payload, root=0)
    mpio_config = RVBinaryConfig.model_validate(mpio_raw)
    serial_config = RVBinaryConfig.model_validate(serial_raw)
    comm.Barrier()

    generate_mock_mpio(mpio_config, comm)
    expected: dict[str, str] | None = None
    if comm.Get_rank() == 0:
        serial = generate_mock(serial_config)
        expected = {"table": digest(serial.table_index)}
        for floor in (0.0, 0.4):
            expected[f"binary:{floor}"] = digest(serial.binary_beta_indexes[floor])
            expected[f"nonbinary:{floor}"] = digest(serial.nonbinary_beta_indexes[floor])
    expected = comm.bcast(expected, root=0)

    reopened = _artifact_set(mpio_config)
    assert digest(reopened.table_index) == expected["table"]
    for floor in (0.0, 0.4):
        assert digest(reopened.binary_beta_indexes[floor]) == expected[f"binary:{floor}"]
        assert digest(reopened.nonbinary_beta_indexes[floor]) == expected[f"nonbinary:{floor}"]
