"""MPI distribution and collective HDF5 helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

import h5py
import numpy as np

from rvbinary import __version__
from rvbinary.config import RVBinaryConfig
from rvbinary.io.hdf5 import ArtifactMetadata, payload_identity
from rvbinary.io.hdf5 import _write_metadata as _write_hdf5_metadata
from rvbinary.io.legacy import open_mock_store
from rvbinary.io.observations import EpochData, read_epochs
from rvbinary.mock.simulation import derive_beta_realization, simulate_sample
from rvbinary.mock.writer import (
    ArtifactSet,
    _artifact_set,
    _batch_id,
    _canonical_source_hashes,
    _config_digest,
    _paths,
    _source_batches,
    _write_shards,
    _write_source_batch,
    rebuild_mock_indexes,
)
from rvbinary.models.base import OrbitModel
from rvbinary.models.factory import make_model, model_payload, validate_model_stars

Index = TypeVar("Index")


class Communicator(Protocol):
    """Subset of the mpi4py communicator API used by RVBinary."""

    def Get_rank(self) -> int: ...

    def Get_size(self) -> int: ...

    def Barrier(self) -> None: ...

    def bcast(self, value: object, root: int = 0) -> object: ...

    def allgather(self, value: object) -> list[object]: ...

    def Allreduce(self, send: object, receive: object) -> None: ...


def partition_indices(
    indices: range | Sequence[Index], rank: int, size: int
) -> range | list[Index]:
    """Split ordered indices into balanced contiguous rank-local partitions."""
    if size <= 0:
        raise ValueError("size must be positive")
    if rank < 0 or rank >= size:
        raise ValueError("rank must satisfy 0 <= rank < size")
    count = len(indices)
    width, remainder = divmod(count, size)
    start = rank * width + min(rank, remainder)
    stop = start + width + (1 if rank < remainder else 0)
    if isinstance(indices, range) and indices.step == 1:
        return range(indices.start + start, indices.start + stop)
    return list(indices[start:stop])


def _contiguous_ranges(indices: Sequence[int]) -> list[range]:
    if not indices:
        return []
    ranges: list[range] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            ranges.append(range(start, previous + 1))
            start = value
        previous = value
    ranges.append(range(start, previous + 1))
    return ranges


def _prepare_sources(config: RVBinaryConfig) -> list[int]:
    from rvbinary.mock.writer import _validate_existing

    existing = _validate_existing(config, None)
    if not _source_batches(config):
        _write_source_batch(config, read_epochs(config.observations.mock_epochs))
    for _, data, _ in _source_batches(config):
        validate_model_stars(config, data)
    return sorted(set(range(config.mock.samples)) - set(existing))


def _broadcast_root_call(comm: Communicator, function: Callable[[], object]) -> object:
    rank = comm.Get_rank()
    result: object = None
    if rank == 0:
        try:
            result = (True, function())
        except BaseException as exc:
            result = (False, f"{type(exc).__name__}: {exc}")
    received = comm.bcast(result, root=0)
    ok, payload = received  # type: ignore[misc]
    if not ok:
        raise RuntimeError(f"MPI rank-zero operation failed: {payload}")
    return payload


def _check_model_ranks(config: RVBinaryConfig, comm: Communicator) -> OrbitModel:
    model = None
    try:
        model = make_model(config)
        payload = json.dumps(model_payload(config), sort_keys=True)
        result = (None, payload)
    except Exception as exc:
        result = (str(exc), "")
    results = comm.allgather(result)
    if any(item[0] is not None for item in results):
        raise RuntimeError(f"MPI model preflight failed: {results}")
    if len({item[1] for item in results}) != 1:
        raise RuntimeError("MPI ranks have inconsistent model/table fingerprints")
    assert model is not None
    return model


def generate_mock_mpi(config: RVBinaryConfig, comm: Communicator) -> ArtifactSet:
    """Generate rank-local immutable shards and rebuild indexes on rank zero."""
    _check_model_ranks(config, comm)
    if config.execution.hdf5_mode == "mpio":
        return generate_mock_mpio(config, comm)
    rank = comm.Get_rank()
    size = comm.Get_size()
    missing = list(_broadcast_root_call(comm, lambda: _prepare_sources(config)))
    comm.Barrier()

    local = list(partition_indices(missing, rank, size))
    local_error: str | None = None
    try:
        for sample_range in _contiguous_ranges(local):
            for _, data, _ in _source_batches(config):
                _write_shards(config, data, sample_range)
    except BaseException as exc:
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = [error for error in comm.allgather(local_error) if error is not None]
    if errors:
        raise RuntimeError("MPI shard generation failed; " + "; ".join(errors))

    comm.Barrier()

    def rebuild() -> None:
        rebuild_mock_indexes(config)

    _broadcast_root_call(comm, rebuild)
    comm.Barrier()
    return _artifact_set(config)


def extend_mock_stars_mpi(
    config: RVBinaryConfig, new_epochs: EpochData, comm: Communicator
) -> ArtifactSet:
    """Append one star batch with sample ranges distributed across MPI ranks."""
    _check_model_ranks(config, comm)
    if config.execution.hdf5_mode == "mpio":
        raise RuntimeError("MPI star extension currently requires hdf5_mode=shards")
    rank = comm.Get_rank()
    size = comm.Get_size()

    def prepare() -> tuple[list[int], str, bool]:
        from rvbinary.mock.writer import _validate_existing

        existing = _validate_existing(config, None)
        if not existing:
            raise ValueError("cannot extend stars before generating mock samples")
        if existing != list(range(existing[0], existing[-1] + 1)):
            raise ValueError("star extension requires contiguous sample coverage")
        table_index = _paths(config)[0]
        with open_mock_store(table_index) as store:
            duplicate = np.intersect1d(store.targetids, np.unique(new_epochs.targetid))
        if duplicate.size:
            listed = ", ".join(str(int(value)) for value in duplicate)
            raise ValueError(f"duplicate TARGETID across star batches: {listed}")
        source, created = _write_source_batch(config, new_epochs)
        return existing, str(source), created

    existing, source_name, created_source = _broadcast_root_call(comm, prepare)  # type: ignore[misc]
    local = list(partition_indices(existing, rank, size))
    local_error: str | None = None
    try:
        for sample_range in _contiguous_ranges(local):
            _write_shards(config, new_epochs, sample_range)
    except BaseException as exc:
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = [error for error in comm.allgather(local_error) if error is not None]
    if errors:
        if rank == 0 and created_source:
            Path(source_name).unlink(missing_ok=True)
        raise RuntimeError("MPI star extension failed; " + "; ".join(errors))

    def rebuild() -> None:
        rebuild_mock_indexes(config)

    _broadcast_root_call(comm, rebuild)
    comm.Barrier()
    return _artifact_set(config)


def _mpio_temp_paths(comm: Communicator, outputs: list[Path]) -> list[Path]:
    rank = comm.Get_rank()
    names: object = None
    if rank == 0:
        generated = []
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{output.name}.", suffix=".mpio", dir=output.parent
            )
            os.close(descriptor)
            Path(name).unlink()
            generated.append(name)
        names = generated
    return [Path(name) for name in comm.bcast(names, root=0)]  # type: ignore[arg-type]


def _create_sample_datasets(
    group: h5py.Group,
    arrays: dict[str, np.ndarray] | None,
    shapes: dict[str, tuple[tuple[int, ...], np.dtype]],
) -> None:
    group.attrs["array_names"] = json.dumps(sorted(shapes))
    for name in sorted(shapes):
        shape, dtype = shapes[name]
        dataset = group.create_dataset(name, shape=shape, dtype=dtype)
        if arrays is not None:
            dataset[...] = arrays[name]


def generate_mock_mpio(config: RVBinaryConfig, comm: Communicator) -> ArtifactSet:
    """Collectively write fixed-shape native HDF5 products with MPI-enabled h5py."""
    if not h5py.get_config().mpi:
        raise RuntimeError(
            "the mpio backend requires an MPI-enabled h5py build; use hdf5_mode=shards "
            "or install RVBinary in an MPI HDF5 environment"
        )
    model = _check_model_ranks(config, comm)
    rank = comm.Get_rank()
    size = comm.Get_size()
    table_path, binary_paths, nonbinary_paths = _paths(config)
    outputs = [table_path, *binary_paths.values(), *nonbinary_paths.values()]

    def prepare() -> None:
        if any(path.exists() for path in outputs):
            raise ValueError("mpio generation requires new output files")
        if not _source_batches(config):
            _write_source_batch(config, read_epochs(config.observations.mock_epochs))
        for _, data, _ in _source_batches(config):
            validate_model_stars(config, data)

    _broadcast_root_call(comm, prepare)
    comm.Barrier()
    batches = _source_batches(config)
    metadata = comm.bcast(
        ArtifactMetadata(
            package_version=__version__,
            config_digest=_config_digest(config),
            seed=config.project.random_seed,
            sample_start=0,
            sample_end=config.mock.samples,
            source_hashes=_canonical_source_hashes(config, batches),
            statistic_mode=config.statistics.mode,
        )
        if rank == 0
        else None,
        root=0,
    )
    temporary = _mpio_temp_paths(comm, outputs)
    handles = [h5py.File(path, "w", driver="mpio", comm=comm) for path in temporary]
    local_samples = set(partition_indices(range(config.mock.samples), rank, size))
    try:
        for handle in handles:
            _write_hdf5_metadata(handle, metadata, completion=False)
            handle.create_group("payloads", track_order=True)
        for _, data, _ in batches:
            batch = _batch_id(data)
            identity = payload_identity(0, config.mock.samples, batch)
            product_groups = []
            for handle in handles:
                payload = handle["payloads"].create_group(identity)
                payload.attrs["sample_start"] = 0
                payload.attrs["sample_end"] = config.mock.samples
                payload.attrs["star_batch_id"] = batch
                product_groups.append(payload.create_group("samples", track_order=True))
            counts = np.unique(data.targetid, return_counts=True)[1]
            pair_count = int(np.sum(counts * (counts - 1) // 2))
            table_shapes = {
                name: (
                    (len(data.targetid),),
                    np.dtype(np.int64 if name in {"TARGETID", "STAR_OFFSET"} else np.float64),
                )
                for name in (
                    "TARGETID",
                    "VRAD",
                    "VRAD_ERR",
                    "PERIOD",
                    "q",
                    "ECC",
                    "THETA",
                    "INCL",
                    "OMEGA",
                    "DELTAT",
                    "STAR_OFFSET",
                )
            }
            beta_shapes = {
                "TARGETID": ((pair_count,), np.dtype(np.int64)),
                "BETA": ((pair_count,), np.dtype(np.float64)),
            }
            for sample_id in range(config.mock.samples):
                table_arrays: dict[str, np.ndarray] | None = None
                binary_arrays: dict[float, dict[str, np.ndarray]] = {}
                nonbinary_arrays: dict[float, dict[str, np.ndarray]] = {}
                if sample_id in local_samples:
                    sample = simulate_sample(data, model, config.project.random_seed, sample_id)
                    table_arrays = {
                        name: np.asarray(values) for name, values in sample.arrays.items()
                    }
                    for floor in binary_paths:
                        beta = derive_beta_realization(
                            sample,
                            floor,
                            config.project.random_seed,
                            config.statistics.mode,
                        )
                        binary_ids, binary_values = beta.flattened("binary")
                        nonbinary_ids, nonbinary_values = beta.flattened("nonbinary")
                        binary_arrays[floor] = {
                            "TARGETID": binary_ids,
                            "BETA": binary_values,
                        }
                        nonbinary_arrays[floor] = {
                            "TARGETID": nonbinary_ids,
                            "BETA": nonbinary_values,
                        }
                _create_sample_datasets(
                    product_groups[0].create_group(str(sample_id)),
                    table_arrays,
                    table_shapes,
                )
                offset = 1
                for floor in binary_paths:
                    _create_sample_datasets(
                        product_groups[offset].create_group(str(sample_id)),
                        binary_arrays.get(floor),
                        beta_shapes,
                    )
                    offset += 1
                for floor in nonbinary_paths:
                    _create_sample_datasets(
                        product_groups[offset].create_group(str(sample_id)),
                        nonbinary_arrays.get(floor),
                        beta_shapes,
                    )
                    offset += 1
        for handle in handles:
            handle.flush()
        comm.Barrier()
        for handle in handles:
            handle.attrs.modify("completion", True)
            handle.flush()
    finally:
        for handle in handles:
            handle.close()

    comm.Barrier()

    def publish() -> None:
        installed: list[tuple[Path, Path]] = []
        try:
            for source, destination in zip(temporary, outputs, strict=True):
                os.replace(source, destination)
                installed.append((source, destination))
        except BaseException:
            for source, destination in reversed(installed):
                os.replace(destination, source)
            raise

    _broadcast_root_call(comm, publish)
    sources = tuple(path for path, _, _ in batches)
    return ArtifactSet(
        table_path,
        binary_paths,
        nonbinary_paths,
        tuple(outputs),
        sources,
    )
