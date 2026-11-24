"""Build subsample, period-limit, and error-floor PDF grids."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from rvbinary.config import RVBinaryConfig
from rvbinary.io.legacy import MockStore, open_mock_store
from rvbinary.io.observations import read_epochs, read_member_ids, select_members
from rvbinary.mock.writer import ArtifactSet, _config_digest, _floor_label
from rvbinary.parallel import Communicator, _broadcast_root_call, partition_indices
from rvbinary.pdf.histogram import (
    BetaSample,
    PDFModel,
    apply_period_limit,
    beta_edges,
    histogram_counts,
    pdf_from_counts,
)
from rvbinary.statistics import star_statistic


@dataclass(frozen=True)
class PDFArtifact:
    """One persisted PDF grid point."""

    path: Path
    subsample: str
    error_floor_kms: float | None
    period_limit_days: float | None


def _artifact_digest(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        payload = {
            "schema": str(handle.attrs.get("schema", "")),
            "package_version": str(handle.attrs.get("package_version", "")),
            "config_digest": str(handle.attrs.get("config_digest", "")),
            "seed": int(handle.attrs.get("seed", 0)),
            "source_hashes": str(handle.attrs.get("source_hashes", "")),
            "statistic_mode": str(handle.attrs.get("statistic_mode", "")),
            "payloads": sorted(handle.get("payloads", {})),
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _preflight(config: RVBinaryConfig, artifacts: ArtifactSet) -> dict[str, str]:
    paths = {
        "table": artifacts.table_index,
        **{
            f"binary:{floor}": path
            for floor, path in artifacts.binary_beta_indexes.items()
        },
        **{
            f"nonbinary:{floor}": path
            for floor, path in artifacts.nonbinary_beta_indexes.items()
        },
    }
    expected_floors = {float(value) for value in config.mock.error_floor.values_kms}
    if set(artifacts.binary_beta_indexes) != expected_floors or set(
        artifacts.nonbinary_beta_indexes
    ) != expected_floors:
        raise ValueError("mock error-floor products do not match the configured grid")
    digests = {}
    for name, path in paths.items():
        with h5py.File(path, "r") as handle:
            mode = str(handle.attrs.get("statistic_mode", ""))
        if mode != config.statistics.mode:
            raise ValueError(
                f"statistic mode mismatch for {name}: {mode!r} != "
                f"{config.statistics.mode!r}"
            )
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("config_digest", "")) != _config_digest(config):
                raise ValueError(
                    f"incompatible config_digest/noise model for {name}; "
                    "regenerate native mocks in a new output directory"
                )
        digests[name] = _artifact_digest(path)
    return digests


def _periods_by_star(
    table_store: MockStore, sample_id: int, targetids: np.ndarray
) -> np.ndarray:
    sample = table_store.sample(sample_id)
    ids = np.asarray(sample.arrays["TARGETID"], dtype=np.int64)
    periods = np.asarray(sample.arrays["PERIOD"], dtype=np.float64)
    result = np.empty(targetids.size, dtype=np.float64)
    for index, targetid in enumerate(targetids):
        values = periods[ids == targetid]
        if values.size == 0 or not np.allclose(values, values[0], rtol=0.0, atol=0.0):
            raise ValueError(f"PERIOD is missing or inconsistent for TARGETID {targetid}")
        result[index] = values[0]
    return result


def _beta_by_star(
    store: MockStore, sample_id: int, targetids: np.ndarray, mode: str
) -> BetaSample:
    sample = store.sample(sample_id)
    ids = np.asarray(sample.arrays["TARGETID"], dtype=np.int64)
    values = np.asarray(sample.arrays["BETA"], dtype=np.float64)
    groups = tuple(values[ids == targetid] for targetid in targetids)
    missing = [int(targetid) for targetid, group in zip(targetids, groups) if group.size == 0]
    if missing:
        raise ValueError(f"mock beta products are missing TARGETID values: {missing}")
    return BetaSample(groups, mode)  # type: ignore[arg-type]


def _period_path_label(limit: float | None) -> str:
    if limit is None:
        return "unrestricted"
    bits = int(np.asarray(limit, dtype=np.float64).view(np.uint64).item())
    return f"days-float64-{bits:016x}"


def _save_pdf(
    path: Path,
    model: PDFModel,
    *,
    config: RVBinaryConfig,
    sample_count: int,
    draw_count: int,
    period_limit: float | None,
    error_floor: float | None,
    source_digests: dict[str, str],
    error_floor_mode: str = "grid",
    binary_out_of_range_count: int = 0,
    nonbinary_out_of_range_count: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                edges=model.edges,
                binary_density=model.binary_density,
                nonbinary_density=model.nonbinary_density,
                binary_counts=model.binary_counts,
                nonbinary_counts=model.nonbinary_counts,
                statistic_mode=config.statistics.mode,
                legacy_statistic=config.statistics.mode == "legacy",
                sample_count=sample_count,
                draw_count=draw_count,
                period_limit_days=np.nan if period_limit is None else period_limit,
                error_floor_kms=np.nan if error_floor is None else error_floor,
                error_floor_mode=error_floor_mode,
                binary_out_of_range_count=binary_out_of_range_count,
                nonbinary_out_of_range_count=nonbinary_out_of_range_count,
                source_digests=json.dumps(
                    source_digests, sort_keys=True, separators=(",", ":")
                ),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _legacy_paths(config: RVBinaryConfig) -> tuple[Path, Path, Path]:
    precomputed = config.mock.precomputed
    if precomputed is None or precomputed.format != "binary2":
        raise ValueError("a precomputed Binary2 mock directory is required")
    root = precomputed.directory
    paths = (
        root / "mcs_table_combined.h5",
        root / "mcs_beta_combined.h5",
        root / "mcs_nberr_combined.h5",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("missing precomputed Binary2 file(s): " + ", ".join(missing))
    return paths


def _legacy_star_order(table_path: Path) -> np.ndarray:
    with h5py.File(table_path, "r") as handle:
        try:
            item = handle["table_0"]
            if isinstance(item, h5py.Group):
                name = "TARGETID" if "TARGETID" in item else "ID"
                epoch_ids = np.asarray(item[name][()], dtype=np.int64)
            elif isinstance(item, h5py.Dataset):
                names = item.dtype.names or ()
                name = "TARGETID" if "TARGETID" in names else "ID"
                epoch_ids = np.asarray(item.fields(name)[()], dtype=np.int64)
            else:
                raise TypeError("table_0 must be a group or compound dataset")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"unable to read legacy star order from {table_path}") from exc
    if epoch_ids.ndim != 1 or epoch_ids.size == 0:
        raise ValueError("legacy table_0 identifiers must be one-dimensional and non-empty")
    _, first = np.unique(epoch_ids, return_index=True)
    return epoch_ids[np.sort(first)]


def _legacy_source_digests(paths: tuple[Path, Path, Path]) -> dict[str, str]:
    return {
        path.name: f"size={path.stat().st_size};mtime_ns={path.stat().st_mtime_ns}"
        for path in paths
    }


def _validate_legacy_sample_coverage(
    paths: tuple[Path, Path, Path], sample_count: int
) -> None:
    expected = list(range(sample_count))
    pattern = re.compile(r"list_(\d+)$")
    for path in paths[1:]:
        with h5py.File(path, "r") as handle:
            actual = sorted(
                int(match.group(1))
                for name in handle.keys()
                if (match := pattern.fullmatch(name)) is not None
            )
        if actual != expected:
            raise ValueError(
                f"legacy sample coverage in {path} does not match 0:{sample_count}"
            )


def _legacy_setup(config: RVBinaryConfig) -> tuple[tuple[Path, Path, Path], np.ndarray]:
    paths = _legacy_paths(config)
    _validate_legacy_sample_coverage(paths, config.mock.samples)
    return paths, _legacy_star_order(paths[0])


def _legacy_member_positions(
    config: RVBinaryConfig, targetids: np.ndarray
) -> dict[str, np.ndarray]:
    position_by_id = {int(targetid): index for index, targetid in enumerate(targetids)}
    result = {}
    for subsample in config.observations.subsamples:
        member_ids = read_member_ids(subsample.members)
        missing = [int(value) for value in member_ids if int(value) not in position_by_id]
        if missing:
            raise ValueError(
                f"subsample {subsample.name} has TARGETID values absent from legacy mocks: "
                + ", ".join(str(value) for value in missing)
            )
        result[subsample.name] = np.array(
            [position_by_id[int(value)] for value in member_ids], dtype=np.int64
        )
    return result


def _legacy_sample_statistics(
    handle: h5py.File,
    sample_id: int,
    positions: np.ndarray,
    mode: str,
) -> np.ndarray:
    try:
        dataset = handle[f"list_{sample_id}/arrays"]
    except (KeyError, OSError) as exc:
        raise ValueError(f"legacy mock is missing list_{sample_id}/arrays") from exc
    if dataset.ndim != 1 or positions.size and int(np.max(positions)) >= dataset.size:
        raise ValueError(f"legacy list_{sample_id}/arrays does not match table_0 star order")
    arrays = dataset[np.sort(positions)]
    groups = [np.asarray(values, dtype=np.float64) for values in arrays]
    return star_statistic(groups, mode=mode)  # type: ignore[arg-type]


def _legacy_histogram_counts(
    values: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, int]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("legacy histogram values must be a non-empty finite vector")
    included = (samples >= edges[0]) & (samples <= edges[-1])
    counts = np.histogram(samples[included], bins=edges)[0].astype(
        np.int64, copy=False
    )
    return counts, int(np.count_nonzero(~included))


def build_precomputed_pdf_grid(
    config: RVBinaryConfig,
    comm: Communicator | None = None,
) -> list[PDFArtifact]:
    """Stream unrestricted PDFs from a precomputed Binary2 three-file store."""
    rank = 0 if comm is None else comm.Get_rank()
    size = 1 if comm is None else comm.Get_size()
    if comm is None:
        paths, targetids = _legacy_setup(config)
    else:
        paths, raw_targetids = cast(
            tuple[tuple[Path, Path, Path], np.ndarray],
            _broadcast_root_call(comm, lambda: _legacy_setup(config)),
        )
        targetids = np.asarray(raw_targetids, dtype=np.int64)
    _, binary_path, nonbinary_path = paths
    edges = beta_edges(config.pdf.bins)
    output: list[PDFArtifact] = []
    if comm is None:
        positions_by_subsample = _legacy_member_positions(config, targetids)
    else:
        positions_by_subsample = cast(
            dict[str, np.ndarray],
            _broadcast_root_call(
                comm, lambda: _legacy_member_positions(config, targetids)
            ),
        )
    union_positions = np.unique(np.concatenate(list(positions_by_subsample.values())))
    union_lookup = {int(position): index for index, position in enumerate(union_positions)}
    union_indices = {
        name: np.array([union_lookup[int(position)] for position in positions], dtype=np.int64)
        for name, positions in positions_by_subsample.items()
    }
    local_binary = {
        name: np.zeros(edges.size - 1, dtype=np.int64) for name in positions_by_subsample
    }
    local_nonbinary = {
        name: np.zeros(edges.size - 1, dtype=np.int64) for name in positions_by_subsample
    }
    local_outside = {name: [0, 0] for name in positions_by_subsample}

    local_sample_ids: range | list[int] = partition_indices(
        range(config.mock.samples), rank, size
    )

    def accumulate() -> None:
        with h5py.File(binary_path, "r") as binary, h5py.File(
            nonbinary_path, "r"
        ) as nonbinary:
            for sample_id in local_sample_ids:
                binary_statistics = _legacy_sample_statistics(
                    binary, sample_id, union_positions, config.statistics.mode
                )
                nonbinary_statistics = _legacy_sample_statistics(
                    nonbinary, sample_id, union_positions, config.statistics.mode
                )
                for name, indices in union_indices.items():
                    sample_counts, outside = _legacy_histogram_counts(
                        binary_statistics[indices], edges
                    )
                    local_binary[name] += sample_counts
                    local_outside[name][0] += outside
                    sample_counts, outside = _legacy_histogram_counts(
                        nonbinary_statistics[indices], edges
                    )
                    local_nonbinary[name] += sample_counts
                    local_outside[name][1] += outside

    if comm is None:
        accumulate()
    else:
        local_error: str | None = None
        try:
            accumulate()
        except BaseException as exc:
            local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        errors = [error for error in comm.allgather(local_error) if error is not None]
        if errors:
            raise RuntimeError(
                "MPI legacy PDF generation failed; " + "; ".join(str(error) for error in errors)
            )

    for subsample in config.observations.subsamples:
        name = subsample.name
        if comm is None:
            counts = local_binary[name], local_nonbinary[name]
            outside_counts = tuple(local_outside[name])
        else:
            binary_counts = np.zeros_like(local_binary[name])
            nonbinary_counts = np.zeros_like(local_nonbinary[name])
            outside_counts_array = np.zeros(2, dtype=np.int64)
            comm.Allreduce(local_binary[name], binary_counts)
            comm.Allreduce(local_nonbinary[name], nonbinary_counts)
            comm.Allreduce(
                np.asarray(local_outside[name], dtype=np.int64),
                outside_counts_array,
            )
            counts = binary_counts, nonbinary_counts
            outside_counts = (
                int(outside_counts_array[0]),
                int(outside_counts_array[1]),
            )

        def publish() -> PDFArtifact:
            model = pdf_from_counts(*counts, edges)
            directory = (
                Path(config.project.output)
                / "pdf"
                / subsample.name
                / "errorfloor-per-star"
                / "period-unrestricted"
            )
            path = directory / "pdf.npz"
            _save_pdf(
                path,
                model,
                config=config,
                sample_count=config.mock.samples,
                draw_count=config.mock.samples * len(positions_by_subsample[name]),
                period_limit=None,
                error_floor=None,
                source_digests=_legacy_source_digests(paths),
                error_floor_mode="per_star_baked_in",
                binary_out_of_range_count=outside_counts[0],
                nonbinary_out_of_range_count=outside_counts[1],
            )
            return PDFArtifact(path, subsample.name, None, None)

        if comm is None:
            output.append(publish())
        else:
            artifact = cast(PDFArtifact, _broadcast_root_call(comm, publish))
            if rank == 0:
                output.append(artifact)
    return output


def build_pdf_grid(
    config: RVBinaryConfig,
    mock_store: ArtifactSet,
    comm: Communicator | None = None,
) -> list[PDFArtifact]:
    """Build all configured subsample, period-limit, and error-floor PDFs."""
    rank = 0 if comm is None else comm.Get_rank()
    size = 1 if comm is None else comm.Get_size()
    source_digests = _preflight(config, mock_store)
    periods: list[float | None] = [
        *([None] if config.model.include_unrestricted else []),
        *[float(value) for value in config.model.period_limits_days],
    ]
    if not periods:
        raise ValueError("at least one unrestricted or period-limited model is required")
    edges = beta_edges(config.pdf.bins)
    output: list[PDFArtifact] = []

    with ExitStack() as stack:
        table_store = stack.enter_context(open_mock_store(mock_store.table_index))
        binary_stores = {
            floor: stack.enter_context(open_mock_store(path))
            for floor, path in mock_store.binary_beta_indexes.items()
        }
        nonbinary_stores = {
            floor: stack.enter_context(open_mock_store(path))
            for floor, path in mock_store.nonbinary_beta_indexes.items()
        }
        sample_ids = table_store.sample_ids
        if sample_ids != list(range(config.mock.samples)):
            raise ValueError("mock table sample coverage does not match configuration")
        for stores in (binary_stores, nonbinary_stores):
            if any(store.sample_ids != sample_ids for store in stores.values()):
                raise ValueError("mock table and beta sample coverage do not match")

        available_ids = table_store.targetids
        for subsample in config.observations.subsamples:
            member_ids = read_member_ids(subsample.members)
            if subsample.epochs is not None:
                select_members(read_epochs(subsample.epochs), member_ids)
            missing = np.setdiff1d(member_ids, available_ids)
            if missing.size:
                raise ValueError(
                    f"subsample {subsample.name} has TARGETID values absent from mocks: "
                    + ", ".join(str(int(value)) for value in missing)
                )
            accumulators = {
                (floor, limit): (
                    np.zeros(edges.size - 1, dtype=np.int64),
                    np.zeros(edges.size - 1, dtype=np.int64),
                )
                for floor in binary_stores
                for limit in periods
            }
            local_sample_ids = partition_indices(sample_ids, rank, size)
            for sample_id in local_sample_ids:
                star_periods = _periods_by_star(table_store, sample_id, member_ids)
                for floor, binary_store in binary_stores.items():
                    binary = _beta_by_star(
                        binary_store, sample_id, member_ids, config.statistics.mode
                    )
                    nonbinary = _beta_by_star(
                        nonbinary_stores[floor],
                        sample_id,
                        member_ids,
                        config.statistics.mode,
                    )
                    nonbinary_stats = nonbinary.statistics
                    for limit in periods:
                        selected = (
                            binary
                            if limit is None
                            else apply_period_limit(
                                binary, nonbinary, star_periods, limit
                            )
                        )
                        binary_counts, nonbinary_counts = accumulators[(floor, limit)]
                        binary_counts += histogram_counts(selected.statistics, edges)
                        nonbinary_counts += histogram_counts(nonbinary_stats, edges)

            for floor in binary_stores:
                for limit in periods:
                    local_binary, local_nonbinary = accumulators[(floor, limit)]
                    if comm is None:
                        counts = (local_binary, local_nonbinary)
                    else:
                        binary_counts = np.zeros_like(local_binary)
                        nonbinary_counts = np.zeros_like(local_nonbinary)
                        comm.Allreduce(local_binary, binary_counts)
                        comm.Allreduce(local_nonbinary, nonbinary_counts)
                        counts = (binary_counts, nonbinary_counts)
                    if rank != 0:
                        continue
                    model = pdf_from_counts(*counts, edges)
                    directory = (
                        Path(config.project.output)
                        / "pdf"
                        / subsample.name
                        / f"errorfloor-{_floor_label(floor)}"
                        / f"period-{_period_path_label(limit)}"
                    )
                    path = directory / "pdf.npz"
                    _save_pdf(
                        path,
                        model,
                        config=config,
                        sample_count=len(sample_ids),
                        draw_count=len(sample_ids) * len(member_ids),
                        period_limit=limit,
                        error_floor=floor,
                        source_digests=source_digests,
                    )
                    if config.pdf.export_legacy_npy:
                        np.save(directory / "log_fprob_DM91_0.npy", model.nonbinary_density)
                        np.save(directory / "log_fprob_DM91_1.npy", model.binary_density)
                    output.append(PDFArtifact(path, subsample.name, floor, limit))
    return output
