"""Serial generation of immutable mock shards and atomic indexes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import h5py
import numpy as np

from rvbinary import __version__
from rvbinary.config import RVBinaryConfig
from rvbinary.io.hdf5 import (
    ArtifactMetadata,
    MockSampleRecord,
    build_external_index,
    parse_payload_identity,
    write_mock_shard,
)
from rvbinary.io.legacy import inspect_hdf5, open_mock_store
from rvbinary.io.observations import EpochData, read_epochs
from rvbinary.mock.simulation import NOISE_MODEL, derive_beta_realization, simulate_sample
from rvbinary.models.factory import make_model, model_payload, validate_model_stars

SOURCE_BATCH_SCHEMA = "rvbinary.source-batch/v1"


@dataclass(frozen=True)
class ArtifactSet:
    """Indexes and immutable shards produced by one serial writer operation."""

    table_index: Path
    binary_beta_indexes: Mapping[float, Path]
    nonbinary_beta_indexes: Mapping[float, Path]
    shards: tuple[Path, ...]
    source_batches: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binary_beta_indexes", MappingProxyType(dict(self.binary_beta_indexes))
        )
        object.__setattr__(
            self,
            "nonbinary_beta_indexes",
            MappingProxyType(dict(self.nonbinary_beta_indexes)),
        )
        object.__setattr__(self, "shards", tuple(Path(path) for path in self.shards))
        object.__setattr__(
            self, "source_batches", tuple(Path(path) for path in self.source_batches)
        )


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_payload(config: RVBinaryConfig) -> dict[str, object]:
    return model_payload(config)


def _config_digest(config: RVBinaryConfig) -> str:
    # Runtime paths, requested sample count, and parallel settings do not change
    # the scientific meaning of an individual sample shard.
    return _json_digest(
        {
            "noise_model": NOISE_MODEL,
            "model": _model_payload(config),
            "error_floors_kms": [float(value) for value in config.mock.error_floor.values_kms],
            "statistic_mode": config.statistics.mode,
        }
    )


class _Hasher(Protocol):
    def update(self, data: bytes) -> object: ...


def _hash_array(hasher: _Hasher, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    hasher.update(name.encode("ascii"))
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype=">u8").tobytes())
    hasher.update(array.tobytes())


def _epochs_digest(data: EpochData) -> str:
    return _source_digest(_source_arrays(data))


def _batch_id(data: EpochData) -> str:
    targetids = np.unique(np.asarray(data.targetid, dtype=np.int64))
    hasher = hashlib.sha256()
    _hash_array(hasher, "TARGETID", targetids)
    return f"stars-{hasher.hexdigest()[:20]}"


def _source_path(config: RVBinaryConfig, batch: str) -> Path:
    return Path(config.project.output) / "mock" / "sources" / f"batch-{batch}.h5"


def _source_arrays(data: EpochData) -> dict[str, np.ndarray]:
    order = np.argsort(data.targetid, kind="stable")
    return {
        "TARGETID": np.asarray(data.targetid, dtype=np.int64)[order],
        "MJD": np.asarray(data.mjd, dtype=np.float64)[order],
        "VRAD_ERR": np.asarray(data.vrad_err_kms, dtype=np.float64)[order],
        "Mass": np.asarray(data.mass_solar, dtype=np.float64)[order],
        "meanLOGG": np.asarray(data.logg_cgs, dtype=np.float64)[order],
    }


def _source_digest(arrays: Mapping[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for name in ("TARGETID", "MJD", "VRAD_ERR", "Mass", "meanLOGG"):
        _hash_array(hasher, name, np.asarray(arrays[name]))
    return hasher.hexdigest()


def _write_source_batch(config: RVBinaryConfig, data: EpochData) -> tuple[Path, bool]:
    validate_model_stars(config, data)
    batch = _batch_id(data)
    path = _source_path(config, batch)
    arrays = _source_arrays(data)
    digest = _source_digest(arrays)
    if path.exists():
        existing, existing_digest = _read_source_batch(path)
        if _batch_id(existing) != batch or existing_digest != digest:
            raise ValueError(f"incompatible immutable source batch: {path}")
        return path, False

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = SOURCE_BATCH_SCHEMA
            handle.attrs["batch_id"] = batch
            handle.attrs["digest"] = digest
            handle.attrs["completion"] = False
            for column, values in arrays.items():
                handle.create_dataset(column, data=values)
            handle.flush()
            handle.attrs.modify("completion", True)
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path, True


def _read_source_batch(path: Path) -> tuple[EpochData, str]:
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema") != SOURCE_BATCH_SCHEMA:
                raise ValueError(f"unsupported source-batch schema: {path}")
            if not bool(handle.attrs.get("completion", False)):
                raise ValueError(f"incomplete source batch: {path}")
            required = {"TARGETID", "MJD", "VRAD_ERR", "Mass", "meanLOGG"}
            if set(handle) != required:
                raise ValueError(f"source batch columns mismatch: {path}")
            arrays = {name: np.asarray(handle[name]) for name in required}
            stored_batch = str(handle.attrs.get("batch_id", ""))
            stored_digest = str(handle.attrs.get("digest", ""))
    except OSError as exc:
        raise ValueError(f"unable to read source batch {path}: {exc}") from exc

    digest = _source_digest(arrays)
    data = EpochData(
        targetid=np.asarray(arrays["TARGETID"], dtype=np.int64),
        mjd=np.asarray(arrays["MJD"], dtype=np.float64),
        vrad_kms=np.zeros(len(arrays["TARGETID"]), dtype=np.float64),
        vrad_err_kms=np.asarray(arrays["VRAD_ERR"], dtype=np.float64),
        mass_solar=np.asarray(arrays["Mass"], dtype=np.float64),
        logg_cgs=np.asarray(arrays["meanLOGG"], dtype=np.float64),
    )
    if stored_batch != _batch_id(data):
        raise ValueError(f"source batch TARGETID digest mismatch: {path}")
    if stored_digest != digest:
        raise ValueError(f"source batch content digest mismatch: {path}")
    return data, digest


def _source_batches(config: RVBinaryConfig) -> list[tuple[Path, EpochData, str]]:
    root = Path(config.project.output) / "mock" / "sources"
    batches = []
    for path in sorted(root.glob("batch-*.h5")):
        data, digest = _read_source_batch(path)
        batches.append((path, data, digest))
    return batches


def _canonical_source_hashes(
    config: RVBinaryConfig, batches: list[tuple[Path, EpochData, str]]
) -> dict[str, str]:
    payload = _model_payload(config)
    hashes = {"model-config": _json_digest(payload)}
    if config.model.name == "ms17":
        hashes["ms17-period-table"] = str(payload["table_sha256"])
    for _, data, digest in batches:
        hashes[f"epochs:{_batch_id(data)}"] = digest
    return hashes


def _floor_label(floor: float) -> str:
    bits = int(np.asarray(float(floor), dtype=np.float64).view(np.uint64).item())
    return f"float64-{bits:016x}"


def _paths(config: RVBinaryConfig) -> tuple[Path, dict[float, Path], dict[float, Path]]:
    root = Path(config.project.output) / "mock"
    floors = [float(value) for value in config.mock.error_floor.values_kms]
    table = root / "indexes" / "tables.h5"
    binary = {
        floor: root / "indexes" / "beta" / f"errorfloor-{_floor_label(floor)}.h5"
        for floor in floors
    }
    nonbinary = {
        floor: root / "indexes" / "nberr" / f"errorfloor-{_floor_label(floor)}.h5"
        for floor in floors
    }
    return table, binary, nonbinary


def _shard_paths(
    config: RVBinaryConfig, start: int, end: int, batch: str
) -> tuple[Path, dict[float, Path], dict[float, Path]]:
    root = Path(config.project.output) / "mock"
    stem = f"samples-{start}-{end}__batch-{batch}.h5"
    floors = [float(value) for value in config.mock.error_floor.values_kms]
    table = root / "tables" / stem
    binary = {
        floor: root / "beta" / f"errorfloor-{_floor_label(floor)}" / stem
        for floor in floors
    }
    nonbinary = {
        floor: root / "nberr" / f"errorfloor-{_floor_label(floor)}" / stem
        for floor in floors
    }
    return table, binary, nonbinary


def _all_indexes(config: RVBinaryConfig) -> list[Path]:
    table, binary, nonbinary = _paths(config)
    return [table, *binary.values(), *nonbinary.values()]


def _metadata(
    config: RVBinaryConfig,
    start: int,
    end: int,
    data: EpochData,
) -> ArtifactMetadata:
    epoch_digest = _epochs_digest(data)
    return ArtifactMetadata(
        package_version=__version__,
        config_digest=_config_digest(config),
        seed=config.project.random_seed,
        sample_start=start,
        sample_end=end,
        source_hashes=_canonical_source_hashes(config, [(Path(), data, epoch_digest)]),
        statistic_mode=config.statistics.mode,
    )


def _index_metadata(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        return {
            "package_version": str(handle.attrs["package_version"]),
            "config_digest": str(handle.attrs["config_digest"]),
            "seed": int(handle.attrs["seed"]),
            "statistic_mode": str(handle.attrs["statistic_mode"]),
        }


def _validate_existing(config: RVBinaryConfig, requested: range | None) -> list[int]:
    indexes = _all_indexes(config)
    present = [path.exists() for path in indexes]
    if any(present) and not all(present):
        raise ValueError("mock artifact set is incomplete")
    if not any(present):
        return []

    expected = {
        "package_version": __version__,
        "config_digest": _config_digest(config),
        "seed": config.project.random_seed,
        "statistic_mode": config.statistics.mode,
    }
    sample_ids: list[int] | None = None
    canonical_hashes: dict[str, str] | None = None
    for path in indexes:
        report = inspect_hdf5(path)
        if not report.usable:
            raise ValueError(f"existing mock index is not usable: {path}")
        metadata = _index_metadata(path)
        mismatched = [name for name, value in expected.items() if metadata[name] != value]
        if mismatched:
            raise ValueError(f"incompatible index metadata: {', '.join(mismatched)}")
        with h5py.File(path, "r") as handle:
            source_hashes = json.loads(str(handle.attrs["source_hashes"]))
        if canonical_hashes is None:
            canonical_hashes = source_hashes
        elif source_hashes != canonical_hashes:
            raise ValueError("incompatible source_hashes across mock indexes")
        if sample_ids is None:
            sample_ids = report.sample_ids
        elif report.sample_ids != sample_ids:
            raise ValueError("mock indexes contain incompatible sample coverage")
    persisted_hashes = _canonical_source_hashes(config, _source_batches(config))
    if canonical_hashes != persisted_hashes:
        raise ValueError("incompatible source_hashes between indexes and source batches")
    existing = sample_ids or []
    if requested is not None and set(existing).intersection(requested):
        raise ValueError("requested sample range overlaps existing samples")
    return existing


def _normalize_range(config: RVBinaryConfig, sample_range: range | None) -> range:
    requested = range(config.mock.samples) if sample_range is None else sample_range
    if not isinstance(requested, range):
        raise TypeError("sample_range must be a range or None")
    if requested.step != 1 or len(requested) == 0 or requested.start < 0:
        raise ValueError("sample_range must be a non-empty contiguous non-negative range")
    if requested.stop > config.mock.samples:
        raise ValueError("sample_range exceeds configured mock.samples")
    return requested


def _write_shards(
    config: RVBinaryConfig,
    data: EpochData,
    sample_range: range,
) -> tuple[Path, dict[float, Path], dict[float, Path]]:
    start, end = sample_range.start, sample_range.stop
    batch = _batch_id(data)
    table_path, binary_paths, nonbinary_paths = _shard_paths(config, start, end, batch)
    paths = [table_path, *binary_paths.values(), *nonbinary_paths.values()]
    collisions = [path for path in paths if path.exists()]
    if collisions:
        raise ValueError(f"immutable shard already exists: {collisions[0]}")

    table_records: list[MockSampleRecord] = []
    binary_records: dict[float, list[MockSampleRecord]] = {
        floor: [] for floor in binary_paths
    }
    nonbinary_records: dict[float, list[MockSampleRecord]] = {
        floor: [] for floor in nonbinary_paths
    }
    validate_model_stars(config, data)
    model = make_model(config)
    for sample_id in sample_range:
        sample = simulate_sample(data, model, config.project.random_seed, sample_id)
        table_records.append(MockSampleRecord(sample_id, batch, sample.arrays))
        for floor in binary_paths:
            beta = derive_beta_realization(
                sample, floor, config.project.random_seed, config.statistics.mode
            )
            binary_ids, binary_values = beta.flattened("binary")
            nonbinary_ids, nonbinary_values = beta.flattened("nonbinary")
            binary_records[floor].append(
                MockSampleRecord(
                    sample_id, batch, {"TARGETID": binary_ids, "BETA": binary_values}
                )
            )
            nonbinary_records[floor].append(
                MockSampleRecord(
                    sample_id,
                    batch,
                    {"TARGETID": nonbinary_ids, "BETA": nonbinary_values},
                )
            )

    metadata = _metadata(config, start, end, data)
    write_mock_shard(table_path, table_records, metadata)
    for floor, path in binary_paths.items():
        write_mock_shard(path, binary_records[floor], metadata)
    for floor, path in nonbinary_paths.items():
        write_mock_shard(path, nonbinary_records[floor], metadata)
    return table_path, binary_paths, nonbinary_paths


def _candidate_path(index: Path) -> Path:
    index.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{index.name}.", suffix=".candidate", dir=index.parent
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _backup_path(index: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{index.name}.", suffix=".backup", dir=index.parent
    )
    os.close(descriptor)
    return Path(name)


def _replace_indexes(
    index_to_shards: Mapping[Path, list[Path]], *, append: bool
) -> None:
    candidates: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    installed: list[Path] = []
    retained_backups: set[Path] = set()
    try:
        for index, shards in index_to_shards.items():
            candidate = _candidate_path(index)
            candidates[index] = candidate
            if append:
                shutil.copy2(index, candidate)
            build_external_index(shards, candidate, append=append)
            report = inspect_hdf5(candidate)
            if not report.usable:
                raise ValueError(f"candidate mock index failed validation: {index}")
        for index in candidates:
            if index.exists():
                backup = _backup_path(index)
                shutil.copy2(index, backup)
                backups[index] = backup
            else:
                backups[index] = None
        try:
            for index, candidate in candidates.items():
                os.replace(candidate, index)
                installed.append(index)
        except BaseException as publication_error:
            rollback_failures: list[tuple[Path | None, Path, BaseException]] = []
            for index in reversed(installed):
                rollback_backup = backups[index]
                try:
                    if rollback_backup is None:
                        index.unlink(missing_ok=True)
                    else:
                        os.replace(rollback_backup, index)
                except BaseException as exc:
                    if rollback_backup is not None:
                        retained_backups.add(rollback_backup)
                    rollback_failures.append((rollback_backup, index, exc))
            if rollback_failures:
                details = "; ".join(
                    f"backup {backup} -> live {index}: {type(error).__name__}: {error}"
                    for backup, index, error in rollback_failures
                )
                raise RuntimeError(
                    f"mock index rollback failed; retained recovery path(s): {details}"
                ) from publication_error
            raise
    finally:
        for candidate in candidates.values():
            candidate.unlink(missing_ok=True)
        for cleanup_backup in backups.values():
            if cleanup_backup is not None and cleanup_backup not in retained_backups:
                cleanup_backup.unlink(missing_ok=True)


def _linked_shards(indexes: list[Path]) -> tuple[Path, ...]:
    targets: set[Path] = set()
    for index in indexes:
        physical = False
        with h5py.File(index, "r") as handle:
            for identity in handle["payloads"]:
                link = handle["payloads"].get(identity, getlink=True)
                if isinstance(link, h5py.HardLink):
                    physical = True
                    continue
                if not isinstance(link, h5py.ExternalLink):
                    raise ValueError(f"index payload is not an external link: {index}")
                targets.add((index.parent / link.filename).resolve())
        if physical:
            targets.add(index.resolve())
    return tuple(sorted(targets))


def _artifact_set(config: RVBinaryConfig) -> ArtifactSet:
    table, binary, nonbinary = _paths(config)
    indexes = [table, *binary.values(), *nonbinary.values()]
    sources = tuple(path for path, _, _ in _source_batches(config))
    return ArtifactSet(table, binary, nonbinary, _linked_shards(indexes), sources)


def _validate_recovery_shards(
    config: RVBinaryConfig, product: str, shards: list[Path]
) -> None:
    if not shards:
        raise ValueError(f"no completed {product} shards were found")
    expected_metadata = {
        "package_version": __version__,
        "config_digest": _config_digest(config),
        "seed": config.project.random_seed,
        "statistic_mode": config.statistics.mode,
    }
    expected_hashes = _canonical_source_hashes(config, _source_batches(config))
    source_hashes: dict[str, str] = {}
    coverage: dict[str, set[int]] = {}
    for shard in shards:
        report = inspect_hdf5(shard)
        if not report.usable:
            raise ValueError(f"incomplete or invalid {product} shard: {shard}")
        metadata = _index_metadata(shard)
        mismatched = [
            name for name, value in expected_metadata.items() if metadata[name] != value
        ]
        if mismatched:
            raise ValueError(
                f"incompatible {product} shard metadata: {', '.join(mismatched)}"
            )
        with h5py.File(shard, "r") as handle:
            shard_hashes = json.loads(str(handle.attrs["source_hashes"]))
            for name, digest in shard_hashes.items():
                if name in source_hashes and source_hashes[name] != digest:
                    raise ValueError(f"incompatible source hash for {name}")
                source_hashes[name] = digest
            for identity in handle["payloads"]:
                start, end, batch = parse_payload_identity(identity)
                sample_ids = coverage.setdefault(batch, set())
                overlap = sample_ids.intersection(range(start, end))
                if overlap:
                    raise ValueError(
                        f"duplicate sample coverage for {product} batch {batch}"
                    )
                sample_ids.update(range(start, end))

    expected_batches = {
        _batch_id(data) for _, data, _ in _source_batches(config)
    }
    expected_samples = set(range(config.mock.samples))
    if set(coverage) != expected_batches or any(
        sample_ids != expected_samples for sample_ids in coverage.values()
    ):
        raise ValueError(
            f"incomplete sample coverage for {product}; expected samples "
            f"0..{config.mock.samples - 1} for every source batch"
        )
    if source_hashes != expected_hashes:
        raise ValueError(f"incompatible source hashes for {product} shards")


def rebuild_mock_indexes(config: RVBinaryConfig) -> ArtifactSet:
    """Rebuild all external-link indexes from complete immutable shards."""
    table_index, binary_indexes, nonbinary_indexes = _paths(config)
    root = Path(config.project.output) / "mock"
    mapping: dict[Path, list[Path]] = {
        table_index: sorted((root / "tables").glob("samples-*.h5"))
    }
    for floor, index in binary_indexes.items():
        mapping[index] = sorted(
            (root / "beta" / f"errorfloor-{_floor_label(floor)}").glob(
                "samples-*.h5"
            )
        )
    for floor, index in nonbinary_indexes.items():
        mapping[index] = sorted(
            (root / "nberr" / f"errorfloor-{_floor_label(floor)}").glob(
                "samples-*.h5"
            )
        )

    for index, shards in mapping.items():
        _validate_recovery_shards(config, str(index.relative_to(root)), shards)
    _replace_indexes(mapping, append=False)
    return _artifact_set(config)


def generate_mock(
    config: RVBinaryConfig, sample_range: range | None = None
) -> ArtifactSet:
    """Generate a serial sample shard and atomically append it to all indexes."""
    requested = _normalize_range(config, sample_range)
    existing = _validate_existing(config, requested)
    batches = _source_batches(config)
    created_source: Path | None = None
    if not batches:
        data = read_epochs(config.observations.mock_epochs)
        source_path, created = _write_source_batch(config, data)
        created_source = source_path if created else None
        batches = _source_batches(config)
    table_index, binary_indexes, nonbinary_indexes = _paths(config)
    mapping: dict[Path, list[Path]] = {
        path: [] for path in [table_index, *binary_indexes.values(), *nonbinary_indexes.values()]
    }
    try:
        for _, data, _ in batches:
            table_shard, binary_shards, nonbinary_shards = _write_shards(
                config, data, requested
            )
            mapping[table_index].append(table_shard)
            for floor, path in binary_shards.items():
                mapping[binary_indexes[floor]].append(path)
            for floor, path in nonbinary_shards.items():
                mapping[nonbinary_indexes[floor]].append(path)
        _replace_indexes(mapping, append=bool(existing))
    except Exception:
        if created_source is not None:
            created_source.unlink(missing_ok=True)
        raise
    return _artifact_set(config)


def extend_mock_stars(config: RVBinaryConfig, new_epochs: EpochData) -> ArtifactSet:
    """Append one disjoint star batch across every existing sample."""
    existing = _validate_existing(config, None)
    if not existing:
        raise ValueError("cannot extend stars before generating mock samples")
    if existing != list(range(existing[0], existing[-1] + 1)):
        raise ValueError("star extension requires contiguous existing sample coverage")
    table_index, _, _ = _paths(config)
    with open_mock_store(table_index) as store:
        duplicate = np.intersect1d(store.targetids, np.unique(new_epochs.targetid))
    if duplicate.size:
        listed = ", ".join(str(int(value)) for value in duplicate)
        raise ValueError(f"duplicate TARGETID across star batches: {listed}")

    requested = range(existing[0], existing[-1] + 1)
    source_path, created_source = _write_source_batch(config, new_epochs)
    table_index, binary_indexes, nonbinary_indexes = _paths(config)
    try:
        table_shard, binary_shards, nonbinary_shards = _write_shards(
            config, new_epochs, requested
        )
        mapping: dict[Path, list[Path]] = {table_index: [table_shard]}
        mapping.update(
            {binary_indexes[floor]: [path] for floor, path in binary_shards.items()}
        )
        mapping.update(
            {
                nonbinary_indexes[floor]: [path]
                for floor, path in nonbinary_shards.items()
            }
        )
        _replace_indexes(mapping, append=True)
    except BaseException:
        if created_source:
            source_path.unlink(missing_ok=True)
        raise
    return _artifact_set(config)
