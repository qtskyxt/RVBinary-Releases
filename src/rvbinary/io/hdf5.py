"""Versioned HDF5 storage for immutable mock-sample payloads."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = "rvbinary.mock/v1"
_PAYLOAD_RE = re.compile(
    r"^samples-(?P<start>\d+)-(?P<end>\d+)__batch-(?P<batch>[A-Za-z0-9_.-]+)$"
)


@dataclass(frozen=True)
class ArtifactMetadata:
    """Reproducibility metadata shared by native mock artifacts."""

    package_version: str
    config_digest: str
    seed: int
    sample_start: int
    sample_end: int
    source_hashes: Mapping[str, str]
    statistic_mode: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class MockSampleRecord:
    """One storage payload identified by sample number and immutable star batch."""

    sample_id: int
    star_batch_id: str
    arrays: Mapping[str, NDArray[np.generic]]

    @property
    def targetid(self) -> NDArray[np.generic]:
        """Return canonical target identifiers without numeric conversion."""
        return np.asarray(self.arrays["TARGETID"])


def payload_identity(sample_start: int, sample_end: int, star_batch_id: str) -> str:
    """Build the stable on-disk identity for a sample range and star batch."""
    if sample_start < 0 or sample_end <= sample_start:
        raise ValueError("sample range must be a non-empty half-open interval")
    if not star_batch_id or re.fullmatch(r"[A-Za-z0-9_.-]+", star_batch_id) is None:
        raise ValueError("star_batch_id may contain only letters, digits, '.', '_', and '-'")
    return f"samples-{sample_start}-{sample_end}__batch-{star_batch_id}"


def parse_payload_identity(identity: str) -> tuple[int, int, str]:
    """Parse an on-disk payload identity."""
    match = _PAYLOAD_RE.fullmatch(identity)
    if match is None:
        raise ValueError(f"invalid payload identity: {identity}")
    start = int(match.group("start"))
    end = int(match.group("end"))
    if end <= start:
        raise ValueError(f"invalid payload identity: {identity}")
    return start, end, match.group("batch")


def _validate_metadata(metadata: ArtifactMetadata) -> None:
    if metadata.sample_start < 0 or metadata.sample_end <= metadata.sample_start:
        raise ValueError("sample interval must be a non-empty half-open interval")
    if metadata.statistic_mode not in {"consistent", "legacy"}:
        raise ValueError(f"unsupported statistic mode: {metadata.statistic_mode}")
    if not metadata.package_version or not metadata.config_digest or not metadata.created_at:
        raise ValueError("package_version, config_digest, and created_at must be non-empty")
    if any(not key or not value for key, value in metadata.source_hashes.items()):
        raise ValueError("source hashes must have non-empty paths and digests")


def _validated_records(
    samples: Iterable[MockSampleRecord], metadata: ArtifactMetadata
) -> list[MockSampleRecord]:
    records = list(samples)
    if not records:
        raise ValueError("at least one mock sample record is required")
    sample_ids_by_batch: dict[str, set[int]] = {}
    for record in records:
        payload_identity(metadata.sample_start, metadata.sample_end, record.star_batch_id)
        sample_ids = sample_ids_by_batch.setdefault(record.star_batch_id, set())
        if record.sample_id in sample_ids:
            raise ValueError(
                "duplicate payload identity: "
                f"sample {record.sample_id}, batch {record.star_batch_id}"
            )
        sample_ids.add(record.sample_id)
        if not metadata.sample_start <= record.sample_id < metadata.sample_end:
            raise ValueError(f"sample {record.sample_id} is outside the metadata interval")
        identity = payload_identity(
            metadata.sample_start, metadata.sample_end, record.star_batch_id
        )
        if "TARGETID" not in record.arrays:
            raise ValueError(f"{identity} is missing TARGETID")
        if not record.arrays:
            raise ValueError(f"{identity} has no arrays")
        row_count: int | None = None
        for name, values in record.arrays.items():
            array = np.asarray(values)
            if not name or "/" in name:
                raise ValueError(f"{identity} contains an invalid array name: {name!r}")
            if array.ndim == 0:
                raise ValueError(f"{identity}/{name} must have at least one dimension")
            if row_count is None:
                row_count = len(array)
            elif len(array) != row_count:
                raise ValueError(f"{identity} arrays must share their first dimension")
        targetid = np.asarray(record.arrays["TARGETID"])
        if targetid.ndim != 1 or targetid.dtype.kind not in "iu":
            raise ValueError(f"{identity}/TARGETID must be a one-dimensional integer array")
    expected = set(range(metadata.sample_start, metadata.sample_end))
    for batch, sample_ids in sample_ids_by_batch.items():
        if sample_ids != expected:
            raise ValueError(f"batch {batch} does not cover the complete sample interval")
    return records


def _write_metadata(
    handle: h5py.File, metadata: ArtifactMetadata, *, completion: bool
) -> None:
    handle.attrs["schema"] = SCHEMA_VERSION
    handle.attrs["package_version"] = metadata.package_version
    handle.attrs["config_digest"] = metadata.config_digest
    handle.attrs["seed"] = metadata.seed
    handle.attrs["sample_start"] = metadata.sample_start
    handle.attrs["sample_end"] = metadata.sample_end
    handle.attrs["sample_stop"] = metadata.sample_end
    handle.attrs["source_hashes"] = json.dumps(
        dict(metadata.source_hashes), sort_keys=True, separators=(",", ":")
    )
    handle.attrs["completion"] = completion
    handle.attrs["statistic_mode"] = metadata.statistic_mode
    handle.attrs["created_at"] = metadata.created_at


def _metadata_from_attrs(attrs: h5py.AttributeManager) -> ArtifactMetadata:
    required = (
        "package_version",
        "config_digest",
        "seed",
        "sample_start",
        "sample_end",
        "source_hashes",
        "statistic_mode",
        "created_at",
    )
    missing = [name for name in required if name not in attrs]
    if missing:
        raise ValueError(f"artifact is missing metadata: {', '.join(missing)}")
    sample_end = int(attrs["sample_end"])
    if "sample_stop" in attrs and int(attrs["sample_stop"]) != sample_end:
        raise ValueError("artifact sample_end and sample_stop metadata disagree")
    return ArtifactMetadata(
        package_version=str(attrs["package_version"]),
        config_digest=str(attrs["config_digest"]),
        seed=int(attrs["seed"]),
        sample_start=int(attrs["sample_start"]),
        sample_end=sample_end,
        source_hashes=json.loads(str(attrs["source_hashes"])),
        statistic_mode=str(attrs["statistic_mode"]),
        created_at=str(attrs["created_at"]),
    )


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def write_mock_shard(
    path: Path, samples: Iterable[MockSampleRecord], metadata: ArtifactMetadata
) -> None:
    """Atomically write native mock records to one physical HDF5 shard."""
    path = Path(path)
    _validate_metadata(metadata)
    records = _validated_records(samples, metadata)
    temporary = _temporary_path(path)
    try:
        with h5py.File(temporary, "w") as handle:
            _write_metadata(handle, metadata, completion=False)
            payloads = handle.create_group("payloads", track_order=True)
            records_by_batch: dict[str, list[MockSampleRecord]] = {}
            for record in records:
                records_by_batch.setdefault(record.star_batch_id, []).append(record)
            for batch, batch_records in sorted(records_by_batch.items()):
                identity = payload_identity(metadata.sample_start, metadata.sample_end, batch)
                group = payloads.create_group(identity)
                group.attrs["sample_start"] = metadata.sample_start
                group.attrs["sample_end"] = metadata.sample_end
                group.attrs["star_batch_id"] = batch
                sample_groups = group.create_group("samples", track_order=True)
                for record in sorted(batch_records, key=lambda item: item.sample_id):
                    sample_group = sample_groups.create_group(str(record.sample_id))
                    sample_group.attrs["array_names"] = json.dumps(sorted(record.arrays))
                    for name, values in record.arrays.items():
                        sample_group.create_dataset(name, data=np.asarray(values))
            handle.flush()
            handle.attrs.modify("completion", True)
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_native_complete(path: Path) -> tuple[ArtifactMetadata, list[str]]:
    if not path.is_file():
        raise ValueError(f"HDF5 shard does not exist: {path}")
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema") != SCHEMA_VERSION:
                raise ValueError(f"unsupported HDF5 schema in {path}")
            if not bool(handle.attrs.get("completion", False)):
                raise ValueError(f"incomplete HDF5 artifact: {path}")
            metadata = _metadata_from_attrs(handle.attrs)
            if "payloads" not in handle:
                raise ValueError(f"native HDF5 artifact has no payloads group: {path}")
            identities = list(handle["payloads"])
            for identity in identities:
                parse_payload_identity(identity)
                try:
                    handle["payloads"][identity]
                except KeyError as exc:
                    raise ValueError(f"broken payload link in {path}: {identity}") from exc
    except OSError as exc:
        raise ValueError(f"unable to open HDF5 shard {path}: {exc}") from exc
    return metadata, identities


def _compatible(reference: ArtifactMetadata, candidate: ArtifactMetadata) -> None:
    fields = ("package_version", "config_digest", "seed", "statistic_mode")
    mismatched = [name for name in fields if getattr(reference, name) != getattr(candidate, name)]
    if mismatched:
        raise ValueError(f"incompatible shard metadata: {', '.join(mismatched)}")


def _validate_payload_ranges(identities: Iterable[str]) -> None:
    ranges_by_batch: dict[str, list[tuple[int, int]]] = {}
    for identity in identities:
        start, end, batch = parse_payload_identity(identity)
        ranges_by_batch.setdefault(batch, []).append((start, end))
    for batch, ranges in ranges_by_batch.items():
        previous_end = -1
        for start, end in sorted(ranges):
            if start < previous_end:
                raise ValueError(f"duplicate payload identity for batch {batch}")
            previous_end = end


def _existing_link_targets(index: Path) -> tuple[ArtifactMetadata, dict[str, tuple[Path, str]]]:
    metadata, identities = _require_native_complete(index)
    targets: dict[str, tuple[Path, str]] = {}
    with h5py.File(index, "r") as handle:
        for identity in identities:
            link = handle["payloads"].get(identity, getlink=True)
            if not isinstance(link, h5py.ExternalLink):
                raise ValueError("append requires an external-link index")
            targets[identity] = ((index.parent / link.filename).resolve(), link.path)
    return metadata, targets


def build_external_index(
    shards: Iterable[Path], output: Path, append: bool = False
) -> None:
    """Atomically build an index of relative links to immutable shard payloads."""
    output = Path(output).resolve()
    shard_paths = [Path(path).resolve() for path in shards]
    if not shard_paths:
        raise ValueError("at least one shard is required")
    if output.resolve() in shard_paths:
        raise ValueError("output path cannot also be a shard")

    reference: ArtifactMetadata | None = None
    targets: dict[str, tuple[Path, str]] = {}
    if append and output.exists():
        reference, targets = _existing_link_targets(output)

    source_hashes: dict[str, str] = dict(reference.source_hashes) if reference else {}
    for shard in shard_paths:
        metadata, identities = _require_native_complete(shard)
        if reference is None:
            reference = metadata
        else:
            _compatible(reference, metadata)
        for name, digest in metadata.source_hashes.items():
            if name in source_hashes and source_hashes[name] != digest:
                raise ValueError(f"incompatible source hash for {name}")
            source_hashes[name] = digest
        for identity in identities:
            if identity in targets:
                raise ValueError(f"duplicate payload identity: {identity}")
            targets[identity] = (shard, f"/payloads/{identity}")

    assert reference is not None
    _validate_payload_ranges(targets)
    ranges = [parse_payload_identity(identity)[:2] for identity in targets]
    combined = ArtifactMetadata(
        package_version=reference.package_version,
        config_digest=reference.config_digest,
        seed=reference.seed,
        sample_start=min(start for start, _ in ranges),
        sample_end=max(end for _, end in ranges),
        source_hashes=source_hashes,
        statistic_mode=reference.statistic_mode,
    )
    temporary = _temporary_path(output)
    try:
        with h5py.File(temporary, "w") as handle:
            _write_metadata(handle, combined, completion=False)
            payloads = handle.create_group("payloads", track_order=True)
            for identity, (target_file, target_path) in sorted(targets.items()):
                relative = os.path.relpath(target_file, output.parent)
                payloads[identity] = h5py.ExternalLink(relative, target_path)
            handle.flush()
            for identity in payloads:
                try:
                    payloads[identity]
                except KeyError as exc:
                    raise ValueError(f"broken payload link: {identity}") from exc
            handle.attrs.modify("completion", True)
            handle.flush()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_dataset(source: h5py.Dataset, destination: h5py.Group, block_bytes: int) -> None:
    dataset = destination.create_dataset(
        source.name.rsplit("/", 1)[-1], shape=source.shape, dtype=source.dtype
    )
    for name, value in source.attrs.items():
        dataset.attrs[name] = value
    if source.shape == ():
        dataset[()] = source[()]
        return
    rows = max(1, block_bytes // max(1, source.dtype.itemsize * int(np.prod(source.shape[1:]))))
    for start in range(0, source.shape[0], rows):
        stop = min(start + rows, source.shape[0])
        dataset[start:stop] = source[start:stop]


def _copy_group(source: h5py.Group, destination: h5py.Group, block_bytes: int) -> None:
    for name, value in source.attrs.items():
        destination.attrs[name] = value
    for name in source:
        item = source[name]
        if isinstance(item, h5py.Group):
            _copy_group(item, destination.create_group(name), block_bytes)
        else:
            _copy_dataset(item, destination, block_bytes)


def compact_index(index: Path, output: Path, block_bytes: int = 64 << 20) -> None:
    """Atomically materialize an external-link index using bounded block copies."""
    index = Path(index)
    output = Path(output)
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    metadata, identities = _require_native_complete(index)
    temporary = _temporary_path(output)
    try:
        with h5py.File(index, "r") as source, h5py.File(temporary, "w") as destination:
            _write_metadata(destination, metadata, completion=False)
            payloads = destination.create_group("payloads", track_order=True)
            for identity in identities:
                try:
                    source_group = source["payloads"][identity]
                except KeyError as exc:
                    raise ValueError(f"broken payload link: {identity}") from exc
                _copy_group(source_group, payloads.create_group(identity), block_bytes)
            destination.flush()
            destination.attrs.modify("completion", True)
            destination.flush()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
