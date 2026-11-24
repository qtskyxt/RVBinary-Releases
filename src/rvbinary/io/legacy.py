"""Read-only inspection and adapters for native and Binary2 HDF5 artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

import h5py
import numpy as np
from numpy.typing import NDArray

from rvbinary.io.hdf5 import SCHEMA_VERSION, MockSampleRecord, parse_payload_identity

LEGACY_SCHEMA = "legacy.binary2"
_LEGACY_SAMPLE_RE = re.compile(r"^(?:table|list)_(\d+)$")


@dataclass(frozen=True)
class InspectionReport:
    """Structural and usability summary for one HDF5 artifact."""

    path: Path
    schema: str
    sample_ids: list[int]
    dataset_shapes: Mapping[str, tuple[int, ...]]
    broken_links: list[str]
    duplicate_sample_ids: list[int]
    completion: bool
    resumable: bool
    usable: bool
    malformed: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegacyArraySample:
    """One sample from a standalone legacy list or arrays artifact."""

    sample_id: int
    arrays: Mapping[str, NDArray[np.generic]]

    @property
    def values(self) -> NDArray[np.generic]:
        """Return the sole array when the legacy sample has one value payload."""
        if len(self.arrays) != 1:
            raise ValueError("legacy array sample contains multiple named arrays")
        return next(iter(self.arrays.values()))


def _legacy_sample_ids(handle: h5py.File) -> list[int]:
    sample_ids = []
    for name in handle:
        match = _LEGACY_SAMPLE_RE.fullmatch(name)
        if match:
            sample_ids.append(int(match.group(1)))
    if not sample_ids and "arrays" in handle:
        arrays = handle.get("arrays")
        if isinstance(arrays, h5py.Dataset) and arrays.ndim == 1:
            sample_ids.append(0)
        elif isinstance(arrays, h5py.Dataset) and arrays.ndim >= 2:
            sample_ids.extend(range(arrays.shape[0]))
        elif isinstance(arrays, h5py.Group):
            for name in arrays:
                match = _LEGACY_SAMPLE_RE.fullmatch(name)
                if match:
                    sample_ids.append(int(match.group(1)))
    return sorted(sample_ids)


_REQUIRED_NATIVE_METADATA = (
    "schema",
    "package_version",
    "config_digest",
    "seed",
    "sample_start",
    "sample_end",
    "source_hashes",
    "completion",
    "statistic_mode",
    "created_at",
)


def _validate_native_metadata(
    handle: h5py.File, malformed: list[str]
) -> tuple[int, int] | None:
    for name in _REQUIRED_NATIVE_METADATA:
        if name not in handle.attrs:
            malformed.append(f"/@{name}")
    try:
        start = int(handle.attrs["sample_start"])
        end = int(handle.attrs["sample_end"])
        stop = int(handle.attrs.get("sample_stop", end))
        if start < 0 or end <= start or stop != end:
            malformed.append("/@sample_interval")
    except (KeyError, TypeError, ValueError, OverflowError):
        if "/@sample_interval" not in malformed:
            malformed.append("/@sample_interval")
        root_interval = None
    else:
        root_interval = (start, end)
    try:
        source_hashes = json.loads(str(handle.attrs["source_hashes"]))
        if not isinstance(source_hashes, dict) or any(
            not isinstance(name, str) or not isinstance(digest, str)
            for name, digest in source_hashes.items()
        ):
            malformed.append("/@source_hashes")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        malformed.append("/@source_hashes")
    if handle.attrs.get("statistic_mode") not in {"consistent", "legacy"}:
        malformed.append("/@statistic_mode")
    return root_interval


def _validate_payload(
    handle: h5py.File,
    identity: str,
    malformed: list[str],
    broken: list[str],
    shapes: dict[str, tuple[int, ...]],
) -> tuple[int, int, str] | None:
    object_path = f"/payloads/{identity}"
    try:
        start, end, batch = parse_payload_identity(identity)
    except ValueError:
        malformed.append(object_path)
        return None
    try:
        payload = handle[object_path]
    except KeyError:
        broken.append(object_path)
        return start, end, batch
    if not isinstance(payload, h5py.Group):
        malformed.append(object_path)
        return start, end, batch

    expected_attrs = {
        "sample_start": start,
        "sample_end": end,
        "star_batch_id": batch,
    }
    for name, expected in expected_attrs.items():
        attr_path = f"{object_path}/@{name}"
        if name not in payload.attrs or payload.attrs[name] != expected:
            malformed.append(attr_path)

    if "samples" not in payload or not isinstance(payload.get("samples"), h5py.Group):
        malformed.append(f"{object_path}/samples")
        return start, end, batch
    samples = payload["samples"]
    expected_samples = {str(sample_id) for sample_id in range(start, end)}
    actual_samples = set(samples)
    for missing in sorted(expected_samples - actual_samples, key=int):
        malformed.append(f"{object_path}/samples/{missing}")
    for extra in sorted(actual_samples - expected_samples):
        malformed.append(f"{object_path}/samples/{extra}")

    for sample_name in sorted(expected_samples & actual_samples, key=int):
        sample_path = f"{object_path}/samples/{sample_name}"
        sample = samples.get(sample_name)
        if not isinstance(sample, h5py.Group):
            malformed.append(sample_path)
            continue
        manifest_path = f"{sample_path}/@array_names"
        try:
            array_names = json.loads(str(sample.attrs["array_names"]))
            if (
                not isinstance(array_names, list)
                or not array_names
                or any(not isinstance(name, str) or not name for name in array_names)
                or len(set(array_names)) != len(array_names)
                or "TARGETID" not in array_names
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            malformed.append(manifest_path)
            array_names = list(sample)
            if "TARGETID" not in array_names:
                array_names.append("TARGETID")

        for name in array_names:
            dataset_path = f"{sample_path}/{name}"
            link = sample.get(name, getlink=True)
            item = sample.get(name)
            if item is None:
                if isinstance(link, (h5py.ExternalLink, h5py.SoftLink)):
                    broken.append(dataset_path)
                else:
                    malformed.append(dataset_path)
                continue
            if not isinstance(item, h5py.Dataset):
                malformed.append(dataset_path)
                continue
            if name == "TARGETID" and (item.ndim != 1 or item.dtype.kind not in "iu"):
                malformed.append(dataset_path)
            shapes[dataset_path] = item.shape
        for name in set(sample) - set(array_names):
            dataset_path = f"{sample_path}/{name}"
            link = sample.get(name, getlink=True)
            item = sample.get(name)
            if item is None and isinstance(link, (h5py.ExternalLink, h5py.SoftLink)):
                broken.append(dataset_path)
            else:
                malformed.append(dataset_path)
    return start, end, batch


def _inspect_native(
    handle: h5py.File,
) -> tuple[list[int], dict[str, tuple[int, ...]], list[str], list[int], list[str]]:
    shapes: dict[str, tuple[int, ...]] = {}
    broken: list[str] = []
    malformed: list[str] = []
    root_interval = _validate_native_metadata(handle, malformed)
    payloads = handle.get("payloads")
    if not isinstance(payloads, h5py.Group):
        malformed.append("/payloads")
        return [], shapes, broken, [], malformed
    if not payloads:
        malformed.append("/payloads")

    coverage: dict[str, dict[int, int]] = {}
    for identity in payloads:
        parsed = _validate_payload(handle, identity, malformed, broken, shapes)
        if parsed is None:
            continue
        start, end, batch = parsed
        counts = coverage.setdefault(batch, {})
        for sample_id in range(start, end):
            counts[sample_id] = counts.get(sample_id, 0) + 1
    sample_ids = sorted({sample_id for counts in coverage.values() for sample_id in counts})
    duplicate_ids = sorted(
        {
            sample_id
            for counts in coverage.values()
            for sample_id, count in counts.items()
            if count > 1
        }
    )
    if root_interval is not None and set(sample_ids) != set(range(*root_interval)):
        malformed.append("/@sample_interval")
    return sample_ids, shapes, broken, duplicate_ids, malformed


def inspect_hdf5(path: Path) -> InspectionReport:
    """Inspect native or legacy structure without modifying the artifact."""
    path = Path(path)
    shapes: dict[str, tuple[int, ...]] = {}
    broken: list[str] = []
    malformed: list[str] = []
    try:
        with h5py.File(path, "r") as handle:
            raw_schema = handle.attrs.get("schema")
            schema = (
                SCHEMA_VERSION
                if raw_schema is None and "payloads" in handle
                else str(raw_schema or LEGACY_SCHEMA)
            )
            completion = bool(handle.attrs.get("completion", schema == LEGACY_SCHEMA))
            if schema == SCHEMA_VERSION:
                sample_ids, shapes, broken, duplicate_ids, malformed = _inspect_native(handle)
            elif schema == LEGACY_SCHEMA:
                sample_ids = _legacy_sample_ids(handle)
                duplicate_ids = []
                if not sample_ids:
                    malformed.append("/")
                handle.visititems(
                    lambda name, obj: shapes.__setitem__(f"/{name}", obj.shape)
                    if isinstance(obj, h5py.Dataset)
                    else None
                )
            else:
                sample_ids = []
                duplicate_ids = []
                completion = False
    except OSError as exc:
        raise ValueError(f"unable to inspect HDF5 artifact {path}: {exc}") from exc

    usable = (
        schema in {SCHEMA_VERSION, LEGACY_SCHEMA}
        and completion
        and not broken
        and not duplicate_ids
        and not malformed
    )
    return InspectionReport(
        path=path,
        schema=schema,
        sample_ids=sample_ids,
        dataset_shapes=shapes,
        broken_links=broken,
        duplicate_sample_ids=duplicate_ids,
        completion=completion,
        resumable=schema == SCHEMA_VERSION and usable,
        usable=usable,
        malformed=malformed,
    )


def _structured_arrays(dataset: h5py.Dataset) -> dict[str, NDArray[np.generic]]:
    values = dataset[()]
    if values.dtype.names is None:
        return {dataset.name.rsplit("/", 1)[-1]: np.asarray(values)}
    return {name: np.asarray(values[name]) for name in values.dtype.names}


def _object_arrays(item: h5py.Dataset | h5py.Group) -> dict[str, NDArray[np.generic]]:
    if isinstance(item, h5py.Dataset):
        return _structured_arrays(item)
    arrays: dict[str, NDArray[np.generic]] = {}
    for name in item:
        child = item[name]
        if isinstance(child, h5py.Dataset):
            arrays.update(_structured_arrays(child))
    if not arrays:
        raise ValueError(f"legacy group {item.name} contains no readable datasets")
    return arrays


def _legacy_object(
    handle: h5py.File, prefix: str, sample_id: int
) -> h5py.Dataset | h5py.Group | None:
    numbered = f"{prefix}_{sample_id}"
    if numbered in handle:
        return handle[numbered]
    return None


def _legacy_list_arrays(
    handle: h5py.File, sample_id: int
) -> dict[str, NDArray[np.generic]] | None:
    item = _legacy_object(handle, "list", sample_id)
    if item is not None:
        return _object_arrays(item)
    if "arrays" not in handle:
        return None
    arrays = handle["arrays"]
    if isinstance(arrays, h5py.Dataset):
        if arrays.ndim == 1:
            if sample_id != 0:
                return None
            return {"arrays": np.asarray(arrays[:])}
        if arrays.ndim == 0 or sample_id >= arrays.shape[0]:
            return None
        return {"arrays": np.asarray(arrays[sample_id])}
    numbered = f"list_{sample_id}"
    if numbered in arrays:
        return _object_arrays(arrays[numbered])
    return None


class MockStore:
    """Context-managed read-only view over native or legacy mock samples."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        report = inspect_hdf5(self.path)
        if not report.usable:
            raise ValueError(f"HDF5 artifact is not usable: {self.path}")
        self.schema = report.schema
        self.sample_ids = report.sample_ids
        self._handles: dict[str, h5py.File] = {"table": h5py.File(self.path, "r")}
        if self.schema == LEGACY_SCHEMA and self.path.name == "mcs_table_combined.h5":
            for key, filename in (
                ("beta", "mcs_beta_combined.h5"),
                ("nberr", "mcs_nberr_combined.h5"),
            ):
                sibling = self.path.with_name(filename)
                if sibling.is_file():
                    self._handles[key] = h5py.File(sibling, "r")

    def __enter__(self) -> MockStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close every physical file backing this logical store."""
        for handle in self._handles.values():
            handle.close()

    @property
    def targetids(self) -> NDArray[np.generic]:
        """Return sorted unique target identifiers across the first sample."""
        if not self.sample_ids:
            return np.array([], dtype=np.int64)
        return np.unique(self.sample(self.sample_ids[0]).targetid)

    def sample_batches(self, sample_id: int) -> list[MockSampleRecord]:
        """Return immutable storage records for a sample without concatenating batches."""
        if sample_id not in self.sample_ids:
            raise KeyError(f"sample {sample_id} is not present")
        if self.schema == SCHEMA_VERSION:
            payloads = self._handles["table"]["payloads"]
            records = []
            for identity in payloads:
                sample_start, sample_end, batch = parse_payload_identity(identity)
                if not sample_start <= sample_id < sample_end:
                    continue
                group = payloads[identity]["samples"][str(sample_id)]
                arrays = {
                    name: np.asarray(group[name][()])
                    for name in group
                    if isinstance(group[name], h5py.Dataset)
                }
                records.append(MockSampleRecord(sample_id, batch, arrays))
            return records
        return [self._legacy_sample(sample_id)]

    def array_sample(self, sample_id: int) -> LegacyArraySample:
        """Read one sample from a standalone legacy list or root arrays artifact."""
        if self.schema != LEGACY_SCHEMA:
            raise ValueError("array_sample is only available for legacy artifacts")
        if sample_id not in self.sample_ids:
            raise KeyError(f"sample {sample_id} is not present")
        arrays = _legacy_list_arrays(self._handles["table"], sample_id)
        if arrays is None:
            raise ValueError("legacy artifact has no standalone list or arrays payload")
        return LegacyArraySample(sample_id, arrays)

    def _legacy_sample(self, sample_id: int) -> MockSampleRecord:
        table = _legacy_object(self._handles["table"], "table", sample_id)
        if table is None:
            raise KeyError(f"legacy table sample {sample_id} is not present")
        arrays = _object_arrays(table)
        if "ID" in arrays:
            arrays["TARGETID"] = arrays.pop("ID")
        if "TARGETID" not in arrays:
            raise ValueError(f"legacy table sample {sample_id} has no ID column")
        for key, output_name in (("beta", "BETA"), ("nberr", "NBERR")):
            handle = self._handles.get(key)
            if handle is None:
                continue
            sibling_arrays = _legacy_list_arrays(handle, sample_id)
            if sibling_arrays is None:
                continue
            if len(sibling_arrays) == 1:
                arrays[output_name] = next(iter(sibling_arrays.values()))
            else:
                arrays.update({name.upper(): value for name, value in sibling_arrays.items()})
        return MockSampleRecord(sample_id, "legacy", arrays)

    def sample(self, sample_id: int) -> MockSampleRecord:
        """Read one sample, combining native star batches in stable TARGETID order."""
        batches = self.sample_batches(sample_id)
        if len(batches) == 1:
            return batches[0]
        targetids = np.concatenate([batch.targetid for batch in batches])
        unique, counts = np.unique(targetids, return_counts=True)
        duplicate_stars = unique[counts > 1]
        for targetid in duplicate_stars:
            owners = {
                batch.star_batch_id for batch in batches if np.any(batch.targetid == targetid)
            }
            if len(owners) > 1:
                raise ValueError(f"duplicate TARGETID across star batches: {targetid}")
        names = set(batches[0].arrays)
        if any(set(batch.arrays) != names for batch in batches[1:]):
            raise ValueError("star batches must contain identical array names")
        arrays = {name: np.concatenate([batch.arrays[name] for batch in batches]) for name in names}
        order = np.argsort(arrays["TARGETID"], kind="stable")
        arrays = {name: values[order] for name, values in arrays.items()}
        return MockSampleRecord(sample_id, "combined", arrays)


def open_mock_store(path: Path) -> MockStore:
    """Open a native artifact or auto-detected legacy Binary2 table store."""
    path = Path(path)
    if path.is_dir():
        legacy_table = path / "mcs_table_combined.h5"
        if legacy_table.is_file():
            path = legacy_table
        else:
            raise ValueError(f"directory contains no mcs_table_combined.h5: {path}")
    return MockStore(path)
