from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from rvbinary.io.hdf5 import ArtifactMetadata, MockSampleRecord, write_mock_shard
from rvbinary.io.legacy import LegacyArraySample, inspect_hdf5, open_mock_store


def write_legacy_table(path: Path, targetids: np.ndarray | None = None) -> Path:
    ids = targetids if targetids is not None else np.array([11, 11, 22, 22], dtype=np.int64)
    dtype = np.dtype([("ID", ids.dtype), ("VRAD", "f8")])
    rows = np.empty(4, dtype=dtype)
    rows["ID"] = ids
    rows["VRAD"] = [1.0, 1.1, 2.0, 2.1]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("table_0", data=rows)
    return path


def test_legacy_id_is_exposed_as_targetid_without_float_conversion(tmp_path: Path) -> None:
    exact_ids = np.array([2**53 + 1, 2**53 + 1, 2**53 + 3, 2**53 + 3], dtype=np.int64)
    legacy = write_legacy_table(tmp_path / "mcs_table_combined.h5", exact_ids)

    with open_mock_store(legacy) as store:
        sample = store.sample(0)

    assert sample.targetid.dtype == np.dtype("int64")
    assert np.array_equal(sample.targetid, exact_ids)
    assert "ID" not in sample.arrays


def test_open_mock_store_discovers_optional_legacy_siblings(tmp_path: Path) -> None:
    table = write_legacy_table(tmp_path / "mcs_table_combined.h5")
    with h5py.File(tmp_path / "mcs_beta_combined.h5", "w") as handle:
        handle.create_dataset("list_0", data=np.array([0.5, 1.5]))
    with h5py.File(tmp_path / "mcs_nberr_combined.h5", "w") as handle:
        handle.create_dataset("arrays", data=np.array([0.25, 0.75]))

    with open_mock_store(table) as store:
        sample = store.sample(0)

    assert np.array_equal(sample.arrays["BETA"], np.array([0.5, 1.5]))
    assert np.array_equal(sample.arrays["NBERR"], np.array([0.25, 0.75]))


def test_legacy_arrays_dataset_selects_the_requested_sample_row(tmp_path: Path) -> None:
    table = write_legacy_table(tmp_path / "mcs_table_combined.h5")
    with h5py.File(table, "a") as handle:
        handle.copy("table_0", "table_1")
    with h5py.File(tmp_path / "mcs_nberr_combined.h5", "w") as handle:
        handle.create_dataset("arrays", data=np.array([[0.25, 0.75], [1.25, 1.75]]))

    with open_mock_store(table) as store:
        sample = store.sample(1)

    assert np.array_equal(sample.arrays["NBERR"], np.array([1.25, 1.75]))


def test_standalone_legacy_list_samples_are_readable(tmp_path: Path) -> None:
    path = tmp_path / "mcs_beta_combined.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("list_0", data=np.array([0.5, 1.5]))
        handle.create_dataset("list_1", data=np.array([2.5, 3.5]))

    report = inspect_hdf5(path)
    with open_mock_store(path) as store:
        sample = store.array_sample(1)

    assert report.sample_ids == [0, 1]
    assert report.usable is True
    assert isinstance(sample, LegacyArraySample)
    assert sample.sample_id == 1
    assert np.array_equal(sample.values, np.array([2.5, 3.5]))


def test_standalone_one_dimensional_root_arrays_is_sample_zero(tmp_path: Path) -> None:
    path = tmp_path / "mcs_nberr_combined.h5"
    expected = np.array([0.25, 0.75])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("arrays", data=expected)

    report = inspect_hdf5(path)
    with open_mock_store(path) as store:
        sample = store.array_sample(0)

    assert report.sample_ids == [0]
    assert report.usable is True
    assert np.array_equal(sample.values, expected)


def test_standalone_root_arrays_reads_only_requested_sample_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "mcs_nberr_combined.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("arrays", data=np.array([[0.25, 0.75], [1.25, 1.75]]))
    requested_keys: list[object] = []
    original_getitem = h5py.Dataset.__getitem__

    def record_getitem(dataset: h5py.Dataset, key: object) -> object:
        if dataset.name == "/arrays":
            requested_keys.append(key)
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_getitem)

    report = inspect_hdf5(path)
    with open_mock_store(path) as store:
        sample = store.array_sample(1)

    assert report.sample_ids == [0, 1]
    assert report.usable is True
    assert np.array_equal(sample.values, np.array([1.25, 1.75]))
    assert requested_keys == [1]


def test_legacy_table_only_can_be_inspected_and_opened(tmp_path: Path) -> None:
    table = write_legacy_table(tmp_path / "mcs_table_combined.h5")

    report = inspect_hdf5(table)
    with open_mock_store(table) as store:
        sample = store.sample(0)

    assert report.schema == "legacy.binary2"
    assert report.sample_ids == [0]
    assert report.broken_links == []
    assert report.usable is True
    assert np.array_equal(sample.targetid, np.array([11, 11, 22, 22]))


def test_inspection_reports_broken_external_links(tmp_path: Path) -> None:
    path = tmp_path / "broken.h5"
    with h5py.File(path, "w") as handle:
        payloads = handle.create_group("payloads")
        payloads["samples-2-3__batch-a"] = h5py.ExternalLink("missing.h5", "/payloads/x")
        handle.attrs["schema"] = "rvbinary.mock/v1"

    report = inspect_hdf5(path)

    assert report.sample_ids == [2]
    assert report.broken_links == ["/payloads/samples-2-3__batch-a"]
    assert report.usable is False


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("schema", "/@schema"),
        ("metadata", "/@config_digest"),
        ("payload_attr", "/payloads/samples-0-1__batch-a/@star_batch_id"),
        ("samples_group", "/payloads/samples-0-1__batch-a/samples"),
        ("sample_group", "/payloads/samples-0-1__batch-a/samples/0"),
        ("targetid", "/payloads/samples-0-1__batch-a/samples/0/TARGETID"),
    ],
)
def test_inspection_reports_malformed_native_structure(
    tmp_path: Path, defect: str, expected: str
) -> None:
    path = tmp_path / f"{defect}.h5"
    metadata = ArtifactMetadata("0.1.0", "digest", 7, 0, 1, {}, "consistent")
    sample = MockSampleRecord(0, "a", {"TARGETID": np.array([11], dtype=np.int64)})
    write_mock_shard(path, [sample], metadata)
    payload_path = "/payloads/samples-0-1__batch-a"
    with h5py.File(path, "r+") as handle:
        if defect == "schema":
            del handle.attrs["schema"]
        elif defect == "metadata":
            del handle.attrs["config_digest"]
        elif defect == "payload_attr":
            del handle[payload_path].attrs["star_batch_id"]
        elif defect == "samples_group":
            del handle[f"{payload_path}/samples"]
        elif defect == "sample_group":
            del handle[f"{payload_path}/samples/0"]
        else:
            del handle[f"{payload_path}/samples/0/TARGETID"]

    report = inspect_hdf5(path)

    assert expected in report.malformed
    assert report.usable is False
    assert report.resumable is False


def test_inspection_reports_overlapping_ranges_within_star_batch(tmp_path: Path) -> None:
    def write_range(path: Path, start: int, end: int) -> None:
        metadata = ArtifactMetadata("0.1.0", "digest", 7, start, end, {}, "consistent")
        samples = [
            MockSampleRecord(
                sample_id,
                "same-batch",
                {"TARGETID": np.array([11], dtype=np.int64)},
            )
            for sample_id in range(start, end)
        ]
        write_mock_shard(path, samples, metadata)

    path = tmp_path / "overlap.h5"
    second = tmp_path / "second.h5"
    write_range(path, 0, 2)
    write_range(second, 1, 3)
    with h5py.File(path, "r+") as handle:
        handle.attrs["sample_end"] = 3
        identity = "samples-1-3__batch-same-batch"
        handle["payloads"][identity] = h5py.ExternalLink(second.name, f"/payloads/{identity}")

    report = inspect_hdf5(path)

    assert report.sample_ids == [0, 1, 2]
    assert report.duplicate_sample_ids == [1]
    assert report.usable is False
    assert report.resumable is False


@pytest.mark.parametrize("replacement", ["dangling", "group"])
def test_inspection_rejects_non_dataset_manifested_payload(
    tmp_path: Path, replacement: str
) -> None:
    path = tmp_path / f"{replacement}.h5"
    metadata = ArtifactMetadata("0.1.0", "digest", 7, 0, 1, {}, "consistent")
    sample = MockSampleRecord(
        0,
        "a",
        {
            "TARGETID": np.array([11], dtype=np.int64),
            "VRAD": np.array([1.5]),
        },
    )
    write_mock_shard(path, [sample], metadata)
    vrad_path = "/payloads/samples-0-1__batch-a/samples/0/VRAD"
    with h5py.File(path, "r+") as handle:
        del handle[vrad_path]
        if replacement == "dangling":
            handle[vrad_path] = h5py.ExternalLink("missing.h5", "/VRAD")
        else:
            handle.create_group(vrad_path)

    report = inspect_hdf5(path)

    defects = report.broken_links if replacement == "dangling" else report.malformed
    assert vrad_path in defects
    assert report.usable is False
    assert report.resumable is False


def test_inspection_rejects_root_interval_that_disagrees_with_payload_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coverage.h5"
    metadata = ArtifactMetadata("0.1.0", "digest", 7, 0, 1, {}, "consistent")
    sample = MockSampleRecord(0, "a", {"TARGETID": np.array([11], dtype=np.int64)})
    write_mock_shard(path, [sample], metadata)
    with h5py.File(path, "r+") as handle:
        handle.attrs["sample_start"] = 1
        handle.attrs["sample_end"] = 2
        handle.attrs["sample_stop"] = 2

    report = inspect_hdf5(path)

    assert "/@sample_interval" in report.malformed
    assert report.usable is False
    assert report.resumable is False
