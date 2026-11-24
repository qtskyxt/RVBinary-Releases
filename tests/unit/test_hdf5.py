from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from rvbinary.io.hdf5 import (
    SCHEMA_VERSION,
    ArtifactMetadata,
    MockSampleRecord,
    build_external_index,
    compact_index,
    write_mock_shard,
)
from rvbinary.io.legacy import inspect_hdf5, open_mock_store


def metadata(start: int, end: int) -> ArtifactMetadata:
    return ArtifactMetadata(
        package_version="0.1.0",
        config_digest="config-abc",
        seed=17,
        sample_start=start,
        sample_end=end,
        source_hashes={"epochs.fits": "sha256:123"},
        statistic_mode="consistent",
        created_at="2026-09-04T12:00:00+00:00",
    )


def test_artifact_metadata_generates_creation_time() -> None:
    generated = ArtifactMetadata(
        package_version="0.1.0",
        config_digest="config-abc",
        seed=17,
        sample_start=0,
        sample_end=1,
        source_hashes={},
        statistic_mode="consistent",
    )

    assert generated.created_at.endswith("+00:00")


def record(sample_id: int, batch: str = "stars-11-22") -> MockSampleRecord:
    return MockSampleRecord(
        sample_id=sample_id,
        star_batch_id=batch,
        arrays={
            "TARGETID": np.array([11, 11, 22, 22], dtype=np.int64),
            "VRAD": np.array([sample_id + 0.1, sample_id + 0.2, 2.1, 2.2]),
        },
    )


def write_shard(path: Path, sample_ids: list[int], batch: str = "stars-11-22") -> Path:
    write_mock_shard(
        path,
        [record(sample_id, batch) for sample_id in sample_ids],
        metadata(min(sample_ids), max(sample_ids) + 1),
    )
    return path


def payload(path: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    with open_mock_store(path) as store:
        return {
            (sample_id, item.star_batch_id): {
                name: values.copy() for name, values in item.arrays.items()
            }
            for sample_id in store.sample_ids
            for item in store.sample_batches(sample_id)
        }


def test_write_mock_shard_stores_complete_versioned_metadata_and_payload_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard.h5"
    write_shard(path, [4, 5])

    with h5py.File(path, "r") as handle:
        assert handle.attrs["schema"] == SCHEMA_VERSION
        assert handle.attrs["package_version"] == "0.1.0"
        assert handle.attrs["config_digest"] == "config-abc"
        assert handle.attrs["seed"] == 17
        assert handle.attrs["sample_start"] == 4
        assert handle.attrs["sample_end"] == 6
        assert handle.attrs["sample_stop"] == 6
        assert json.loads(handle.attrs["source_hashes"]) == {"epochs.fits": "sha256:123"}
        assert bool(handle.attrs["completion"]) is True
        assert handle.attrs["statistic_mode"] == "consistent"
        assert handle.attrs["created_at"] == "2026-09-04T12:00:00+00:00"
        assert sorted(handle["payloads"]) == ["samples-4-6__batch-stars-11-22"]
        assert sorted(handle["payloads/samples-4-6__batch-stars-11-22/samples"]) == [
            "4",
            "5",
        ]
        assert json.loads(
            handle["payloads/samples-4-6__batch-stars-11-22/samples/4"].attrs[
                "array_names"
            ]
        ) == ["TARGETID", "VRAD"]


def test_write_mock_shard_replaces_only_after_complete_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "shard.h5"
    path.write_bytes(b"previous artifact")
    real_replace = os.replace
    observed: dict[str, object] = {}

    def observe_replace(source: str | Path, destination: str | Path) -> None:
        observed["old"] = Path(destination).read_bytes()
        with h5py.File(source, "r") as handle:
            observed["temporary_completion"] = bool(handle.attrs["completion"])
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", observe_replace)
    write_shard(path, [0])

    assert observed == {"old": b"previous artifact", "temporary_completion": True}
    assert bool(h5py.File(path, "r").attrs["completion"]) is True


def test_external_index_and_compact_roundtrip_survives_directory_move(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    shards = [write_shard(tree / "a.h5", [0, 1]), write_shard(tree / "b.h5", [2, 3])]
    index = tree / "index.h5"
    build_external_index(shards, index)

    with h5py.File(index, "r") as handle:
        link = handle["payloads"].get("samples-0-2__batch-stars-11-22", getlink=True)
        assert isinstance(link, h5py.ExternalLink)
        assert not Path(link.filename).is_absolute()

    moved = tmp_path / "moved"
    shutil.move(tree, moved)
    moved_index = moved / "index.h5"
    assert inspect_hdf5(moved_index).sample_ids == [0, 1, 2, 3]

    compact = tmp_path / "compact.h5"
    compact_index(moved_index, compact, block_bytes=32)
    actual = payload(moved_index)
    copied = payload(compact)
    assert actual.keys() == copied.keys()
    for identity in actual:
        for name in actual[identity]:
            assert np.array_equal(actual[identity][name], copied[identity][name])


def test_build_external_index_rejects_duplicate_payload_identity(tmp_path: Path) -> None:
    first = write_shard(tmp_path / "first.h5", [0])
    second = write_shard(tmp_path / "second.h5", [0])

    with pytest.raises(ValueError, match="duplicate payload identity"):
        build_external_index([first, second], tmp_path / "index.h5")


def test_build_external_index_rejects_broken_shard_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        build_external_index([tmp_path / "missing.h5"], tmp_path / "index.h5")


def test_external_index_rejects_output_as_shard_before_creating_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = write_shard(tmp_path / "index.h5", [0])
    before = output.read_bytes()
    temporary_requested = False

    def reject_temporary(_destination: Path) -> Path:
        nonlocal temporary_requested
        temporary_requested = True
        raise AssertionError("temporary index must not be created")

    monkeypatch.setattr("rvbinary.io.hdf5._temporary_path", reject_temporary)

    with pytest.raises(ValueError, match="output path cannot also be a shard"):
        build_external_index([output], output)

    assert temporary_requested is False
    assert output.read_bytes() == before
    with h5py.File(output, "r") as handle:
        assert all(
            not (
                isinstance(link := handle["payloads"].get(name, getlink=True), h5py.ExternalLink)
                and Path(link.filename).resolve() == output.resolve()
            )
            for name in handle["payloads"]
        )


def test_external_index_merges_disjoint_source_hashes(tmp_path: Path) -> None:
    first_metadata = metadata(0, 1)
    second_metadata = ArtifactMetadata(
        package_version=first_metadata.package_version,
        config_digest=first_metadata.config_digest,
        seed=first_metadata.seed,
        sample_start=0,
        sample_end=1,
        source_hashes={"new-stars.fits": "sha256:456"},
        statistic_mode=first_metadata.statistic_mode,
        created_at=first_metadata.created_at,
    )
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    write_mock_shard(first, [record(0, "original")], first_metadata)
    write_mock_shard(second, [record(0, "extension")], second_metadata)

    index = tmp_path / "index.h5"
    build_external_index([first, second], index)

    with h5py.File(index, "r") as handle:
        assert json.loads(handle.attrs["source_hashes"]) == {
            "epochs.fits": "sha256:123",
            "new-stars.fits": "sha256:456",
        }


def test_append_rejects_collision_without_replacing_existing_index(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "first.h5", [0])
    index = tmp_path / "index.h5"
    build_external_index([shard], index)
    before = index.read_bytes()
    collision = write_shard(tmp_path / "collision.h5", [0])

    with pytest.raises(ValueError, match="duplicate payload identity"):
        build_external_index([collision], index, append=True)

    assert index.read_bytes() == before


def test_native_store_combines_disjoint_star_batches_in_targetid_order(tmp_path: Path) -> None:
    first = MockSampleRecord(
        sample_id=0,
        star_batch_id="late",
        arrays={"TARGETID": np.array([22, 22]), "VRAD": np.array([2.0, 2.1])},
    )
    second = MockSampleRecord(
        sample_id=0,
        star_batch_id="early",
        arrays={"TARGETID": np.array([11, 11]), "VRAD": np.array([1.0, 1.1])},
    )
    shard_a = tmp_path / "a.h5"
    shard_b = tmp_path / "b.h5"
    write_mock_shard(shard_a, [first], metadata(0, 1))
    write_mock_shard(shard_b, [second], metadata(0, 1))
    index = tmp_path / "index.h5"
    build_external_index([shard_a, shard_b], index)

    with open_mock_store(index) as store:
        combined = store.sample(0)

    assert np.array_equal(combined.targetid, np.array([11, 11, 22, 22]))
    assert np.array_equal(combined.arrays["VRAD"], np.array([1.0, 1.1, 2.0, 2.1]))


def test_external_links_resolve_when_output_uses_directory_alias(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    shard = write_shard(physical / "shard.h5", [0])
    alias_parent = tmp_path / "aliases" / "deep"
    alias_parent.mkdir(parents=True)
    alias = alias_parent / "output"
    alias.symlink_to(physical, target_is_directory=True)
    index = alias / "index.h5"

    build_external_index([shard], index)

    assert inspect_hdf5(index).usable
    with open_mock_store(index) as store:
        assert store.sample_ids == [0]
