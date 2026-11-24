from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.table import Table

from rvbinary.config import RVBinaryConfig
from rvbinary.io.legacy import inspect_hdf5, open_mock_store
from rvbinary.io.observations import EpochData
from rvbinary.mock.writer import ArtifactSet, extend_mock_stars, generate_mock


def write_epochs(path: Path, targetids: list[int]) -> Path:
    ids = np.repeat(np.asarray(targetids, dtype=np.int64), 3)
    epoch = np.tile(np.array([59000.0, 59002.0, 59007.0]), len(targetids))
    Table(
        {
            "TARGETID": ids,
            "MJD": epoch,
            "VRAD": np.zeros(ids.size),
            "VRAD_ERR": np.tile(np.array([0.5, 0.7, 0.9]), len(targetids)),
            "Mass": np.repeat(np.linspace(0.9, 1.1, len(targetids)), 3),
            "meanLOGG": np.repeat(np.linspace(4.4, 4.2, len(targetids)), 3),
        }
    ).write(path)
    return path


def make_config(
    tmp_path: Path, name: str, targetids: list[int], *, samples: int = 4
) -> RVBinaryConfig:
    root = tmp_path / name
    root.mkdir()
    epochs = write_epochs(root / "epochs.fits", targetids)
    members = root / "members.fits"
    Table({"TARGETID": np.asarray(targetids, dtype=np.int64)}).write(members)
    return RVBinaryConfig.model_validate(
        {
            "project": {"name": name, "output": root / "result", "random_seed": 7123},
            "observations": {
                "mock_epochs": epochs,
                "subsamples": [{"name": "all", "members": members}],
            },
            "mock": {"samples": samples, "error_floor": {"values_kms": [0.4, 0.0]}},
            "statistics": {"mode": "legacy"},
        }
    )


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linked_shards(index: Path) -> set[Path]:
    with h5py.File(index, "r") as handle:
        return {
            (index.parent / handle["payloads"].get(name, getlink=True).filename).resolve()
            for name in handle["payloads"]
        }


def assert_store_equal(first: Path, second: Path) -> None:
    with open_mock_store(first) as left, open_mock_store(second) as right:
        assert left.sample_ids == right.sample_ids
        for sample_id in left.sample_ids:
            left_arrays = left.sample(sample_id).arrays
            right_arrays = right.sample(sample_id).arrays
            assert set(left_arrays) == set(right_arrays)
            for name in left_arrays:
                np.testing.assert_array_equal(left_arrays[name], right_arrays[name])


def epoch_subset(path: Path, targetids: list[int]) -> EpochData:
    table = Table.read(path)
    rows = np.isin(np.asarray(table["TARGETID"]), targetids)
    return EpochData(
        targetid=np.asarray(table["TARGETID"][rows], dtype=np.int64),
        mjd=np.asarray(table["MJD"][rows], dtype=np.float64),
        vrad_kms=np.asarray(table["VRAD"][rows], dtype=np.float64),
        vrad_err_kms=np.asarray(table["VRAD_ERR"][rows], dtype=np.float64),
        mass_solar=np.asarray(table["Mass"][rows], dtype=np.float64),
        logg_cgs=np.asarray(table["meanLOGG"][rows], dtype=np.float64),
    )


def test_generate_mock_writes_tables_and_unrestricted_pair_beta_per_floor(tmp_path: Path) -> None:
    config = make_config(tmp_path, "serial", [11, 22], samples=2)

    artifacts = generate_mock(config)

    assert isinstance(artifacts, ArtifactSet)
    assert inspect_hdf5(artifacts.table_index).sample_ids == [0, 1]
    assert set(artifacts.binary_beta_indexes) == {0.0, 0.4}
    assert set(artifacts.nonbinary_beta_indexes) == {0.0, 0.4}
    with h5py.File(artifacts.binary_beta_indexes[0.4], "r") as handle:
        assert handle.attrs["statistic_mode"] == "legacy"
    with open_mock_store(artifacts.table_index) as store:
        assert set(store.sample(0).arrays) == {
            "TARGETID", "VRAD", "VRAD_ERR", "PERIOD", "q", "ECC",
            "THETA", "INCL", "OMEGA", "DELTAT", "STAR_OFFSET",
        }
    with open_mock_store(artifacts.binary_beta_indexes[0.0]) as store:
        assert set(store.sample(0).arrays) == {"TARGETID", "BETA"}
        assert store.sample(0).arrays["TARGETID"].tolist() == [11, 11, 11, 22, 22, 22]


def test_sample_append_matches_one_shot_and_preserves_old_shards(tmp_path: Path) -> None:
    appended_config = make_config(tmp_path, "appended", [11, 22], samples=4)
    initial = generate_mock(appended_config, sample_range=range(0, 2))
    old_shards = set(initial.shards)
    old_bytes = {path: file_digest(path) for path in old_shards}

    appended = generate_mock(appended_config, sample_range=range(2, 4))
    full_config = make_config(tmp_path, "full", [11, 22], samples=4)
    full = generate_mock(full_config)

    assert inspect_hdf5(appended.table_index).sample_ids == [0, 1, 2, 3]
    assert {path: file_digest(path) for path in old_shards} == old_bytes
    assert_store_equal(appended.table_index, full.table_index)
    for floor in (0.0, 0.4):
        assert_store_equal(appended.binary_beta_indexes[floor], full.binary_beta_indexes[floor])
        assert_store_equal(
            appended.nonbinary_beta_indexes[floor], full.nonbinary_beta_indexes[floor]
        )


def test_append_rejects_overlap_without_changing_indexes(tmp_path: Path) -> None:
    config = make_config(tmp_path, "overlap", [11, 22])
    artifacts = generate_mock(config, sample_range=range(0, 3))
    before = file_digest(artifacts.table_index)

    with pytest.raises(ValueError, match="overlap"):
        generate_mock(config, sample_range=range(2, 4))

    assert file_digest(artifacts.table_index) == before


def test_incompatible_config_digest_preserves_existing_index(tmp_path: Path) -> None:
    config = make_config(tmp_path, "digest", [11, 22])
    artifacts = generate_mock(config, sample_range=range(0, 2))
    before = file_digest(artifacts.table_index)
    config.model.log10_period_mean = 3.7

    with pytest.raises(ValueError, match="config_digest"):
        generate_mock(config, sample_range=range(2, 4))

    assert file_digest(artifacts.table_index) == before


def test_extend_stars_matches_full_generation_and_preserves_old_payloads(tmp_path: Path) -> None:
    extension_config = make_config(tmp_path, "extension", [11, 33], samples=3)
    initial = generate_mock(extension_config)
    old_shards = set(initial.shards)
    old_bytes = {path: file_digest(path) for path in old_shards}

    full_config = make_config(tmp_path, "extension-source", [11, 22, 33], samples=3)
    new_epochs = epoch_subset(full_config.observations.mock_epochs, [22])
    extended = extend_mock_stars(extension_config, new_epochs)
    full = generate_mock(full_config)

    assert {path: file_digest(path) for path in old_shards} == old_bytes
    assert linked_shards(extended.table_index).issuperset(
        old_shards & linked_shards(initial.table_index)
    )
    assert_store_equal(extended.table_index, full.table_index)
    for floor in (0.0, 0.4):
        assert_store_equal(extended.binary_beta_indexes[floor], full.binary_beta_indexes[floor])
        assert_store_equal(
            extended.nonbinary_beta_indexes[floor], full.nonbinary_beta_indexes[floor]
        )


def test_extend_rejects_duplicate_targetid_without_changing_index(tmp_path: Path) -> None:
    config = make_config(tmp_path, "duplicate", [11, 22], samples=2)
    artifacts = generate_mock(config)
    before = file_digest(artifacts.table_index)
    duplicate = epoch_subset(config.observations.mock_epochs, [22])

    with pytest.raises(ValueError, match="duplicate TARGETID.*22"):
        extend_mock_stars(config, duplicate)

    assert file_digest(artifacts.table_index) == before


def test_source_batches_persist_normalized_generation_inputs(tmp_path: Path) -> None:
    config = make_config(tmp_path, "sources", [11, 22], samples=2)
    expected = Table.read(config.observations.mock_epochs)

    artifacts = generate_mock(config)

    assert len(artifacts.source_batches) == 1
    with h5py.File(artifacts.source_batches[0], "r") as handle:
        assert handle.attrs["schema"] == "rvbinary.source-batch/v1"
        assert bool(handle.attrs["completion"])
        assert len(str(handle.attrs["digest"])) == 64
        assert set(handle) == {"TARGETID", "MJD", "VRAD_ERR", "Mass", "meanLOGG"}
        assert np.asarray(handle["TARGETID"]).dtype == np.dtype(np.int64)
        for name in ("TARGETID", "MJD", "VRAD_ERR", "Mass", "meanLOGG"):
            np.testing.assert_array_equal(np.asarray(handle[name]), np.asarray(expected[name]))


def test_extend_then_append_matches_append_then_extend_without_external_source(
    tmp_path: Path,
) -> None:
    extend_first_config = make_config(tmp_path, "extend-first", [11, 33], samples=4)
    append_first_config = make_config(tmp_path, "append-first", [11, 33], samples=4)
    source_config = make_config(tmp_path, "new-source", [22], samples=4)
    new_epochs = epoch_subset(source_config.observations.mock_epochs, [22])

    generate_mock(extend_first_config, sample_range=range(0, 2))
    extend_mock_stars(extend_first_config, new_epochs)
    source_config.observations.mock_epochs.unlink()
    extend_then_append = generate_mock(extend_first_config, sample_range=range(2, 4))

    generate_mock(append_first_config, sample_range=range(0, 2))
    generate_mock(append_first_config, sample_range=range(2, 4))
    append_then_extend = extend_mock_stars(append_first_config, new_epochs)

    assert_store_equal(extend_then_append.table_index, append_then_extend.table_index)
    for floor in (0.0, 0.4):
        assert_store_equal(
            extend_then_append.binary_beta_indexes[floor],
            append_then_extend.binary_beta_indexes[floor],
        )
        assert_store_equal(
            extend_then_append.nonbinary_beta_indexes[floor],
            append_then_extend.nonbinary_beta_indexes[floor],
        )
    with open_mock_store(extend_then_append.table_index) as store:
        for sample_id in store.sample_ids:
            assert set(store.sample(sample_id).targetid) == {11, 22, 33}


@pytest.mark.parametrize("failed_live_replace", [2, 3])
def test_index_set_replacement_rolls_back_every_live_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_live_replace: int
) -> None:
    config = make_config(tmp_path, f"rollback-{failed_live_replace}", [11, 22])
    artifacts = generate_mock(config, sample_range=range(0, 2))
    indexes = [
        artifacts.table_index,
        *artifacts.binary_beta_indexes.values(),
        *artifacts.nonbinary_beta_indexes.values(),
    ]
    before = {path: path.read_bytes() for path in indexes}
    old_shards = set(artifacts.shards)
    real_replace = os.replace
    live_replacements = 0

    def fail_selected_live_replace(source: Path | str, destination: Path | str) -> None:
        nonlocal live_replacements
        destination_path = Path(destination)
        if destination_path in indexes:
            live_replacements += 1
            if live_replacements == failed_live_replace:
                raise OSError("injected live-index replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr("rvbinary.mock.writer.os.replace", fail_selected_live_replace)

    with pytest.raises(OSError, match="injected live-index"):
        generate_mock(config, sample_range=range(2, 4))

    assert {path: path.read_bytes() for path in indexes} == before
    recovery_shards = (
        set(Path(config.project.output).glob("mock/tables/**/*.h5"))
        | set(Path(config.project.output).glob("mock/beta/**/*.h5"))
        | set(Path(config.project.output).glob("mock/nberr/**/*.h5"))
    ) - old_shards
    assert recovery_shards
    assert all(inspect_hdf5(path).usable for path in recovery_shards)
    transaction_files = [
        path
        for path in Path(config.project.output).rglob(".*")
        if path.is_file() and ("candidate" in path.name or "backup" in path.name)
    ]
    assert transaction_files == []


def test_source_hash_mismatch_is_rejected_before_writing_shards(tmp_path: Path) -> None:
    config = make_config(tmp_path, "source-mismatch", [11, 22])
    artifacts = generate_mock(config, sample_range=range(0, 2))
    existing_shards = set(Path(config.project.output).glob("mock/**/*.h5"))
    mismatched_index = artifacts.binary_beta_indexes[0.0]
    with h5py.File(mismatched_index, "r+") as handle:
        handle.attrs["source_hashes"] = json.dumps(
            {"epochs:tampered": "0" * 64, "model-config": "1" * 64},
            sort_keys=True,
            separators=(",", ":"),
        )
    existing_bytes = {path: path.read_bytes() for path in existing_shards}

    with pytest.raises(ValueError, match="incompatible source_hashes"):
        generate_mock(config, sample_range=range(2, 4))

    assert set(Path(config.project.output).glob("mock/**/*.h5")) == existing_shards
    assert {path: path.read_bytes() for path in existing_shards} == existing_bytes


def test_keyboard_interrupt_during_extension_removes_uncommitted_source_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path, "interrupt-extension", [11, 33], samples=4)
    artifacts = generate_mock(config, sample_range=range(0, 2))
    source_config = make_config(tmp_path, "interrupt-source", [22], samples=4)
    new_epochs = epoch_subset(source_config.observations.mock_epochs, [22])
    original_sources = set(artifacts.source_batches)
    real_replace = os.replace

    def interrupt_first_live_replace(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == artifacts.table_index:
            raise KeyboardInterrupt("injected extension interruption")
        real_replace(source, destination)

    with monkeypatch.context() as context:
        context.setattr(
            "rvbinary.mock.writer.os.replace", interrupt_first_live_replace
        )
        with pytest.raises(KeyboardInterrupt, match="injected extension"):
            extend_mock_stars(config, new_epochs)

    assert set(Path(config.project.output).glob("mock/sources/*.h5")) == original_sources
    appended = generate_mock(config, sample_range=range(2, 4))
    with open_mock_store(appended.table_index) as store:
        for sample_id in store.sample_ids:
            assert set(store.sample(sample_id).targetid) == {11, 33}


def test_failed_rollback_retains_backup_and_names_recovery_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path, "failed-rollback", [11, 22], samples=4)
    artifacts = generate_mock(config, sample_range=range(0, 2))
    indexes = [
        artifacts.table_index,
        *artifacts.binary_beta_indexes.values(),
        *artifacts.nonbinary_beta_indexes.values(),
    ]
    original = {path: path.read_bytes() for path in indexes}
    real_replace = os.replace
    publications = 0
    failed_backup: Path | None = None
    failed_live: Path | None = None

    def fail_publication_and_restore(source: Path | str, destination: Path | str) -> None:
        nonlocal publications, failed_backup, failed_live
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path in indexes and source_path.name.endswith(".backup"):
            failed_backup = source_path
            failed_live = destination_path
            raise OSError("injected rollback restoration failure")
        if destination_path in indexes:
            publications += 1
            if publications == 2:
                raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        "rvbinary.mock.writer.os.replace", fail_publication_and_restore
    )

    with pytest.raises(RuntimeError, match="mock index rollback failed") as raised:
        generate_mock(config, sample_range=range(2, 4))

    assert failed_backup is not None
    assert failed_live is not None
    assert str(failed_backup) in str(raised.value)
    assert str(failed_live) in str(raised.value)
    assert failed_backup.read_bytes() == original[failed_live]
    retained_backups = set(Path(config.project.output).rglob("*.backup"))
    assert retained_backups == {failed_backup}
    assert not list(Path(config.project.output).rglob("*.candidate"))


def test_old_noise_shards_cannot_be_appended_or_recovered(tmp_path: Path) -> None:
    from rvbinary.mock.writer import _json_digest, rebuild_mock_indexes

    config = make_config(tmp_path, "old-noise", [11, 22], samples=2)
    artifacts = generate_mock(config, sample_range=range(0, 1))
    old_digest = _json_digest({
        "model": config.model.model_dump(mode="json"),
        "error_floors_kms": [float(v) for v in config.mock.error_floor.values_kms],
        "statistic_mode": config.statistics.mode,
    })
    for path in [
        *artifacts.shards, artifacts.table_index,
        *artifacts.binary_beta_indexes.values(), *artifacts.nonbinary_beta_indexes.values(),
    ]:
        with h5py.File(path, "r+") as handle:
            handle.attrs["config_digest"] = old_digest
    before = {p: file_digest(p) for p in artifacts.shards}

    from rvbinary.pdf.builder import _preflight

    with pytest.raises(ValueError, match="config_digest"):
        _preflight(config, artifacts)
    with pytest.raises(ValueError, match="config_digest"):
        generate_mock(config, sample_range=range(1, 2))
    with pytest.raises(ValueError, match="config_digest"):
        rebuild_mock_indexes(config)

    assert before == {p: file_digest(p) for p in artifacts.shards}
