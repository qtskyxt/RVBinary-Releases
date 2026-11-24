from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from rvbinary.config import RVBinaryConfig
from rvbinary.io.legacy import open_mock_store
from rvbinary.mock.writer import generate_mock, rebuild_mock_indexes


def make_config(tmp_path: Path, *, samples: int = 5) -> RVBinaryConfig:
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
    Table({"TARGETID": np.array([11, 22], dtype=np.int64)}).write(members)
    return RVBinaryConfig.model_validate(
        {
            "project": {"name": "merge", "output": tmp_path / "out", "random_seed": 71},
            "observations": {
                "mock_epochs": epochs,
                "subsamples": [{"name": "all", "members": members}],
            },
            "mock": {"samples": samples, "error_floor": {"values_kms": [0.0, 0.4]}},
        }
    )


def scientific_digest(index: Path) -> str:
    digest = hashlib.sha256()
    with open_mock_store(index) as store:
        for sample_id in store.sample_ids:
            sample = store.sample(sample_id)
            for name in sorted(sample.arrays):
                digest.update(name.encode("ascii"))
                digest.update(np.ascontiguousarray(sample.arrays[name]).tobytes())
    return digest.hexdigest()


def test_merge_only_rebuilds_external_indexes_without_changing_shards(tmp_path: Path) -> None:
    config = make_config(tmp_path, samples=5)
    first = generate_mock(config, sample_range=range(0, 2))
    complete = generate_mock(config, sample_range=range(2, 5))
    expected = scientific_digest(complete.table_index)
    shard_bytes = {path: path.read_bytes() for path in complete.shards}
    for index in [
        complete.table_index,
        *complete.binary_beta_indexes.values(),
        *complete.nonbinary_beta_indexes.values(),
    ]:
        index.unlink()

    rebuilt = rebuild_mock_indexes(config)

    assert scientific_digest(rebuilt.table_index) == expected
    assert {path: path.read_bytes() for path in rebuilt.shards} == shard_bytes
    assert rebuilt.table_index.stat().st_size < sum(path.stat().st_size for path in first.shards)


def test_merge_only_rejects_missing_sample_coverage(tmp_path: Path) -> None:
    config = make_config(tmp_path, samples=5)
    partial = generate_mock(config, sample_range=range(0, 2))
    for index in [
        partial.table_index,
        *partial.binary_beta_indexes.values(),
        *partial.nonbinary_beta_indexes.values(),
    ]:
        index.unlink()

    with pytest.raises(ValueError, match="sample coverage"):
        rebuild_mock_indexes(config)

