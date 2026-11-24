from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from rvbinary.io.observations import (
    InputDataError,
    read_epochs,
    read_member_ids,
    select_members,
)
from tests.data.make_fixtures import EPOCH_COLUMNS, write_epoch_fits, write_member_fits


def test_read_epochs_normalizes_arrays_in_stable_target_order(tmp_path: Path) -> None:
    path = write_epoch_fits(tmp_path)

    data = read_epochs(path)

    assert data.targetid.dtype == np.int64
    assert data.targetid.tolist() == [10, 10, 20, 20]
    assert data.mjd.tolist() == [2.0, 4.0, 3.0, 1.0]
    assert data.vrad_kms.tolist() == [4.0, 10.0, 14.0, 10.0]
    assert data.vrad_err_kms.tolist() == [1.0, 1.0, 2.0, 2.0]
    assert data.mass_solar.tolist() == [0.9, 0.9, 1.2, 1.2]
    assert data.logg_cgs.tolist() == [4.4, 4.4, 4.1, 4.1]


def test_read_epochs_preserves_large_int64_targetids(tmp_path: Path) -> None:
    lower = 2**53
    upper = lower + 1
    targetids = np.array([upper, lower, upper, lower], dtype=np.int64)
    path = write_epoch_fits(tmp_path, overrides={"TARGETID": targetids})

    data = read_epochs(path)

    assert data.targetid.tolist() == [lower, lower, upper, upper]


def test_read_member_ids_rejects_masked_targetids(tmp_path: Path) -> None:
    path = tmp_path / "members.fits"
    targetids = MaskedColumn([10, 20], dtype=np.int64, mask=[False, True])
    Table({"TARGETID": targetids}).write(path)

    with pytest.raises(InputDataError, match=r"members\.fits.*TARGETID.*unmasked"):
        read_member_ids(path)


def test_read_member_ids_accepts_int64_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "members.fits"
    info = np.iinfo(np.int64)
    Table({"TARGETID": np.array([info.max, info.min], dtype=np.int64)}).write(path)

    ids = read_member_ids(path)

    assert ids.dtype == np.int64
    assert ids.tolist() == [info.min, info.max]


def test_read_member_ids_rejects_unsigned_values_above_int64(tmp_path: Path) -> None:
    path = tmp_path / "members.fits"
    too_large = np.uint64(np.iinfo(np.int64).max) + np.uint64(1)
    Table({"TARGETID": np.array([too_large], dtype=np.uint64)}).write(path)

    with pytest.raises(InputDataError, match=r"members\.fits.*TARGETID.*fit in int64"):
        read_member_ids(path)


def test_read_member_ids_rejects_non_integral_floats(tmp_path: Path) -> None:
    path = tmp_path / "members.fits"
    Table({"TARGETID": np.array([10.5], dtype=np.float64)}).write(path)

    with pytest.raises(InputDataError, match=r"members\.fits.*TARGETID.*integers"):
        read_member_ids(path)


@pytest.mark.parametrize(
    "column",
    ["TARGETID", "MJD", "VRAD", "VRAD_ERR", "Mass", "meanLOGG"],
)
def test_read_epochs_rejects_each_missing_required_column(
    tmp_path: Path, column: str
) -> None:
    path = write_epoch_fits(tmp_path, omit=column)

    with pytest.raises(InputDataError, match=rf"epochs\.fits.*{column}"):
        read_epochs(path)


@pytest.mark.parametrize("column", ["MJD", "VRAD", "VRAD_ERR", "Mass", "meanLOGG"])
def test_read_epochs_rejects_nonfinite_numeric_values(tmp_path: Path, column: str) -> None:
    values = EPOCH_COLUMNS[column].copy()
    values[0] = np.nan
    path = write_epoch_fits(tmp_path, overrides={column: values})

    with pytest.raises(InputDataError, match=rf"epochs\.fits.*{column}.*finite"):
        read_epochs(path)


@pytest.mark.parametrize("uncertainty", [0.0, -0.1])
def test_read_epochs_requires_positive_uncertainties(
    tmp_path: Path, uncertainty: float
) -> None:
    values = EPOCH_COLUMNS["VRAD_ERR"].copy()
    values[0] = uncertainty
    path = write_epoch_fits(tmp_path, overrides={"VRAD_ERR": values})

    with pytest.raises(InputDataError, match=r"epochs\.fits.*VRAD_ERR.*positive"):
        read_epochs(path)


@pytest.mark.parametrize(
    ("column", "replacement"),
    [("Mass", 1.3), ("meanLOGG", 4.2)],
)
def test_read_epochs_requires_one_stellar_value_per_target(
    tmp_path: Path, column: str, replacement: float
) -> None:
    values = EPOCH_COLUMNS[column].copy()
    values[2] = replacement
    path = write_epoch_fits(tmp_path, overrides={column: values})

    with pytest.raises(InputDataError, match=rf"epochs\.fits.*TARGETID 20.*{column}"):
        read_epochs(path)


def test_read_member_ids_accepts_targetid_only_and_returns_sorted_ids(tmp_path: Path) -> None:
    path = write_member_fits(tmp_path, [20, 10])

    ids = read_member_ids(path)

    assert ids.dtype == np.int64
    assert ids.tolist() == [10, 20]


def test_read_member_ids_rejects_missing_targetid(tmp_path: Path) -> None:
    path = tmp_path / "members.fits"
    Table({"OTHER": [10]}).write(path)

    with pytest.raises(InputDataError, match=r"members\.fits.*TARGETID"):
        read_member_ids(path)


def test_read_member_ids_rejects_duplicate_members(tmp_path: Path) -> None:
    path = write_member_fits(tmp_path, [10, 10])

    with pytest.raises(InputDataError, match=r"members\.fits.*TARGETID.*unique"):
        read_member_ids(path)


def test_select_members_filters_and_keeps_stable_sorted_target_order(tmp_path: Path) -> None:
    data = read_epochs(write_epoch_fits(tmp_path))

    selected = select_members(data, np.array([20], dtype=np.int64))

    assert selected.targetid.tolist() == [20, 20]
    assert selected.mjd.tolist() == [3.0, 1.0]


def test_select_members_rejects_ids_missing_from_epochs(tmp_path: Path) -> None:
    data = read_epochs(write_epoch_fits(tmp_path))

    with pytest.raises(InputDataError, match=r"TARGETID.*30.*epoch"):
        select_members(data, np.array([10, 30], dtype=np.int64))


def test_select_members_rejects_a_retained_star_with_one_epoch(tmp_path: Path) -> None:
    columns = {name: values[:3] for name, values in EPOCH_COLUMNS.items()}
    data = read_epochs(write_epoch_fits(tmp_path, overrides=columns))

    with pytest.raises(InputDataError, match=r"TARGETID 10.*at least two epochs"):
        select_members(data, np.array([10, 20], dtype=np.int64))
