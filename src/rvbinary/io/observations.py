"""Read and validate epoch observations and member identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table
from numpy.typing import NDArray

EPOCH_COLUMNS = ("TARGETID", "MJD", "VRAD", "VRAD_ERR", "Mass", "meanLOGG")


class InputDataError(ValueError):
    """Raised when an input dataset cannot satisfy the public data contract."""


@dataclass(frozen=True)
class EpochData:
    """Normalized arrays for one row per radial-velocity epoch."""

    targetid: NDArray[np.int64]
    mjd: NDArray[np.float64]
    vrad_kms: NDArray[np.float64]
    vrad_err_kms: NDArray[np.float64]
    mass_solar: NDArray[np.float64]
    logg_cgs: NDArray[np.float64]


def _read_table(path: Path) -> Table:
    try:
        return Table.read(path)
    except Exception as exc:
        raise InputDataError(f"{path}: unable to read FITS table: {exc}") from exc


def _require_columns(table: Table, path: Path, required: tuple[str, ...]) -> None:
    missing = [name for name in required if name not in table.colnames]
    if missing:
        raise InputDataError(f"{path}: missing required column(s): {', '.join(missing)}")


def _numeric_column(table: Table, path: Path, name: str) -> NDArray[np.float64]:
    column = table[name]
    if np.any(np.ma.getmaskarray(column)):
        raise InputDataError(f"{path}: {name} values must be finite and unmasked")
    try:
        values = np.asarray(column, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputDataError(f"{path}: {name} values must be numeric and finite") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise InputDataError(f"{path}: {name} values must be finite")
    return values


def _targetids(table: Table, path: Path) -> NDArray[np.int64]:
    column = table["TARGETID"]
    if np.any(np.ma.getmaskarray(column)):
        raise InputDataError(f"{path}: TARGETID values must be finite and unmasked")

    values = np.asarray(column)
    if values.ndim != 1 or values.dtype.kind not in "iuf":
        raise InputDataError(f"{path}: TARGETID values must be numeric integers")

    info = np.iinfo(np.int64)
    if values.dtype.kind == "u":
        out_of_range = np.any(values > info.max)
    elif values.dtype.kind == "i":
        out_of_range = np.any((values < info.min) | (values > info.max))
    else:
        if not np.all(np.isfinite(values)):
            raise InputDataError(f"{path}: TARGETID values must be finite")
        if np.any(values != np.trunc(values)):
            raise InputDataError(f"{path}: TARGETID values must be integers")
        out_of_range = np.any((values < info.min) | (values >= 2**63))

    if out_of_range:
        raise InputDataError(f"{path}: TARGETID values must fit in int64")
    return values.astype(np.int64)


def _take(data: EpochData, indices: NDArray[np.int64]) -> EpochData:
    return EpochData(
        targetid=data.targetid[indices],
        mjd=data.mjd[indices],
        vrad_kms=data.vrad_kms[indices],
        vrad_err_kms=data.vrad_err_kms[indices],
        mass_solar=data.mass_solar[indices],
        logg_cgs=data.logg_cgs[indices],
    )


def read_epochs(path: Path) -> EpochData:
    """Read an epoch FITS table into validated, stably grouped arrays."""
    path = Path(path)
    table = _read_table(path)
    _require_columns(table, path, EPOCH_COLUMNS)
    if len(table) == 0:
        raise InputDataError(f"{path}: epoch table must contain at least one row")

    data = EpochData(
        targetid=_targetids(table, path),
        mjd=_numeric_column(table, path, "MJD"),
        vrad_kms=_numeric_column(table, path, "VRAD"),
        vrad_err_kms=_numeric_column(table, path, "VRAD_ERR"),
        mass_solar=_numeric_column(table, path, "Mass"),
        logg_cgs=_numeric_column(table, path, "meanLOGG"),
    )
    if np.any(data.vrad_err_kms <= 0):
        raise InputDataError(f"{path}: VRAD_ERR values must be positive")

    for targetid in np.unique(data.targetid):
        rows = data.targetid == targetid
        for field_name, column_name in (("mass_solar", "Mass"), ("logg_cgs", "meanLOGG")):
            if np.unique(getattr(data, field_name)[rows]).size != 1:
                raise InputDataError(
                    f"{path}: TARGETID {targetid} must have exactly one {column_name} value"
                )

    order = np.argsort(data.targetid, kind="stable")
    return _take(data, order)


def read_member_ids(path: Path) -> NDArray[np.int64]:
    """Read a member FITS table containing a unique TARGETID column."""
    path = Path(path)
    table = _read_table(path)
    _require_columns(table, path, ("TARGETID",))
    if len(table) == 0:
        raise InputDataError(f"{path}: member table must contain at least one TARGETID")
    targetids = _targetids(table, path)
    if np.unique(targetids).size != targetids.size:
        raise InputDataError(f"{path}: member TARGETID values must be unique")
    return np.sort(targetids, kind="stable")


def select_members(data: EpochData, ids: NDArray[np.int64]) -> EpochData:
    """Select requested stars and retain stable, sorted TARGETID groups."""
    member_ids = np.asarray(ids, dtype=np.int64)
    if member_ids.ndim != 1 or member_ids.size == 0:
        raise InputDataError("member TARGETID array must be one-dimensional and non-empty")
    if np.unique(member_ids).size != member_ids.size:
        raise InputDataError("member TARGETID values must be unique")

    missing = np.setdiff1d(member_ids, data.targetid)
    if missing.size:
        listed = ", ".join(str(value) for value in missing)
        raise InputDataError(f"member TARGETID value(s) {listed} have no epoch data")

    mask = np.isin(data.targetid, member_ids)
    indices = np.flatnonzero(mask)
    indices = indices[np.argsort(data.targetid[indices], kind="stable")]
    selected = _take(data, indices)
    for targetid, count in zip(*np.unique(selected.targetid, return_counts=True), strict=True):
        if count < 2:
            raise InputDataError(f"TARGETID {targetid} must have at least two epochs")
    return selected
