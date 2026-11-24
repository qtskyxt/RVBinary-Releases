"""Small FITS-table builders shared by observation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

EPOCH_COLUMNS = {
    "TARGETID": np.array([20, 10, 20, 10], dtype=np.int64),
    "MJD": np.array([3.0, 2.0, 1.0, 4.0]),
    "VRAD": np.array([14.0, 4.0, 10.0, 10.0]),
    "VRAD_ERR": np.array([2.0, 1.0, 2.0, 1.0]),
    "Mass": np.array([1.2, 0.9, 1.2, 0.9]),
    "meanLOGG": np.array([4.1, 4.4, 4.1, 4.4]),
}


def write_epoch_fits(
    directory: Path,
    *,
    omit: str | None = None,
    overrides: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a minimal epoch table, optionally malformed for one test."""
    columns = {name: values.copy() for name, values in EPOCH_COLUMNS.items()}
    if overrides:
        columns.update(overrides)
    if omit:
        columns.pop(omit)
    path = directory / "epochs.fits"
    Table(columns).write(path)
    return path


def write_member_fits(directory: Path, targetids: list[int] | np.ndarray) -> Path:
    """Write a minimal member table containing only the public identifier."""
    path = directory / "members.fits"
    Table({"TARGETID": np.asarray(targetids, dtype=np.int64)}).write(path)
    return path


if __name__ == "__main__":
    output = Path(__file__).parent
    write_epoch_fits(output)
    write_member_fits(output, [10, 20])
