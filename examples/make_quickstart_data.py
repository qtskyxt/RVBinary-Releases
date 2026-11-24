"""Create deterministic, low-cost FITS inputs for the README quick start."""

from pathlib import Path

import numpy as np
from astropy.table import Table

output = Path(__file__).resolve().parent / "synthetic_data"
(output / "FEH0").mkdir(parents=True, exist_ok=True)
targetids = np.repeat(np.array([10, 20, 30, 40], dtype=np.int64), 3)
Table(
    {
        "TARGETID": targetids,
        "MJD": np.tile([59000.0, 59003.0, 59010.0], 4),
        "VRAD": np.zeros(targetids.size),
        "VRAD_ERR": np.tile([0.7, 0.8, 0.9], 4),
        "Mass": np.repeat([0.85, 0.95, 1.05, 1.15], 3),
        "meanLOGG": np.repeat([4.5, 4.4, 4.3, 4.2], 3),
    }
).write(output / "all_epochs.fits", overwrite=True)
Table({"TARGETID": np.array([10, 20, 30, 40], dtype=np.int64)}).write(
    output / "FEH0" / "loa_mem.fits", overwrite=True
)
