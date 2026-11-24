from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

from rvbinary.config import RVBinaryConfig
from rvbinary.mock.writer import generate_mock
from rvbinary.pdf.builder import build_pdf_grid

# epoch-shared/v1 reference: per-epoch NumPy draws for target 22, samples 0:3.
# Nonbinary star statistics: 1.7088037686264919, 1.4509536425101968,
# 2.152822264129979; pair denominators use the reported [0.5, 0.7, 0.9] errors.
FROZEN_BINARY_COUNTS = np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
FROZEN_NONBINARY_COUNTS = np.array([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
FROZEN_BINARY_PDF = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
FROZEN_NONBINARY_PDF = np.array(
    [2.0 / 3.0, 0.3902364428770174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)


def _config(tmp_path: Path) -> RVBinaryConfig:
    epochs = tmp_path / "epochs.fits"
    members = tmp_path / "members.fits"
    Table(
        {
            "TARGETID": np.repeat(np.array([11, 22], dtype=np.int64), 3),
            "MJD": np.tile([59000.0, 59002.0, 59007.0], 2),
            "VRAD": np.zeros(6),
            "VRAD_ERR": np.tile([0.5, 0.7, 0.9], 2),
            "Mass": np.repeat([0.9, 1.1], 3),
            "meanLOGG": np.repeat([4.4, 4.2], 3),
        }
    ).write(epochs)
    Table({"TARGETID": np.array([22], dtype=np.int64)}).write(members)
    return RVBinaryConfig.model_validate(
        {
            "project": {"name": "legacy-regression", "output": tmp_path / "out", "random_seed": 91},
            "observations": {
                "mock_epochs": epochs,
                "subsamples": [{"name": "FEH0", "members": members}],
            },
            "model": {"include_unrestricted": True, "period_limits_days": [1e-12]},
            "mock": {"samples": 3, "error_floor": {"values_kms": [0.0]}},
            "pdf": {"bins": 12},
            "statistics": {"mode": "consistent"},
        }
    )


def test_consistent_mode_matches_frozen_pdf(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifacts = build_pdf_grid(config, generate_mock(config))
    unrestricted = next(item for item in artifacts if item.period_limit_days is None)

    with np.load(unrestricted.path, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["binary_counts"], FROZEN_BINARY_COUNTS)
        np.testing.assert_array_equal(stored["nonbinary_counts"], FROZEN_NONBINARY_COUNTS)
        np.testing.assert_allclose(stored["binary_density"], FROZEN_BINARY_PDF, rtol=1e-12)
        np.testing.assert_allclose(
            stored["nonbinary_density"], FROZEN_NONBINARY_PDF, rtol=1e-12
        )
