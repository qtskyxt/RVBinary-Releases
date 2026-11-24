from pathlib import Path

import h5py
import numpy as np
import pytest

from rvbinary.config import MS17Config
from rvbinary.mock.writer import _config_digest, generate_mock
from rvbinary.pdf.builder import build_pdf_grid
from tests.integration.test_pdf_grid import make_config
from tests.unit.test_ms17 import table_file


def test_ms17_native_products_pdf_and_changed_table_guard(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.model = MS17Config(period_table=table_file(tmp_path), include_unrestricted=True)
    config.mock.samples = 2
    artifacts = generate_mock(config)
    digest = _config_digest(config)
    with h5py.File(artifacts.table_index, "r") as handle:
        assert handle.attrs["config_digest"] == digest
    assert len(build_pdf_grid(config, artifacts)) == 2
    a = np.loadtxt(config.model.period_table)
    a[:, 4] *= 2
    np.savetxt(config.model.period_table, a)
    assert _config_digest(config) != digest
    config.mock.samples = 3
    with pytest.raises(ValueError, match="config_digest"):
        generate_mock(config, sample_range=range(2, 3))


def test_ms17_rejects_unsupported_star_before_shards(tmp_path: Path) -> None:
    from astropy.table import Table

    config = make_config(tmp_path)
    table = Table.read(config.observations.mock_epochs)
    table["Mass"][0:3] = 0.79
    table.write(config.observations.mock_epochs, overwrite=True)
    config.model = MS17Config(period_table=table_file(tmp_path))
    with pytest.raises(ValueError, match="TARGETID 11"):
        generate_mock(config)
    assert not list(Path(config.project.output).rglob("*.h5"))


@pytest.mark.parametrize("case", ["outside", "zero_support"])
def test_invalid_period_support_creates_no_source_files(tmp_path: Path, case: str) -> None:
    config = make_config(tmp_path)
    path = table_file(tmp_path)
    a = np.loadtxt(path)
    if case == "outside":
        a[:, 1] += 8
    else:
        for i in range(3):
            a[i * 3 : (i + 1) * 3, 1] = [-10.0, 0.0, 10.0]
            a[i * 3 : (i + 1) * 3, 4] = [1.0, 0.0, 0.0]
    np.savetxt(path, a)
    from rvbinary.models.ms17 import PeriodTable

    with pytest.raises(ValueError):
        PeriodTable.load(path)
    config.model = MS17Config(period_table=path)
    with pytest.raises(ValueError):
        generate_mock(config)
    assert not list(Path(config.project.output).rglob("*.h5"))


def test_mpi_preflight_checks_existing_source_batches(tmp_path: Path) -> None:
    from astropy.table import Table

    from rvbinary.io.observations import read_epochs
    from rvbinary.mock.writer import _write_source_batch
    from rvbinary.parallel import _prepare_sources

    config = make_config(tmp_path)
    table = Table.read(config.observations.mock_epochs)
    table["Mass"][0:3] = 0.79
    table.write(config.observations.mock_epochs, overwrite=True)
    _write_source_batch(config, read_epochs(config.observations.mock_epochs))
    config.model = MS17Config(period_table=table_file(tmp_path))
    with pytest.raises(ValueError, match="TARGETID 11"):
        _prepare_sources(config)
