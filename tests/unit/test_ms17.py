from pathlib import Path

import numpy as np
import pytest

from rvbinary.models.base import PrimaryStar
from rvbinary.models.ms17 import MS17Model, PeriodTable


def table_file(tmp_path: Path) -> Path:
    path = tmp_path / "period.dat"
    np.savetxt(
        path,
        [
            [m, p, 0, 0, d]
            for m in [0.8, 1.0, 1.2]
            for p, d in [(0.2, 0.02), (1.0, 0.04), (1.8, 0.08)]
        ],
    )
    return path


def test_trial_table_domain_and_bad_grid(tmp_path: Path) -> None:
    path = table_file(tmp_path)
    table = PeriodTable.load(path)
    assert table.mass_bounds == (0.8, 1.2)
    assert table.period_bounds == (0.2, 1.8)
    with pytest.raises(ValueError, match="mass"):
        table.sample_log_period(0.79, np.random.default_rng(0))
    with pytest.raises(ValueError, match="mass"):
        MS17Model(table).sample_star(PrimaryStar(1, 1.2, 4.5), np.random.default_rng(0))
    a = np.loadtxt(path)
    np.savetxt(path, a[:-1])
    with pytest.raises(ValueError, match="grid"):
        PeriodTable.load(path)


def test_ms17_repeatability_and_clearance(tmp_path: Path) -> None:
    model = MS17Model(PeriodTable.load(table_file(tmp_path)))
    primary = PrimaryStar(11, 1.0, 4.5)
    first = model.sample_star(primary, np.random.default_rng(91))
    assert first == model.sample_star(primary, np.random.default_rng(91))
    from rvbinary.models.dm91 import _roche_lobe_fraction

    rng = np.random.default_rng(8)
    for _ in range(300):
        o = model.sample_star(primary, rng)
        assert 0.2 <= np.log10(o.period_days) <= 1.8
        assert _roche_lobe_fraction(o.mass_ratio) * o.periastron_distance_m >= primary.radius_m
        if o.period_days <= 2:
            assert o.eccentricity == 0


def test_ms17_config_and_dispatch(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from rvbinary.config import MS17Config
    from rvbinary.models.factory import make_model, model_payload

    c = MS17Config(period_table=table_file(tmp_path))
    config = SimpleNamespace(model=c)
    assert isinstance(make_model(config), MS17Model)
    assert model_payload(config)["table_sha256"] == PeriodTable.load(c.period_table).sha256
    with pytest.raises(ValueError):
        MS17Config(period_table=c.period_table, log10_period_mean=4.8)


def test_twin_mixture_matches_integrated_piecewise_density(tmp_path: Path) -> None:
    model = MS17Model(PeriodTable.load(table_file(tmp_path)))
    rng = np.random.default_rng(42)
    q = np.array([model.sample_mass_ratio(1.0, 0.5, rng) for _ in range(10000)])
    # Independent analytic integrals of low-q q^.3 and high-q q^(-.5),
    # with the source's F=.3 enhancement relative to the high-q integral.
    high = 2 * (1 - np.sqrt(0.3))
    low = 0.3 ** (-0.8) * (0.3**1.3 - 0.1**1.3) / 1.3
    twin = high * 0.3 / 0.7
    expected = (2 * (1 - np.sqrt(0.95)) + twin) / (low + high + twin)
    assert abs(float(np.mean(q > 0.95)) - expected) < 0.015


def test_inverse_power_draw_handles_singular_exponent() -> None:
    from rvbinary.models.ms17 import _power_draw

    for exponent in [-1.0, -1e15, 0.6]:
        value = _power_draw(1e-5, 0.5, exponent, np.random.default_rng(1))
        assert 1e-5 <= value <= 0.5


@pytest.mark.parametrize("bad", ["negative", "nan", "duplicate"])
def test_trial_table_rejects_invalid_values(tmp_path: Path, bad: str) -> None:
    path = table_file(tmp_path)
    a = np.loadtxt(path)
    if bad == "negative":
        a[0, 4] = -1
    elif bad == "nan":
        a[0, 1] = np.nan
    else:
        a[-1] = a[0]
    np.savetxt(path, a)
    with pytest.raises(ValueError):
        PeriodTable.load(path)


def test_impossible_ms17_primary_fails_with_context(tmp_path: Path) -> None:
    from rvbinary.models.base import SimulationError

    model = MS17Model(PeriodTable.load(table_file(tmp_path)), max_attempts=2)
    with pytest.raises(SimulationError, match="sample 3, TARGETID 11"):
        model.sample_star(PrimaryStar(11, 1.0, -5.0), np.random.default_rng(1), sample_index=3)


def test_ms17_yaml_resolves_external_table_relative_to_config(tmp_path: Path) -> None:
    import yaml

    from rvbinary.config import load_config

    table_file(tmp_path)
    raw = {
        "project": {"name": "trial", "output": "out", "random_seed": 1},
        "observations": {"mock_epochs": "epochs.fits", "subsamples": []},
        "model": {"name": "ms17", "period_table": "period.dat"},
    }
    path = tmp_path / "trial.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config.model.name == "ms17"
    assert config.model.period_table == tmp_path / "period.dat"


def test_umi_encounter_cutoff_and_missing_table(tmp_path: Path) -> None:
    from astropy.constants import pc

    table = PeriodTable.load(table_file(tmp_path))
    model = MS17Model(table, environment="umi_legacy")
    maximum_axis = 1 / np.sqrt(
        np.pi * 8000 * (10e9 * 365.2425 * 86400) * (0.006 / 0.4**4 / pc.value**3)
    )
    o = model.sample_star(PrimaryStar(1, 1.0, 4.5), np.random.default_rng(0))
    assert o.semi_major_axis_m <= maximum_axis
    with pytest.raises(FileNotFoundError):
        PeriodTable.load(tmp_path / "absent.dat")


def test_ms17_defaults_to_verified_bundled_trial_table() -> None:
    from types import SimpleNamespace

    from rvbinary.config import MS17Config
    from rvbinary.models.factory import make_model, model_payload

    c = MS17Config()
    assert c.period_table is None
    model = make_model(SimpleNamespace(model=c))
    assert isinstance(model, MS17Model)
    assert model.table.sha256 == "490592bec1d0a2c9708053102c38f0d105bb7c8230c23566d41e3a0faca73c50"
    assert model_payload(SimpleNamespace(model=c))["table_sha256"] == model.table.sha256
