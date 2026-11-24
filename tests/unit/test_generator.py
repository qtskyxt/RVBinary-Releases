import csv
import io
import json

import numpy as np
import pytest

from rvbinary.mock.generator import (
    GeneratorConfig,
    batch_config_json,
    batch_to_csv,
    generate_orbits,
)
from rvbinary.models.base import SimulationError
from rvbinary.models.dm91 import DM91Model


def test_reproducible_prefix():
    short = generate_orbits(GeneratorConfig(count=5))
    long = generate_orbits(GeneratorConfig(count=8))
    assert short.orbits == long.orbits[:5]
    assert short == generate_orbits(short.config)
    assert short.orbits != generate_orbits(GeneratorConfig(count=5, seed=7)).orbits


@pytest.mark.parametrize("field,value", [
    ("count", 0), ("count", 10001), ("count", True), ("count", 1.5),
    ("seed", -1), ("seed", 4294967296), ("seed", True), ("seed", 1.5),
    ("mass_solar", -1), ("mass_solar", float("inf")),
    ("logg_cgs", float("nan")), ("logg_cgs", 400),
    ("log10_period_mean", float("inf")), ("log10_period_sigma", 0),
])
def test_generator_validation(field, value):
    with pytest.raises(ValueError):
        GeneratorConfig(**{field: value})


def test_physical_constraints():
    batch = generate_orbits(GeneratorConfig(count=100))
    assert len(batch.orbits) == 100
    for index, orbit in enumerate(batch.orbits):
        assert orbit.primary.target_id == index
        assert 0.1 <= orbit.mass_ratio <= 1
        assert 0 <= orbit.eccentricity < 1
        assert 0 < orbit.period_days < 1e8
        assert orbit.periastron_distance_m >= orbit.primary_radius_m


def test_sampling_failure_is_atomic(monkeypatch):
    original = DM91Model.sample_star

    def fail_third(self, primary, rng, *, sample_index=None):
        if sample_index == 2:
            raise SimulationError("sample 2: impossible population")
        return original(self, primary, rng, sample_index=sample_index)

    monkeypatch.setattr(DM91Model, "sample_star", fail_third)
    with pytest.raises(SimulationError, match="sample 2"):
        generate_orbits(GeneratorConfig(count=5))


def test_export_contract():
    batch = generate_orbits(GeneratorConfig(count=8))
    rows = list(csv.DictReader(io.StringIO(batch_to_csv(batch))))
    assert len(rows) == 8
    assert list(rows[0]) == [
        "orbit_id", "primary_mass_solar", "primary_logg_cgs", "period_days",
        "mass_ratio", "eccentricity", "inclination_rad",
        "argument_periastron_rad", "mean_anomaly_rad",
    ]
    assert float(rows[0]["inclination_rad"]) == batch.orbits[0].inclination_rad
    assert float(rows[0]["period_days"]) == batch.orbits[0].period_days
    config = json.loads(batch_config_json(batch))
    assert config["schema_version"] == 1
    assert config["generator_kind"] == "dm91_orbits"
    assert config["parameters"]["count"] == 8
    assert config["parameters"]["seed"] == 20261010
    assert config["units"]["inclination_rad"] == "rad"
    assert config["software"]["rvbinary"]
    assert config["software"]["numpy"] == np.__version__
    assert config["seed_scheme"]
    assert config["model_constraints"]
