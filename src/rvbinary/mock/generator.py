"""Small, reproducible DM91 orbit populations, independent of the GUI."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from importlib import metadata
from math import isfinite

from rvbinary import __version__
from rvbinary.mock.orbit import rng_for_sample
from rvbinary.models.base import BinaryOrbit, PrimaryStar
from rvbinary.models.dm91 import DM91Model

FIELDS = (
    "orbit_id", "primary_mass_solar", "primary_logg_cgs", "period_days",
    "mass_ratio", "eccentricity", "inclination_rad",
    "argument_periastron_rad", "mean_anomaly_rad",
)


@dataclass(frozen=True)
class GeneratorConfig:
    """A lightweight generator configuration, not a pipeline configuration."""

    count: int = 1000
    seed: int = 20261010
    mass_solar: float = 1.0
    logg_cgs: float = 4.5
    log10_period_mean: float = 4.8
    log10_period_sigma: float = 2.3

    def __post_init__(self) -> None:
        for name, lower, upper in (("count", 1, 10000), ("seed", 0, 4294967295)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        for name in ("mass_solar", "logg_cgs", "log10_period_mean", "log10_period_sigma"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.mass_solar <= 0 or self.log10_period_sigma <= 0:
            raise ValueError("mass_solar and log10_period_sigma must be positive")
        try:
            PrimaryStar(0, self.mass_solar, self.logg_cgs).radius_m
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError("logg_cgs must give a finite positive stellar radius") from exc


@dataclass(frozen=True)
class OrbitBatch:
    """A complete, immutable population with the inputs that generated it."""

    config: GeneratorConfig
    orbits: tuple[BinaryOrbit, ...]


def generate_orbits(config: GeneratorConfig) -> OrbitBatch:
    """Generate all requested orbits, or raise without returning a partial batch."""
    model = DM91Model(config.log10_period_mean, config.log10_period_sigma)
    orbits = tuple(
        model.sample_star(
            PrimaryStar(index, config.mass_solar, config.logg_cgs),
            rng_for_sample(config.seed, index),
            sample_index=index,
        )
        for index in range(config.count)
    )
    return OrbitBatch(config, orbits)


def orbit_row(orbit: BinaryOrbit) -> dict[str, int | float]:
    """Return one CSV record with explicit units in the column names."""
    return dict(zip(FIELDS, (
        orbit.primary.target_id, orbit.primary.mass_solar, orbit.primary.logg_cgs,
        orbit.period_days, orbit.mass_ratio, orbit.eccentricity,
        orbit.inclination_rad, orbit.argument_periastron_rad, orbit.mean_anomaly_rad,
    ), strict=True))


def batch_to_csv(batch: OrbitBatch) -> str:
    """Export every orbit using Python's round-trippable float representation."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(orbit_row(orbit) for orbit in batch.orbits)
    return output.getvalue()


def batch_config_json(batch: OrbitBatch) -> str:
    """Serialize the successful batch's configuration and software provenance."""
    software = {"rvbinary": __version__}
    for name in ("numpy", "astropy", "matplotlib"):
        software[name] = metadata.version(name)
    return json.dumps({
        "schema_version": 1,
        "generator_kind": "dm91_orbits",
        "parameters": asdict(batch.config),
        "seed_scheme": "rng_for_sample: SeedSequence([seed, orbit_id]); target_id=orbit_id",
        "model_constraints": {
            "implementation": "rvbinary.models.dm91.DM91Model",
            "mass_ratio_range": [0.1, 1.0],
            "log10_period_upper_exclusive": 8.0,
            "period_eccentricity_filter": "stellar radius and Roche lobe at periastron",
            "max_attempts": 10000,
        },
        "units": {
            "orbit_id": "index", "primary_mass_solar": "solar mass",
            "primary_logg_cgs": "log10(g / cm s^-2)", "period_days": "day",
            "mass_ratio": "dimensionless", "eccentricity": "dimensionless",
            "inclination_rad": "rad", "argument_periastron_rad": "rad",
            "mean_anomaly_rad": "rad",
        },
        "software": software,
    }, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
