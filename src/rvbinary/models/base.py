"""Shared records and errors for orbit simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np
from astropy.constants import G, M_sun


class SimulationError(RuntimeError):
    """Raised when a bounded simulation algorithm cannot produce a result."""


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class PrimaryStar:
    """Primary-star properties needed by the DM91 orbit model."""

    target_id: int
    mass_solar: float
    logg_cgs: float

    def __post_init__(self) -> None:
        _require_finite("mass_solar", self.mass_solar)
        _require_finite("logg_cgs", self.logg_cgs)
        if self.mass_solar <= 0.0:
            raise ValueError("mass_solar must be positive")

    @property
    def mass_kg(self) -> float:
        return self.mass_solar * M_sun.value

    @property
    def radius_m(self) -> float:
        gravity_m_s2 = 10.0**self.logg_cgs / 100.0
        radius = np.sqrt(G.value * self.mass_kg / gravity_m_s2)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                f"TARGETID {self.target_id}: stellar radius is not finite and positive"
            )
        return float(radius)


@dataclass(frozen=True)
class BinaryOrbit:
    """One sampled binary orbit, with angles in radians and period in days."""

    primary: PrimaryStar
    mass_ratio: float
    period_days: float
    eccentricity: float
    inclination_rad: float
    argument_periastron_rad: float
    mean_anomaly_rad: float

    def __post_init__(self) -> None:
        for name in (
            "mass_ratio",
            "period_days",
            "eccentricity",
            "inclination_rad",
            "argument_periastron_rad",
            "mean_anomaly_rad",
        ):
            _require_finite(name, getattr(self, name))
        if not 0.1 <= self.mass_ratio <= 1.0:
            raise ValueError("mass_ratio must be in [0.1, 1.0]")
        if self.period_days <= 0.0:
            raise ValueError("period_days must be positive")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("eccentricity must be in [0, 1)")
        if not 0.0 <= self.inclination_rad <= np.pi / 2.0:
            raise ValueError("inclination_rad must be in [0, pi/2]")
        if not 0.0 <= self.argument_periastron_rad < 2.0 * np.pi:
            raise ValueError("argument_periastron_rad must be in [0, 2*pi)")
        if not 0.0 <= self.mean_anomaly_rad < 2.0 * np.pi:
            raise ValueError("mean_anomaly_rad must be in [0, 2*pi)")

    @property
    def period_seconds(self) -> float:
        return self.period_days * 86400.0

    @property
    def semi_major_axis_m(self) -> float:
        total_mass = self.primary.mass_kg * (1.0 + self.mass_ratio)
        return float(
            (G.value * total_mass * self.period_seconds**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
        )

    @property
    def primary_radius_m(self) -> float:
        return self.primary.radius_m

    @property
    def periastron_distance_m(self) -> float:
        return self.semi_major_axis_m * (1.0 - self.eccentricity)


class OrbitModel(Protocol):
    def sample_star(
        self, primary: PrimaryStar, rng: np.random.Generator, *, sample_index: int | None = None
    ) -> BinaryOrbit: ...
