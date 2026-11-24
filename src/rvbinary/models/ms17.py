"""Restricted, table-driven MS17 trial prescription; not a full MS17 model."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
from astropy.constants import G, pc

from rvbinary.models.base import BinaryOrbit, PrimaryStar, SimulationError
from rvbinary.models.dm91 import _roche_lobe_fraction

MS17_IMPLEMENTATION = "ms17-lowmass-trial/v1"


@dataclass(frozen=True)
class PeriodTable:
    masses: np.ndarray
    log_periods: np.ndarray
    density: np.ndarray
    sha256: str

    @classmethod
    def load(cls, path: Path | None = None) -> PeriodTable:
        content = (
            files("rvbinary").joinpath("data/Pdist_small.dat").read_bytes()
            if path is None
            else Path(path).read_bytes()
        )
        if path is None and hashlib.sha256(content).hexdigest() != (
            "490592bec1d0a2c9708053102c38f0d105bb7c8230c23566d41e3a0faca73c50"
        ):
            raise ValueError("bundled MS17 trial table checksum mismatch")
        data = np.loadtxt(io.BytesIO(content), ndmin=2)
        if data.shape[1] != 5 or not np.all(np.isfinite(data)):
            raise ValueError("MS17 trial table must contain five finite columns")
        if np.any(data[:, 0] <= 0) or np.any(data[:, 4] < 0):
            raise ValueError("MS17 table requires positive mass and nonnegative density")
        masses, periods = np.unique(data[:, 0]), np.unique(data[:, 1])
        if (
            len(masses) < 2
            or len(periods) < 2
            or len(data) != len(masses) * len(periods)
            or len(np.unique(data[:, :2], axis=0)) != len(data)
        ):
            raise ValueError("MS17 table must be a complete unique rectangular grid")
        for axis in (masses, periods):
            if not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-8, atol=1e-10):
                raise ValueError("MS17 table must have a regular grid")
        ordered = data[np.lexsort((data[:, 1], data[:, 0]))]
        density = ordered[:, 4].reshape(len(masses), len(periods)).copy()
        if np.any(density.sum(axis=1) <= 0):
            raise ValueError("MS17 table rows need positive normalization")
        low, high = max(0.2, float(periods[0])), min(7.98, float(periods[-1]))
        if high <= low:
            raise ValueError("MS17 table has no supported period interval")
        edges = np.clip(np.r_[periods[0], (periods[:-1] + periods[1:]) / 2, periods[-1]], low, high)
        normalization = (density * np.diff(edges)).sum(axis=1)
        if not np.all(np.isfinite(normalization)) or np.any(normalization <= 0):
            raise ValueError("MS17 supported period interval has zero/invalid normalization")
        for array in (masses, periods, density):
            array.flags.writeable = False
        return cls(masses, periods, density, hashlib.sha256(content).hexdigest())

    @property
    def mass_bounds(self) -> tuple[float, float]:
        return float(self.masses[0]), float(self.masses[-1])

    @property
    def period_bounds(self) -> tuple[float, float]:
        return float(self.log_periods[0]), float(self.log_periods[-1])

    def validate_mass(self, mass: float) -> None:
        if not np.isfinite(mass) or not self.masses[0] <= mass <= self.masses[-1]:
            raise ValueError(f"MS17 mass {mass} outside table domain {self.mass_bounds}")
        if not 0.8 <= mass < 1.2:
            raise ValueError("MS17 trial mass must satisfy 0.8 <= mass < 1.2 solar masses")

    def sample_log_period(self, mass: float, rng: np.random.Generator) -> float:
        self.validate_mass(mass)
        row = int(np.argmin(np.abs(np.log10(self.masses) - np.log10(mass))))
        nodes = self.log_periods
        low, high = max(0.2, float(nodes[0])), min(7.98, float(nodes[-1]))
        if high <= low:
            raise ValueError("MS17 table has no supported period interval")
        edges = np.clip(np.r_[nodes[0], (nodes[:-1] + nodes[1:]) / 2, nodes[-1]], low, high)
        weights = self.density[row] * np.diff(edges)
        if weights.sum() <= 0:
            raise ValueError("MS17 supported period interval has zero normalization")
        index = int(rng.choice(len(nodes), p=weights / weights.sum()))
        return float(rng.uniform(edges[index], edges[index + 1]))


def _integral(low: float, high: float, exponent: float) -> float:
    power = exponent + 1
    return float(
        np.log(high / low)
        if abs(power) < 1e-10
        else low**power * np.expm1(power * np.log(high / low)) / power
    )


def _power_draw(low: float, high: float, exponent: float, rng: np.random.Generator) -> float:
    u = float(rng.random())
    if u == 0:
        return low
    power = exponent + 1
    if abs(power) < 1e-10:
        return float(np.exp(np.log(low) + u * np.log(high / low)))
    # Stable even for the very negative eta immediately above logP=0.5.
    log_value = np.logaddexp(np.log1p(-u) + power * np.log(low), np.log(u) + power * np.log(high))
    return float(np.clip(np.exp(log_value / power), low, high))


@dataclass(frozen=True)
class MS17Model:
    table: PeriodTable
    environment: str = "none"
    max_attempts: int = 10_000

    def __post_init__(self) -> None:
        if self.environment not in ("none", "umi_legacy") or self.max_attempts <= 0:
            raise ValueError("invalid MS17 environment or max_attempts")

    def sample_mass_ratio(self, mass: float, log_period: float, rng: np.random.Generator) -> float:
        gamma = -0.5 - 0.3 * max(0.0, log_period - 5.0)
        f1 = 0.3 - 0.15 * np.log10(mass)
        twin_end = 8.0 - mass
        fraction = float(
            f1 if log_period < 1 else f1 * max(0.0, 1 - (log_period - 1) / (twin_end - 1))
        )
        high = _integral(0.3, 1.0, gamma)
        low = 0.3 ** (gamma - 0.3) * _integral(0.1, 0.3, 0.3)
        twin = high * fraction / (1 - fraction)
        component = int(rng.choice(3, p=np.array([low, high, twin]) / (low + high + twin)))
        if component == 0:
            return _power_draw(0.1, 0.3, 0.3, rng)
        if component == 1:
            return _power_draw(0.3, 1.0, gamma, rng)
        return float(rng.uniform(0.95, 1.0))

    def sample_star(
        self, primary: PrimaryStar, rng: np.random.Generator, *, sample_index: int | None = None
    ) -> BinaryOrbit:
        self.table.validate_mass(primary.mass_solar)
        radius = primary.radius_m
        maximum_axis = np.inf
        if self.environment == "umi_legacy":
            age = 10.0e9 * 365.2425 * 86400
            density = 0.006 / 0.4**4 / pc.value**3
            maximum_axis = 1 / np.sqrt(np.pi * 8000 * age * density)
        for _ in range(self.max_attempts):
            log_period = self.table.sample_log_period(primary.mass_solar, rng)
            period = 10.0**log_period
            q = self.sample_mass_ratio(primary.mass_solar, log_period, rng)
            axis = (
                G.value * primary.mass_kg * (1 + q) * (period * 86400) ** 2 / (4 * np.pi**2)
            ) ** (1 / 3)
            roche = _roche_lobe_fraction(q) * axis
            if axis > maximum_axis or roche < radius:
                continue
            if period <= 2:
                ecc = 0.0
            else:
                ceiling = min(1 - radius / roche, 1 - (period / 2) ** (-2 / 3))
                if ceiling <= 1e-5:
                    continue
                eta = 0.0 if log_period <= 0.5 else 0.6 - 0.7 / (min(log_period, 6.0) - 0.5)
                ecc = _power_draw(1e-5, ceiling, eta, rng)
            orbit = BinaryOrbit(
                primary,
                q,
                period,
                ecc,
                float(np.arccos(rng.uniform(0, 1))),
                float(rng.uniform(0, 2 * np.pi)),
                float(rng.uniform(0, 2 * np.pi)),
            )
            if _roche_lobe_fraction(q) * orbit.periastron_distance_m >= radius:
                return orbit
        raise SimulationError(
            f"sample {sample_index}, TARGETID {primary.target_id}: "
            f"MS17 trial sampling failed after {self.max_attempts} attempts"
        )
