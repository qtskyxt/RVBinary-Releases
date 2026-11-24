"""Duquennoy-Mayor (1991) binary-orbit population model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.constants import G

from rvbinary.models.base import BinaryOrbit, PrimaryStar, SimulationError


def _roche_lobe_fraction(mass_ratio: float) -> float:
    inverse_q = 1.0 / mass_ratio
    inverse_q_cuberoot = inverse_q ** (1.0 / 3.0)
    inverse_q_two_thirds = inverse_q_cuberoot**2
    return 0.49 * inverse_q_two_thirds / (
        0.6 * inverse_q_two_thirds + np.log1p(inverse_q_cuberoot)
    )


@dataclass(frozen=True)
class DM91Model:
    """Sample the accepted legacy DM91 orbital-parameter distributions."""

    log10_period_mean: float = 4.8
    log10_period_sigma: float = 2.3
    max_attempts: int = 10_000

    def __post_init__(self) -> None:
        if not np.isfinite(self.log10_period_mean):
            raise ValueError("log10_period_mean must be finite")
        if not np.isfinite(self.log10_period_sigma) or self.log10_period_sigma <= 0.0:
            raise ValueError("log10_period_sigma must be finite and positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

    def _sample_mass_ratio(self, context: str, rng: np.random.Generator) -> float:
        for _ in range(self.max_attempts):
            mass_ratio = rng.random() * 0.9 + 0.1
            proposal = rng.random()
            probability = np.exp(-((mass_ratio - 0.23) ** 2) / (2.0 * 0.42**2))
            if proposal <= probability:
                return float(mass_ratio)
        raise SimulationError(
            f"{context}: mass-ratio rejection sampler failed "
            f"after {self.max_attempts} attempts"
        )

    def _sample_period(
        self,
        primary: PrimaryStar,
        mass_ratio: float,
        rng: np.random.Generator,
        context: str,
    ) -> tuple[float, float]:
        minimum_axis = primary.radius_m
        roche_fraction = _roche_lobe_fraction(mass_ratio)
        total_mass = primary.mass_kg * (1.0 + mass_ratio)

        for _ in range(self.max_attempts):
            log10_period = rng.normal(self.log10_period_mean, self.log10_period_sigma)
            eccentric_draw = rng.random()
            if not np.isfinite(log10_period) or log10_period >= 8.0:
                continue
            period_days = 10.0**log10_period
            period_seconds = period_days * 86400.0
            semi_major_axis = (
                G.value * total_mass * period_seconds**2 / (4.0 * np.pi**2)
            ) ** (1.0 / 3.0)
            if not np.isfinite(semi_major_axis) or minimum_axis >= semi_major_axis:
                continue
            eccentricity_limit = 1.0 - minimum_axis / semi_major_axis
            trial_eccentricity = eccentric_draw * eccentricity_limit
            roche_at_periastron = (
                roche_fraction * semi_major_axis * (1.0 - trial_eccentricity)
            )
            if roche_at_periastron >= minimum_axis:
                return float(period_days), float(trial_eccentricity)

        raise SimulationError(
            f"{context}: period rejection sampler failed "
            f"after {self.max_attempts} attempts"
        )

    def _sample_inclination(self, context: str, rng: np.random.Generator) -> float:
        for _ in range(self.max_attempts):
            inclination = rng.random() * np.pi / 2.0
            if rng.random() <= np.sin(inclination):
                return float(inclination)
        raise SimulationError(
            f"{context}: inclination rejection sampler failed "
            f"after {self.max_attempts} attempts"
        )

    def sample_star(
        self,
        primary: PrimaryStar,
        rng: np.random.Generator,
        *,
        sample_index: int | None = None,
    ) -> BinaryOrbit:
        """Draw one orbit using only the supplied random-number generator."""
        context = f"TARGETID {primary.target_id}"
        if sample_index is not None:
            context = f"sample {sample_index}, {context}"
        mass_ratio = self._sample_mass_ratio(context, rng)
        period_days, eccentricity = self._sample_period(
            primary, mass_ratio, rng, context
        )

        return BinaryOrbit(
            primary=primary,
            mass_ratio=mass_ratio,
            period_days=period_days,
            eccentricity=eccentricity,
            inclination_rad=self._sample_inclination(context, rng),
            argument_periastron_rad=float(rng.random() * 2.0 * np.pi),
            mean_anomaly_rad=float(rng.random() * 2.0 * np.pi),
        )
