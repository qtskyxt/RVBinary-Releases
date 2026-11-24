"""Deterministic random streams and binary radial velocities."""

from __future__ import annotations

import numpy as np
from astropy.constants import G
from numpy.typing import ArrayLike, NDArray

from rvbinary.models.base import BinaryOrbit, SimulationError


def rng_for_sample(seed: int, sample_index: int) -> np.random.Generator:
    """Return the rank-independent random stream for one sample index."""
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    return np.random.default_rng(np.random.SeedSequence([seed, sample_index]))


def solve_kepler(
    mean_anomaly: ArrayLike,
    eccentricity: float,
    *,
    max_iterations: int = 64,
    tolerance: float = 1e-13,
    context: str = "orbit",
) -> NDArray[np.float64]:
    """Solve Kepler's equation with Newton steps and a bounded bracket fallback."""
    mean = np.asarray(mean_anomaly, dtype=np.float64)
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean_anomaly values must be finite")
    if not np.isfinite(eccentricity) or not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must be finite and in [0, 1)")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    wrapped = np.remainder(mean, 2.0 * np.pi)
    eccentric_anomaly = wrapped.copy() if eccentricity < 0.8 else np.full_like(wrapped, np.pi)

    for _ in range(max_iterations):
        residual = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - wrapped
        derivative = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        step = residual / derivative
        eccentric_anomaly -= step
        if np.all(np.abs(step) <= tolerance):
            return eccentric_anomaly.astype(np.float64, copy=False)

    # Reflect near 2*pi to avoid cancellation in high-eccentricity orbits.
    # Newton's step may oscillate above tolerance even at floating-point accuracy.
    result = eccentric_anomaly.copy()
    for index in np.flatnonzero(np.abs(step).ravel() > tolerance):
        angle = float(wrapped.flat[index])
        reflected = angle > np.pi
        target = 2.0 * np.pi - angle if reflected else angle
        lower, upper = 0.0, np.pi
        for _ in range(max_iterations):
            midpoint = (lower + upper) / 2.0
            residual = midpoint - eccentricity * np.sin(midpoint) - target
            if residual == 0.0 or (upper - lower) / 2.0 <= tolerance:
                result.flat[index] = 2.0 * np.pi - midpoint if reflected else midpoint
                break
            if residual > 0.0:
                upper = midpoint
            else:
                lower = midpoint
        else:
            raise SimulationError(
                f"{context}: Kepler solver failed to converge after {max_iterations} iterations"
            )
    return result.astype(np.float64, copy=False)


def radial_velocity(orbit: BinaryOrbit, delta_days: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the legacy primary-star radial velocity in km/s."""
    offsets = np.asarray(delta_days, dtype=np.float64)
    if not np.all(np.isfinite(offsets)):
        raise ValueError("delta_days values must be finite")

    mean_anomaly = orbit.mean_anomaly_rad + 2.0 * np.pi * offsets / orbit.period_days
    eccentric_anomaly = solve_kepler(
        mean_anomaly,
        orbit.eccentricity,
        context=f"TARGETID {orbit.primary.target_id}",
    )
    true_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 + orbit.eccentricity) * np.sin(eccentric_anomaly / 2.0),
        np.sqrt(1.0 - orbit.eccentricity) * np.cos(eccentric_anomaly / 2.0),
    )
    amplitude_m_s = (
        orbit.mass_ratio
        * np.sin(orbit.inclination_rad)
        / np.sqrt(1.0 - orbit.eccentricity**2)
        * (
            2.0
            * np.pi
            * G.value
            * orbit.primary.mass_kg
            / (orbit.period_seconds * (1.0 + orbit.mass_ratio) ** 2)
        )
        ** (1.0 / 3.0)
    )
    velocity_m_s = amplitude_m_s * (
        np.cos(true_anomaly + orbit.argument_periastron_rad)
        + orbit.eccentricity * np.cos(orbit.argument_periastron_rad)
    )
    return np.asarray(velocity_m_s / 1000.0, dtype=np.float64)
