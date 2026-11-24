"""Figures for sampled populations and individual primary-star RV curves."""

from __future__ import annotations

import csv
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from rvbinary.mock.generator import OrbitBatch
from rvbinary.mock.orbit import radial_velocity
from rvbinary.models.base import BinaryOrbit


def distribution_figure(batch: OrbitBatch) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    values = (
        [np.log10(o.period_days) for o in batch.orbits],
        [o.mass_ratio for o in batch.orbits],
        [o.eccentricity for o in batch.orbits],
        [np.degrees(o.inclination_rad) for o in batch.orbits],
    )
    labels = ("log10(P / day)", "Mass ratio q", "Eccentricity e", "Inclination (deg)")
    for axis, data, label in zip(axes.flat, values, labels, strict=True):
        axis.hist(data, bins=40, color="#408a9c", edgecolor="white", linewidth=0.4)
        axis.set(xlabel=label, ylabel="Orbit count")
    return figure


def period_eccentricity_figure(batch: OrbitBatch) -> Figure:
    indexes = np.linspace(0, len(batch.orbits) - 1, min(len(batch.orbits), 2000), dtype=int)
    selected = [batch.orbits[index] for index in indexes]
    figure, axis = plt.subplots(figsize=(9, 3.5), layout="constrained")
    axis.scatter(
        [np.log10(o.period_days) for o in selected],
        [o.eccentricity for o in selected],
        s=10,
        alpha=0.5,
        color="#408a9c",
        rasterized=True,
    )
    axis.set(xlabel="log10(P / day)", ylabel="Eccentricity e")
    return figure


def rv_series(
    orbit: BinaryOrbit,
    span_days: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    span = 2.0 * orbit.period_days if span_days is None else span_days
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Time span must be finite and positive")
    times = np.linspace(0, span, 512)
    return times, radial_velocity(orbit, times)


def rv_observations(
    orbit: BinaryOrbit,
    span_days: float | None = None,
    count: int = 50,
    sigma_km_s: float = 1.0,
    seed: int = 20261010,
) -> tuple[NDArray[np.float64], ...]:
    """Independent, reproducible Gaussian measurements on a uniform time grid."""
    span = 2.0 * orbit.period_days if span_days is None else span_days
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Time span must be finite and positive")
    if type(count) is not int or not 2 <= count <= 10000:
        raise ValueError("Observation count must be an integer in [2, 10000]")
    if not np.isfinite(sigma_km_s) or sigma_km_s < 0:
        raise ValueError("Noise sigma must be finite and nonnegative")
    if type(seed) is not int or not 0 <= seed <= 4294967295:
        raise ValueError("Noise seed must be an integer in [0, 4294967295]")
    times = np.linspace(0, span, count)
    truth = radial_velocity(orbit, times)
    noisy = truth + np.random.default_rng(seed).normal(0, sigma_km_s, count)
    return times, truth, noisy, np.full(count, sigma_km_s)


def observations_to_csv(times, truth, noisy, errors) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["time_days", "true_rv_km_s", "observed_rv_km_s", "sigma_km_s"])
    writer.writerows(zip(times, truth, noisy, errors, strict=True))
    return stream.getvalue()


def rv_figure(
    orbit: BinaryOrbit,
    span_days: float | None = None,
    *,
    observations: tuple[NDArray[np.float64], ...] | None = None,
) -> Figure:
    times, velocities = rv_series(orbit, span_days)
    figure, axis = plt.subplots(figsize=(9, 3.5), layout="constrained")
    axis.plot(times, velocities, color="#d28a42", linewidth=1.5, label="True RV")
    if observations is not None:
        observed_times, _, noisy, errors = observations
        axis.errorbar(
            observed_times,
            noisy,
            yerr=errors,
            fmt="o",
            markersize=3,
            color="#408a9c",
            alpha=0.75,
            capsize=2,
            label="Noisy RV (1σ)",
        )
        axis.legend()
    axis.set(xlabel="Time since initial epoch (day)", ylabel="Primary RV (km/s)")
    return figure
