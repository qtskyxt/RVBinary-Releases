"""Deterministic quadrature and discrete error-floor posterior utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from rvbinary.inference.likelihood import mixture_log_likelihood, observed_densities
from rvbinary.pdf.histogram import PDFModel

Prior = str | NDArray[np.float64] | Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _logsumexp(values: NDArray[np.float64]) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return -np.inf
    maximum = float(np.max(values[finite]))
    return maximum + float(np.log(np.sum(np.exp(values[finite] - maximum))))


def _prior_density(prior: Prior, grid: NDArray[np.float64]) -> NDArray[np.float64]:
    if isinstance(prior, str):
        if prior != "uniform":
            raise ValueError(f"unsupported binary-fraction prior: {prior}")
        density = np.ones_like(grid)
    elif callable(prior):
        density = np.asarray(prior(grid), dtype=np.float64)
    else:
        density = np.asarray(prior, dtype=np.float64)
    if (
        density.shape != grid.shape
        or not np.all(np.isfinite(density))
        or np.any(density < 0.0)
    ):
        raise ValueError("prior density must be finite, non-negative, and match the grid")
    step = grid[1] - grid[0]
    normalization = step * (
        0.5 * density[0] + np.sum(density[1:-1]) + 0.5 * density[-1]
    )
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("prior density must have positive finite mass")
    return density / normalization


def marginal_log_evidence(
    observed: NDArray[np.float64],
    pdf: PDFModel,
    prior: Prior = "uniform",
    quadrature_points: int = 2049,
) -> float:
    """Integrate the fixed-model likelihood over binary fraction."""
    if quadrature_points < 3:
        raise ValueError("quadrature_points must be at least 3")
    grid = np.linspace(0.0, 1.0, quadrature_points)
    density = _prior_density(prior, grid)
    binary_values, nonbinary_values = observed_densities(observed, pdf)
    log_values = np.array(
        [
            mixture_log_likelihood(fraction, binary_values, nonbinary_values)
            for fraction in grid
        ],
        dtype=np.float64,
    )
    weights = np.full(quadrature_points, 1.0 / (quadrature_points - 1))
    weights[[0, -1]] *= 0.5
    with np.errstate(divide="ignore"):
        terms = log_values + np.log(density) + np.log(weights)
    return _logsumexp(terms)


def normalize_discrete_posterior(
    log_evidence: NDArray[np.float64], prior_weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Normalize evidence times prior weights over a discrete model grid."""
    evidence = np.asarray(log_evidence, dtype=np.float64)
    weights = np.asarray(prior_weights, dtype=np.float64)
    if evidence.ndim != 1 or evidence.size == 0 or weights.shape != evidence.shape:
        raise ValueError("log evidence and prior weights must be matching non-empty vectors")
    if np.any(np.isnan(evidence)) or np.any(np.isposinf(evidence)):
        raise ValueError("log evidence must not contain NaN or positive infinity")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("prior weights must be finite, non-negative, and have positive mass")
    with np.errstate(divide="ignore"):
        log_weights = evidence + np.log(weights)
    normalization = _logsumexp(log_weights)
    if not np.isfinite(normalization):
        raise ValueError("discrete posterior has zero total evidence")
    return np.exp(log_weights - normalization)


def discrete_credible_interval(
    grid: NDArray[np.float64], probability: NDArray[np.float64], mass: float
) -> tuple[float, float]:
    """Return an equal-tailed interval on a sorted discrete model grid."""
    values = np.asarray(grid, dtype=np.float64)
    weights = np.asarray(probability, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or weights.shape != values.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or weights.sum() <= 0.0
    ):
        raise ValueError("grid and probability must be valid matching vectors")
    if not np.isfinite(mass) or mass <= 0.0 or mass > 1.0:
        raise ValueError("credible mass must be within (0, 1]")
    order = np.argsort(values)
    sorted_values = values[order]
    if np.any(np.diff(sorted_values) <= 0.0):
        raise ValueError("credible-interval grid values must be unique")
    normalized = weights[order] / weights.sum()
    cumulative = np.cumsum(normalized)
    tail = (1.0 - mass) / 2.0
    lower = min(int(np.searchsorted(cumulative, tail, side="left")), values.size - 1)
    upper = min(
        int(np.searchsorted(cumulative, 1.0 - tail, side="left")), values.size - 1
    )
    return float(sorted_values[lower]), float(sorted_values[upper])
