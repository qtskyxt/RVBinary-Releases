"""Period selection and normalized histogram primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rvbinary.statistics import StatisticMode, star_statistic


@dataclass(frozen=True)
class BetaSample:
    """Pair-beta arrays grouped in a stable star order."""

    by_star: tuple[NDArray[np.float64], ...]
    statistic_mode: StatisticMode

    def __post_init__(self) -> None:
        groups = tuple(np.asarray(values, dtype=np.float64) for values in self.by_star)
        if not groups:
            raise ValueError("a beta sample must contain at least one star")
        star_statistic(list(groups), mode=self.statistic_mode)
        object.__setattr__(self, "by_star", groups)

    @property
    def statistics(self) -> NDArray[np.float64]:
        """Reduce pair values with the sample's declared statistic convention."""
        return star_statistic(list(self.by_star), mode=self.statistic_mode)


@dataclass(frozen=True)
class PDFModel:
    """Normalized binary and non-binary histogram densities."""

    edges: NDArray[np.float64]
    binary_density: NDArray[np.float64]
    nonbinary_density: NDArray[np.float64]
    binary_counts: NDArray[np.int64]
    nonbinary_counts: NDArray[np.int64]


def apply_period_limit(
    binary: BetaSample,
    nonbinary: BetaSample,
    periods: NDArray[np.float64],
    limit_days: float,
) -> BetaSample:
    """Substitute non-binary pair values for stars above a period limit."""
    values = np.asarray(periods, dtype=np.float64)
    if binary.statistic_mode != nonbinary.statistic_mode:
        raise ValueError("binary and nonbinary statistic modes must match")
    if len(binary.by_star) != len(nonbinary.by_star) or values.shape != (
        len(binary.by_star),
    ):
        raise ValueError("binary, nonbinary, and periods must describe the same stars")
    if not np.isfinite(limit_days) or limit_days <= 0.0:
        raise ValueError("period limit must be finite and positive")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("periods must be finite and positive")
    selected = tuple(
        nonbinary.by_star[index] if period > limit_days else binary.by_star[index]
        for index, period in enumerate(values)
    )
    return BetaSample(selected, binary.statistic_mode)


def beta_edges(bins: int) -> NDArray[np.float64]:
    """Return deterministic hybrid linear/log edges over the legacy 1--100 range."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    if bins == 1:
        return np.array([1.0, 100.0], dtype=np.float64)
    linear_bins = max(1, min(bins - 1, round(bins * 0.1)))
    linear = np.linspace(1.0, 2.0, linear_bins + 1)
    logarithmic = np.geomspace(2.0, 100.0, bins - linear_bins + 1)[1:]
    return np.concatenate((linear, logarithmic)).astype(np.float64, copy=False)


def histogram_counts(
    values: NDArray[np.float64], edges: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Count finite values without silently discarding underflow or overflow."""
    samples = np.asarray(values, dtype=np.float64)
    boundaries = np.asarray(edges, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("histogram values must be a non-empty finite vector")
    if (
        boundaries.ndim != 1
        or boundaries.size < 2
        or not np.all(np.isfinite(boundaries))
        or np.any(np.diff(boundaries) <= 0.0)
    ):
        raise ValueError("histogram edges must be finite and strictly increasing")
    if np.any(samples < boundaries[0]) or np.any(samples > boundaries[-1]):
        raise ValueError(
            f"values outside histogram edges [{boundaries[0]}, {boundaries[-1]}]"
        )
    return np.histogram(samples, bins=boundaries)[0].astype(np.int64, copy=False)


def pdf_from_counts(
    binary_counts: NDArray[np.int64],
    nonbinary_counts: NDArray[np.int64],
    edges: NDArray[np.float64],
) -> PDFModel:
    """Normalize fixed-size count accumulators into probability densities."""
    boundaries = np.asarray(edges, dtype=np.float64)
    binary = np.asarray(binary_counts, dtype=np.int64)
    nonbinary = np.asarray(nonbinary_counts, dtype=np.int64)
    expected = (boundaries.size - 1,)
    if binary.shape != expected or nonbinary.shape != expected:
        raise ValueError("histogram counts must match the edge count")
    if np.any(binary < 0) or np.any(nonbinary < 0):
        raise ValueError("histogram counts must be non-negative")
    if binary.sum() == 0 or nonbinary.sum() == 0:
        raise ValueError("both histogram products must contain samples")
    widths = np.diff(boundaries)
    return PDFModel(
        edges=boundaries,
        binary_density=binary / (binary.sum() * widths),
        nonbinary_density=nonbinary / (nonbinary.sum() * widths),
        binary_counts=binary,
        nonbinary_counts=nonbinary,
    )


def build_pdf(
    binary_stats: NDArray[np.float64],
    nonbinary_stats: NDArray[np.float64],
    bins: int | NDArray[np.float64],
) -> PDFModel:
    """Build normalized histogram PDFs from star-level statistics."""
    binary = np.asarray(binary_stats, dtype=np.float64)
    nonbinary = np.asarray(nonbinary_stats, dtype=np.float64)
    if isinstance(bins, int):
        combined = np.concatenate((binary, nonbinary))
        if not np.all(np.isfinite(combined)) or combined.size == 0:
            raise ValueError("PDF statistics must be finite and non-empty")
        lower = float(np.min(combined))
        upper = float(np.max(combined))
        if lower == upper:
            upper = np.nextafter(upper, np.inf)
        edges = np.linspace(lower, upper, bins + 1)
    else:
        edges = np.asarray(bins, dtype=np.float64)
    return pdf_from_counts(
        histogram_counts(binary, edges),
        histogram_counts(nonbinary, edges),
        edges,
    )
