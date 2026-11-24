"""Binary/non-binary mixture likelihood on persisted PDF edges."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rvbinary.pdf.histogram import PDFModel


def _validated_pdf(
    pdf: PDFModel,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    edges = np.asarray(pdf.edges, dtype=np.float64)
    binary = np.asarray(pdf.binary_density, dtype=np.float64)
    nonbinary = np.asarray(pdf.nonbinary_density, dtype=np.float64)
    expected = (edges.size - 1,)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("PDF edges must be finite and strictly increasing")
    if binary.shape != expected or nonbinary.shape != expected:
        raise ValueError("PDF densities must match the persisted edges")
    if (
        not np.all(np.isfinite(binary))
        or not np.all(np.isfinite(nonbinary))
        or np.any(binary < 0.0)
        or np.any(nonbinary < 0.0)
    ):
        raise ValueError("PDF densities must be finite and non-negative")
    widths = np.diff(edges)
    if not np.isclose(np.sum(binary * widths), 1.0) or not np.isclose(
        np.sum(nonbinary * widths), 1.0
    ):
        raise ValueError("PDF densities must integrate to one")
    return edges, binary, nonbinary


def _density_indices(
    observed: NDArray[np.float64], edges: NDArray[np.float64]
) -> NDArray[np.int64]:
    values = np.asarray(observed, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("observed statistics must be a non-empty finite vector")
    outside = (values < edges[0]) | (values > edges[-1])
    if np.any(outside):
        bad = values[outside]
        raise ValueError(
            f"observed statistic outside PDF support [{edges[0]}, {edges[-1]}]: "
            f"{bad[0]}"
        )
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = edges.size - 2
    return indices.astype(np.int64, copy=False)


def observed_densities(
    observed: NDArray[np.float64], pdf: PDFModel
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resolve observed statistics to fixed binary and non-binary densities."""
    edges, binary, nonbinary = _validated_pdf(pdf)
    indices = _density_indices(observed, edges)
    return binary[indices], nonbinary[indices]


def mixture_log_likelihood(
    fraction: float,
    binary_values: NDArray[np.float64],
    nonbinary_values: NDArray[np.float64],
) -> float:
    """Evaluate a mixture after PDF validation and bin lookup are complete."""
    if fraction == 0.0:
        mixture = nonbinary_values
    elif fraction == 1.0:
        mixture = binary_values
    else:
        with np.errstate(divide="ignore"):
            log_mixture = np.logaddexp(
                np.log1p(-fraction) + np.log(nonbinary_values),
                np.log(fraction) + np.log(binary_values),
            )
        return float(np.sum(log_mixture))
    if np.any(mixture == 0.0):
        return -np.inf
    return float(np.sum(np.log(mixture)))


def log_likelihood(
    f_binary: float, observed: NDArray[np.float64], pdf: PDFModel
) -> float:
    """Evaluate a stable binary-fraction mixture log likelihood."""
    fraction = float(f_binary)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("binary fraction must be finite and within [0, 1]")
    binary_values, nonbinary_values = observed_densities(observed, pdf)
    return mixture_log_likelihood(fraction, binary_values, nonbinary_values)
