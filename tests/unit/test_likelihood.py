from __future__ import annotations

import numpy as np
import pytest

from rvbinary.inference.likelihood import log_likelihood
from rvbinary.pdf.histogram import PDFModel


@pytest.fixture
def pdf() -> PDFModel:
    return PDFModel(
        edges=np.array([1.0, 2.0, 4.0]),
        binary_density=np.array([0.25, 0.375]),
        nonbinary_density=np.array([0.5, 0.25]),
        binary_counts=np.array([1, 3]),
        nonbinary_counts=np.array([2, 2]),
    )


def test_mixture_likelihood_endpoints(pdf: PDFModel) -> None:
    observed = np.array([1.5, 3.0])

    assert log_likelihood(0.0, observed, pdf) == pytest.approx(
        np.log(0.5) + np.log(0.25)
    )
    assert log_likelihood(1.0, observed, pdf) == pytest.approx(
        np.log(0.25) + np.log(0.375)
    )


def test_mixture_likelihood_uses_stable_weighted_density(pdf: PDFModel) -> None:
    observed = np.array([1.5, 3.0])

    expected = np.log(0.75 * 0.5 + 0.25 * 0.25) + np.log(
        0.75 * 0.25 + 0.25 * 0.375
    )

    assert log_likelihood(0.25, observed, pdf) == pytest.approx(expected)


def test_likelihood_rejects_observations_outside_pdf_support(pdf: PDFModel) -> None:
    with pytest.raises(ValueError, match="outside PDF support"):
        log_likelihood(0.5, np.array([0.9, 2.0]), pdf)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan])
def test_likelihood_rejects_invalid_binary_fraction(
    pdf: PDFModel, fraction: float
) -> None:
    with pytest.raises(ValueError, match="binary fraction"):
        log_likelihood(fraction, np.array([1.5]), pdf)
