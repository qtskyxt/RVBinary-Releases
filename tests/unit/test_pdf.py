from __future__ import annotations

import numpy as np
import pytest

from rvbinary.pdf.histogram import BetaSample, apply_period_limit, build_pdf


def test_period_limit_substitutes_nonbinary_values_by_star() -> None:
    binary = BetaSample((np.array([2.0, 4.0]), np.array([8.0])), "consistent")
    nonbinary = BetaSample((np.array([1.0, 1.5]), np.array([0.5])), "consistent")

    limited = apply_period_limit(
        binary, nonbinary, np.array([5.0, 500.0]), limit_days=100.0
    )

    np.testing.assert_array_equal(limited.by_star[0], binary.by_star[0])
    np.testing.assert_array_equal(limited.by_star[1], nonbinary.by_star[1])


def test_period_limit_rejects_mismatched_star_counts() -> None:
    binary = BetaSample((np.array([2.0]),), "consistent")
    nonbinary = BetaSample((np.array([1.0]), np.array([1.5])), "consistent")

    with pytest.raises(ValueError, match="same stars"):
        apply_period_limit(binary, nonbinary, np.array([5.0]), limit_days=100.0)


def test_pdf_density_integrates_to_one() -> None:
    edges = np.array([1.0, 2.0, 4.0, 8.0])

    pdf = build_pdf(
        np.array([1.2, 1.8, 3.0, 7.0]),
        np.array([1.1, 1.4, 2.5, 5.0]),
        edges,
    )

    assert np.sum(pdf.binary_density * np.diff(pdf.edges)) == pytest.approx(1.0)
    assert np.sum(pdf.nonbinary_density * np.diff(pdf.edges)) == pytest.approx(1.0)
    np.testing.assert_array_equal(pdf.binary_counts, np.array([2, 1, 1]))
    np.testing.assert_array_equal(pdf.nonbinary_counts, np.array([2, 1, 1]))


def test_pdf_rejects_values_outside_explicit_edges() -> None:
    with pytest.raises(ValueError, match="outside histogram edges"):
        build_pdf(np.array([0.5]), np.array([1.5]), np.array([1.0, 2.0]))
