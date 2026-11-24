from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from rvbinary.statistics import pairwise_beta, star_statistic


@dataclass(frozen=True)
class EpochDataStub:
    targetid: NDArray[np.int64] | NDArray[np.float64]
    vrad_kms: NDArray[np.float64]
    vrad_err_kms: NDArray[np.float64]


def test_pairwise_beta_groups_all_pairs_by_stably_sorted_targetid() -> None:
    data = EpochDataStub(
        targetid=np.array([20, 10, 20, 10, 10], dtype=np.int64),
        vrad_kms=np.array([14.0, 4.0, 10.0, 10.0, 7.0]),
        vrad_err_kms=np.array([2.0, 1.0, 2.0, 1.0, 1.0]),
    )

    pairs = pairwise_beta(data)

    assert len(pairs) == 2
    np.testing.assert_allclose(
        pairs[0], np.array([6.0, 3.0, 3.0], dtype=np.float64) / np.sqrt(2.0)
    )
    np.testing.assert_allclose(pairs[1], [4.0 / np.sqrt(8.0)])
    assert all(group.dtype == np.float64 for group in pairs)


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("targetid", np.array([[10, 10], [20, 20]], dtype=np.int64)),
        ("vrad_kms", np.array([[1.0, 2.0], [3.0, 4.0]])),
        ("vrad_err_kms", np.array([[1.0, 1.0], [1.0, 1.0]])),
    ],
)
def test_pairwise_beta_requires_one_dimensional_arrays(
    field: str, values: NDArray[np.generic]
) -> None:
    data = EpochDataStub(
        targetid=np.array([10, 10, 20, 20], dtype=np.int64),
        vrad_kms=np.array([1.0, 2.0, 3.0, 4.0]),
        vrad_err_kms=np.ones(4),
    )
    object.__setattr__(data, field, values)

    with pytest.raises(ValueError, match="one-dimensional"):
        pairwise_beta(data)


def test_pairwise_beta_requires_matching_array_shapes() -> None:
    data = EpochDataStub(
        targetid=np.array([10, 10], dtype=np.int64),
        vrad_kms=np.array([1.0]),
        vrad_err_kms=np.array([1.0, 1.0]),
    )

    with pytest.raises(ValueError, match="matching shapes"):
        pairwise_beta(data)


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("targetid", np.array([10.0, np.nan])),
        ("vrad_kms", np.array([1.0, np.inf])),
        ("vrad_err_kms", np.array([1.0, np.nan])),
    ],
)
def test_pairwise_beta_requires_finite_values(
    field: str, values: NDArray[np.float64]
) -> None:
    data = EpochDataStub(
        targetid=np.array([10, 10], dtype=np.int64),
        vrad_kms=np.array([1.0, 2.0]),
        vrad_err_kms=np.array([1.0, 1.0]),
    )
    object.__setattr__(data, field, values)

    with pytest.raises(ValueError, match=f"{field}.*finite"):
        pairwise_beta(data)


@pytest.mark.parametrize("uncertainty", [0.0, -0.1])
def test_pairwise_beta_requires_positive_uncertainties(uncertainty: float) -> None:
    data = EpochDataStub(
        targetid=np.array([10, 10], dtype=np.int64),
        vrad_kms=np.array([1.0, 2.0]),
        vrad_err_kms=np.array([1.0, uncertainty]),
    )

    with pytest.raises(ValueError, match="vrad_err_kms.*positive"):
        pairwise_beta(data)


def test_pairwise_beta_rejects_a_star_with_fewer_than_two_epochs() -> None:
    data = EpochDataStub(
        targetid=np.array([10, 10, 20], dtype=np.int64),
        vrad_kms=np.array([1.0, 2.0, 3.0]),
        vrad_err_kms=np.ones(3),
    )

    with pytest.raises(ValueError, match=r"TARGETID 20.*at least two epochs"):
        pairwise_beta(data)


def test_consistent_statistic_adds_one_after_rms() -> None:
    pairs = [np.array([1.0, 3.0])]

    assert star_statistic(pairs)[0] == pytest.approx(1 + np.sqrt(5))


def test_legacy_statistic_adds_one_to_each_pair_before_rms() -> None:
    pairs = [np.array([1.0, 3.0])]

    assert star_statistic(pairs, mode="legacy")[0] == pytest.approx(np.sqrt(10.0))


def test_star_statistic_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="unsupported statistic mode"):
        star_statistic([np.array([1.0])], mode="other")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "group",
    [
        np.array([], dtype=np.float64),
        np.array([[1.0]], dtype=np.float64),
        np.array([np.nan], dtype=np.float64),
    ],
)
def test_star_statistic_rejects_invalid_pair_groups(group: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="pair group 0"):
        star_statistic([group])
