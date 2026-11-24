from __future__ import annotations

import numpy as np

from rvbinary.io.observations import EpochData
from rvbinary.mock.simulation import MockSample, derive_beta_realization, simulate_sample
from rvbinary.models.dm91 import DM91Model


def epoch_data(targetids: list[int] | None = None) -> EpochData:
    ids = targetids or [11, 11, 22, 22, 22]
    size = len(ids)
    values = np.asarray(ids, dtype=np.int64)
    masses = np.where(values == 11, 0.9, 1.1).astype(np.float64)
    logg = np.where(values == 11, 4.4, 4.2).astype(np.float64)
    return EpochData(
        targetid=values,
        mjd=np.arange(size, dtype=np.float64) + 59000.0,
        vrad_kms=np.zeros(size, dtype=np.float64),
        vrad_err_kms=np.linspace(0.5, 0.9, size, dtype=np.float64),
        mass_solar=masses,
        logg_cgs=logg,
    )


def test_simulate_sample_stores_required_epoch_columns_and_offsets() -> None:
    sample = simulate_sample(epoch_data(), DM91Model(), seed=7, sample_index=3)

    assert set(sample.arrays) == {
        "TARGETID",
        "VRAD",
        "VRAD_ERR",
        "PERIOD",
        "q",
        "ECC",
        "THETA",
        "INCL",
        "OMEGA",
        "DELTAT",
        "STAR_OFFSET",
    }
    assert sample.arrays["TARGETID"].dtype == np.int64
    assert sample.arrays["STAR_OFFSET"].tolist() == [0, 1, 0, 1, 2]
    assert sample.arrays["DELTAT"].tolist() == [0.0, 1.0, 0.0, 1.0, 2.0]
    assert all(not array.flags.writeable for array in sample.arrays.values())


def test_star_streams_depend_on_exact_targetid_not_star_position() -> None:
    lower = 2**53
    upper = lower + 1
    ordered = epoch_data([lower, lower, upper, upper])
    reordered = epoch_data([upper, upper, lower, lower])

    first = simulate_sample(ordered, DM91Model(), seed=19, sample_index=5)
    second = simulate_sample(reordered, DM91Model(), seed=19, sample_index=5)

    for targetid in (lower, upper):
        first_rows = first.arrays["TARGETID"] == targetid
        second_rows = second.arrays["TARGETID"] == targetid
        for name in ("PERIOD", "q", "ECC", "INCL", "OMEGA"):
            np.testing.assert_array_equal(
                first.arrays[name][first_rows], second.arrays[name][second_rows]
            )
    assert first.arrays["PERIOD"][0] != first.arrays["PERIOD"][2]


def test_period_limit_does_not_regenerate_or_mutate_orbits() -> None:
    sample = simulate_sample(epoch_data(), DM91Model(), seed=7, sample_index=3)

    low = sample.with_period_limit(10.0)
    high = sample.with_period_limit(1000.0)

    assert low.orbits_digest == high.orbits_digest == sample.orbits_digest
    assert low.sample is sample
    assert high.sample is sample


def test_error_floor_stream_uses_exact_float_bits_and_grid_order_is_irrelevant() -> None:
    sample = simulate_sample(epoch_data(), DM91Model(), seed=31, sample_index=2)

    positive_zero = derive_beta_realization(sample, 0.0, seed=31, statistic_mode="consistent")
    other = derive_beta_realization(sample, 0.4, seed=31, statistic_mode="consistent")
    negative_zero = derive_beta_realization(sample, -0.0, seed=31, statistic_mode="consistent")
    repeated = derive_beta_realization(sample, 0.0, seed=31, statistic_mode="consistent")

    assert all(
        np.array_equal(a, b)
        for a, b in zip(
            positive_zero.binary_by_star, repeated.binary_by_star, strict=True
        )
    )
    assert all(
        np.array_equal(a, b)
        for a, b in zip(
            positive_zero.nonbinary_by_star, repeated.nonbinary_by_star, strict=True
        )
    )
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(positive_zero.binary_by_star, other.binary_by_star, strict=True)
    )
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            positive_zero.binary_by_star, negative_zero.binary_by_star, strict=True
        )
    )


def test_shared_epoch_noise_is_reused_across_every_pair() -> None:
    zeros = np.zeros(3, dtype=np.float64)
    sample = MockSample(
        sample_index=4,
        arrays={
            "TARGETID": np.array([99, 99, 99], dtype=np.int64),
            "VRAD": zeros,
            "VRAD_ERR": np.ones(3, dtype=np.float64),
            "PERIOD": np.ones(3, dtype=np.float64),
            "q": zeros,
            "ECC": zeros,
            "THETA": zeros,
            "INCL": zeros,
            "OMEGA": zeros,
            "DELTAT": np.arange(3, dtype=np.float64),
            "STAR_OFFSET": np.arange(3, dtype=np.int64),
        },
    )

    beta = derive_beta_realization(sample, 0.0, seed=123, statistic_mode="consistent")
    distances = np.sort(beta.binary_by_star[0])

    # Three shared noisy epochs lie on a line: the largest distance is the
    # sum of the other two, for both binary and nonbinary realizations.
    assert np.isclose(distances[2], distances[0] + distances[1])
    nonbinary_distances = np.sort(beta.nonbinary_by_star[0])
    assert np.isclose(nonbinary_distances[2], nonbinary_distances[0] + nonbinary_distances[1])
    assert not np.array_equal(beta.binary_by_star[0], beta.nonbinary_by_star[0])


def test_statistic_mode_changes_only_reduction_and_is_explicit_metadata() -> None:
    sample = simulate_sample(epoch_data(), DM91Model(), seed=41, sample_index=8)

    consistent = derive_beta_realization(sample, 0.2, seed=41, statistic_mode="consistent")
    legacy = derive_beta_realization(sample, 0.2, seed=41, statistic_mode="legacy")

    assert consistent.statistic_mode == "consistent"
    assert legacy.statistic_mode == "legacy"
    assert all(
        np.array_equal(a, b)
        for a, b in zip(consistent.binary_by_star, legacy.binary_by_star, strict=True)
    )
    assert not np.array_equal(consistent.binary_statistics, legacy.binary_statistics)


def test_pair_statistics_match_one_noisy_velocity_per_epoch_with_unequal_errors() -> None:
    from rvbinary.mock.simulation import _pair_beta

    velocity = np.array([0.0, 12.0, -4.0, 3.0])
    errors = np.array([0.4, 0.9, 1.2, 0.6])
    floor = 0.7
    noisy = velocity + np.random.default_rng(42).normal(0.0, np.hypot(errors, floor))
    first, second = np.triu_indices(4, k=1)
    expected = np.abs(noisy[first] - noisy[second]) / np.hypot(errors[first], errors[second])

    actual = _pair_beta(velocity, errors, floor, np.random.default_rng(42))

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
