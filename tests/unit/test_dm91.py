from __future__ import annotations

import numpy as np
import pytest

from rvbinary.mock.orbit import rng_for_sample
from rvbinary.models.base import BinaryOrbit, PrimaryStar, SimulationError
from rvbinary.models.dm91 import DM91Model

PRIMARY = PrimaryStar(target_id=314159, mass_solar=0.8, logg_cgs=4.2)


def roche_lobe_at_periastron(orbit: BinaryOrbit) -> float:
    mass_ratio = orbit.mass_ratio
    inverse_q = 1.0 / mass_ratio
    fraction = 0.49 * inverse_q ** (2.0 / 3.0) / (
        0.6 * inverse_q ** (2.0 / 3.0) + np.log(1.0 + inverse_q ** (1.0 / 3.0))
    )
    return fraction * orbit.semi_major_axis_m * (1.0 - orbit.eccentricity)


def test_sample_stream_depends_only_on_seed_and_index() -> None:
    first = rng_for_sample(42, 17).normal(size=8)
    repeated = rng_for_sample(42, 17).normal(size=8)
    neighboring = rng_for_sample(42, 18).normal(size=8)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, neighboring)


def test_sample_stream_has_fixed_seed_vector() -> None:
    values = rng_for_sample(42, 17).normal(size=8)

    np.testing.assert_array_equal(
        values,
        [
            0.47391659609722864,
            0.09953683103870832,
            0.1894574374622115,
            -1.7294146753143156,
            -1.0741446442623357,
            -1.491955422175234,
            1.52276691863938,
            0.4063081918170648,
        ],
    )


def test_dm91_parameters_and_period_override() -> None:
    model = DM91Model(log10_period_mean=3.2, log10_period_sigma=0.1)

    orbit = model.sample_star(PRIMARY, np.random.default_rng(9))

    assert 0.1 <= orbit.mass_ratio <= 1.0
    assert 0.0 <= orbit.eccentricity < 1.0
    assert orbit.period_days > 0.0
    assert 0.0 <= orbit.inclination_rad <= np.pi / 2.0
    assert 0.0 <= orbit.argument_periastron_rad < 2.0 * np.pi
    assert 0.0 <= orbit.mean_anomaly_rad < 2.0 * np.pi
    assert np.log10(orbit.period_days) == pytest.approx(3.0343654573)


def test_sampled_orbit_keeps_primary_inside_periastron() -> None:
    orbit = DM91Model().sample_star(PRIMARY, np.random.default_rng(71))

    assert orbit.primary_radius_m < orbit.semi_major_axis_m
    assert orbit.periastron_distance_m >= orbit.primary_radius_m


def test_dm91_fixed_seed_orbit_vector() -> None:
    orbit = DM91Model().sample_star(PRIMARY, rng_for_sample(8675309, 23))

    np.testing.assert_allclose(
        [
            orbit.mass_ratio,
            orbit.period_days,
            orbit.eccentricity,
            orbit.inclination_rad,
            orbit.argument_periastron_rad,
            orbit.mean_anomaly_rad,
        ],
        [
            0.27925426301366102,
            3663345.6982816313,
            0.68387747085429207,
            1.3337191521035585,
            3.6959774211277168,
            5.5710230313840379,
        ],
        rtol=1e-14,
        atol=0.0,
    )


@pytest.mark.parametrize("seed", range(64))
def test_dm91_many_seed_physical_domains(seed: int) -> None:
    orbit = DM91Model().sample_star(PRIMARY, np.random.default_rng(seed))

    assert 0.1 <= orbit.mass_ratio <= 1.0
    assert 0.0 <= orbit.eccentricity < 1.0
    assert 0.0 < orbit.period_days < 1e8
    assert orbit.periastron_distance_m >= orbit.primary_radius_m
    assert orbit.primary_radius_m <= roche_lobe_at_periastron(orbit)
    assert 0.0 <= orbit.inclination_rad <= np.pi / 2.0


def test_seed_58_preserves_the_roche_accepted_eccentricity() -> None:
    orbit = DM91Model().sample_star(PRIMARY, np.random.default_rng(58))

    assert orbit.eccentricity == pytest.approx(0.14685470621132019)
    assert orbit.primary_radius_m <= roche_lobe_at_periastron(orbit)


def test_sampling_failure_names_the_star_and_rejection_stage() -> None:
    primary = PrimaryStar(target_id=2718, mass_solar=1.0, logg_cgs=15.0)
    model = DM91Model(log10_period_mean=-10.0, log10_period_sigma=0.01, max_attempts=3)

    with pytest.raises(
        SimulationError,
        match=r"sample 12.*TARGETID 2718.*period.*3 attempts",
    ):
        model.sample_star(primary, np.random.default_rng(4), sample_index=12)
