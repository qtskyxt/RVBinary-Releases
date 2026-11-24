from __future__ import annotations

import numpy as np
import pytest

from rvbinary.mock.orbit import radial_velocity, solve_kepler
from rvbinary.models.base import BinaryOrbit, PrimaryStar, SimulationError

PRIMARY = PrimaryStar(target_id=99, mass_solar=1.0, logg_cgs=4.0)


def test_kepler_solver_satisfies_equation_for_physical_domain() -> None:
    mean_anomaly = np.array([0.0, 0.3, np.pi, 5.8, 20.0 * np.pi + 0.2])

    eccentric_anomaly = solve_kepler(mean_anomaly, 0.93, context="TARGETID 99")

    wrapped_mean = np.remainder(mean_anomaly, 2.0 * np.pi)
    np.testing.assert_allclose(
        eccentric_anomaly - 0.93 * np.sin(eccentric_anomaly),
        wrapped_mean,
        atol=1e-12,
        rtol=0.0,
    )


def test_kepler_solver_failure_is_bounded_and_contextual() -> None:
    with pytest.raises(SimulationError, match=r"TARGETID 99.*Kepler.*1 iterations"):
        solve_kepler(np.array([1.0]), 0.999, max_iterations=1, context="TARGETID 99")


@pytest.mark.parametrize("eccentricity", [0.999, 0.9999, 0.99999])
def test_kepler_solver_handles_roundoff_near_periastron(eccentricity):
    mean = np.array([6.28316320591006, 6.283184898078539, 6.28318512678944])
    result = solve_kepler(mean, eccentricity)
    np.testing.assert_allclose(
        result - eccentricity * np.sin(result), mean, atol=2e-15, rtol=0,
    )
    assert np.all((result >= 0) & (result <= 2 * np.pi))


def test_circular_edge_on_radial_velocity_has_legacy_phase_and_amplitude() -> None:
    orbit = BinaryOrbit(
        primary=PRIMARY,
        mass_ratio=1.0,
        period_days=10.0,
        eccentricity=0.0,
        inclination_rad=np.pi / 2.0,
        argument_periastron_rad=0.0,
        mean_anomaly_rad=0.0,
    )

    velocities = radial_velocity(orbit, np.array([0.0, 2.5, 5.0, 7.5, 10.0]))

    np.testing.assert_allclose(
        velocities,
        [62.2547637732, 0.0, -62.2547637732, 0.0, 62.2547637732],
        atol=1e-9,
    )
    assert velocities.dtype == np.float64


def test_fixed_seed_orbit_has_fixed_radial_velocity_vector() -> None:
    orbit = BinaryOrbit(
        primary=PrimaryStar(target_id=314159, mass_solar=0.8, logg_cgs=4.2),
        mass_ratio=0.27925426301366102,
        period_days=3663345.6982816313,
        eccentricity=0.68387747085429207,
        inclination_rad=1.3337191521035585,
        argument_periastron_rad=3.6959774211277168,
        mean_anomaly_rad=5.5710230313840379,
    )

    velocities = radial_velocity(orbit, np.array([0.0, 1.25, 20.0, 365.25]))

    np.testing.assert_allclose(
        velocities,
        [
            -0.21390222656806346,
            -0.21390305598327106,
            -0.21391549759845052,
            -0.21414471899786588,
        ],
        rtol=1e-14,
        atol=0.0,
    )


def test_radial_velocity_rejects_nonfinite_epoch_offsets() -> None:
    orbit = BinaryOrbit(
        primary=PRIMARY,
        mass_ratio=0.5,
        period_days=100.0,
        eccentricity=0.2,
        inclination_rad=0.7,
        argument_periastron_rad=1.2,
        mean_anomaly_rad=2.1,
    )

    with pytest.raises(ValueError, match="delta_days.*finite"):
        radial_velocity(orbit, np.array([0.0, np.nan]))
