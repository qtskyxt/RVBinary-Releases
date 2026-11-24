import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from rvbinary.gui.plots import (
    distribution_figure,
    period_eccentricity_figure,
    rv_series,
)
from rvbinary.mock.generator import GeneratorConfig, generate_orbits
from rvbinary.mock.orbit import radial_velocity


@pytest.mark.parametrize("count", [1, 10000])
def test_plot_limits(count):
    batch = generate_orbits(GeneratorConfig(count=count))
    histogram = distribution_figure(batch)
    assert len(histogram.axes) == 4
    assert all(len(ax.patches) == 40 for ax in histogram.axes)
    scatter = period_eccentricity_figure(batch)
    assert len(scatter.axes[0].collections[0].get_offsets()) == min(count, 2000)
    assert len(batch.orbits) == count
    plt.close(histogram)
    plt.close(scatter)


def test_rv_series_contract():
    orbit = generate_orbits(GeneratorConfig(count=1)).orbits[0]
    times, velocities = rv_series(orbit)
    assert len(times) == 512
    assert times[0] == 0
    assert times[-1] == 2 * orbit.period_days
    np.testing.assert_array_equal(velocities, radial_velocity(orbit, times))
    assert rv_series(orbit, 7.0)[0][-1] == 7.0
    for span in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            rv_series(orbit, span)


def test_noisy_observations_and_export():
    import csv
    import io

    from rvbinary.gui import plots

    orbit = generate_orbits(GeneratorConfig(count=1)).orbits[0]
    times, truth, noisy, errors = plots.rv_observations(orbit, 7.0, 50, 1.5, 42)
    np.testing.assert_allclose(times, np.linspace(0, 7, 50))
    np.testing.assert_array_equal(truth, radial_velocity(orbit, times))
    np.testing.assert_array_equal(noisy, truth + np.random.default_rng(42).normal(0, 1.5, 50))
    np.testing.assert_array_equal(errors, np.full(50, 1.5))
    assert not np.array_equal(noisy, plots.rv_observations(orbit, 7, 50, 1.5, 43)[2])
    np.testing.assert_array_equal(plots.rv_observations(orbit, 7, 50, 0, 42)[2], truth)
    rows = list(csv.DictReader(io.StringIO(plots.observations_to_csv(times, truth, noisy, errors))))
    assert len(rows) == 50
    assert float(rows[0]["observed_rv_km_s"]) == noisy[0]
    fig = plots.rv_figure(orbit, 7, observations=(times, truth, noisy, errors))
    assert len(fig.axes[0].containers) == 1
    plt.close(fig)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_km_s": -1},
        {"sigma_km_s": float("nan")},
        {"count": 1},
        {"count": 2.5},
        {"seed": -1},
        {"span_days": 0},
    ],
)
def test_invalid_observations(kwargs):
    from rvbinary.gui import plots

    orbit = generate_orbits(GeneratorConfig(count=1)).orbits[0]
    with pytest.raises(ValueError):
        plots.rv_observations(orbit, **kwargs)
