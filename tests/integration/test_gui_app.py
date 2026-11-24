from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from rvbinary.models.base import SimulationError

PAGE = Path(__file__).parents[2] / "src/rvbinary/gui/app.py"


def generated_page(count=10):
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not page.exception
    page.number_input(key="count").set_value(count)
    page.button(key="generate").click().run()
    assert not page.exception
    return page


def test_default_generation_and_orbit_selection():
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    page.button(key="generate").click().run()
    assert not page.exception
    assert len(page.session_state["batch"].orbits) == 1000
    assert len(page.dataframe[0].value) == 100
    page.number_input(key="orbit_id").set_value(999).run()
    page.toggle(key="custom_span").set_value(True).run()
    page.number_input(key="span_days").set_value(7.0).run()
    assert not page.exception
    assert page.number_input(key="orbit_id").value == 999


def test_changed_inputs_preserve_generated_batch():
    page = generated_page()
    old = page.session_state["batch"]
    page.number_input(key="seed").set_value(7).run()
    assert page.session_state["batch"] == old
    assert page.warning
    page.button(key="generate").click().run()
    assert page.session_state["batch"].config.seed == 7
    assert page.session_state["batch"].orbits != old.orbits
    page.number_input(key="orbit_id").set_value(9).run()
    page.number_input(key="count").set_value(1)
    page.button(key="generate").click().run()
    assert not page.exception
    assert page.number_input(key="orbit_id").value == 0


def test_sampling_failure_preserves_old_result(monkeypatch):
    from rvbinary.mock import generator

    page = generated_page()
    old = page.session_state["batch"]

    def fail(config):
        raise SimulationError("sample 3: failed to converge")

    monkeypatch.setattr(generator, "generate_orbits", fail)
    page.button(key="generate").click().run()
    assert not page.exception
    assert page.error
    assert page.session_state["batch"] == old


def test_rv_failure_does_not_destroy_export(monkeypatch):
    from rvbinary.gui import plots

    def fail(orbit, span_days=None, **kwargs):
        raise SimulationError("curve calculation failed")

    monkeypatch.setattr(plots, "rv_figure", fail)
    page = generated_page()
    assert page.error
    assert len(page.session_state["batch"].orbits) == 10
    assert len(page.get("download_button")) == 2


def test_language_and_noise_preserve_orbits():
    page = generated_page()
    old = page.session_state["batch"]
    page.number_input(key="noise_sigma").set_value(2.0).run()
    page.number_input(key="noise_seed").set_value(123).run()
    page.selectbox(key="language").set_value("English").run()
    assert not page.exception
    assert page.session_state["batch"] == old
    assert page.number_input(key="noise_sigma").value == 2.0
    assert page.number_input(key="noise_seed").value == 123
    assert page.button(key="generate").label == "Generate orbits"
    assert all(not any("\u4e00" <= c <= "\u9fff" for c in w.label) for w in page.number_input)
    assert len(page.get("download_button")) == 3
    page.selectbox(key="language").set_value("中文").run()
    assert page.session_state["batch"] == old
    assert page.number_input(key="noise_sigma").value == 2.0
