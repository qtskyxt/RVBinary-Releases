from dataclasses import replace

from rvbinary.gui.state import parameters_changed
from rvbinary.mock.generator import GeneratorConfig, generate_orbits


def test_parameters_changed_does_not_modify_batch():
    config = GeneratorConfig(count=1)
    batch = generate_orbits(config)
    assert parameters_changed(config, None)
    assert not parameters_changed(config, batch)
    assert parameters_changed(replace(config, seed=3), batch)
    assert batch.config == config
