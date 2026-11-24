"""Compare editable inputs with the successful batch's immutable configuration."""

from rvbinary.mock.generator import GeneratorConfig, OrbitBatch


def parameters_changed(config: GeneratorConfig, batch: OrbitBatch | None) -> bool:
    return batch is None or config != batch.config
