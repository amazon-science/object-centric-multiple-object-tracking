"""Perceptual grouping models."""
import dataclasses

from hydra_zen import builds

from ocl import memory


@dataclasses.dataclass
class MemoryConfig:
    """Configuration class of Predictor."""


TransitionConfig = builds(
    memory.SelfSupervisedMemory,
    builds_bases=(MemoryConfig,),
    populate_full_signature=True,
)


def register_configs(config_store):
    config_store.store(group="schemas", name="memory", node=MemoryConfig)
    config_store.store(group="memory", name="mem", node=TransitionConfig)
