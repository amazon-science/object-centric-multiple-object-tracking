"""Perceptual grouping models."""
import dataclasses

from hydra_zen import builds

from ocl import predictor


@dataclasses.dataclass
class PredictorConfig:
    """Configuration class of Predictor."""


TransitionConfig = builds(
    predictor.TransformerBlock,
    builds_bases=(PredictorConfig,),
    populate_full_signature=True,
)



def register_configs(config_store):
    config_store.store(group="schemas", name="perceptual_grouping", node=PredictorConfig)
    config_store.store(group="perceptual_grouping", name="slot_attention", node=TransitionConfig)

