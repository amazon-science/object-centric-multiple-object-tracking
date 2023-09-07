"""Train a slot attention type model."""
import dataclasses
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.cloud_io import load as pl_load
import hydra
import hydra_zen
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pluggy import PluginManager
import torch
import ocl.hooks
from ocl import base
from ocl.combined_model import CombinedModel
from ocl.config.datasets import DataModuleConfig
from ocl.config.metrics import MetricConfig
from ocl.config.plugins import PluginConfig
from ocl.plugins import Plugin
from ocl.utils import Combined, Recurrent, RoutableMixin
from typing import Union
TrainerConf = hydra_zen.builds(
    pl.Trainer, max_epochs=100, zen_partial=False, populate_full_signature=True
)


@dataclasses.dataclass
class TrainingConfig:
    """Configuration of a training run."""

    dataset: DataModuleConfig
    models: Any # When provided with dict wrap in `utils.Combined`, otherwise interpret as model.
    losses: Dict[str, Any]
    visualizations: Dict[str, Any] = dataclasses.field(default_factory=dict)
    plugins: Dict[str, PluginConfig] = dataclasses.field(default_factory=dict)
    trainer: TrainerConf = TrainerConf
    training_vis_frequency: Optional[int] = None
    training_metrics: Optional[Dict[str, MetricConfig]] = None
    evaluation_metrics: Optional[Dict[str, MetricConfig]] = None
    load_checkpoint: Optional[str] = None
    seed: Optional[int] = None
    experiment: Optional[Any] = None
    root_output_folder: Optional[str] = None


hydra.core.config_store.ConfigStore.instance().store(
    name="training_config",
    node=TrainingConfig,
)


def create_plugin_manager() -> PluginManager:
    pm = PluginManager("ocl")
    pm.add_hookspecs(ocl.hooks)
    return pm


def build_and_register_datamodule_from_config(
    config: TrainingConfig,
    hooks: base.PluggyHookRelay,
    plugin_manager: Optional[PluginManager] = None,
    **datamodule_kwargs,
) -> pl.LightningDataModule:
    datamodule = hydra_zen.instantiate(
        config.dataset, hooks=hooks, _convert_="all", **datamodule_kwargs
    )

    if plugin_manager:
        plugin_manager.register(datamodule)

    return datamodule


def build_and_register_plugins_from_config(
    config: TrainingConfig, plugin_manager: Optional[PluginManager] = None
) -> Dict[str, Plugin]:
    plugins = hydra_zen.instantiate(config.plugins)
    # Use lexicographical sorting to allow to influence registration order. This is necessary in
    # some cases as certain plugins might need to be called before others. Pluggy calls hooks
    # according to FILO (first in last out) and this is slightly unintuitive. We thus register
    # plugins in reverse order to their sorting position, leading to a FIFO (first in first out)
    # behavior with regard to the sorted position.
    if plugin_manager:
        for plugin_name in sorted(plugins.keys())[::-1]:
            plugin_manager.register(plugins[plugin_name])

    return plugins


def build_model_from_config(
    config: TrainingConfig,
    hooks: base.PluggyHookRelay,
    checkpoint_path: Optional[str] = None,
) -> pl.LightningModule:
    models = hydra_zen.instantiate(config.models, _convert_="all")
    losses = hydra_zen.instantiate(config.losses, _convert_="all")
    visualizations = hydra_zen.instantiate(config.visualizations, _convert_="all")

    training_metrics = hydra_zen.instantiate(config.training_metrics)
    evaluation_metrics = hydra_zen.instantiate(config.evaluation_metrics)

    # train_vis_freq = config.training_vis_frequency if config.training_vis_frequency else 10
    train_vis_freq = 10

    if checkpoint_path is None:
        model = CombinedModel(
            models=models,
            losses=losses,
            visualizations=visualizations,
            hooks=hooks,
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
            vis_log_frequency=train_vis_freq,
        )
    else:
        model = CombinedModel.load_from_checkpoint(
            checkpoint_path,
            strict=False,
            models=models,
            losses=losses,
            visualizations=visualizations,
            hooks=hooks,
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
            vis_log_frequency=train_vis_freq,
        )
        # checkpoint = pl_load(checkpoint_path)
        # model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


@hydra.main(config_name="training_config", config_path="../../configs/", version_base="1.1")
def train(config: TrainingConfig):
    # Set all relevant random seeds. If `config.seed is None`, the function samples a random value.
    # The function takes care of correctly distributing the seed across nodes in multi-node training,
    # and assigns each dataloader worker a different random seed.
    # IMPORTANTLY, we need to take care not to set a custom `worker_init_fn` function on the
    # dataloaders (or take care of worker seeding ourselves).
    pl.seed_everything(config.seed, workers=True)

    pm = create_plugin_manager()

    datamodule = build_and_register_datamodule_from_config(config, pm.hook, pm)

    build_and_register_plugins_from_config(config, pm)

    if config.load_checkpoint:
        checkpoint_path = hydra.utils.to_absolute_path(config.load_checkpoint)
    else:
        checkpoint_path = None

    model = build_model_from_config(config, pm.hook, checkpoint_path)

    callbacks = []
    if config.trainer.logger is not False:
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    checkpoint_path_ = 'checkpoints'

    checkpoint_callback  = ModelCheckpoint(dirpath = checkpoint_path_, every_n_epochs = 10)
    callbacks.append(checkpoint_callback)

    trainer: pl.Trainer = hydra_zen.instantiate(config.trainer, callbacks=callbacks, _convert_="all", gpus=[4,5,6,7]) #gpus=[5,6]

    trainer.fit(model, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    train()
