from typing import Optional

import pytorch_lightning

from ocl.cli import train


def build_from_train_config(
    config: train.TrainingConfig, checkpoint_path: Optional[str], seed: bool = True
):
    if seed:
        pytorch_lightning.seed_everything(config.seed, workers=True)

    pm = train.create_plugin_manager()
    datamodule = train.build_and_register_datamodule_from_config(config, pm.hook, pm)
    train.build_and_register_plugins_from_config(config, pm)
    model = train.build_model_from_config(config, pm.hook, checkpoint_path)

    return datamodule, model, pm
