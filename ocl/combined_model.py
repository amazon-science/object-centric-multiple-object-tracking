"""Implementation of combined model."""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Dict

import pytorch_lightning as pl
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from ocl import base, path_defaults, tree_utils, utils
from ocl.visualization_types import Visualization

# from slot_attention.tasks import Task

if TYPE_CHECKING:
    import torchmetrics


class CombinedModel(pl.LightningModule):
    def __init__(
        self,
        models: Dict[str, Any],
        losses: Dict[str, Any],
        visualizations: Dict[str, Any],
        hooks: base.PluggyHookRelay,
        training_metrics: Dict[str, torchmetrics.Metric] = None,
        evaluation_metrics: Dict[str, torchmetrics.Metric] = None,
        vis_log_frequency: int = 10,  #100
    ):
        super().__init__()
        if isinstance(models, Dict):
            models = utils.Combined(models)
        self.models = models
        self.losses = losses
        self.visualizations = visualizations
        self.hooks = hooks
        self.vis_log_frequency = vis_log_frequency

        if training_metrics is None:
            training_metrics = {}
        self.training_metrics = torch.nn.ModuleDict(training_metrics)

        if evaluation_metrics is None:
            evaluation_metrics = {}
        self.evaluation_metrics = torch.nn.ModuleDict(evaluation_metrics)

    def configure_optimizers(self):
        return self.hooks.configure_optimizers(model=self)

    def __getattribute__(self, name):
        """Forward pytorch lightning module hooks to the plugin manager.

        We need to implement `__getattribute__` as the model hooks are defined in a superclass of
        `pl.LightningModule` and thus `__getattr__` would never get called for them. This makes the
        call a bit more clumsy.
        """
        if not name.startswith("__") and hasattr(pl.core.hooks.ModelHooks, name):
            # A pytorch lighting hook is being called.
            try:
                hook_caller = getattr(self.hooks, name)
                return partial(hook_caller, model=self)
            except AttributeError:
                pass
        return super().__getattribute__(name)

    def forward(self, input_data: dict):
        # Maybe we should use something like a read only dict to prevent existing keys from being
        # overwritten.
        data: Dict[str, Any]
        data = {path_defaults.INPUT: input_data, path_defaults.GLOBAL_STEP: self.global_step}
        phase = input_data['phase']
        return self.models.forward(inputs=data, phase=phase)

    def _compute_losses(self, inputs, phase="train"):
        quantities_to_log = {}
        for name, loss in self.losses.items():
            quantities_to_log[f"{phase}/{name}"] = loss(inputs=inputs)

        losses = []
        for loss in quantities_to_log.values():
            losses.append(loss)
        total_loss = torch.stack(losses).sum()

        # Log total loss only if there is more than one task
        if len(losses) > 1:
            quantities_to_log[f"{phase}/loss_total"] = total_loss

        return total_loss, quantities_to_log

    def training_step(self, batch, batch_idx):
        batch_size = batch["batch_size"]
        batch['phase'] = 'train'
        # with profile(activities=[
        #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_train"):
        #         outputs = self(batch)
        #         total_loss, quantities_to_log = self._compute_losses(outputs)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        outputs = self(batch)
        total_loss, quantities_to_log = self._compute_losses(outputs)
        quantities_to_log.update(self._compute_metrics(outputs, self.training_metrics))
        self.log_dict(quantities_to_log, on_step=True, on_epoch=False, batch_size=batch_size)

        if self.trainer.global_step % self.vis_log_frequency == 0:
            self._log_visualizations(outputs)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["batch_size"]
        batch['phase'] = 'val'
        outputs = self(batch)
        total_loss, quantities_to_log = self._compute_losses(outputs, phase="val")

        quantities_to_log.update(
            self._compute_metrics(outputs, self.evaluation_metrics, phase="val")
        )
        self.log_dict(
            quantities_to_log, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
        )

        if batch_idx == 0:
            self._log_visualizations(outputs, phase="val")

    def _compute_metrics(self, outputs, metric_fns, phase="train"):
        metrics = {}
        if len(metric_fns) > 0:
            for metric_name, metric in metric_fns.items():
                if phase == "val":
                    # Call update instead of forward to avoid unnecessary metric compute on batch.
                    metric.update(**outputs)
                else:
                    metric(**outputs)
                metrics[f"{phase}/{metric_name}"] = metric

        return metrics

    def _log_visualizations(self, outputs, phase="train"):
        if self.logger is None:
            return
        logger_experiment = self.logger.experiment
        visualizations = {}
        for name, vis in self.visualizations.items():
            visualizations[name] = vis(inputs=outputs)

        visualization_iterator = tree_utils.walk_tree_with_paths(
            visualizations, path=None, instance_check=lambda a: isinstance(a, Visualization)
        )
        for path, vis in visualization_iterator:
            str_path = ".".join(path)
            vis.add_to_experiment(
                experiment=logger_experiment,
                tag=f"{phase}/{str_path}",
                global_step=self.trainer.global_step,
            )
