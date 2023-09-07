"""Register metric related configs."""
import dataclasses

from hydra_zen import builds, make_custom_builds_fn

from ocl import metrics


@dataclasses.dataclass
class MetricConfig:
    """Base class for metrics."""

    pass


builds_metric = make_custom_builds_fn(
    populate_full_signature=True,
)

TensorStatisticConfig = builds_metric(metrics.TensorStatistic, builds_bases=(MetricConfig,))
TorchmetricsWrapperConfig = builds_metric(metrics.TorchmetricsWrapper, builds_bases=(MetricConfig,))
ARIMetricConfig = builds_metric(metrics.ARIMetric, builds_bases=(MetricConfig,))
PatchARIMetricConfig = builds_metric(
    metrics.PatchARIMetric,
    builds_bases=(MetricConfig,),
)

MOTMetricConfig = builds_metric(
    metrics.MOTMetric,
    builds_bases=(MetricConfig,),
)
ThreeDIOUMetric = builds_metric(
    metrics.ThreeDIOUMetric,
    builds_bases=(MetricConfig,),
)
UnsupervisedMaskIoUMetricConfig = builds_metric(
    metrics.UnsupervisedMaskIoUMetric,
    builds_bases=(MetricConfig,),
)
MaskCorLocMetricConfig = builds_metric(
    metrics.UnsupervisedMaskIoUMetric,
    matching="best_overlap",
    correct_localization=True,
    builds_bases=(MetricConfig,),
)
AverageBestOverlapMetricConfig = builds_metric(
    metrics.UnsupervisedMaskIoUMetric,
    matching="best_overlap",
    builds_bases=(MetricConfig,),
)
BestOverlapObjectRecoveryMetricConfig = builds_metric(
    metrics.UnsupervisedMaskIoUMetric,
    matching="best_overlap",
    compute_discovery_fraction=True,
    builds_bases=(MetricConfig,),
)
UnsupervisedBboxIoUMetricConfig = builds_metric(
    metrics.UnsupervisedBboxIoUMetric,
    builds_bases=(MetricConfig,),
)
BboxCorLocMetricConfig = builds_metric(
    metrics.UnsupervisedBboxIoUMetric,
    matching="best_overlap",
    correct_localization=True,
    builds_bases=(MetricConfig,),
)
BboxRecallMetricConfig = builds_metric(
    metrics.UnsupervisedBboxIoUMetric,
    matching="best_overlap",
    compute_discovery_fraction=True,
    builds_bases=(MetricConfig,),
)

DatasetSemanticMaskIoUMetricConfig = builds_metric(metrics.DatasetSemanticMaskIoUMetric)

SklearnClusteringConfig = builds(
    metrics.SklearnClustering,
    populate_full_signature=True,
)


def register_configs(config_store):
    config_store.store(group="metrics", name="tensor_statistic", node=TensorStatisticConfig)
    config_store.store(group="metrics", name="torchmetric", node=TorchmetricsWrapperConfig)
    config_store.store(group="metrics", name="ari_metric", node=ARIMetricConfig)
    config_store.store(group="metrics", name="mot_metric", node=MOTMetricConfig)
    config_store.store(group="metrics", name="three_d_iou", node=ThreeDIOUMetric)
    config_store.store(group="metrics", name="patch_ari_metric", node=PatchARIMetricConfig)
    config_store.store(
        group="metrics", name="unsupervised_mask_iou_metric", node=UnsupervisedMaskIoUMetricConfig
    )
    config_store.store(group="metrics", name="mask_corloc_metric", node=MaskCorLocMetricConfig)
    config_store.store(
        group="metrics", name="average_best_overlap_metric", node=AverageBestOverlapMetricConfig
    )
    config_store.store(
        group="metrics",
        name="best_overlap_object_recovery_metric",
        node=BestOverlapObjectRecoveryMetricConfig,
    )
    config_store.store(
        group="metrics", name="unsupervised_bbox_iou_metric", node=UnsupervisedBboxIoUMetricConfig
    )
    config_store.store(group="metrics", name="bbox_corloc_metric", node=BboxCorLocMetricConfig)
    config_store.store(group="metrics", name="bbox_recall_metric", node=BboxRecallMetricConfig)

    config_store.store(
        group="metrics",
        name="dataset_semantic_mask_iou",
        node=DatasetSemanticMaskIoUMetricConfig,
    )
    config_store.store(
        group="clustering",
        name="sklearn_clustering",
        node=SklearnClusteringConfig,
    )
