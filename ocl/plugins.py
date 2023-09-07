import functools
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import webdataset

from ocl import hooks


class Plugin:
    """A plugin which defines a set of hooks to be called by the code."""


class Optimization(Plugin):
    """Optimize (a subset of) the parameters using a optimizer and a LR scheduler."""

    def __init__(
        self, optimizer, lr_scheduler=None, parameter_groups: Optional[List[Dict[str, Any]]] = None
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.parameter_group_specs = parameter_groups
        if self.parameter_group_specs:
            for idx, param_group_spec in enumerate(self.parameter_group_specs):
                if "params" not in param_group_spec:
                    raise ValueError(f'Parameter group {idx + 1} does not contain key "params"')
                param_spec = param_group_spec["params"]
                if isinstance(param_spec, str):
                    param_group_spec["params"] = [param_spec]
                elif isinstance(param_spec, Iterable):
                    param_group_spec["params"] = list(param_spec)
                else:
                    raise ValueError(
                        f'"params" for parameter group {idx + 1} is not of type str or iterable'
                    )

                if "predicate" in param_group_spec:
                    if not callable(param_group_spec["predicate"]):
                        raise ValueError(
                            f'"predicate" for parameter group {idx + 1} is not a callable'
                        )

    def _get_parameter_groups(self, model):
        """Build parameter groups from specification."""
        parameter_groups = []
        for param_group_spec in self.parameter_group_specs:
            param_spec = param_group_spec["params"]
            # Default predicate includes all parameters
            predicate = param_group_spec.get("predicate", lambda name, param: True)

            parameters = []
            for parameter_path in param_spec:
                root = model
                for child in parameter_path.split("."):
                    root = getattr(root, child)
                parameters.extend(
                    param for name, param in root.named_parameters() if predicate(name, param)
                )

            param_group = {
                k: v for k, v in param_group_spec.items() if k not in ("params", "predicate")
            }
            param_group["params"] = parameters
            parameter_groups.append(param_group)

        return parameter_groups

    @hooks.hook_implementation
    def configure_optimizers(self, model):
        if self.parameter_group_specs:
            params_or_param_groups = self._get_parameter_groups(model)
        else:
            params_or_param_groups = model.parameters()

        optimizer = self.optimizer(params_or_param_groups)
        output = {"optimizer": optimizer}
        if self.lr_scheduler:
            output.update(self.lr_scheduler(optimizer))
        return output


def transform_with_duplicate(elements: dict, *, transform, element_key: str, duplicate_key: str):
    """Utility function to fix issues with pickling."""
    element = transform(elements[element_key])
    elements[element_key] = element
    elements[duplicate_key] = element
    return elements


class SingleElementPreprocessing(Plugin):
    """Preprocessing of a single element in the input data.

    This is useful to build preprocessing pipelines based on existing element transformations such
    as those provided by torchvision. The element can optionally be duplicated and stored under a
    different key after the transformation by specifying `duplicate_key`. This is useful to further
    preprocess this element in different ways afterwards.
    """

    def __init__(
        self,
        training_transform: Callable,
        evaluation_transform: Callable,
        element_key: str = "image",
        duplicate_key: Optional[str] = None,
    ):
        self._training_transform = training_transform
        self._evaluation_transform = evaluation_transform
        self.element_key = element_key
        self.duplicate_key = duplicate_key

    @hooks.hook_implementation
    def training_fields(self):
        return (self.element_key,)

    @hooks.hook_implementation
    def training_transform(self):
        if self._training_transform:

            if self.duplicate_key is None:

                def transform(pipeline: webdataset.Processor):
                    return pipeline.map_dict(**{self.element_key: self._training_transform})

            else:

                def transform(pipeline: webdataset.Processor):
                    transform_func = functools.partial(
                        transform_with_duplicate,
                        transform=self._training_transform,
                        element_key=self.element_key,
                        duplicate_key=self.duplicate_key,
                    )
                    return pipeline.map(transform_func)

            return transform
        else:
            return None

    @hooks.hook_implementation
    def evaluation_fields(self):
        return (self.element_key,)

    @hooks.hook_implementation
    def evaluation_transform(self):
        if self._evaluation_transform:

            if self.duplicate_key is None:

                def transform(pipeline: webdataset.Processor):
                    return pipeline.map_dict(**{self.element_key: self._evaluation_transform})

            else:

                def transform(pipeline: webdataset.Processor):
                    transform_func = functools.partial(
                        transform_with_duplicate,
                        transform=self._evaluation_transform,
                        element_key=self.element_key,
                        duplicate_key=self.duplicate_key,
                    )

                    return pipeline.map(transform_func)

            return transform
        else:
            return None


class MultiElementPreprocessing(Plugin):
    """Preprocessing of multiple elements in the input data.

    This is useful preprocessing pipelines based on existing element transformations such as those
    provided by torchvision.
    """

    def __init__(
        self,
        training_transforms: Optional[Dict[str, Any]] = None,
        evaluation_transforms: Optional[Dict[str, Any]] = None,
    ):
        if training_transforms is None:
            training_transforms = {}
        self.training_keys = tuple(training_transforms)
        self._training_transforms = {
            key: transf for key, transf in training_transforms.items() if transf is not None
        }

        if evaluation_transforms is None:
            evaluation_transforms = {}
        self.evaluation_keys = tuple(evaluation_transforms)
        self._evaluation_transforms = {
            key: transf for key, transf in evaluation_transforms.items() if transf is not None
        }

    @hooks.hook_implementation
    def training_fields(self):
        return self.training_keys

    @hooks.hook_implementation
    def training_transform(self):
        if self._training_transforms:

            def transform(pipeline: webdataset.Processor):
                return pipeline.map_dict(**self._training_transforms)

            return transform
        else:
            return None

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self.evaluation_keys

    @hooks.hook_implementation
    def evaluation_transform(self):
        if self._evaluation_transforms:

            def transform(pipeline: webdataset.Processor):
                return pipeline.map_dict(**self._evaluation_transforms)

            return transform
        else:
            return None


class DataPreprocessing(Plugin):
    """Arbitrary preprocessing of input data.

    The transform takes in a dictionary of elements and should return a dictionary of elements.
    Plugin must specify the elements that should be included in the dictionary using
    `training_fields` and `evaluation_fields` arguments.
    """

    def __init__(
        self,
        training_transform: Optional[Callable] = None,
        evaluation_transform: Optional[Callable] = None,
        training_fields: Optional[Sequence[str]] = None,
        evaluation_fields: Optional[Sequence[str]] = None,
    ):
        if training_transform is not None and training_fields is None:
            raise ValueError(
                "If passing `training_transform`, `training_fields` must also be specified."
            )
        if evaluation_transform is not None and evaluation_fields is None:
            raise ValueError(
                "If passing `evaluation_transform`, `evaluation_fields` must also be specified."
            )

        self._training_transform = training_transform
        self._evaluation_transform = evaluation_transform
        self._training_fields = tuple(training_fields) if training_fields else tuple()
        self._evaluation_fields = tuple(evaluation_fields) if evaluation_fields else tuple()

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    @hooks.hook_implementation
    def training_transform(self):
        if self._training_transform:

            def transform(pipeline: webdataset.Processor):
                return pipeline.map(self._training_transform)

            return transform
        else:
            return None

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        if self._evaluation_transform:

            def transform(pipeline: webdataset.Processor):
                return pipeline.map(self._evaluation_transform)

            return transform
        else:
            return None


class SubsetDataset(Plugin):
    """Create a subset of a dataset by discarding samples."""

    def __init__(self, predicate, fields: Sequence[str]):
        """Plugin to create a subset of a dataset by discarding samples.

        Args:
            predicate: Function which determines if elements should be kept (return value is True)
                or discarded (return value is False). The function is only provided with the fields
                specified in the `fields` parameter.
            fields (Sequence[str]): The fields from the input which should be passed on to the
                predicate for evaluation.
        """
        self.predicate = predicate
        self.fields = tuple(fields)

    def _get_transform_function(self):
        def wrapped_predicate(d: dict):
            return self.predicate(*(d[field] for field in self.fields))

        def select(pipeline: webdataset.Processor):
            return pipeline.select(wrapped_predicate)

        return select

    @hooks.hook_implementation
    def training_fields(self):
        return self.fields

    @hooks.hook_implementation(tryfirst=True)
    def training_transform(self):
        return self._get_transform_function()

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self.fields

    @hooks.hook_implementation(tryfirst=True)
    def evaluation_transform(self):
        return self._get_transform_function()


class SampleFramesFromVideo(Plugin):
    def __init__(
        self,
        n_frames_per_video: int,
        training_fields: Sequence[str],
        evaluation_fields: Sequence[str],
        dim: int = 0,
        seed: int = 39480234,
        shuffle_buffer_size: int = 1000,
    ):
        """Sample frames from input tensors.

        Args:
            n_frames_per_video: Number of frames per video to sample. -1 indicates that all frames
                should be sampled.
            training_fields: The fields that should be considered video data and thus sliced
                according to the frame sampling during training.
            evaluation_fields: The fields that should be considered video data and thus sliced
                according to the frame sampling during evaluation.
            dim: The dimension along which to slice the tensors.
            seed: Random number generator seed to deterministic sampling during evaluation.
            shuffle_buffer_size: Size of shuffle buffer used during training. An additional
                shuffling step ensures each batch contains a diverse set of images and not only
                images from the same video.
        """
        self.n_frames_per_video = n_frames_per_video
        self._training_fields = tuple(training_fields)
        self._evaluation_fields = tuple(evaluation_fields)
        self.dim = dim
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size

    def slice_data(self, data, index: int):
        """Small utility method to slice a numpy array along a specified axis."""
        n_dims_before = self.dim
        n_dims_after = data.ndim - 1 - self.dim
        slices = (slice(None),) * n_dims_before + (index,) + (slice(None),) * n_dims_after
        return data[slices]

    def sample_frames_using_key(self, data, fields, seed):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        for sample in data:
            # Initialize random number generator dependent on instance key. This should make the
            # sampling process deterministic, which is useful when sampling frames for the
            # validation/test data.
            key = sample["__key__"]
            # TODO (hornmax): We assume all fields to have the same size. I do not want to check
            # this here as it seems a bit verbose.
            n_frames = sample[fields[0]].shape[self.dim]
            rand = random.Random(int(key) + seed)
            frames_per_video = self.n_frames_per_video if self.n_frames_per_video != -1 else n_frames
            selected_frames = rand.sample(range(n_frames), k=frames_per_video)

            for frame in selected_frames:
                # Slice the fields according to the frame.
                sliced_fields = {field: self.slice_data(sample[field], frame) for field in fields}
                # Leave all fields besides the sliced ones as before, augment the __key__ field to
                # include the frame number.
                sliced_fields["__key__"] = f"{key}_{frame}"
                yield {**sample, **sliced_fields}

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    @hooks.hook_implementation
    def training_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._training_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.sample_frames_using_key, fields=self._training_fields, seed=self.seed
                    )
                ).shuffle(self.shuffle_buffer_size)
            else:
                return pipeline

        return apply_deterministic_sampling

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._evaluation_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.sample_frames_using_key,
                        fields=self._evaluation_fields,
                        seed=self.seed + 1,
                    )
                )
            else:
                return pipeline

        return apply_deterministic_sampling


class SplitConsecutiveFrames(Plugin):
    def __init__(
        self,
        n_consecutive_frames: int,
        training_fields: Sequence[str],
        evaluation_fields: Sequence[str],
        dim: int = 0,
        shuffle_buffer_size: int = 1000,
        drop_last: bool = True,
    ):
        self.n_consecutive_frames = n_consecutive_frames
        self._training_fields = tuple(training_fields)
        self._evaluation_fields = tuple(evaluation_fields)
        self.dim = dim
        self.shuffle_buffer_size = shuffle_buffer_size
        self.drop_last = drop_last

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    def split_to_consecutive_frames(self, data, fields):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        for sample in data:
            key = sample["__key__"]
            n_frames = sample[fields[0]].shape[self.dim]

            splitted_fields = [
                np.array_split(
                    sample[field],
                    range(self.n_consecutive_frames, n_frames, self.n_consecutive_frames),
                    axis=self.dim,
                )
                for field in fields
            ]

            for i, slices in enumerate(zip(*splitted_fields)):
                if self.drop_last and slices[0].shape[self.dim] < self.n_consecutive_frames:
                    # Last slice of not equally divisible input, discard.
                    continue

                sliced_fields = dict(zip(fields, slices))
                sliced_fields["__key__"] = f"{key}_{i}"
                yield {**sample, **sliced_fields}

    @hooks.hook_implementation
    def training_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._training_fields) > 0:
                return pipeline.then(
                    functools.partial(self.split_to_consecutive_frames, fields=self._training_fields)
                ).shuffle(self.shuffle_buffer_size)
            else:
                return pipeline

        return apply_deterministic_sampling

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._evaluation_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.split_to_consecutive_frames,
                        fields=self._evaluation_fields,
                    )
                )
            else:
                return pipeline

        return apply_deterministic_sampling


class RandomStridedWindow(Plugin):
    """Select a random consecutive subsequence of frames in a strided manner.

    Given a sequence of [1, 2, 3, 4, 5, 6, 7, 8, 9] this will return one of
    [1, 2, 3] [4, 5, 6] [7, 8, 9].
    """

    def __init__(
        self,
        n_consecutive_frames: int,
        training_fields: Sequence[str],
        evaluation_fields: Sequence[str],
        dim: int = 0,
    ):
        self.n_consecutive_frames = n_consecutive_frames
        self._training_fields = tuple(training_fields)
        self._evaluation_fields = tuple(evaluation_fields)
        self.dim = dim
        self._random = None

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    @property
    def random(self):
        if not self._random:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info:
                self._random = random.Random(worker_info.seed)
            else:
                self._random = random.Random(torch.initial_seed())

        return self._random

    def split_to_consecutive_frames(self, data, fields):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        for sample in data:
            key = sample["__key__"]
            n_frames = sample[fields[0]].shape[self.dim]
            splitted_fields = []
            for field in fields:
                splitted_field = []
                for step in range(n_frames // self.n_consecutive_frames):
                    ind = []
                    clip_len = (self.n_consecutive_frames-1)*step + self.n_consecutive_frames-1
                    end = n_frames-1-clip_len
                    start_id = self.random.randint(0, end)
                    end_id  = start_id + (self.n_consecutive_frames-1)*step + self.n_consecutive_frames -1
                    for i in range(start_id, end_id+1, step+1):
                        ind.append(i)
                    # ind = self.random.sample(range(0, (step+1)*self.n_consecutive_frames), self.n_consecutive_frames)
                    splitted_field.append(sample[field][sorted(ind)])
                splitted_fields.append(splitted_field)
            # splitted_fields = [
            #     np.array_split(
            #         sample[field],
            #         range(self.n_consecutive_frames, n_frames, self.n_consecutive_frames),
            #         axis=self.dim,
            #     )
            #     for field in fields
            # ]


            n_fragments = len(splitted_fields[0])

            if len(splitted_fields[0][-1] < self.n_consecutive_frames):
                # Discard last fragment if too short.
                n_fragments -= 1

            # fragment_id = self.random.randint(0, n_fragments - 1)
            fragment_id = self.random.randint(0, n_frames // self.n_consecutive_frames-1)
            # fragment_id = 0
            sliced_fields = {
                field_name: splitted_field[fragment_id]
                for field_name, splitted_field in zip(fields, splitted_fields)
            }
            sliced_fields["__key__"] = f"{key}_{fragment_id}"
            yield {**sample, **sliced_fields}

    @hooks.hook_implementation
    def training_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._training_fields) > 0:
                return pipeline.then(
                    functools.partial(self.split_to_consecutive_frames, fields=self._training_fields)
                )
            else:
                return pipeline

        return apply_deterministic_sampling

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._evaluation_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.split_to_consecutive_frames,
                        fields=self._evaluation_fields,
                    )
                )
            else:
                return pipeline

        return apply_deterministic_sampling
