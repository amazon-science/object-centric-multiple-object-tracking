"""Utility modules and functions."""
from __future__ import annotations

import dataclasses
import functools
import inspect
import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ocl import tree_utils
from torchvision.ops.boxes import box_area

class SoftPositionEmbed(nn.Module):
    """Embeding of positions using convex combination of learnable tensors.

    This assumes that the input positions are between 0 and 1.
    """

    def __init__(
        self, n_spatial_dims: int, feature_dim: int, cnn_channel_order=False, savi_style=False
    ):
        """__init__.

        Args:
            n_spatial_dims (int): Number of spatial dimensions.
            feature_dim (int): Dimensionality of the input features.
            cnn_channel_order (bool): Assume features are in CNN channel order (i.e. C x H x W).
            savi_style (bool): Use savi style positional encoding, where positions are normalized
                between -1 and 1 and a single dense layer is used for embedding.
        """
        super().__init__()
        self.savi_style = savi_style
        n_features = n_spatial_dims if savi_style else 2 * n_spatial_dims
        self.dense = nn.Linear(in_features=n_features, out_features=feature_dim)
        self.cnn_channel_order = cnn_channel_order

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor):
        if self.savi_style:
            # Rescale positional encoding to -1 to 1
            positions = (positions - 0.5) * 2
        else:
            positions = torch.cat([positions, 1 - positions], axis=-1)
        emb_proj = self.dense(positions)
        if self.cnn_channel_order:
            emb_proj = emb_proj.permute(*range(inputs.ndim - 3), -1, -3, -2)
        return inputs + emb_proj


class DummyPositionEmbed(nn.Module):
    """Embedding that just passes through inputs without adding any positional embeddings."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor):
        return inputs


def tensor_to_one_hot(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert tensor to one-hot encoding by using maximum across dimension as one-hot element."""
    assert 0 <= dim
    max_idxs = torch.argmax(tensor, dim=dim, keepdim=True)
    shape = [1] * dim + [-1] + [1] * (tensor.ndim - dim - 1)
    one_hot = max_idxs == torch.arange(tensor.shape[dim], device=tensor.device).view(*shape)
    return one_hot.to(torch.long)


def resize_patches_to_image(
    patches: torch.Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    resize_mode: str = "bilinear",
) -> torch.Tensor:
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to.
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns: Tensor of shape (..., C, S, S) where S is the image size.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = math.sqrt(n_patches)
    patch_size = int(math.sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])


def flatten_dict_of_dataclasses(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a dictionary containing dataclasses as values.

    Fields of dataclass are stored with the key "{dataclass_key}.{field_name}" in the flattened dict.
    Entries that are no dataclass are stored with the same key in the flattened dict. Flattening is
    not recursive, only the first level of nesting is considered. Then the first level is a list it
    is also flattened in a similar fashion.

    Example:
    ```
    @dataclasses.dataclass
    class MyDataclass:
        value: int

    dictionary = {"item1": MyDataclass(value=5), "another_item": "no_dataclass"}
    assert _flatten_dict_of_dataclasses(dictionary) == {"item1.value": 5, "item2": "no_dataclass"}
    ```
    """
    flattened_dict = {}
    for key, value in input_dict.items():
        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                flattened_dict[f"{key}.{field.name}"] = getattr(value, field.name)
        elif isinstance(value, Sequence):
            sequence_output = defaultdict(list)
            for element in value:
                if dataclasses.is_dataclass(element):
                    for field in dataclasses.fields(element):
                        sequence_output[f"{key}.{field.name}"].append(getattr(element, field.name))
                else:
                    sequence_output[key].append(value)

            for nest_key, nest_value in sequence_output.items():
                if isinstance(nest_value[0], torch.Tensor):
                    flattened_dict[nest_key] = torch.stack(nest_value, dim=1)
                else:
                    flattened_dict[nest_key] = nest_value
        else:
            flattened_dict[key] = value

    return flattened_dict


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)


class RoutableMixin:
    """Mixin class that allows to connect any element of a (nested) dict with a module input."""

    def __init__(self, input_mapping: Mapping[str, Optional[str]]):
        self.input_mapping = {
            key: value.split(".") for key, value in input_mapping.items() if value is not None
        }

    def _route(method, filter_parameters=True):
        """Pass arguments to a function based on the mapping defined in `self.input_mapping`.

        This method supports both filtering for parameters that match the arguments of the wrapped
        method and passing all arguments defined in `input_mapping`.  If a non-optional argument is
        missing this will raise an exception.  Additional arguments can also be passed to the method
        to override entries in the input dict.  Non-keyword arguments are always directly passed to
        the method.

        Args:
            method: The method to pass the arguments to.
            filter_parameters: Only pass arguments to wrapped method that match the methods
                signature.  This is practical if different methods require different types of input.

        """
        # Run inspection here to reduce compute time when calling method.
        signature = inspect.signature(method)
        valid_parameters = list(signature.parameters)  # Returns the parameter names.
        valid_parameters = valid_parameters[1:]  # Discard "self".
        # Keep track of default parameters. For these we should not fail if they are not in
        # the input dict.
        with_defaults = [
            name
            for name, param in signature.parameters.items()
            if param.default is not inspect.Parameter.empty
        ]

        @functools.wraps(method)
        def method_with_routing(self, *args, inputs=None, **kwargs):
            if not inputs:
                inputs = {}
            if self.input_mapping:
                if not inputs:  # Empty dict.
                    inputs = kwargs

                routed_inputs = {}
                for input_field, input_path in self.input_mapping.items():
                    if filter_parameters and input_field not in valid_parameters:
                        # Skip parameters that are not the function signature.
                        continue
                    if input_field in kwargs.keys():
                        # Skip parameters that are directly provided as kwargs.
                        continue
                    try:
                        element = tree_utils.get_tree_element(inputs, input_path)
                        routed_inputs[input_field] = element
                    except ValueError as e:
                        if input_field in with_defaults:
                            continue
                        else:
                            raise e
                # Support for additional parameters passed via keyword arguments.
                # TODO(hornmax): This is not ideal as it mixes routing args from the input dict
                # and explicitly passed kwargs and thus could lead to collisions.
                for name, element in kwargs.items():
                    if filter_parameters and name not in valid_parameters:
                        continue
                    else:
                        routed_inputs[name] = element
                return method(self, *args, **routed_inputs)
            else:
                return method(self, *args, **kwargs)

        return method_with_routing

    # This is needed in order to allow the decorator to be used in child classes. The documentation
    # looks a bit hacky but I didn't find an alternative approach on how to do it.
    route = staticmethod(functools.partial(_route, filter_parameters=True))
    route.__doc__ = (
        """Route input arguments according to input_mapping and filter non-matching arguments."""
    )
    route_unfiltered = staticmethod(functools.partial(_route, filter_parameters=False))
    route_unfiltered.__doc__ = """Route all input arguments according to input_mapping."""


class DataRouter(nn.Module, RoutableMixin):
    """Data router for modules that don't support the RoutableMixin.

    This allows the usage of modules without RoutableMixin support in the dynamic information flow
    pattern of the code.
    """

    def __init__(self, module: nn.Module, input_mapping: Mapping[str, str]):
        nn.Module.__init__(self)
        RoutableMixin.__init__(self, input_mapping)
        self.module = module
        self._cached_valid_parameters = None

    @RoutableMixin.route_unfiltered
    def forward(self, *args, **kwargs):
        # We need to filter parameters at runtime as we cannot know them prior to initialization.
        if not self._cached_valid_parameters:
            try:
                signature = inspect.signature(self.module.forward)
            except AttributeError:
                if callable(self.module):
                    signature = inspect.signature(self.module.__call__)
                else:
                    signature = inspect.signature(self.module)

            self._cached_valid_parameters = list(signature.parameters)

        kwargs = {
            name: param for name, param in kwargs.items() if name in self._cached_valid_parameters
        }
        return self.module(*args, **kwargs)


class Combined(nn.ModuleDict):
    """Module to combine multiple modules and store their outputs.

    A combined module groups together multiple model components and allows them to access any
    information that was returned in processing steps prior to their own application.

    It functions similarly to `nn.ModuleDict` yet for modules of type `RoutableMixin` and
    additionally implements a forward routine which will return a dict of the outputs of the
    submodules.

    """

    def __init__(self, modules: Dict[str, Union[RoutableMixin, Combined, Recurrent]]):
        super().__init__(modules)

    def forward(self, inputs: Dict[str, Any]):
        # The combined module does not know where it is positioned and thus also does not know in
        # which sub-path results should be written. As we want different modules of a combined
        # module to be able access previous outputs using their global path in the dictionary, we
        # need to somehow keep track of the nesting level and then directly write results into the
        # input dict at the right path.  The prefix variable keeps track of the nesting level.
        prefix: List[str]
        if "prefix" in inputs.keys():
            prefix = inputs["prefix"]
        else:
            prefix = []
            inputs["prefix"] = prefix

        outputs = tree_utils.get_tree_element(inputs, prefix)
        for name, module in self.items():
            # Update prefix state such that nested calls of combined return dict in the correct
            # location.
            prefix.append(name)
            outputs[name] = {}
            # If module is a Combined module, it will return the same dict as set above. If not the
            # dict will be overwritten with the output of the module.
            outputs[name] = module(inputs=inputs)
            # Remove last component of prefix after execution.
            prefix.pop()
        return outputs


class Recurrent(nn.Module):
    """Module to apply another module in a recurrent fashion over a axis.

    This module takes a set of input tensors and applies a module recurrent over them.  The output
    of the previous iteration is kept in the `previous_output` key of input dict and thus can be
    accessed using data routing. After applying the module to the input slices, the outputs are
    stacked along the same axis as the inputs where split.

    Args:
        module: The module that should be applied recurrently along input tensors.
        inputs_to_split: List of paths that should be split for recurrent application.
        initial_input_mapping: Mapping that constructs the first `previous_output` element.  If
            `previous_output` should just be a tensor, use a mapping of the format
            `{"": "input_path"}`.
        split_axis: Axis along which to split the tensors defined by inputs_to_split.
        chunk_size: The size of each slice, when set to 1, the slice dimension is squeezed prior to
            passing to the module.

    """

    def __init__(
        self,
        module,
        inputs_to_split: List[str],
        initial_input_mapping: Dict[str, str],
        split_axis: int = 1,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.module = module
        self.inputs_to_split = [path.split(".") for path in inputs_to_split]
        self.initial_input_mapping = {
            output: input.split(".") for output, input in initial_input_mapping.items()
        }
        self.split_axis = split_axis
        self.chunk_size = chunk_size

    def _build_initial_dict(self, inputs):
        # This allows us to bing the initial input and previous_output into a similar format.
        output_dict = {}
        for output_path, input_path in self.initial_input_mapping.items():
            source = tree_utils.get_tree_element(inputs, input_path)
            if output_path == "":
                # Just the object itself, no dict nesting.
                return source

            output_path = output_path.split(".")
            cur_search = output_dict
            for path_part in output_path[:-1]:
                # Iterate along path and create nodes that do not exist yet.
                try:
                    # Get element prior to last.
                    cur_search = tree_utils.get_tree_element(cur_search, [path_part])
                except ValueError:
                    # Element does not yet exist.
                    cur_search[path_part] = {}
                    cur_search = cur_search[path_part]

            cur_search[output_path[-1]] = source
        return output_dict

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Come up with a better way of handling the initial input without putting restrictions
        # on modules being run recurrently.
        outputs = [self._build_initial_dict(inputs)]
        for split_dict in tree_utils.split_tree(
            inputs, self.inputs_to_split, self.split_axis, self.chunk_size
        ):
            split_dict["previous_output"] = outputs[-1]
            outputs.append(self.module(inputs=split_dict))

        # TODO: When chunk size is larger than 1 then this should be cat and not stack. Otherwise an
        # additional axis would be added. Evtl. this should be configurable.
        stack_fn = functools.partial(torch.stack, dim=self.split_axis)
        # Ignore initial input.
        return tree_utils.reduce_tree(outputs[1:], stack_fn)


class Sequential(nn.Module):
    """Extended sequential module that supports multiple inputs and outputs to layers.

    This allows a stack of layers where for example the first layer takes two inputs and only has
    a single output or where a layer has multiple outputs and the downstream layer takes multiple
    inputs.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *inputs):
        outputs = inputs
        for layer in self.layers:
            if isinstance(outputs, (tuple, list)):
                outputs = layer(*outputs)
            else:
                outputs = layer(outputs)
        return outputs


class CreateSlotMask(nn.Module, RoutableMixin):
    """Module intended to create a mask that marks empty slots.

    Module takes a tensor holding the number of slots per batch entry, and returns a binary mask of
    shape (batch_size, max_slots) where entries exceeding the number of slots are masked out.
    """

    def __init__(self, max_slots: int, n_slots_path: str):
        nn.Module.__init__(self)
        RoutableMixin.__init__(self, {"n_slots": n_slots_path})
        self.max_slots = max_slots

    @RoutableMixin.route
    def forward(self, n_slots: torch.Tensor) -> torch.Tensor:
        (batch_size,) = n_slots.shape

        # Create mask of shape B x K where the first n_slots entries per-row are false, the rest true
        indices = torch.arange(self.max_slots, device=n_slots.device)
        masks = indices.unsqueeze(0).expand(batch_size, -1) >= n_slots.unsqueeze(1)

        return masks

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


class Resize(nn.Module, RoutableMixin):
    """Module resizing tensors."""

    MODES = {"nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"}

    def __init__(
        self,
        input_path: str,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        take_size_from: Optional[str] = None,
        resize_mode: str = "bilinear",
        patch_mode: bool = False,
        channels_last: bool = False,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(self, {"tensor": input_path, "size_tensor": take_size_from})

        if size is not None and take_size_from is not None:
            raise ValueError("`size` and `take_size_from` can not be set at the same time")
        self.size = size

        if resize_mode not in Resize.MODES:
            raise ValueError(f"`mode` must be one of {Resize.MODES}")
        self.resize_mode = resize_mode
        self.patch_mode = patch_mode
        self.channels_last = channels_last
        self.expected_dims = 3 if patch_mode else 4

    @RoutableMixin.route
    def forward(
        self, tensor: torch.Tensor, size_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Resize tensor.

        Args:
            tensor: Tensor to resize. If `patch_mode=False`, assumed to be of shape (..., C, H, W).
                If `patch_mode=True`, assumed to be of shape (..., C, P), where P is the number of
                patches. Patches are assumed to be viewable as a perfect square image. If
                `channels_last=True`, channel dimension is assumed to be the last dimension instead.
            size_tensor: Tensor which size to resize to. If tensor has <=2 dimensions and the last
                dimension of this tensor has length 2, the two entries are taken as height and width.
                Otherwise, the size of the last two dimensions of this tensor are used as height
                and width.

        Returns: Tensor of shape (..., C, H, W), where height and width are either specified by
            `size` or `size_tensor`.
        """
        dims_to_flatten = tensor.ndim - self.expected_dims
        if dims_to_flatten > 0:
            flattened_dims = tensor.shape[: dims_to_flatten + 1]
            tensor = tensor.flatten(0, dims_to_flatten)
        elif dims_to_flatten < 0:
            raise ValueError(
                f"Tensor needs at least {self.expected_dims} dimensions, but only has {tensor.ndim}"
            )

        if self.patch_mode:
            if self.channels_last:
                tensor = tensor.transpose(-2, -1)
            n_channels, n_patches = tensor.shape[-2:]
            patch_size_float = math.sqrt(n_patches)
            patch_size = int(math.sqrt(n_patches))
            if patch_size_float != patch_size:
                raise ValueError(
                    f"The number of patches needs to be a perfect square, but is {n_patches}."
                )
            tensor = tensor.view(-1, n_channels, patch_size, patch_size)
        else:
            if self.channels_last:
                tensor = tensor.permute(0, 3, 1, 2)

        if self.size is None:
            if size_tensor is None:
                raise ValueError("`size` is `None` but no `size_tensor` was passed.")
            if size_tensor.ndim <= 2 and size_tensor.shape[-1] == 2:
                height, width = size_tensor.unbind(-1)
                height = torch.atleast_1d(height)[0].squeeze().detach().cpu()
                width = torch.atleast_1d(width)[0].squeeze().detach().cpu()
                size = (int(height), int(width))
            else:
                size = size_tensor.shape[-2:]
        else:
            size = self.size

        tensor = torch.nn.functional.interpolate(
            tensor,
            size=size,
            mode=self.resize_mode,
        )

        if dims_to_flatten > 0:
            tensor = tensor.unflatten(0, flattened_dims)

        return tensor

def lecun_uniform_(tensor, gain=1.):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    var = gain / float(fan_in)
    a = math.sqrt(3 * var)
    return nn.init._no_grad_uniform_(tensor, -a, a)


def lecun_normal_(tensor, gain=1., mode="fan_in"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        scale_mode = fan_in
    elif mode == "fan_out":
        scale_mode = fan_out
    else:
        raise NotImplementedError
    var = gain / float(scale_mode)
    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(var) / .87962566103423978
    # return nn.init._no_grad_normal_(tensor, 0., std)
    kernel = torch.nn.init._no_grad_trunc_normal_(tensor, 0, 1, -2, 2) * std
    with torch.no_grad():
        tensor[:] = kernel[:]
    return tensor


def lecun_normal_fan_out_(tensor, gain=1.):
    return lecun_normal_(tensor, gain=gain, mode="fan_out")

def lecun_normal_convtranspose_(tensor, gain=1.):
    # for some reason, the convtranspose weights are [in_channels, out_channels, kernel, kernel]
    # but the _calculate_fan_in_and_fan_out treats dim 1 as fan_in and dim 0 as fan_out.
    # so, for convolution weights, have to use fan_out instead of fan_in
    # which is actually using fan_in instead of fan_out
    return lecun_normal_fan_out_(tensor, gain=gain)


init_fn = {
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'lecun_uniform': lecun_uniform_,
    'lecun_normal': lecun_normal_,
    'lecun_normal_fan_out': lecun_normal_fan_out_,
    'ones': nn.init.ones_,
    'zeros': nn.init.zeros_,
    'default': lambda x: x}

