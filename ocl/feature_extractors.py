"""Implementation of feature extractors."""
import enum
import itertools
import math
from typing import List, Optional, Union

import torch
from torch import nn

from ocl import base, path_defaults
from ocl.utils import RoutableMixin


def cnn_compute_positions_and_flatten(features: torch.Tensor):
    """Flatten output image CNN output and return it with positions of the features."""
    # todo(hornmax): see how this works with vision transformer based architectures.
    spatial_dims = features.shape[2:]
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )
    # reorder into format (batch_size, flattened_spatial_dims, feature_dim).
    flattened = torch.permute(features.view(features.shape[:2] + (-1,)), (0, 2, 1)).contiguous()
    return positions, flattened


def transformer_compute_positions(features: torch.Tensor):
    """Compute positions for Transformer features."""
    n_tokens = features.shape[1]
    image_size = math.sqrt(n_tokens)
    image_size_int = int(image_size)
    assert (
        image_size_int == image_size
    ), "Position computation for Transformers requires square image"

    spatial_dims = (image_size_int, image_size_int)
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )

    return positions


class ImageFeatureExtractor(base.FeatureExtractor, RoutableMixin):
    """Feature extractor which operates on images.

    For these we reshape the frame dimension into the batch dimension and process the frames as
    individual images.
    """

    def __init__(self, video_path: Optional[str] = path_defaults.VIDEO):
        base.FeatureExtractor.__init__(self)
        RoutableMixin.__init__(self, {"video": video_path})

    def forward_images(self, images: torch.Tensor):
        pass

    @RoutableMixin.route
    def forward(self, video: torch.Tensor) -> base.FeatureExtractorOutput:
        ndim = video.dim()
        assert ndim == 4 or ndim == 5

        if ndim == 5:
            # Handling video data.
            bs, frames, channels, height, width = video.shape
            images = video.view(bs * frames, channels, height, width).contiguous()
        else:
            images = video

        result = self.forward_images(images)

        if len(result) == 2:
            positions, features = result
            aux_features = None
        elif len(result) == 3:
            positions, features, aux_features = result

        if ndim == 5:
            features = features.unflatten(0, (bs, frames))
            if aux_features is not None:
                aux_features = {k: f.unflatten(0, (bs, frames)) for k, f in aux_features.items()}

        return base.FeatureExtractorOutput(features, positions, aux_features)


class VitFeatureType(enum.Enum):
    BLOCK = 1
    KEY = 2
    VALUE = 3
    QUERY = 4
    CLS = 5


class VitFeatureHook:
    """Auxilliary class used to extract features from timm ViT models.

    Args:
        mode: Type of feature to extract.
        block: Number of block to extract features from. Note that this is not zero-indexed.
    """

    def __init__(self, feature_type: VitFeatureType, block: int, drop_cls_token: bool = True):
        assert isinstance(feature_type, VitFeatureType)
        self.feature_type = feature_type
        self.block = block
        self.drop_cls_token = drop_cls_token
        self.name = f"{feature_type.name.lower()}{block}"
        self.remove_handle = None  # Can be used to remove this hook from the model again

        self._features = None

    @staticmethod
    def create_hook_from_feature_level(feature_level: Union[int, str]):
        feature_level = str(feature_level)
        prefixes = ("key", "query", "value", "block", "cls")
        for prefix in prefixes:
            if feature_level.startswith(prefix):
                _, _, block = feature_level.partition(prefix)
                feature_type = VitFeatureType[prefix.upper()]
                block = int(block)
                break
        else:
            feature_type = VitFeatureType.BLOCK
            try:
                block = int(feature_level)
            except ValueError:
                raise ValueError(f"Can not interpret feature_level '{feature_level}'.")

        return VitFeatureHook(feature_type, block)

    def register_with(self, model):
        import timm

        if not isinstance(model, timm.models.vision_transformer.VisionTransformer):
            raise ValueError(
                "This hook only supports timm.models.vision_transformer.VisionTransformer."
            )

        if self.block > len(model.blocks):
            raise ValueError(
                f"Trying to extract features of block {self.block}, but model only has "
                f"{len(model.blocks)} blocks"
            )

        block = model.blocks[self.block - 1]
        if self.feature_type == VitFeatureType.BLOCK:
            self.remove_handle = block.register_forward_hook(self)
        else:
            if isinstance(block, timm.models.vision_transformer.ParallelBlock):
                raise ValueError(
                    f"ViT with `ParallelBlock` not supported for {self.feature_type} extraction."
                )
            self.remove_handle = block.attn.qkv.register_forward_hook(self)

        return self

    def pop(self) -> torch.Tensor:
        """Remove and return extracted feature from this hook.

        We only allow access to the features this way to not have any lingering references to them.
        """
        assert self._features is not None, "Feature extractor was not called yet!"
        features = self._features
        self._features = None
        return features

    def __call__(self, module, inp, outp):
        if self.feature_type == VitFeatureType.BLOCK:
            features = outp
            if self.drop_cls_token:
                # First token is CLS token.
                features = features[:, 1:]
        elif self.feature_type in {VitFeatureType.KEY, VitFeatureType.QUERY, VitFeatureType.VALUE}:
            # This part is adapted from the timm implementation. Unfortunately, there is no more
            # elegant way to access keys, values, or queries.
            B, N, C = inp[0].shape
            qkv = outp.reshape(B, N, 3, C)  # outp has shape B, N, 3 * H * (C // H)
            q, k, v = qkv.unbind(2)

            if self.feature_type == VitFeatureType.QUERY:
                features = q
            elif self.feature_type == VitFeatureType.KEY:
                features = k
            else:
                features = v
            if self.drop_cls_token:
                # First token is CLS token.
                features = features[:, 1:]
        elif self.feature_type == VitFeatureType.CLS:
            # We ignore self.drop_cls_token in this case as it doesn't make any sense.
            features = outp[:, 0]  # Only get class token.
        else:
            raise ValueError("Invalid VitFeatureType provided.")

        self._features = features


class TimmFeatureExtractor(ImageFeatureExtractor):
    """Feature extractor implementation for timm models.

    Args:
        model_name: Name of model. See `timm.list_models("*")` for available options.
        feature_level: Level of features to return. For CNN-based models, a single integer. For ViT
            models, either a single or a list of feature descriptors. If a list is passed, multiple
            levels of features are extracted and concatenated. A ViT feature descriptor consists of
            the type of feature to extract, followed by an integer indicating the ViT block whose
            features to use. The type of features can be one of "block", "key", "query", "value",
            specifying that the block's output, attention keys, query or value should be used. If
            omitted, assumes "block" as the type. Example: "block1" or ["block1", "value2"].
        aux_features: Features to store as auxilliary features. The format is the same as in the
            `feature_level` argument. Features are stored as a dictionary, using their string
            representation (e.g. "block1") as the key. Only valid for ViT models.
        pretrained: Whether to load pretrained weights.
        freeze: Whether the weights of the feature extractor should be trainable.
        n_blocks_to_unfreeze: Number of blocks that should be trainable, beginning from the last
            block.
        unfreeze_attention: Whether weights of ViT attention layers should be trainable (only valid
            for ViT models). According to http://arxiv.org/abs/2203.09795, finetuning attention
            layers only can yield better results in some cases, while being slightly cheaper in terms
            of computation and memory.
    """

    def __init__(
        self,
        model_name: str,
        feature_level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        aux_features: Optional[Union[int, str, List[Union[int, str]]]] = None,
        pretrained: bool = False,
        freeze: bool = False,
        n_blocks_to_unfreeze: int = 0,
        unfreeze_attention: bool = False,
        video_path: Optional[str] = path_defaults.VIDEO,
    ):
        super().__init__(video_path)
        try:
            import timm
        except ImportError:
            raise Exception("Using timm models requires installation with extra `timm`.")

        register_custom_timm_models()

        self.is_vit = model_name.startswith("vit")

        def feature_level_to_list(feature_level):
            if feature_level is None:
                return []
            elif isinstance(feature_level, (int, str)):
                return [feature_level]
            else:
                return list(feature_level)

        self.feature_levels = feature_level_to_list(feature_level)
        self.aux_features = feature_level_to_list(aux_features)

        if self.is_vit:
            model = timm.create_model(model_name, pretrained=pretrained)
            if hasattr(model, "head"):
                # Classification head adds unused parameters during training, so delete it
                del model.head

            if len(self.feature_levels) > 0 or len(self.aux_features) > 0:
                self._feature_hooks = [
                    VitFeatureHook.create_hook_from_feature_level(level).register_with(model)
                    for level in itertools.chain(self.feature_levels, self.aux_features)
                ]
                if len(self.feature_levels) > 0:
                    feature_dim = model.num_features * len(self.feature_levels)

                    # Remove modules not needed in computation of features
                    max_block = max(hook.block for hook in self._feature_hooks)
                    new_blocks = model.blocks[:max_block]  # Creates a copy
                    del model.blocks
                    model.blocks = new_blocks
                    model.norm = nn.Identity()
                else:
                    feature_dim = model.num_features
            else:
                self._feature_hooks = None
                feature_dim = model.num_features
        else:
            if len(self.feature_levels) == 0:
                raise ValueError(
                    f"Feature extractor {model_name} requires specifying `feature_level`"
                )
            elif len(self.feature_levels) != 1:
                raise ValueError(
                    f"Feature extractor {model_name} only supports a single `feature_level`"
                )
            elif not isinstance(self.feature_levels[0], int):
                raise ValueError("`feature_level` needs to be an integer")

            if len(self.aux_features) > 0:
                raise ValueError("`aux_features` not supported by feature extractor {model_name}")

            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=self.feature_levels,
            )
            feature_dim = model.feature_info.channels()[0]

        self.model = model
        self._feature_dim = feature_dim

        if freeze:
            self.model.requires_grad_(False)

        if n_blocks_to_unfreeze > 0:
            if not self.is_vit:
                raise NotImplementedError(
                    "`unfreeze_n_blocks` option only implemented for ViT models"
                )
            self.model.blocks[-n_blocks_to_unfreeze:].requires_grad_(True)
            if self.model.norm is not None:
                self.model.norm.requires_grad_(True)

        if unfreeze_attention:
            if not self.is_vit:
                raise ValueError("`unfreeze_attention` option only works with ViT models")
            for module in self.model.modules():
                if isinstance(module, timm.models.vision_transformer.Attention):
                    module.requires_grad_(True)

    @property
    def feature_dim(self):
        return self._feature_dim

    def forward_images(self, images: torch.Tensor):
        if self.is_vit:
            features = self.model.forward_features(images)
            if self._feature_hooks is not None:
                hook_features = [hook.pop() for hook in self._feature_hooks]

            if len(self.feature_levels) == 0:
                # Remove class token when not using hooks.
                features = features[:, 1:]
                positions = transformer_compute_positions(features)
            else:
                features = hook_features[: len(self.feature_levels)]
                positions = transformer_compute_positions(features[0])
                features = torch.cat(features, dim=-1)

            if len(self.aux_features) > 0:
                aux_hooks = self._feature_hooks[len(self.feature_levels) :]
                aux_features = hook_features[len(self.feature_levels) :]
                aux_features = {hook.name: feat for hook, feat in zip(aux_hooks, aux_features)}
            else:
                aux_features = None
        else:
            features = self.model(images)[0]
            positions, features = cnn_compute_positions_and_flatten(features)
            aux_features = None

        return positions, features, aux_features


class SlotAttentionFeatureExtractor(ImageFeatureExtractor):
    """Feature extractor as used in slot attention paper."""

    def __init__(self, video_path: Optional[str] = path_defaults.VIDEO):
        super().__init__(video_path)
        self.layers = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
        )

    @property
    def feature_dim(self):
        return 64

    def forward_images(self, images: torch.Tensor):
        features = self.layers(images)
        positions, flattened = cnn_compute_positions_and_flatten(features)
        return positions, flattened


class SAViFeatureExtractor(ImageFeatureExtractor):
    """Feature extractor as used in the slot attention for video paper."""

    def __init__(self, larger_input_arch=False, video_path: Optional[str] = path_defaults.VIDEO):
        """Feature extractor as used in the slot attention for video paper.

        Args:
            larger_input_arch: Use the architecture for larger image datasets such as MOVi++, which
                contains more a stride in the first layer and a higher number of feature channels in
                the CNN backbone.
            video_path: Path of input video or also image.
        """
        super().__init__()
        self.larger_input_arch = larger_input_arch
        if larger_input_arch:
            self.layers = nn.Sequential(
                nn.Conv2d(3, out_channels=64, kernel_size=5, stride=2, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(3, out_channels=32, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels=32, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels=32, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels=32, kernel_size=5, padding="same"),
            )

    @property
    def feature_dim(self):
        return 64 if self.larger_input_arch else 32

    def forward_images(self, images: torch.Tensor):
        features = self.layers(images)
        positions, flattened = cnn_compute_positions_and_flatten(features)
        return positions, flattened


def register_custom_timm_models():
    import timm

    @timm.models.registry.register_model
    def resnet34_savi(pretrained=False, **kwargs):
        """ResNet34 as used in SAVi and SAVi++.

        As of now, no official code including the ResNet was released, so we can only guess which of
        the numerous ResNet variants was used. This modifies the basic timm ResNet34 to have 1x1
        strides in the stem, and replaces batch norm with group norm. Gives 16x16 feature maps with
        an input size of 224x224.

        From SAVi:
        > For the modified SAVi (ResNet) model on MOVi++, we replace the convolutional backbone [...]
        > with a ResNet-34 backbone. We use a modified ResNet root block without strides
        > (i.e. 1×1 stride), resulting in 16×16 feature maps after the backbone [w. 128x128 images].
        > We further use group normalization throughout the ResNet backbone.

        From SAVi++:
        > We used a ResNet-34 backbone with modified root convolutional layer that has 1×1 stride.
        > For all layers, we replaced the batch normalization operation by group normalization.
        """
        if pretrained:
            raise ValueError("No pretrained weights available for `savi_resnet34`.")
        from timm.models import layers, resnet

        model_args = dict(
            block=resnet.BasicBlock, layers=[3, 4, 6, 3], norm_layer=layers.GroupNorm, **kwargs
        )
        model = resnet._create_resnet("resnet34", pretrained=pretrained, **model_args)
        model.conv1.stride = (1, 1)
        model.maxpool.stride = (1, 1)
        return model
