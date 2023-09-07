"""Implementations of perceptual grouping algorithms."""
import math
from typing import Dict, Optional

import torch
from torch import nn

from ocl import base, path_defaults
from ocl.utils import RoutableMixin
from ocl.predictor import TransformerBlock

class SlotAttention(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp

    def step(self, slots, k, v, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            # Masked slots should not take part in the competition for features. By replacing their
            # dot-products with -inf, their attention values will become zero within the softmax.
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks)
        return slots, attn

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)

        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)

        return slots, attn, v


class SlotAttentionGrouping(base.PerceptualGrouping, RoutableMixin):
    """Implementation of SlotAttention for perceptual grouping.

    Args:
        feature_dim: Dimensionality of features to slot attention (after positional encoding).
        object_dim: Dimensionality of slots.
        kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
            `object_dim` is used.
        n_heads: Number of heads slot attention uses.
        iters: Number of slot attention iterations.
        eps: Epsilon in slot attention.
        ff_mlp: Optional module applied slot-wise after GRU update.
        positional_embedding: Optional module applied to the features before slot attention, adding
            positional encoding.
        use_projection_bias: Whether to use biases in key, value, query projections.
        use_implicit_differentiation: Whether to use implicit differentiation trick. If true,
            performs one more iteration of slot attention that is used for the gradient step after
            `iters` iterations of slot attention without gradients. Faster and more memory efficient
            than the standard version, but can not backpropagate gradients to the conditioning input.
        input_dim: Dimensionality of features before positional encoding is applied. Specifying this
            is optional but can be convenient to structure configurations.
    """

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        input_dim: Optional[int] = None,
        feature_path: Optional[str] = path_defaults.FEATURES,
        conditioning_path: Optional[str] = path_defaults.CONDITIONING,
        slot_mask_path: Optional[str] = None,
    ):
        base.PerceptualGrouping.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "extracted_features": feature_path,
                "conditioning": conditioning_path,
                "slot_masks": slot_mask_path,
            },
        )

        self._object_dim = object_dim
        self.slot_attention = SlotAttention(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            if slot_mask_path is None:
                raise ValueError("Need `slot_mask_path` for `use_empty_slot_for_masked_slots`")
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

    @property
    def object_dim(self):
        return self._object_dim

    @RoutableMixin.route
    def forward(
        self,
        extracted_features: base.FeatureExtractorOutput,
        conditioning: base.ConditioningOutput,
        slot_masks: Optional[torch.Tensor] = None,
    ):
        if self.positional_embedding:
            features = self.positional_embedding(
                extracted_features.features, extracted_features.positions
            )
        else:
            features = extracted_features.features
        slots, attn, v = self.slot_attention(features, conditioning, slot_masks)

        if slot_masks is not None and self.empty_slot is not None:
            slots[slot_masks] = self.empty_slot.to(dtype=slots.dtype)

        return base.PerceptualGroupingOutput(slots, feature_attributions=attn, is_empty=slot_masks, value=v)


class StickBreakingGrouping(base.Conditioning, RoutableMixin):
    """Perceptual grouping based on a stick-breaking process.

    The idea is to pick a random feature from a yet unexplained part of the feature map, then see
    which parts of the feature map are "explained" by this feature using a kernel distance. This
    process is iterated until some termination criterion is reached. In principle, this process
    allows to extract a variable number of slots per image.

    This is based on Engelcke et al, GENESIS-V2: Inferring Unordered Object Representations without
    Iterative Refinement, http://arxiv.org/abs/2104.09958. Our implementation here differs a bit from
    the one described there:

    - It only implements one kernel distance, the Gaussian kernel
    - It does not take features positions into account when computing the kernel distances
    - It L2-normalises the input features to get comparable scales of the kernel distance
    - It has multiple termination criteria, namely termination based on fraction explained, mean
      mask value, and min-max mask value. GENESIS-V2 implements termination based on mean mask
      value, but does not mention it in the paper. Note that by default, all termination criteria
      are disabled.

    Args:
        object_dim: Dimensionality of extracted slots.
        feature_dim: Dimensionality of features to operate on.
        n_slots: Maximum number of slots.
        kernel_var: Variance in Gaussian kernel.
        learn_kernel_var: Whether kernel variance should be included as trainable parameter.
        max_unexplained: If fraction of unexplained features drops under this value, drop the slot.
        min_slot_mask: If slot mask has lower average value than this value, drop the slot.
        min_max_mask_value: If slot mask's maximum value is lower than this value, drop the slot.
        early_termination: If true, all slots after the first dropped slot are also dropped.
        add_unexplained: If true, add a slot that covers all unexplained parts at the point when the
            first slot was dropped.
        eps: Minimum value for masks.
        detach_features: If true, detach input features such that no gradient flows through this
            operation.
    """

    def __init__(
        self,
        object_dim: int,
        feature_dim: int,
        n_slots: int,
        kernel_var: float = 1.0,
        learn_kernel_var: bool = False,
        max_unexplained: float = 0.0,
        min_slot_mask: float = 0.0,
        min_max_mask_value: float = 0.0,
        early_termination: bool = False,
        add_unexplained: bool = False,
        eps: float = 1e-8,
        detach_features: bool = False,
        use_input_layernorm: bool = False,
        features_path: Optional[str] = path_defaults.FEATURES,
    ):
        base.Conditioning.__init__(self)
        RoutableMixin.__init__(self, {"features": features_path})
        self.n_slots = n_slots
        self.object_dim = object_dim

        assert kernel_var > 0.0
        if learn_kernel_var:
            self.kernel_logvar = nn.Parameter(torch.tensor(math.log(kernel_var)))
        else:
            self.register_buffer("kernel_logvar", torch.tensor(math.log(kernel_var)))

        assert 0.0 <= max_unexplained < 1.0
        self.max_unexplained = max_unexplained
        assert 0.0 <= min_slot_mask < 1.0
        self.min_slot_mask = min_slot_mask
        assert 0.0 <= min_max_mask_value < 1.0
        self.min_max_mask_value = min_max_mask_value

        self.early_termination = early_termination
        self.add_unexplained = add_unexplained
        if add_unexplained and not early_termination:
            raise ValueError("`add_unexplained=True` only works with `early_termination=True`")

        self.eps = eps
        self.log_eps = math.log(eps)
        self.detach_features = detach_features

        if use_input_layernorm:
            self.in_proj = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim)
            )
            torch.nn.init.xavier_uniform_(self.in_proj[-1].weight)
            torch.nn.init.zeros_(self.in_proj[-1].bias)
        else:
            self.in_proj = nn.Linear(feature_dim, feature_dim)
            torch.nn.init.xavier_uniform_(self.in_proj.weight)
            torch.nn.init.zeros_(self.in_proj.bias)

        self.out_proj = nn.Linear(feature_dim, object_dim)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        torch.nn.init.zeros_(self.out_proj.bias)

    @RoutableMixin.route
    def forward(self, features: base.FeatureExtractorOutput) -> Dict[str, torch.Tensor]:
        features = features.features
        bs, n_features, feature_dim = features.shape
        if self.detach_features:
            features = features.detach()

        proj_features = torch.nn.functional.normalize(self.in_proj(features), dim=-1)

        # The scope keep tracks of the unexplained parts of the feature map
        log_scope = torch.zeros_like(features[:, :, 0])
        # Seeds are used for random sampling of features
        log_seeds = torch.rand_like(log_scope).clamp_min(self.eps).log()

        slot_masks = []
        log_scopes = []

        # Always iterate for `n_iters` steps for batching reasons. Termination is modeled afterwards.
        n_iters = self.n_slots - 1 if self.add_unexplained else self.n_slots
        for _ in range(n_iters):
            log_scopes.append(log_scope)

            # Sample random features from unexplained parts of the feature map
            rand_idxs = torch.argmax(log_scope + log_seeds, dim=1)
            cur_centers = proj_features.gather(
                1, rand_idxs.view(bs, 1, 1).expand(-1, -1, feature_dim)
            )

            # Compute similarity between selected features and other features. alpha can be
            # considered an attention mask.
            dists = torch.sum((cur_centers - proj_features) ** 2, dim=-1)
            log_alpha = (-dists / self.kernel_logvar.exp()).clamp_min(self.log_eps)

            # To get the slot mask, we subtract already explained parts from alpha using the scope
            mask = (log_scope + log_alpha).exp()
            slot_masks.append(mask)

            # Update scope by masking out parts explained by the current iteration
            log_1m_alpha = (1 - log_alpha.exp()).clamp_min(self.eps).log()
            log_scope = log_scope + log_1m_alpha

        if self.add_unexplained:
            slot_masks.append(log_scope.exp())
            log_scopes.append(log_scope)

        slot_masks = torch.stack(slot_masks, dim=1)
        scopes = torch.stack(log_scopes, dim=1).exp()

        # Compute criteria for ignoring slots
        empty_slots = torch.zeros_like(slot_masks[:, :, 0], dtype=torch.bool)
        # When fraction of unexplained features drops under threshold, ignore slot,
        empty_slots |= scopes.mean(dim=-1) < self.max_unexplained
        # or when slot's mean mask is under threshold, ignore slot,
        empty_slots |= slot_masks.mean(dim=-1) < self.min_slot_mask
        # or when slot's masks maximum value is under threshold, ignore slot.
        empty_slots |= slot_masks.max(dim=-1).values < self.min_max_mask_value

        if self.early_termination:
            # Simulate early termination by marking all slots after the first empty slot as empty
            empty_slots = torch.cummax(empty_slots, dim=1).values
            if self.add_unexplained:
                # After termination, add one more slot using the unexplained parts at that point
                first_empty = torch.argmax(empty_slots.to(torch.int32), dim=1).unsqueeze(-1)
                empty_slots.scatter_(1, first_empty, torch.zeros_like(first_empty, dtype=torch.bool))

                idxs = first_empty.view(bs, 1, 1).expand(-1, -1, n_features)
                unexplained = scopes.gather(1, idxs)
                slot_masks.scatter_(1, idxs, unexplained)

        # Create slot representations as weighted average of feature map
        slots = torch.einsum("bkp,bpd->bkd", slot_masks, features)
        slots = slots / slot_masks.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        slots = self.out_proj(slots)

        # Zero-out masked slots
        slots.masked_fill_(empty_slots.view(bs, slots.shape[1], 1), 0.0)

        return base.PerceptualGroupingOutput(
            slots, feature_attributions=slot_masks, is_empty=empty_slots
        )


