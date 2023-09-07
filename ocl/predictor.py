import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ocl.utils import RoutableMixin, init_fn
from typing import Callable, Optional, Tuple, Union, Any
from ocl.path_defaults import OBJECTS
DType = Any
import dataclasses
from torchtyping import TensorType
from ocl import base, path_defaults

class MLP(nn.Module):
    def __init__(self,
                 input_size: int,  # FIXME: added because or else can't instantiate submodules
                 hidden_size: int,
                 output_size: int,  # if not given, should be inputs.shape[-1] at forward
                 num_hidden_layers: int = 1,
                 activation_fn: nn.Module = nn.ReLU,
                 layernorm: Optional[str] = None,
                 activate_output: bool = False,
                 residual: bool = False,
                 weight_init=None
                 ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.activation_fn = activation_fn
        self.layernorm = layernorm
        self.activate_output = activate_output
        self.residual = residual
        self.weight_init = weight_init
        if self.layernorm == "pre":
            self.layernorm_module = nn.LayerNorm(input_size, eps=1e-6)
        elif self.layernorm == "post":
            self.layernorm_module = nn.LayerNorm(output_size, eps=1e-6)
        ## mlp
        self.model = nn.ModuleList()
        self.model.add_module("dense_mlp_0", nn.Linear(self.input_size, self.hidden_size))
        self.model.add_module("dense_mlp_0_act", self.activation_fn())
        for i in range(1, self.num_hidden_layers):
            self.model.add_module(f"den_mlp_{i}", nn.Linear(self.hidden_size, self.hidden_size))
            self.model.add_module(f"dense_mlp_{i}_act", self.activation_fn())
        self.model.add_module(f"dense_mlp_{self.num_hidden_layers}", nn.Linear(self.hidden_size, self.output_size))
        if self.activate_output:
            self.model.add_module(f"dense_mlp_{self.num_hidden_layers}_act", self.activation_fn())
        for name, module in self.model.named_children():
            if 'act' not in name:
                nn.init.xavier_uniform_(module.weight)
                # init_fn[weight_init['linear_w']](module.weight)
                # init_fn[weight_init['linear_b']](module.bias)

    def forward(self, inputs: torch.Tensor, train: bool = False) -> torch.Tensor:
        del train  # Unused

        x = inputs
        if self.layernorm == "pre":
            x = self.layernorm_module(x)
        for layer in self.model:
            x = layer(x)
        if self.residual:
            x = x + inputs
        if self.layernorm == "post":
            x = self.layernorm_module(x)
        return x




class GeneralizedDotProductAttention(nn.Module):
    """Multi-head dot-product attention with customizable normalization axis.
    This module supports logging of attention weights in a variable collection.
    """

    def __init__(self,
                 dtype: DType = torch.float32,
                 # precision: Optional[] # not used
                 epsilon: float = 1e-8,
                 inverted_attn: bool = False,
                 renormalize_keys: bool = False,
                 attn_weights_only: bool = False
                ):
        super().__init__()

        self.dtype = dtype
        self.epsilon = epsilon
        self.inverted_attn = inverted_attn
        self.renormalize_keys = renormalize_keys
        self.attn_weights_only = attn_weights_only

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                train: bool = False, **kwargs) -> torch.Tensor:
        """Computes multi-head dot-product attention given query, key, and value.
        Args:
            query: Queries with shape of `[batch..., q_num, num_heads, qk_features]`.
            key: Keys with shape of `[batch..., kv_num, num_heads, qk_features]`.
            value: Values with shape of `[batch..., kv_num, num_heads, v_features]`.
            train: Indicating whether we're training or evaluating.
            **kwargs: Additional keyword arguments are required when used as attention
                function in nn.MultiHeadDotPRoductAttention, but they will be ignored here.
        Returns:
            Output of shape `[batch..., q_num, num_heads, v_features]`.
        """
        del train # Unused.

        assert query.ndim == key.ndim == value.ndim, (
            "Queries, keys, and values must have the same rank.")
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
            "Query, key, and value batch dimensions must match.")
        assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
            "Query, key, and value num_heads dimensions must match.")
        assert key.shape[-3] == value.shape[-3], (
            "Key and value cardinality dimensions must match.")
        assert query.shape[-1] == key.shape[-1], (
            "Query and key feature dimensions must match.")

        if kwargs.get("bias") is not None:
            raise NotImplementedError(
                "Support for masked attention is not yet implemented.")

        if "dropout_rate" in kwargs:
            if kwargs["dropout_rate"] > 0.:
                raise NotImplementedError("Support for dropout is not yet implemented.")

        # Temperature normalization.
        qk_features = query.shape[-1]
        query = query / (qk_features ** 0.5) # torch.sqrt(qk_features)

        # attn.shape = (batch..., num_heads, q_num, kv_num)
        attn = torch.matmul(query.permute(0, 2, 1, 3), key.permute(0, 2, 3, 1)) # bhqd @ bhdk -> bhqk

        if self.inverted_attn:
            attention_dim = -2 # Query dim
        else:
            attention_dim = -1 # Key dim

        # Softmax normalization (by default over key dim)
        attn = torch.softmax(attn, dim=attention_dim, dtype=self.dtype)

        if self.renormalize_keys:
            # Corresponds to value aggregation via weighted mean (as opposed to sum).
            normalizer = torch.sum(attn, axis=-1, keepdim=True) + self.epsilon
            attn_n = attn / normalizer
        else:
            attn_n = attn

        if self.attn_weights_only:
            return attn_n

        # Aggregate values using a weighted sum with weights provided by `attn`
        updates = torch.einsum("bhqk,bkhd->bqhd", attn_n, value)

        return updates, attn # FIXME: return attention too, as no option for intermediate storing in module in torch.

class TransformerBlock(nn.Module, RoutableMixin):
    def __init__(self,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 qkv_size: int = 128,
                 mlp_size: int = 256,
                 pre_norm: bool = False,
                 weight_init=None,
                 # object_features_path: Optional[str] = OBJECTS,
                 ):
        nn.Module.__init__(self)
        # RoutableMixin.__init__(self, {"object_features": object_features_path})

        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.weight_init = weight_init

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "embed dim must be divisible by num_heads"
        self.head_dim = qkv_size // num_heads

        # submodules
        ## MHA #
        self.attn = GeneralizedDotProductAttention()
        ## mlps
        self.mlp = MLP(
            input_size=embed_dim, hidden_size=mlp_size,
            output_size=embed_dim, weight_init=weight_init)
        ## layernorms
        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        ## weights
        self.dense_q = nn.Linear(embed_dim, qkv_size)
        self.dense_k = nn.Linear(embed_dim, qkv_size)
        self.dense_v = nn.Linear(embed_dim, qkv_size)
        # init_fn[weight_init['linear_w']](self.dense_q.weight)
        # init_fn[weight_init['linear_b']](self.dense_q.bias)
        # init_fn[weight_init['linear_w']](self.dense_k.weight)
        # init_fn[weight_init['linear_b']](self.dense_k.bias)
        # init_fn[weight_init['linear_w']](self.dense_v.weight)
        # init_fn[weight_init['linear_b']](self.dense_v.bias)
        if self.num_heads > 1:
            self.dense_o = nn.Linear(qkv_size, embed_dim)
            # nn.init.xavier_uniform_(self.w_o.weight)
            # init_fn[weight_init['linear_w']](self.dense_o.weight)
            # init_fn[weight_init['linear_b']](self.dense_o.bias)
            self.multi_head = True
        else:
            self.multi_head = False


    # @RoutableMixin.route
    def forward(self, object_features: torch.Tensor):  # TODO: add general attention for q, k, v, not just for x = qkv
        assert object_features.ndim == 3
        B, L, _ = object_features.shape
        head_dim = self.embed_dim // self.num_heads

        if self.pre_norm:
            # Self-attention.
            x = self.layernorm_query(object_features)
            q = self.dense_q(x).view(B, L, self.num_heads, head_dim)
            k = self.dense_k(x).view(B, L, self.num_heads, head_dim)
            v = self.dense_v(x).view(B, L, self.num_heads, head_dim)
            x, _ = self.attn(query=q, key=k, value=v)
            if self.multi_head:
                x = self.dense_o(x.reshape(B, L, self.qkv_size)).view(B, L, self.embed_dim)
            else:
                x = x.squeeze(-2)
            x = x + object_features

            y = x

            # MLP
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            z = z + y
        else:
            # Self-attention on queries.
            x = object_features
            # q = self.dense_q(x).view(B, L, self.num_heads, head_dim)
            # k = self.dense_k(x).view(B, L, self.num_heads, head_dim)
            # v = self.dense_v(x).view(B, L, self.num_heads, head_dim)
            # x, _ = self.attn(query=q, key=k, value=v)
            # if self.multi_head:
            #     x = self.dense_o(x.reshape(B, L, self.qkv_size)).view(B, L, self.embed_dim)
            # else:
            #     x = x.squeeze(-2)
            # x = x + object_features
            # x = self.layernorm_query(x)

            y = x

            # MLP
            z = self.mlp(y)
            z = z + y
            z = self.layernorm_mlp(z)
        return z






@dataclasses.dataclass
class PredictorOutput:
    objects: TensorType["batch_size", "n_objects", "object_dim"]  # noqa: F821
    is_empty: Optional[TensorType["batch_size", "n_objects"]] = None  # noqa: F821
    feature_attributions: Optional[
        TensorType["batch_size", "n_objects", "n_spatial_features"]  # noqa: F821
    ] = None