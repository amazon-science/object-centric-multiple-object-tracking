"""Neural network backbones."""
from typing import Callable, List, Optional, Union

import torch
from torch import nn

from ocl.utils import Residual


class ReLUSquared(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.relu(x, inplace=self.inplace) ** 2


def get_activation_fn(name: str, inplace: bool = True, leaky_relu_slope: Optional[float] = None):
    if callable(name):
        return name

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "relu_squared":
        return ReLUSquared(inplace=inplace)
    elif name == "leaky_relu":
        if leaky_relu_slope is None:
            raise ValueError("Slope of leaky ReLU was not defined")
        return nn.LeakyReLU(leaky_relu_slope, inplace=inplace)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function {name}")


def build_mlp(
    input_dim: int,
    output_dim: int,
    features: List[int],
    activation_fn: Union[str, Callable] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_layer_norm: bool = False,
    residual: bool = False,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(get_activation_fn(activation_fn))
        current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)
    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    if residual:
        return Residual(nn.Sequential(*layers))
    return nn.Sequential(*layers)


def build_two_layer_mlp(
    input_dim, output_dim, hidden_dim, initial_layer_norm: bool = False, residual: bool = False
):
    """Build a two layer MLP, with optional initial layer norm.

    Separate class as this type of construction is used very often for slot attention and
    transformers.
    """
    return build_mlp(
        input_dim, output_dim, [hidden_dim], initial_layer_norm=initial_layer_norm, residual=residual
    )


def build_transformer_encoder(
    input_dim: int,
    output_dim: int,
    n_layers: int,
    n_heads: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    activation_fn: Union[str, Callable] = "relu",
    layer_norm_eps: float = 1e-5,
    use_output_transform: bool = True,
):
    if hidden_dim is None:
        hidden_dim = 4 * input_dim

    layers = []
    for _ in range(n_layers):
        layers.append(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=activation_fn,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
                norm_first=True,
            )
        )

    if use_output_transform:
        layers.append(nn.LayerNorm(input_dim, eps=layer_norm_eps))
        output_transform = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.xavier_uniform_(output_transform.weight)
        nn.init.zeros_(output_transform.bias)
        layers.append(output_transform)

    return nn.Sequential(*layers)


def build_transformer_decoder(
    input_dim: int,
    output_dim: int,
    n_layers: int,
    n_heads: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    activation_fn: Union[str, Callable] = "relu",
    layer_norm_eps: float = 1e-5,
    return_attention_weights: bool = False,
    attention_weight_type: Union[int, str] = -1,
):
    if hidden_dim is None:
        hidden_dim = 4 * input_dim

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=input_dim,
        nhead=n_heads,
        dim_feedforward=hidden_dim,
        dropout=dropout,
        activation=activation_fn,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
        norm_first=True,
    )

    if return_attention_weights:
        return TransformerDecoder(
            decoder_layer,
            n_layers,
            return_attention_weights=True,
            attention_weight_type=attention_weight_type,
        )
    else:
        return nn.TransformerDecoder(decoder_layer, n_layers)


class TransformerDecoder(nn.TransformerDecoder):
    """Modified nn.TransformerDecoder class that returns attention weights over memory."""

    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_attention_weights=False,
        attention_weight_type: Union[int, str] = "mean",
    ):
        super(TransformerDecoder, self).__init__(decoder_layer, num_layers, norm)

        if return_attention_weights:
            self.attention_hooks = []
            for layer in self.layers:
                self.attention_hooks.append(self._prepare_layer(layer))
        else:
            self.attention_hooks = None

        if isinstance(attention_weight_type, int):
            if attention_weight_type >= num_layers or attention_weight_type < -num_layers:
                raise ValueError(
                    f"Index {attention_weight_type} exceeds number of layers {num_layers}"
                )
        elif attention_weight_type != "mean":
            raise ValueError("`weights` needs to be a number or 'mean'.")
        self.weights = attention_weight_type

    def _prepare_layer(self, layer):
        assert isinstance(layer, nn.TransformerDecoderLayer)

        def _mha_block(self, x, mem, attn_mask, key_padding_mask):
            x = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )[0]
            return self.dropout2(x)

        # Patch _mha_block method to compute attention weights
        layer._mha_block = _mha_block.__get__(layer, nn.TransformerDecoderLayer)

        class AttentionHook:
            def __init__(self):
                self._attention = None

            def pop(self) -> torch.Tensor:
                assert self._attention is not None, "Forward was not called yet!"
                attention = self._attention
                self._attention = None
                return attention

            def __call__(self, module, inp, outp):
                self._attention = outp[1]

        hook = AttentionHook()
        layer.multihead_attn.register_forward_hook(hook)
        return hook

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        if self.attention_hooks is not None:
            attentions = []
            for hook in self.attention_hooks:
                attentions.append(hook.pop())

            if self.weights == "mean":
                attentions = torch.stack(attentions, dim=-1)
                # Take mean over all layers
                attention = attentions.mean(dim=-1)
            else:
                attention = attentions[self.weights]

            return output, attention.transpose(1, 2)
        else:
            return output
