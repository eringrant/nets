"""Minimal implementation of a Transformer model.

Adapted from:
  - https://github.com/paganpasta/eqxvision
  - https://github.com/neelnanda-io/TransformerLens
"""
from collections.abc import Callable, Sequence
from math import prod
from typing import Self

import equinox as eqx
import equinox.nn as enn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax import Array

from nets.models.feedforward import MLP, Linear, StopGradient, trunc_normal_init


class TokenEmbed(eqx.Module):
  """Abstract class for token (example or label) embedding modules."""


class LinearTokenEmbed(enn.Linear, TokenEmbed):
  """Linear token embedding layer."""

  def __init__(
    self: Self,
    input_shape: int | Sequence[int],
    embedding_size: int,
    init_stddev: float | None = 1.0,
    *,
    trainable: bool = True,
    key: Array,
  ) -> None:
    """Initialize a linear token embedding layer."""
    if isinstance(input_shape, int):
      input_shape = (input_shape,)
    super().__init__(
      in_features=prod(input_shape),
      out_features=embedding_size,
      use_bias=False,
      key=key,
    )

    # Reinitialize weight from truncated normal distribution, reusing `key`.
    self.weight: Array = trunc_normal_init(self.weight, key=key, stddev=init_stddev)
    if not trainable:
      self.weight = StopGradient(self.weight)


class PosEmbed(eqx.Module):
  """Abstract class for positional embedding modules."""

  MAX_SEQ_LEN = 32


class LinearPosEmbed(enn.Embedding, PosEmbed):
  """Simple learned linear positional embedding."""

  def __init__(
    self: Self,
    embedding_size: int,
    *,
    key: Array,
  ) -> None:
    """Initialize a linear positional embedding."""
    super().__init__(
      num_embeddings=LinearPosEmbed.MAX_SEQ_LEN,
      embedding_size=embedding_size,
      key=key,
    )

  def __call__(self: Self, x: Array, key: Array) -> Array:
    """Pad sequence to max length for compatibility with the weight matrix."""
    del key
    pad_size = self.num_embeddings - x.size
    return super().__call__(jnp.pad(x, (0, pad_size), "constant"))


# Adapted from
# https://github.com/google-deepmind/emergent_in_context_learning/blob/eba75a4208b8927cc1e981384a2cc7e014677095/modules/embedding.py#L27-L56
class SinusoidalPosEmbed(enn.Embedding, PosEmbed):
  """Sinusoidal positional embedding."""

  embedding_size: int = eqx.static_field()

  def __init__(
    self: Self,
    embedding_size: int,
    max_time: float = 30.0,
    *,
    key: Array,
  ) -> None:
    """Initialize a sinusoidal positional embedding.

    Args:
      embedding_size: Size of each embedding vector.
      max_time: Position scaling factor.
      key: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation (keyword-only argument).
    """
    if embedding_size % 2 == 1:
      msg = "Embedding size must be even if using sinusoidal encoding."
      raise ValueError(msg)

    # Generate a sequence of positions and frequencies.
    pos = jnp.arange(SinusoidalPosEmbed.MAX_SEQ_LEN, dtype=jnp.float32)
    freqs = jnp.arange(0, embedding_size, 2, dtype=jnp.float32)
    inverse_freqs = 1.0 / (max_time ** (freqs / embedding_size))

    # Combine [seq_len] and [emb_size / 2] to [seq_len, emb_size / 2].
    pos_emb = jnp.einsum("i,j->ij", pos, inverse_freqs)

    # Concatenate sines and cosines.
    pos_emb = jnp.concatenate([jnp.sin(pos_emb), jnp.cos(pos_emb)], -1)

    super().__init__(
      num_embeddings=SinusoidalPosEmbed.MAX_SEQ_LEN,
      embedding_size=embedding_size,
      weight=pos_emb,
      key=key,
    )


class AttentionBlock(eqx.Module):
  """An attention block as used in a transformer."""

  num_heads: int
  scale: float
  causal: bool
  qkv: Linear
  attn_drop: enn.Dropout
  proj: Linear
  proj_drop: enn.Dropout

  def __init__(
    self: Self,
    dim: int,
    num_heads: int,
    qk_scale: float | None = None,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    *,
    causal: bool,
    qkv_bias: bool = False,
    key: Array,
  ) -> None:
    """Initialize an attention block.

    Args:
      dim: The feature dimensions of the input.
      num_heads: The number of attention heads.
      causal: Whether or not to use causal attention.
      qkv_bias: Whether to use bias a bias term in the query-key-value computation.
      qk_scale: Scalar multiplier for the query-value (unnormalized attention)
          computation.
      attn_drop: Dropout rate for attention.
      proj_drop: Dropout rate for projection.
      key: A `jax.random.PRNGKey` used to provide randomness for parameter
          initialisation.
    """
    super().__init__()
    keys = jrandom.split(key, 2)
    self.num_heads = num_heads
    head_dim = dim // num_heads

    # TODO(eringrant): Different default from neelnanda-io/TransformerLens.
    self.scale = qk_scale or head_dim**-0.5

    self.causal = causal

    self.qkv = Linear(
      in_features=dim,
      out_features=dim * 3,
      use_bias=qkv_bias,
      key=keys[0],
    )
    self.attn_drop = enn.Dropout(attn_drop)
    self.proj = Linear(in_features=dim, out_features=dim, key=keys[1])
    self.proj_drop = enn.Dropout(proj_drop)

  def __call__(self: Self, x: Array, key: Array) -> Sequence[Array]:
    """Apply the attention block to the input."""
    n, c = x.shape
    keys = jrandom.split(key, 2)
    qkv = jax.vmap(self.qkv)(x)
    qkv = jnp.reshape(qkv, (n, 3, self.num_heads, c // self.num_heads))
    qkv = jnp.transpose(qkv, axes=(1, 2, 0, 3))
    q, k, v = jnp.split(qkv, indices_or_sections=3)

    attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * self.scale
    if self.causal:
      mask = jnp.arange(n)[:, None] >= jnp.arange(n)[None, :]
      attn = attn * mask[None, None, :, :] + (1 - mask)[None, None, :, :] * -1e6

    attn = jnn.softmax(attn, axis=-1)
    attn = self.attn_drop(attn, key=keys[0])

    x = jnp.reshape(jnp.transpose((attn @ v), axes=(0, 2, 1, 3)), (n, c))
    x = jax.vmap(self.proj)(x)
    x = self.proj_drop(x, key=keys[1])

    return x, attn


class TransformerBlock(eqx.Module):
  """A transformer block as used in a transformer."""

  norm1: eqx.Module
  attn: AttentionBlock
  drop_path1: enn.Dropout

  norm2: eqx.Module | None
  mlp: MLP | None
  drop_path2: enn.Dropout | None

  def __init__(
    self: Self,
    dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    qk_scale: float | None = None,
    mlp_drop: float = 4.0,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    path_drop: float = 0.0,
    act: Callable = jnn.gelu,
    norm_layer: eqx.Module = enn.LayerNorm,
    *,
    causal: bool,
    qkv_bias: bool = False,
    key: Array,
  ) -> None:
    """Initialize a transformer block.

    Args:
      dim: The feature dimensions of the input.
      num_heads: The number of equal parts to split the input along the `dim`.
      causal: Whether or not to use causal attention.
      mlp_ratio: For computing hidden dimension of the `MLP` (=`dim * mlp_ratio`).
      qkv_bias: To add `bias` within the `qkv` computation.
      qk_scale: For scaling the `query` `value` computation.
      mlp_drop: Dropout rate for the `MLP`.
      attn_drop: Dropout rate for the `AttentionBlock`.
      proj_drop: Dropout rate for the `projection.
      path_drop: Dropout rate for the non-residual pathway.
      act: Activation applied on the intermediate outputs.
      norm_layer: Normalisation applied to the intermediate outputs.
      key: A `jax.random.PRNGKey` used to provide randomness for parameter
          initialisation.
    """
    super().__init__()
    keys = jrandom.split(key, 2)

    self.norm1 = norm_layer(dim) if norm_layer else enn.Identity()
    self.attn = AttentionBlock(
      dim,
      num_heads=num_heads,
      causal=causal,
      qkv_bias=qkv_bias,
      qk_scale=qk_scale,
      attn_drop=attn_drop,
      proj_drop=proj_drop,
      key=keys[0],
    )
    self.drop_path1 = enn.Dropout(path_drop) if path_drop > 0.0 else enn.Identity()

    if mlp_ratio > 0:
      self.norm2 = norm_layer(dim) if norm_layer else enn.Identity()

      hidden_dim = int(dim * mlp_ratio)
      self.mlp = MLP(
        in_features=dim,
        hidden_features=(hidden_dim, hidden_dim),
        out_features=dim,
        activation=act,
        dropout_probs=(mlp_drop, mlp_drop, mlp_drop),
        key=keys[1],
      )

      self.drop_path2 = enn.Dropout(path_drop) if path_drop > 0.0 else enn.Identity()

    else:
      self.norm2 = None
      self.mlp = None
      self.drop_path2 = None

  def __call__(self: Self, x: Array, key: Array) -> Array:
    """Apply the transformer block to the input."""
    keys = jrandom.split(key, 4)

    # Attention block.
    y = jax.vmap(self.norm1)(x)
    y, attn = self.attn(y, key=keys[0])
    x = x + self.drop_path1(y, key=keys[1])

    if self.norm2 is not None and self.mlp is not None and self.drop_path2 is not None:
      # MLP head.
      y = jax.vmap(self.norm2)(x)
      y = jax.vmap(self.mlp)(y, key=jrandom.split(keys[2], x.shape[0]))
      x = x + self.drop_path2(y, key=keys[3])

    return x


class Transformer(eqx.Module):
  """A transformer model."""

  embed_dim: int

  blocks: Sequence[AttentionBlock]
  norm: eqx.Module

  unembed: Linear
  inference: bool

  def __init__(
    self: Self,
    num_classes: int | None,
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    qk_scale: float | None = None,
    mlp_drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    path_drop_rate: float = 0.0,
    norm_layer: eqx.Module = enn.LayerNorm,
    *,
    causal: bool,
    qkv_bias: bool = True,
    key: Array,
  ) -> None:
    """Initialize a transformer model.

    Args:
      num_classes: Number of classes in the classification task, or `None` to omit
          output projection.
      embed_dim: The input embedding dimension.
      depth: Number of `TransformerBlock`s in the network.
      num_heads: Number of attention heads within each `AttentionBlock`.
      causal: Whether or not to use causal attention.
      mlp_ratio: For computing hidden dimension of the `MLP` (=`dim * mlp_ratio`).
      qkv_bias: Whether to use bias a bias term in the query-key-value computation.
      qk_scale: Scalar multiplier for the query-value computation; defaults to
         `1 / sqrt(head_dim)`.
      mlp_drop_rate: Dropout rate used within the `MLP`.
      attn_drop_rate: Dropout rate used within the `AttentionBlock`s.
      path_drop_rate: Dropout rate used within `TransformerBlock`s.
      norm_layer: Normalisation applied to the intermediate outputs.
      key: A `jax.random.PRNGKey` used to provide randomness for parameter
          initialisation.
    """
    super().__init__()
    keys = jrandom.split(key, depth + 3)

    # Switch to inference mode for evaluation via `model = eqx.tree_inference(model)`.
    self.inference = False

    # Size of the embedding and thus the residual stream.
    self.embed_dim = embed_dim

    # TODO(eringrant): Check scaling convention for path drop.
    pdr = np.linspace(0, path_drop_rate, depth)
    self.blocks = [
      TransformerBlock(
        dim=embed_dim,
        num_heads=num_heads,
        causal=causal,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        mlp_drop=mlp_drop_rate,
        attn_drop=attn_drop_rate,
        path_drop=pdr[i],
        norm_layer=norm_layer,
        key=keys[i + 2],
      )
      for i in range(depth)
    ]

    self.norm = norm_layer(embed_dim) if norm_layer else enn.Identity()

    self.unembed = (
      enn.Identity()
      if num_classes is None
      else Linear(in_features=embed_dim, out_features=num_classes, key=keys[depth + 2])
    )

  def __call__(self: Self, x: Array, key: Array) -> Array:
    """Apply the transformer to the input."""
    keys = jrandom.split(key, len(self.blocks))

    # `x` should be a sequence of embeddings.
    if len(x.shape) != 2:
      msg = f"Expected `x` to be a sequence of embeddings, but got {x.shape}."
      raise ValueError(msg)
    if x.shape[1] != self.embed_dim:
      msg = (
        f"Expected `x` to have embedding dimension {self.embed_dim}, "
        f"but got {x.shape[1]}."
      )
      raise ValueError(msg)

    # Residual stream.
    residual = x
    for key_, blk in zip(keys, self.blocks, strict=True):
      residual = blk(residual, key=key_)
    residual = jax.vmap(self.norm)(residual)

    # Unembedding.
    return jax.vmap(self.unembed)(residual)


# TODO(eringrant): Factor out `Transformer` rather than passing kwargs.
class SequenceClassifier(eqx.Module):
  """A model that ingests image-label pairs as a sequence."""

  embed_dim: int
  num_classes: int

  example_embed: TokenEmbed
  example_embed_drop: enn.Dropout
  deterministic_example_embed_drop: bool

  label_embed: TokenEmbed
  label_embed_drop: enn.Dropout

  pos_embed: PosEmbed
  pos_embed_drop: enn.Dropout

  transformer: Transformer

  unembed: Linear
  inference: bool

  def __init__(
    self: Self,
    example_shape: tuple[int],
    num_classes: int,
    embed_dim: int,
    *,
    train_embed: bool = True,
    example_embed_cls: type[TokenEmbed] = LinearTokenEmbed,
    example_embed_drop_rate: float = 0.0,
    deterministic_example_embed_drop: bool = False,
    label_embed_drop_rate: float = 0.0,
    pos_embed_cls: type[PosEmbed] = SinusoidalPosEmbed,
    pos_embed_drop_rate: float = 0.0,
    transformer_depth: int,
    transformer_num_heads: int,
    transformer_mlp_ratio: float = 4.0,
    transformer_qk_scale: float | None = None,
    transformer_mlp_drop_rate: float = 0.0,
    transformer_attn_drop_rate: float = 0.0,
    transformer_path_drop_rate: float = 0.0,
    transformer_norm_layer: eqx.Module = enn.LayerNorm,
    transformer_causal: bool,
    transformer_qkv_bias: bool = True,
    key: Array,
  ) -> None:
    """A model that ingests image-label pairs as a sequence.

    Args:
      example_shape: The shape of input examples.
      num_classes: The number of classes to predict.
      embed_dim: The dimension of the embedding, and thus the residual stream.
      train_embed: Whether or not to train the embedding layer.
      example_embed_cls: The example embedding class.
      example_embed_drop_rate: The dropout probability for example embeddings.
      deterministic_example_embed_drop: Whether to associate a deterministic
        dropout pattern with each exemplar.
      label_embed_drop_rate: The dropout probability for label embeddings.
      pos_embed_cls: The positional embedding class.
      pos_embed_drop_rate: The dropout probability for positional embeddings.
      transformer_depth: Number of `TransformerBlock`s in the network.
      transformer_num_heads: Number of attention heads within each `AttentionBlock`.
      transformer_causal: Whether or not to use causal attention.
      transformer_mlp_ratio: For computing hidden dimension of the `MLP`
        (=`dim * mlp_ratio`).
      transformer_qkv_bias: Whether to use bias a bias term in the query-key-value
        computation.
      transformer_qk_scale: Scalar multiplier for the query-value computation; defaults
        to `1 / sqrt(head_dim)`.
      transformer_mlp_drop_rate: Dropout rate used within the `MLP`.
      transformer_attn_drop_rate: Dropout rate used within the `AttentionBlock`s.
      transformer_path_drop_rate: Dropout rate used within `TransformerBlock`s.
      transformer_norm_layer: Normalisation applied to the intermediate outputs.
      key: A key for randomness in parameter initialization.
    """
    super().__init__()
    keys = jrandom.split(key, 5)

    # Switch to inference mode for evaluation via `model = eqx.tree_inference(model)`.
    self.inference = False

    # Example embeddings.
    self.embed_dim = embed_dim
    self.num_classes = num_classes

    self.example_embed = example_embed_cls(
      example_shape,
      embed_dim,
      trainable=train_embed,
      key=keys[0],
    )
    self.example_embed_drop = enn.Dropout(p=example_embed_drop_rate)
    self.deterministic_example_embed_drop = deterministic_example_embed_drop

    # Label embeddings initialized like
    # https://github.com/google-deepmind/emergent_in_context_learning/blob/eba75a4208b8927cc1e981384a2cc7e014677095/modules/embedding.py#L152-L154
    self.label_embed = LinearTokenEmbed(
      num_classes,
      embed_dim,
      trainable=train_embed,
      key=keys[1],
      init_stddev=0.02,
    )
    self.label_embed_drop = enn.Dropout(p=label_embed_drop_rate)

    # Positional embeddings
    self.pos_embed = pos_embed_cls(embed_dim, key=keys[2])
    self.pos_embed_drop = enn.Dropout(p=pos_embed_drop_rate)

    self.transformer = Transformer(
      embed_dim=embed_dim,
      num_classes=None,  # Handle output projection elsewhere.
      depth=transformer_depth,
      num_heads=transformer_num_heads,
      mlp_ratio=transformer_mlp_ratio,
      qk_scale=transformer_qk_scale,
      mlp_drop_rate=transformer_mlp_drop_rate,
      attn_drop_rate=transformer_attn_drop_rate,
      path_drop_rate=transformer_path_drop_rate,
      norm_layer=transformer_norm_layer,
      causal=transformer_causal,
      qkv_bias=transformer_qkv_bias,
      key=keys[3],
    )

    self.unembed = (
      enn.Identity()
      if num_classes is None
      else Linear(in_features=embed_dim, out_features=num_classes, key=keys[4])
    )

  def __call__(self: Self, examples: Array, labels: Array, key: Array) -> Array:
    """Process the sequence of `examples` and `labels`."""
    keys = jrandom.split(key, 3)

    num_pairs = examples.shape[0]
    if num_pairs != labels.shape[0]:
      msg = "Expected `examples` and `labels` to have the same length."
      raise ValueError(msg)

    # Example embedding.
    def deterministic_examplar_dropout(example: Array) -> Array:
      return self.example_embed_drop(
        example,
        # Hash example for determinism.
        key=jax.random.fold_in(jax.random.PRNGKey(0), example.sum()),
        inference=False,
      )

    example_embedding = jax.vmap(self.example_embed)(examples)
    if self.deterministic_example_embed_drop:
      example_embedding = jax.vmap(deterministic_examplar_dropout)(example_embedding)
    else:
      example_embedding = self.example_embed_drop(example_embedding)

    # Label embedding.
    onehot_labels = jnn.one_hot(labels, self.num_classes)
    label_embedding = self.label_embed_drop(
      jax.vmap(self.label_embed)(onehot_labels),
      key=keys[0],
    )

    # Interleave example and label embeddings, except for the final (query) label.
    if example_embedding.dtype != label_embedding.dtype:
      msg = (
        "Expected `example_embedding` and `label_embedding` to have the same "
        f"dtype, but got {example_embedding.dtype} and {label_embedding.dtype}."
      )
      raise ValueError(msg)
    tok_embedding = jnp.empty(
      (num_pairs * 2 - 1, self.embed_dim),
      dtype=example_embedding.dtype,
    )
    tok_embedding = tok_embedding.at[0::2, :].set(example_embedding)  # noqa: PD008
    tok_embedding = tok_embedding.at[1::2, :].set(  # noqa: PD008
      label_embedding[:-1, :],
    )

    # Positional embedding.
    seq_len, _ = tok_embedding.shape
    pos = jnp.arange(seq_len, dtype=int)
    pos_embedding = self.pos_embed_drop(jax.vmap(self.pos_embed)(pos), key=keys[1])

    embeddings = tok_embedding + pos_embedding
    residual = self.transformer(embeddings, key=keys[2])
    unembeddings = jax.vmap(self.unembed)(residual)

    # Discard labels predicted for labels, i.e., undo interleaving.
    return unembeddings[0::2, :]
