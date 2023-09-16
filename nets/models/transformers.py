"""Minimal implementation of a Transformer model.

Adapted from:
  - https://github.com/paganpasta/eqxvision
  - https://github.com/neelnanda-io/TransformerLens
"""
import numpy as np
from math import prod
from math import sqrt

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
import equinox.nn as enn

from jaxtyping import Array
from jax.random import KeyArray
from collections.abc import Callable
from collections.abc import Sequence


def trunc_normal_init(
  weight: Array, key: KeyArray, stddev: float | None = None
) -> Array:
  _, in_ = weight.shape
  stddev = stddev or sqrt(1.0 / max(1.0, in_))
  return stddev * jax.random.truncated_normal(
    key=key,
    shape=weight.shape,
    lower=-2,
    upper=2,
  )


def lecun_normal_init(
  weight: Array, key: KeyArray, scale: float | None = None
) -> Array:
  """Adapted from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/initializers.py."""
  _, in_ = weight.shape
  scale /= max(1.0, in_)

  stddev = np.sqrt(scale)
  # Adjust stddev for truncation.
  # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
  distribution_stddev = jnp.asarray(0.87962566103423978, dtype=float)
  stddev = stddev / distribution_stddev

  return trunc_normal_init(weight, key, stddev=stddev)


class StopGradient(eqx.Module):
  array: jnp.ndarray

  def __jax_array__(self):
    return jax.lax.stop_gradient(self.array)


class Linear(enn.Linear):
  """Linear layer with variance scaling init (LeCun init)."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    use_bias: bool = True,
    trainable: bool = True,
    *,
    key: KeyArray,
    init_scale: float | None = 1.0,
  ):
    super().__init__(
      in_features=in_features,
      out_features=out_features,
      use_bias=use_bias,
      key=key,
    )

    # Reinitialize weight from variance scaling distribution, reusing `key`.
    self.weight: Array = lecun_normal_init(self.weight, key=key, scale=init_scale)
    if not trainable:
      self.weight = StopGradient(self.weight)

    # Reinitialize bias to zeros.
    if use_bias:
      self.bias: Array = jnp.zeros_like(self.bias)

      if not trainable:
        self.bias = StopGradient(self.bias)


class TokenEmbed(eqx.Module):
  """Abstract class for token (example or label) embedding modules."""

  pass


class LinearTokenEmbed(enn.Linear, TokenEmbed):
  """Linear token embedding layer."""

  def __init__(
    self,
    input_shape: int | Sequence[int],
    embedding_size: int,
    init_stddev: float | None = 1.0,
    trainable: bool = True,
    *,
    key: KeyArray,
  ):
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
    self,
    embedding_size: int,
    *,
    key: KeyArray,
  ):
    super().__init__(
      num_embeddings=LinearPosEmbed.MAX_SEQ_LEN,
      embedding_size=embedding_size,
      key=key,
    )

  def __call__(self, x: Array, *, key: KeyArray | None = None) -> Array:
    """Pad sequence to max length for compatibility with the weight matrix."""
    pad_size = self.num_embeddings - x.size
    return super().__call__(jnp.pad(x, (0, pad_size), "constant"))


class SinusoidalPosEmbed(enn.Embedding, PosEmbed):
  """Sinusoidal positional embedding.

  Reproduction of
  https://github.com/deepmind/emergent_in_context_learning/blob/main/modules/embedding.py#L27-L56.
  """

  embedding_size: int = eqx.static_field()

  def __init__(
    self,
    embedding_size: int,
    max_time: float = 30.0,
    *,
    key: KeyArray,
    **kwargs,
  ):
    """**Arguments:**
    - `embedding_size`: Size of each embedding vector.
    - `max_time`: Position scaling factor.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.).
    """
    if embedding_size % 2 == 1:
      raise ValueError("Embedding size must be even if using sinusoidal encoding.")

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


class MLPBlock(eqx.Module):
  """TODO."""

  fc1: eqx.Module
  act: Callable
  drop1: enn.Dropout
  fc2: eqx.Module
  drop2: enn.Dropout

  def __init__(
    self,
    in_features: int,
    hidden_features: int | None = None,
    out_features: int | None = None,
    act: Callable = lambda x: x,
    drop: float | tuple[float] = 0.0,
    *,
    key: KeyArray = None,
  ):
    """TODO.

    Args:
    - `in_features`: The expected dimension of the input.
    - `hidden_features`: Dimensionality of the hidden layer.
    - `out_features`: The dimension of the output feature.
    - `act`: Activation function to be applied to the intermediate layers.
    - `drop`: The probability associated with `Dropout`.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter initialisation.
    """
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    drop_probs = drop if isinstance(drop, tuple) else (drop, drop)
    keys = jrandom.split(key, 2)

    self.fc1 = Linear(
      in_features=in_features, out_features=hidden_features, key=keys[0]
    )
    self.act = act
    self.drop1 = enn.Dropout(drop_probs[0])
    self.fc2 = Linear(
      in_features=hidden_features, out_features=out_features, key=keys[1]
    )
    self.drop2 = enn.Dropout(drop_probs[1])

  def __call__(self, x: Array, *, key: KeyArray) -> Array:
    keys = jrandom.split(key, 2)
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x, key=keys[0])
    x = self.fc2(x)
    x = self.drop2(x, key=keys[1])
    return x


class AttentionBlock(eqx.Module):
  """TODO."""

  num_heads: int
  scale: float
  causal: bool
  qkv: Linear
  attn_drop: enn.Dropout
  proj: Linear
  proj_drop: enn.Dropout

  def __init__(
    self,
    dim: int,
    num_heads: int,
    causal: bool,
    qkv_bias: bool = False,
    qk_scale: float | None = None,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    *,
    key: KeyArray,
  ):
    """TODO.

    Args:
    - `dim`: The feature dimensions of the input.
    - `num_heads`: The number of attention heads.
    - `causal`: Whether or not to use causal attention.
    - `qkv_bias`: Whether to use bias a bias term in the query-key-value computation.
    - `qk_scale`: Scalar multiplier for the query-value (unnormalized attention) computation.
    - `attn_drop`: Dropout rate for attention.
    - `proj_drop`: Dropout rate for projection.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter initialisation.
    """
    super().__init__()
    keys = jrandom.split(key, 2)
    self.num_heads = num_heads
    head_dim = dim // num_heads

    # TODO(eringrant): Different default from neelnanda-io/TransformerLens.
    self.scale = qk_scale or head_dim**-0.5

    self.causal = causal

    self.qkv = Linear(
      in_features=dim, out_features=dim * 3, use_bias=qkv_bias, key=keys[0]
    )
    self.attn_drop = enn.Dropout(attn_drop)
    self.proj = Linear(in_features=dim, out_features=dim, key=keys[1])
    self.proj_drop = enn.Dropout(proj_drop)

  def __call__(self, x: Array, *, key: KeyArray) -> Sequence[Array]:
    N, C = x.shape
    keys = jrandom.split(key, 2)
    qkv = jax.vmap(self.qkv)(x)
    qkv = jnp.reshape(qkv, (N, 3, self.num_heads, C // self.num_heads))
    qkv = jnp.transpose(qkv, axes=(1, 2, 0, 3))
    q, k, v = jnp.split(qkv, indices_or_sections=3)

    attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * self.scale
    if self.causal:
      mask = jnp.arange(N)[:, None] >= jnp.arange(N)[None, :]
      attn = attn * mask[None, None, :, :] + (1 - mask)[None, None, :, :] * -1e6

    attn = jnn.softmax(attn, axis=-1)
    attn = self.attn_drop(attn, key=keys[0])

    x = jnp.reshape(jnp.transpose((attn @ v), axes=(0, 2, 1, 3)), (N, C))
    x = jax.vmap(self.proj)(x)
    x = self.proj_drop(x, key=keys[1])

    return x, attn


class TransformerBlock(eqx.Module):
  """TODO."""

  mlp_ratio: float | None

  norm1: eqx.Module
  attn: AttentionBlock
  drop_path1: enn.Dropout

  norm2: eqx.Module | None
  mlp: MLPBlock | None
  drop_path2: enn.Dropout | None

  def __init__(
    self,
    dim: int,
    num_heads: int,
    causal: bool,
    mlp_ratio: float | None = 4.0,
    qkv_bias: bool = False,
    qk_scale: float | None = None,
    mlp_drop: float = 0.0,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    path_drop: float = 0.0,
    act: Callable = jnn.gelu,
    norm_layer: eqx.Module = enn.LayerNorm,
    *,
    key: KeyArray,
  ) -> None:
    """TODO.

    Args:
    - `dim`: The feature dimensions of the input.
    - `num_heads`: The number of equal parts to split the input along the `dim`.
    - `causal`: Whether or not to use causal attention.
    - `mlp_ratio`: For computing hidden dimension of the `MLPBlock` (=`dim * mlp_ratio`).
    - `qkv_bias`: To add `bias` within the `qkv` computation.
    - `qk_scale`: For scaling the `query` `value` computation.
    - `mlp_drop`: Dropout rate for the `MLPBlock`.
    - `attn_drop`: Dropout rate for the `AttentionBlock`.
    - `proj_drop`: Dropout rate for the `projection.
    - `path_drop`: Dropout rate for the non-residual pathway.
    - `act`: Activation applied on the intermediate outputs.
    - `norm_layer`: Normalisation applied to the intermediate outputs.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter initialisation.
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

    self.mlp_ratio = mlp_ratio
    if self.mlp_ratio is not None:
      self.norm2 = norm_layer(dim) if norm_layer else enn.Identity()
      self.mlp = MLPBlock(
        in_features=dim,
        hidden_features=int(dim * mlp_ratio),
        act=act,
        drop=mlp_drop,
        key=keys[1],
      )
      self.drop_path2 = enn.Dropout(path_drop) if path_drop > 0.0 else enn.Identity()

    else:
      self.norm2 = None
      self.mlp = None
      self.drop_path2 = None

  def __call__(self, x: Array, *, key: KeyArray) -> Array:
    keys = jrandom.split(key, 4)

    # Attention block.
    y = jax.vmap(self.norm1)(x)
    y, attn = self.attn(y, key=keys[0])
    x = x + self.drop_path1(y, key=keys[1])

    if self.mlp_ratio is not None:
      # MLP head.
      y = jax.vmap(self.norm2)(x)
      y = jax.vmap(self.mlp)(y, key=jrandom.split(keys[2], x.shape[0]))
      x = x + self.drop_path2(y, key=keys[3])

    return x


class Transformer(eqx.Module):
  """TODO."""

  embed_dim: int

  blocks: Sequence[AttentionBlock]
  norm: eqx.Module

  unembed: Linear
  inference: bool

  def __init__(
    self,
    num_classes: int | None,
    embed_dim: int,
    depth: int,
    num_heads: int,
    causal: bool,
    mlp_ratio: float | None = 4.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    tok_embed_drop_rate: float = 0.0,
    pos_embed_drop_rate: float = 0.0,
    mlp_drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    path_drop_rate: float = 0.0,
    norm_layer: eqx.Module = enn.LayerNorm,
    *,
    key: KeyArray,
  ) -> None:
    """TODO:

    Args:
    - `num_classes`: Number of classes in the classification task, or `None` to omit output projection.
    - `embed_dim`: The input embedding dimension.
    - `depth`: Number of `TransformerBlock`s in the network.
    - `num_heads`: Number of attention heads within each `AttentionBlock`.
    - `causal`: Whether or not to use causal attention.
    - `mlp_ratio`: For computing hidden dimension of the `MLPBlock`s, or `None` to omit MLP heads.
    - `qkv_bias`: Whether to use bias a bias term in the query-key-value computation.
    - `qk_scale`: Scalar multiplier for the query-value computation; defaults to `1 / sqrt(head_dim)`.
    - `embed_drop_rate`: Dropout rate used for the embedding matrix.
    - `mlp_drop_rate`: Dropout rate used within the `MLPBlock`.
    - `attn_drop_rate`: Dropout rate used within the `AttentionBlock`s.
    - `path_drop_rate`: Dropout rate used within `TransformerBlock`s.
    - `norm_layer`: Normalisation applied to the intermediate outputs.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter initialisation.
    """
    super().__init__()
    keys = jrandom.split(key, depth + 3)

    # Switch to `inference` mode for evaluations via `model = eqx.tree_inference(model)`.
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

  def __call__(self, x: Array, *, key: KeyArray) -> Array:
    keys = jrandom.split(key, len(self.blocks))

    # `x` should be a sequence of embeddings.
    assert len(x.shape) == 2 and x.shape[1] == self.embed_dim

    # Residual stream
    residual = x
    for key_, blk in zip(keys, self.blocks):
      residual = blk(residual, key=key_)
    residual = jax.vmap(self.norm)(residual)

    # Unembedding.
    return jax.vmap(self.unembed)(residual)


# TODO(eringrant): Factorize `Transformer` rather than passing kwargs.
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
    self,
    example_shape: tuple[int],
    num_classes: int,
    embed_dim: int,
    train_embed: bool = True,
    example_embed_cls: type[TokenEmbed] = LinearTokenEmbed,
    example_embed_drop_rate: float = 0.0,
    deterministic_example_embed_drop: bool = False,
    label_embed_drop_rate: float = 0.0,
    pos_embed_cls: type[PosEmbed] = SinusoidalPosEmbed,
    pos_embed_drop_rate: float = 0.0,
    *,
    key: KeyArray,
    **transformer_kwargs,
  ):
    """A model that ingests image-label pairs as a sequence.

    Args:
      - `example_shape`: The shape of input examples.
      - `num_classes`: The number of classes to predict.
      - `embed_dim`: The dimension of the embedding, and thus the residual stream.
      - `train_embed`: Whether or not to train the embedding layer.
      - `example_embed_cls`: The example embedding class.
      - `example_embed_drop_rate`: The dropout probability for example embeddings.
      - `deterministic_example_embed_drop`: Whether to associate a deterministic
        dropout pattern with each exemplar.
      - `pos_embed_cls`: The positional embedding class.
      - `pos_embed_drop_rate`: The dropout probability for positional embeddings.
      - `key`: A key for randomness in parameter initialization.
      - `transformer_kwargs`: Remaining keyword arguments to be passed to the wrapped
        Transformer.
    """
    super().__init__()
    keys = jrandom.split(key, 5)

    # Switch to `inference` mode for evaluations via `model = eqx.tree_inference(model)`.
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
    # https://github.com/deepmind/emergent_in_context_learning/blob/main/modules/embedding.py#L152-L154.
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
      key=keys[3],
      **transformer_kwargs,
    )

    self.unembed = (
      enn.Identity()
      if num_classes is None
      else Linear(in_features=embed_dim, out_features=num_classes, key=keys[4])
    )

  def __call__(self, examples: Array, labels: Array, *, key: KeyArray) -> Array:
    """Process the sequence of `examples` and `labels`."""
    keys = jrandom.split(key, 3)

    num_pairs = examples.shape[0]
    assert num_pairs == labels.shape[0]

    # Example embedding.
    def deterministic_examplar_dropout(example):
      return self.example_embed_drop(
        example,
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
      jax.vmap(self.label_embed)(onehot_labels), key=keys[0]
    )

    # Interleave example and label embeddings, except for the final (query) label.
    assert example_embedding.dtype == label_embedding.dtype
    tok_embedding = jnp.empty(
      (num_pairs * 2 - 1, self.embed_dim), dtype=example_embedding.dtype
    )
    tok_embedding = tok_embedding.at[0::2, :].set(example_embedding)
    tok_embedding = tok_embedding.at[1::2, :].set(label_embedding[:-1, :])

    # Positional embedding.
    seq_len, _ = tok_embedding.shape
    pos = jnp.arange(seq_len, dtype=int)
    pos_embedding = self.pos_embed_drop(jax.vmap(self.pos_embed)(pos), key=keys[1])

    embeddings = tok_embedding + pos_embedding
    residual = self.transformer(embeddings, key=keys[2])
    unembeddings = jax.vmap(self.unembed)(residual)

    # Discard labels predicted for labels, i.e., undo interleaving.
    predictions = unembeddings[0::2, :]

    return predictions
