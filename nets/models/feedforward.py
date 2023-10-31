"""Simple feedforward neural networks."""
from collections.abc import Callable
from math import sqrt
from typing import Self

import equinox as eqx
import equinox.nn as enn
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def trunc_normal_init(weight: Array, key: Array, stddev: float | None = None) -> Array:
  """Truncated normal distribution initialization."""
  _, in_ = weight.shape
  stddev = stddev or sqrt(1.0 / max(1.0, in_))
  return stddev * jax.random.truncated_normal(
    key=key,
    shape=weight.shape,
    lower=-2,
    upper=2,
  )


# Adapted from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/initializers.py.
def lecun_normal_init(
  weight: Array,
  key: Array,
  scale: float = 1.0,
) -> Array:
  """LeCun (variance-scaling) normal distribution initialization."""
  _, in_ = weight.shape
  scale /= max(1.0, in_)

  stddev = np.sqrt(scale)
  # Adjust stddev for truncation.
  # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
  distribution_stddev = jnp.asarray(0.87962566103423978, dtype=float)
  stddev = stddev / distribution_stddev

  return trunc_normal_init(weight, key, stddev=stddev)


class StopGradient(eqx.Module):
  """Stop gradient wrapper."""

  array: Array

  def __jax_array__(self: Self) -> Array:
    """Return the array wrapped with a stop gradient op."""
    return jax.lax.stop_gradient(self.array)


class Linear(enn.Linear):
  """Linear layer."""

  def __init__(
    self: Self,
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    trainable: bool = True,
    key: Array,
    init_scale: float = 1.0,
  ) -> None:
    """Initialize a linear layer."""
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


class MLP(eqx.Module):
  """Multi-layer perceptron."""

  layers: tuple[Linear, ...]
  dropouts: tuple[enn.Dropout, ...]
  activation: Callable
  final_activation: Callable

  def __init__(
    self: Self,
    in_features: int,
    hidden_features: tuple[int, ...],
    out_features: int,
    activation: Callable = jax.nn.relu,
    final_activation: Callable = lambda x: x,
    dropout_probs: tuple[float, ...] | None = None,
    *,
    use_bias: bool = True,
    use_final_bias: bool = True,
    key: Array,
    init_scale: float = 1.0,
  ) -> None:
    """Initialize an MLP."""
    super().__init__()
    self.activation = activation
    self.final_activation = final_activation

    # TODO(eringrant): Canonical dropout usage?
    if dropout_probs is not None:
      if len(dropout_probs) != len(hidden_features) + 1:
        msg = (
          f"Expected {len(hidden_features) + 1} dropout probabilities, "
          f"got {len(dropout_probs)}."
        )
        raise ValueError(
          msg,
        )
    else:
      dropout_probs = (0.0,) * (len(hidden_features) + 1)

    output_key, *hidden_keys = jax.random.split(key, len(hidden_features) + 1)

    layers = []
    dropouts = []

    for key, ins, outs, drop in zip(
      hidden_keys,
      (in_features,) + hidden_features[:-1],
      hidden_features,
      dropout_probs[:-1],
      strict=True,
    ):
      layers.append(
        Linear(
          in_features=ins,
          out_features=outs,
          use_bias=use_bias,
          init_scale=init_scale,
          key=key,
        ),
      )
      dropouts.append(enn.Dropout(drop))

    layers.append(
      Linear(
        in_features=hidden_features[-1],
        out_features=out_features,
        use_bias=use_final_bias,
        init_scale=init_scale,
        key=key,
      ),
    )
    dropouts.append(enn.Dropout(dropout_probs[-1]))

    self.layers = tuple(layers)
    self.dropouts = tuple(dropouts)

  def __call__(self: Self, x: Array, key: Array) -> Array:
    """Apply the MLP to an input."""
    keys = jax.random.split(key, len(self.layers))

    for key, drop, layer in zip(
      keys[:-1],
      self.dropouts[:-1],
      self.layers[:-1],
      strict=True,
    ):
      x = drop(x, key=key)
      x = layer(x)
      x = self.activation(x)

    x = self.dropouts[-1](x, key=keys[-1])
    x = self.layers[-1](x)
    return self.final_activation(x)


if __name__ == "__main__":
  MLP(
    in_features=10,
    out_features=10,
    hidden_features=(8, 4, 2),
    activation=jax.nn.relu,
    final_activation=lambda x: x,
    use_bias=True,
    use_final_bias=True,
    key=jax.random.PRNGKey(0),
  )(jnp.ones(10), key=jax.random.PRNGKey(0))
