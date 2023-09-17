"""Simple feedforward neural networks."""
import numpy as np
from math import sqrt

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
import equinox.nn as enn

from jaxtyping import Array
from jax.random import KeyArray
from collections.abc import Callable


def trunc_normal_init(
  weight: Array, key: KeyArray, stddev: float | None = None
) -> Array:
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
  weight: Array, key: KeyArray, scale: float | None = None
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

  array: jnp.ndarray

  def __jax_array__(self):
    """Return the array wrapped with a stop gradient op."""
    return jax.lax.stop_gradient(self.array)


class Linear(enn.Linear):
  """Linear layer."""

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
    """Initialize an MLP.

    Args:
       in_features: The expected dimension of the input.
       hidden_features: Dimensionality of the hidden layer.
       out_features: The dimension of the output feature.
       act: Activation function to be applied to the intermediate layers.
       drop: The probability associated with `Dropout`.
       key: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation.
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
    """Apply the MLP block to the input."""
    keys = jrandom.split(key, 2)
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x, key=keys[0])
    x = self.fc2(x)
    x = self.drop2(x, key=keys[1])
    return x
