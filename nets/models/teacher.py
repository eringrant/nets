"""Teacher models for the canonical teacher-student setup."""
from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
from jax import Array

from nets import models


class CanonicalTeacher(eqx.Module):
  """Multi-layer perceptron over standard Normal input."""

  input_sampler: Callable
  net: eqx.Module

  def __init__(
    self: Self,
    in_features: int,
    hidden_features: tuple[int, ...],
    out_features: int,
    activation: Callable = jax.nn.relu,
    dropout_probs: tuple[float, ...] | None = None,
    init_scale: float = 1.0,
    *,
    key: Array,
  ) -> None:
    """Initialize a CanonicalTeacher."""
    super().__init__()

    del dropout_probs  # TODO(eringrant): Unused.

    @jax.jit
    def gaussian_sampler(key: Array) -> Array:
      return jax.random.normal(key, shape=(in_features,))

    self.input_sampler = gaussian_sampler
    self.net = models.MLP(
      in_features=in_features,
      hidden_features=hidden_features,
      out_features=out_features,
      activation=activation,
      init_scale=init_scale,
      key=key,
    )

  def __call__(self: Self, key: Array) -> tuple[Array, Array]:
    """Generate the input and output to this teacher."""
    input_key, net_key = jax.random.split(key, 2)

    x = self.input_sampler(input_key)
    y = self.net(x, key=net_key)

    return x, y
