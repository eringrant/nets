"""Teacher models for the canonical teacher-student setup."""
from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from nets import models
from nets.models.feedforward import StopGradient


class CanonicalTeacher(eqx.Module):
  """A canonical teacher model for the teacher-student setup."""

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
      return jax.random.normal(key, shape=(in_features,)) / jnp.sqrt(in_features)

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


class CommitteeTeacher(CanonicalTeacher):
  """A teacher model that is a committee machine."""

  def __init__(
    self: Self,
    in_features: int,
    hidden_features: int,
    activation: Callable = jax.nn.relu,
    dropout_probs: tuple[float, ...] | None = None,
    init_scale: float = 1.0,
    *,
    key: Array,
  ) -> None:
    """Initialize a CommitteeTeacher."""
    super().__init__(
      in_features=in_features,
      hidden_features=(hidden_features,),
      out_features=1,
      activation=activation,
      dropout_probs=dropout_probs,
      init_scale=init_scale,
      key=key,
    )

    # Fix last-layer weights to compute the mean of the hidden-layer activations.
    self.net = eqx.tree_at(
      lambda net: net.layers[-1].weight,
      self.net,
      StopGradient(
        jnp.ones_like(self.net.layers[-1].weight)
        / self.net.layers[-1].weight.shape[-1],
      ),
    )
    self.net = eqx.tree_at(
      lambda net: net.layers[-1].bias,
      self.net,
      StopGradient(jnp.zeros_like(self.net.layers[-1].bias)),
    )
