from typing import Generator
from typing import Sequence
from jaxtyping import Array
from jax.random import KeyArray

from itertools import count
from functools import partial

import jax
from jax import numpy as jnp

from nets.samplers.base import zipfian_weights
from nets.samplers.base import ClassificationSequenceSampler
from nets.samplers.base import QueryType


def zipfian_concentration_parameter(
  num_classes: int, concentration_multiplier: float, zipf_exponent: float
) -> Array:
  return zipfian_weights(num_classes, zipf_exponent) * concentration_multiplier


def generate_dirichlet_multinomial_sequence(
  key: KeyArray,
  num_classes: int,
  context_len: int,
  query_type: QueryType,
  concentration_multiplier: float,
  zipf_exponent: float,
) -> Array:
  """Sample a class sequence from a Dirichlet-multinomial distribution."""
  context_key, query_key = jax.random.split(key, 2)

  dirichlet_key, categorical_key = jax.random.split(context_key)
  seq_probs = jax.random.dirichlet(
    dirichlet_key,
    alpha=zipfian_concentration_parameter(
      num_classes=num_classes,
      concentration_multiplier=concentration_multiplier,
      zipf_exponent=zipf_exponent,
    ),
  )

  class_idx_seq = jax.random.choice(
    key=categorical_key,
    a=num_classes,
    shape=(context_len,),
    p=seq_probs,
  )

  if query_type == QueryType.NATURALISTIC:
    # Query generated as the other sequence elements.
    query_class = jax.random.choice(
      key=query_key,
      a=num_classes,
      replace=True,
      p=seq_probs,
    )

  else:
    # A [num_class] array of class counts in the sequence.
    class_occurrences = jnp.sum(jax.nn.one_hot(class_idx_seq, num_classes), axis=0)

    # Take query from classes either occurring or *not* occurring in the
    # context, according to a biased variant of the sequence-level class
    # distribution. This is not implemented as uniform sampling in order
    # to approximate dataset-level class marginals.
    if query_type == QueryType.SUPPORTED:
      masked_seq_probs = jnp.where(class_occurrences > 0, seq_probs, 0)

    elif query_type == QueryType.UNSUPPORTED:
      masked_seq_probs = jnp.where(class_occurrences == 0, seq_probs, 1)

    else:
      raise ValueError(f"Unknown `query_type: {query_type}.")

    query_class = jax.random.choice(query_key, a=num_classes, p=masked_seq_probs)

  class_idx_seq = jnp.concatenate((class_idx_seq, query_class[jnp.newaxis]), axis=0)

  return class_idx_seq


# TODO(eringrant): D-M sampler doesn't have to be a classification sequence.
class DirichletMultinomialSampler(ClassificationSequenceSampler):
  """Compound Dirichlet-multinomial sequence sampler."""

  def __init__(
    self,
    *,
    context_len: int,
    query_type: QueryType,
    zipf_exponent: float,
    concentration_multiplier: float = 1.0,
    **kwargs,
  ):
    """Compound Dirichlet-multinomial sequence sampler.

    Args:
      context_len:
      query_type: A `QueryType` specifying how to generate each sequence's
        query class from (or independently of) the sequence's context elements.
      concentration_multiplier: The scalar multiplier concentration parameter
        of the Dirichlet-Multinomial compound distribution; a smaller value
        increases burstiness.
      zipf_exponent: The exponent of the power law-scaled concentration
        parameter of the Dirichlet-Multinomial compound distribution; a larger
        value increases the skew of the class marginal distribution.
    """
    dirichlet_multinomial_sampler = partial(
      generate_dirichlet_multinomial_sequence,
      context_len=context_len,
      concentration_multiplier=concentration_multiplier,
      zipf_exponent=zipf_exponent,
      query_type=query_type,
    )

    super().__init__(class_idx_sequence_sampler=dirichlet_multinomial_sampler, **kwargs)
