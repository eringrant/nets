"""Dirichlet-multinomial sequence sampler."""
from functools import partial
from typing import Self

import jax
from jax import Array
from jax import numpy as jnp

from nets.datasets import Dataset, DatasetSplit
from nets.samplers.base import ClassificationSequenceSampler, QueryType, zipfian_weights


def zipfian_concentration_parameter(
  num_classes: int,
  concentration_multiplier: float,
  zipf_exponent: float,
) -> Array:
  """Return a Zipfian-parameterized concentration parameter."""
  return zipfian_weights(num_classes, zipf_exponent) * concentration_multiplier


def generate_dirichlet_multinomial_sequence(
  key: Array,
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
      msg = f"Unknown `query_type: {query_type}."
      raise ValueError(msg)

    query_class = jax.random.choice(query_key, a=num_classes, p=masked_seq_probs)

  return jnp.concatenate((class_idx_seq, query_class[jnp.newaxis]), axis=0)


# TODO(eringrant): D-M sampler doesn't have to be a classification sequence.
class DirichletMultinomialSampler(ClassificationSequenceSampler):
  """Compound Dirichlet-multinomial sequence sampler."""

  def __init__(
    self: Self,
    *,
    key: Array,
    dataset: Dataset,
    class_split: DatasetSplit,
    exemplar_split: DatasetSplit,
    context_len: int,
    query_type: QueryType,
    zipf_exponent: float,
    concentration_multiplier: float = 1.0,
    relabel_sequences: bool = False,
    num_seqs: int | None = None,
  ) -> None:
    """Compound Dirichlet-multinomial sequence sampler.

    Args:
      key: A key for randomness in sampling.
      dataset: The dataset from which to sample.
      class_split: The class split of `dataset` from which to sample.
      exemplar_split: The exemplar split within each class of `dataset` from
        which to sample; this argument is currently unused!
      context_len: The length of the sequence context.
      query_type: A `QueryType` specifying how to generate each sequence's
        query class from (or independently of) the sequence's context elements.
      zipf_exponent: The exponent of the power law-scaled concentration
        parameter of the Dirichlet-Multinomial compound distribution; a larger
        value increases the skew of the class marginal distribution.
      concentration_multiplier: The scalar multiplier concentration parameter
        of the Dirichlet-Multinomial compound distribution; a smaller value
        increases burstiness.
      relabel_sequences: Whether to randomly reassign labels on a sequence-level
        basis; implements few-shot relabelling.
      num_seqs: The total number of context + query sequences to sample using
        `class_idx_sequence_sampler`. If `None`, sample an infinite sequence of
        sequences.
    """
    dirichlet_multinomial_sampler = partial(
      generate_dirichlet_multinomial_sequence,
      context_len=context_len,
      concentration_multiplier=concentration_multiplier,
      zipf_exponent=zipf_exponent,
      query_type=query_type,
    )

    super().__init__(
      key=key,
      dataset=dataset,
      class_split=class_split,
      exemplar_split=exemplar_split,
      class_idx_sequence_sampler=dirichlet_multinomial_sampler,
      relabel_sequences=relabel_sequences,
      num_seqs=num_seqs,
    )
