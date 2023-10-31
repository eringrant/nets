"""`Sampler`s are sequences of samples from a `Dataset`."""
from __future__ import annotations

import copy
from collections.abc import Sequence
from enum import Enum, unique
from functools import partial
from typing import TYPE_CHECKING, Self

import jax
from jax import Array
from jax import nn as jnn
from jax import numpy as jnp
from typing_extensions import Protocol

from nets.datasets import Dataset, DatasetSplit

if TYPE_CHECKING:
  from nets.datasets.base import ExemplarType

# The initial number of sequences to instantiate within an infinite `Sampler`.
# Changing this parameter impacts PRNGs for sequence sampling.
# TODO(eringrant): Change to use `jax.random.fold_in`.
MAX_NUM_SEQS = int(1e7)


@unique
class QueryType(Enum):
  """Types of queries that can be generated from a `Sampler`."""

  # Generate the query in the same manner as the context.
  NATURALISTIC = 1
  # Generate a query from classes in the context.
  SUPPORTED = 2
  # Generate a query from classes that do not occur in the context.
  UNSUPPORTED = 3


def zipfian_weights(num_classes: int, zipf_exponent: float) -> Array:
  """Compute Zipfian weights for `num_classes` classes."""
  return jnp.exp(-zipf_exponent * jnp.log(jnp.arange(1, num_classes + 1)))


def zipfian_distribution(num_classes: int, zipf_exponent: float) -> Array:
  """Compute Zipfian distribution for `num_classes` classes."""
  weights = zipfian_weights(num_classes, zipf_exponent)
  return weights / jnp.sum(weights)


def generate_exemplar_idx_sequence(
  key: Array,
  label_seq: Array,
  dataset_labels: Array,
) -> Array:
  """Generate a sequence of exemplar indices.

  Args:
    key: A key for randomness in sampling.
    label_seq: A sequence of `context_len` class labels for which to sample
      exemplars.
    dataset_labels: Class labels corresponding to the dataset of exemplars
      from which to sample.

  Returns:
    An array of indices into `dataset_labels` corresponding to sampled
    exemplars.
  """
  # Identify valid class exemplars for each element of the sequence
  # via a [context_len + 1, dataset_len] mask.
  exemplar_mask_seq = jnn.one_hot(
    dataset_labels,
    num_classes=dataset_labels.max(),
    dtype=jnp.int_,
  ).T[label_seq]

  @partial(jax.vmap, in_axes=0)
  def _sample_exemplar(key: Array, p: float) -> Array:
    # Note: This samples exemplars independently, and thus with replacement,
    # which gives the possibility of duplicate exemplars in a sequence.
    return jax.random.choice(key, jnp.arange(len(dataset_labels)), shape=(), p=p)

  # Sample an exemplar per class element in the sequence.
  return _sample_exemplar(
    jax.random.split(key, len(exemplar_mask_seq)),
    jax.nn.softmax(jnp.log(exemplar_mask_seq)),
  )

  # A [context_len + 1] array of indices into `dataset_labels`,
  # corresponding to sampled exemplars.


class ClassSampler(Protocol):
  """Protocol for a function that generates a sequence of class indices."""

  def __call__(self: Self, key: Array, num_classes: int) -> Array:
    """Generate a sequence of class indices."""
    ...


class ExemplarSampler(Protocol):
  """Protocol for a function that generates a sequence of exemplar indices."""

  def __call__(
    self: Self,
    key: Array,
    label_seq: Array,
    dataset_labels: Array,
  ) -> Array:
    """Generate a sequence of exemplar indices."""
    ...


def generate_sequence(
  key: Array,
  dataset_labels: Array,
  classes_to_sample: Array,
  generate_class_idx_sequence_fn: ClassSampler,
  generate_exemplar_idx_sequence_fn: ExemplarSampler,
) -> Array:
  """Generate a sequence of examples.

  Args:
    key: A key for randomness in sampling.
    dataset_labels: Class labels corresponding to the dataset of exemplars
      from which to sample; the ith element of `dataset_labels` is the class
      label of the ith example in the dataset.
    classes_to_sample: The subset of class labels in `dataset_labels` to be
      sampled, perhaps corresponding to a class split.
    generate_class_idx_sequence_fn: A function that generates a sequence of
      class indices.
    generate_exemplar_idx_sequence_fn: A function that generates a sequence of
      exemplar indices.

  Returns:
    An array of indices into `dataset_labels` corresponding to sampled
    exemplars.
  """
  class_key, exemplar_key = jax.random.split(key, 2)

  # Sample class indices.
  class_idx_seq = generate_class_idx_sequence_fn(
    key=class_key,
    num_classes=classes_to_sample.size,
  )

  # Translate class indices into class labels.
  label_seq = jnp.take(classes_to_sample, class_idx_seq)

  # Sample exemplar indices.
  return generate_exemplar_idx_sequence_fn(
    key=exemplar_key,
    label_seq=label_seq,
    dataset_labels=dataset_labels,
  )


class Sampler(Sequence):
  """Sampler of sequences drawn from a `nets.datasets.Dataset`."""


class SingletonSampler(Sampler):
  """Sampler of a sequence of examples."""


# TODO(eringrant): Is this faster than a recursive call to `__getitem__`?
def slice_to_array(s: slice, array_length: int) -> Array:
  """Convert a `slice` object to an array of indices."""
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else array_length
  step = s.step if s.step is not None else 1

  return jnp.array(range(start, stop, step))


class EpochSampler(SingletonSampler):
  """Sampler of example-label pairs over multiple epochs."""

  def __init__(
    self: Self,
    key: Array,
    dataset: Dataset,
    num_epochs: int | None = None,
  ) -> None:
    """Sampler of example-label pairs over multiple epochs."""
    self.key = key
    self.dataset = dataset
    self.num_epochs = num_epochs
    self.epoch_count = 0
    self.index_in_epoch = 0

    self.dataset_size = len(self.dataset)

  def __len__(self: Self) -> int:
    """Return the number of example-label pairs in `Sampler`."""
    if self.num_epochs is None:
      return int(float("inf"))  # Infinite length if num_epochs is not set
    return self.num_epochs * self.dataset_size

  def __getitem__(self: Self, index: int | slice) -> ExemplarType:
    """Return exemplar-class pairs at index `index` of `Sampler`."""
    # TODO(eringrant): Simplify this while maintaining type-validity.
    if isinstance(index, slice):
      transformed_index = slice_to_array(index, len(self))
    else:
      transformed_index = index

    epoch_idx = transformed_index // self.dataset_size
    if not isinstance(epoch_idx, int):
      unique_vals = jnp.unique(epoch_idx)
      if unique_vals.size != 1:
        # TODO(eringrant): Implement this case.
        msg = "Array should contain only one unique value."
        raise ValueError(msg)
      epoch_idx = unique_vals[0]
    index_in_epoch = transformed_index % self.dataset_size

    if self.num_epochs is not None and epoch_idx >= self.num_epochs:
      msg = "Reached the end of data generation."
      raise StopIteration(msg)

    epoch_key = jax.random.fold_in(self.key, epoch_idx)
    permuted_index = jax.random.permutation(
      epoch_key,
      jnp.arange(self.dataset_size),
    )[index_in_epoch]

    return self.dataset[permuted_index]


class SequenceSampler(Sampler):
  """Sampler of context + query sequences for in-context learning."""


class ClassificationSequenceSampler(SequenceSampler):
  """Sampler of sequences of example-label pairs."""

  def __init__(
    self: Self,
    *,
    key: Array,
    dataset: Dataset,
    class_split: DatasetSplit,
    exemplar_split: DatasetSplit,
    class_idx_sequence_sampler: ClassSampler,
    relabel_sequences: bool = False,
    num_seqs: int | None = None,
  ) -> None:
    """Sampler of sequences of example-label pairs.

    Args:
      key: A key for randomness in sampling.
      dataset: The dataset from which to sample.
      class_split: The class split of `dataset` from which to sample.
      exemplar_split: The exemplar split within each class of `dataset` from
        which to sample; this argument is currently unused!
      class_idx_sequence_sampler: A function that accepts two keyword
        arguments: `key`, a PRNG key, and `num_classes`, an integer number of
        classes, and returns a sequence of class indices.
      relabel_sequences: Whether to randomly reassign labels on a sequence-level
        basis; implements few-shot relabelling.
      num_seqs: The total number of context + query sequences to sample using
        `class_idx_sequence_sampler`. If `None`, sample an infinite sequence of
        sequences.
    """
    # TODO(eringrant): Use it or lose it.
    del exemplar_split

    # The classes to sample from are determined by the dataset split.
    if class_split == DatasetSplit.ALL:
      dataset_classes = dataset.unique_classes
    elif class_split == DatasetSplit.TRAIN:
      dataset_classes = dataset.train_classes
    elif class_split == DatasetSplit.VALID:
      dataset_classes = dataset.valid_classes
    elif class_split == DatasetSplit.TEST:
      dataset_classes = dataset.test_classes
    else:
      msg = f"Unrecognized split: {class_split}"
      raise ValueError(msg)

    if len(dataset_classes) < 2:
      msg = f"Class split has too few classes: {class_split}"
      raise ValueError(msg)

    self._dataset = dataset
    self.num_seqs = num_seqs

    # Compile functions for sampling at `Sampler.__init__`.
    self.generate_sequence = jax.jit(
      jax.vmap(
        partial(
          generate_sequence,
          # TODO(eringrant): Public interface for `Dataset` labels.
          dataset_labels=self._dataset._labels,  # noqa: SLF001
          classes_to_sample=jnp.asarray(dataset_classes),
          generate_class_idx_sequence_fn=class_idx_sequence_sampler,
          generate_exemplar_idx_sequence_fn=generate_exemplar_idx_sequence,
        ),
      ),
    )

    def relabel_sequence(key: Array, labels: Array) -> Array:
      n = self._dataset.num_observed_classes
      onehot_labels = jnn.one_hot(labels, n)
      perm = jax.random.permutation(key, n)
      relabeling = jnp.eye(n)[perm]
      return (onehot_labels @ relabeling).argmax(axis=-1)

    def do_not_relabel_sequence(key: Array, labels: Array) -> Array:
      del key
      return labels

    if relabel_sequences:
      self.relabel_sequences = jax.jit(jax.vmap(relabel_sequence))
    else:
      self.relabel_sequences = jax.jit(jax.vmap(do_not_relabel_sequence))

    # PRNG depends on `MAX_NUM_SEQS` parameter in the infinite `Sampler` case.
    self._seq_keys = jax.random.split(key, num_seqs or MAX_NUM_SEQS)

  def __len__(self: Self) -> int:
    """Return the number of sequences in `Sampler`."""
    if self.num_seqs is None:
      msg = "An infinite sequence does not have finite length."
      raise AttributeError(msg)

    return self.num_seqs

  def __getitem__(self: Self, index: int | slice) -> ExemplarType:
    """Return exemplar-class pairs for the sequence at `index` of `Sampler`."""
    if isinstance(index, int):
      index_max = index

    # Case checking for infinite and finite `slice`s.
    elif index.stop is None:
      if self.num_seqs is None:
        msg = "cannot slice till the end of an infinite `Sampler`"
        raise ValueError(msg)
      else:  # noqa: RET506
        index_max = self.num_seqs
    else:
      index_max = index.stop

    # Dynamically expand PRNGKey array if necessary.
    if index_max > self._seq_keys.shape[0]:
      if self.num_seqs is None:
        self._seq_keys = jnp.concatenate(
          (self._seq_keys, jax.random.split(self._seq_keys[-1], MAX_NUM_SEQS)),
        )
      else:
        msg = "index out of range"
        raise IndexError(msg)
    seq_key = self._seq_keys[index]

    if isinstance(index, int):
      seq_key = jnp.expand_dims(seq_key, 0)  # type: ignore[arg-type]

    exemplar_idx_seq = self.generate_sequence(seq_key)
    exemplars, labels = self._dataset[exemplar_idx_seq]

    if isinstance(index, int):
      if not (len(exemplars) == 1 and len(labels) == 1):
        msg = "Sequence length should be 1 for integer indexing."
        raise ValueError(msg)
      exemplars = exemplars[0]
      labels = labels[0]

    labels = self.relabel_sequences(
      jax.vmap(jax.random.split)(seq_key)[:, 0, :],
      labels,
    )

    return exemplars, labels

  # TODO(eringrant): Make less brittle (reliant on copying & setting attributes).
  def take(self: Self, count: int) -> Sampler:
    """Return a `Sampler` with the first `count` sequences of `Sampler`."""
    take_dataset = copy.copy(self)
    take_dataset._seq_keys = self._seq_keys[:count]  # noqa: SLF001
    take_dataset.num_seqs = count
    return take_dataset
