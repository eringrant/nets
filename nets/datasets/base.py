"""`Dataset`s are sequences of unique examples."""
from typing import Any
from collections.abc import Sequence
from nptyping import NDArray
from nptyping import Bool
from nptyping import Floating
from nptyping import Int
from jax.random import KeyArray
from jaxtyping import Array

from enum import Enum
from enum import unique
from functools import cached_property
from functools import partial
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.nn as jnn


# Type hints.
IndexType = int | Sequence[int] | slice
ExemplarType = tuple[NDArray[Any, Floating], NDArray[Any, Int]]


@unique
class ExemplarLabeling(Enum):
  """How to assign class labels to exemplars from the underlying dataset."""

  # Use the original labels from the dataset.
  STANDARD = 1
  # Remove all but the first exemplar from each class.
  SINGLE = 2
  # Assign each exemplar a unique label, resulting in as many classes as there
  # are exemplars.
  SEPARATED = 3


@unique
class HoldoutClassLabeling(Enum):
  """How to assign class labels to holdout (validation and testing) splits."""

  # Use the original labels from the dataset.
  STANDARD = 1
  # Relabel validation and test classes with labels from the training set.
  TRAIN_LABELS = 2


@unique
class DatasetSplit(Enum):
  """Which split of the underlying dataset to use."""

  TRAIN = 1
  VALID = 2
  TEST = 3
  ALL = 4


def wrap_labels(labels: Array, num_classes: int, modulus: Array) -> Array:
  """Wrap `num_classes` into `labels`."""
  onehot_labels = jnn.one_hot(labels, num_classes)
  return (onehot_labels @ modulus).argmax(axis=-1)


def get_wrapped_indices(
  prop_labels: float, num_classes: int, offset=0
) -> tuple[int, Array]:
  """Get indices to wrap `num_classes` into `prop_labels` labels."""
  if prop_labels < 1.0:
    num_labels = int(prop_labels * num_classes)
    indices = jnp.arange(num_classes) % num_labels
  else:
    num_labels = num_classes
    indices = jnp.arange(num_classes)
  indices += offset
  return num_labels, indices


class Dataset:
  """A `Dataset` of class exemplars from which to draw sequences."""

  _exemplars: Sequence[Path] | NDArray
  _labels: NDArray

  num_train_classes: int
  prop_train_labels: float
  num_test_classes: int
  prop_test_labels: float
  num_valid_classes: int
  prop_valid_labels: float

  def __init__(
    self,
    key: KeyArray,
    split: DatasetSplit,
    exemplar_labeling: ExemplarLabeling,
    holdout_class_labeling: HoldoutClassLabeling,
    num_train_classes: int,
    prop_train_labels: float,
    num_test_classes: int,
    prop_test_labels: float,
    num_valid_classes: int = 0,
    prop_valid_labels: float = 0,
    num_exemplars_per_class: int = 400,
  ):
    """A `Dataset` of class exemplars from which to draw sequences.

    Args:
      key: A key for randomness in sampling.
      split: Which split of the underlying dataset to use.
      exemplar_labeling: How to assign class labels to exemplars from the underlying
          dataset.
      holdout_class_labeling: How to assign class labels to holdout (validation and
          testing) splits of this `Dataset`.
      num_train_classes: Number of training classes in this `Dataset`.
      prop_train_labels: Size of the training label set proportional to the underlying
          class set. If 1.0, then labels are identical to the underlying class labels;
          if < 1.0, then labels are wrapped in increasing order.
      num_valid_classes: Number of validation classes in this `Dataset`.
      prop_valid_labels: Size of the validation label set proportional to the
          underlying class set. If 1.0, then labels are identical to the underlying
          class labels; if < 1.0, then labels are wrapped in increasing order.
      num_test_classes: Number of testing classes in this `Dataset`.
      prop_test_labels: Size of the testing label set proportional to the underlying
          class set. If 1.0, then labels are identical to the underlying class labels;
          if < 1.0, then labels are wrapped in increasing order.
      num_exemplars_per_class: Number of exemplars per class to draw from the
          underlying dataset.
    """
    self.num_train_classes = num_train_classes
    self.num_valid_classes = num_valid_classes
    self.num_test_classes = num_test_classes
    self.num_exemplars_per_class = num_exemplars_per_class

    if holdout_class_labeling == HoldoutClassLabeling.TRAIN_LABELS:
      if (
        prop_train_labels * num_train_classes < prop_valid_labels * num_valid_classes
        or prop_train_labels * num_train_classes < prop_test_labels * num_test_classes
      ):
        raise ValueError(
          "Relabeling of validation and test sets with train "
          "labels usually assumes more train classes than "
          "validation and test classes, but "
          f"{prop_train_labels * num_train_classes} < "
          f"{prop_valid_labels * num_valid_classes} or "
          f"{prop_train_labels * num_train_classes} < "
          f"{prop_test_labels * num_test_classes}."
        )

      self.num_observed_classes = int(prop_train_labels * self.num_train_classes)

    else:
      self.num_observed_classes = (
        int(prop_train_labels * self.num_train_classes)
        + int(prop_valid_labels * self.num_valid_classes)
        + int(prop_test_labels * self.num_test_classes)
      )

    # TODO(eringrant): Empty valid class set?
    if not all(
      0.0 < p <= 1.0
      for p in (
        prop_train_labels,
        prop_valid_labels,
        prop_test_labels,
      )
    ):
      raise ValueError(
        "One of `prop_{train,valid,test}_labels` was invalid: "
        f"{prop_train_labels}, {prop_valid_labels}, {prop_test_labels}."
      )

    num_train_labels, train_indices = get_wrapped_indices(
      prop_train_labels, num_train_classes
    )
    num_valid_labels, valid_indices = get_wrapped_indices(
      prop_valid_labels,
      num_valid_classes,
      offset=0
      if holdout_class_labeling == HoldoutClassLabeling.TRAIN_LABELS
      else num_train_labels,
    )
    num_test_labels, test_indices = get_wrapped_indices(
      prop_test_labels,
      num_test_classes,
      offset=0
      if holdout_class_labeling == HoldoutClassLabeling.TRAIN_LABELS
      else num_train_labels + num_valid_labels,
    )

    indices = jnp.concatenate((train_indices, valid_indices, test_indices))
    modulus = jnp.eye(self.num_classes, dtype=int)[indices, :]

    self.wrap_labels = jax.jit(
      partial(
        wrap_labels,
        num_classes=self.num_classes,
        modulus=modulus,
      )
    )

  def __len__(self) -> int:
    """Number of exemplars in this `Dataset`."""
    return len(self._exemplars)

  @property
  def num_classes(self) -> int:
    """Number of classes in this `Dataset`."""
    return self.num_train_classes + self.num_valid_classes + self.num_test_classes

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Shape of an exemplar."""
    raise NotImplementedError("To be implemented by the subclass.")

  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""
    raise NotImplementedError("To be implemented by the subclass.")

  @cached_property
  def unique_classes(self) -> Sequence[int]:
    """Deterministic ordering of dataset class labels."""
    return np.unique(self._labels).tolist()

  @cached_property
  def train_classes(self) -> Sequence[int]:
    """Deterministic ordering of training class labels."""
    i = self.num_train_classes
    return self.unique_classes[:i]

  @cached_property
  def valid_classes(self) -> Sequence[int]:
    """Deterministic ordering of validation class labels."""
    i = self.num_train_classes
    j = self.num_train_classes + self.num_valid_classes
    return self.unique_classes[i:j]

  @cached_property
  def test_classes(self) -> Sequence[int]:
    """Deterministic ordering of testing class labels."""
    j = self.num_train_classes + self.num_valid_classes
    k = self.num_train_classes + self.num_valid_classes + self.num_test_classes
    return self.unique_classes[j:k]

  @cached_property
  def _train_idx(self) -> NDArray[Any, Bool]:
    """Mask for the train split."""
    return np.in1d(self._labels, self.train_classes)

  @cached_property
  def _valid_idx(self) -> NDArray[Any, Bool]:
    """Mask for the validation split."""
    return np.in1d(self._labels, self.valid_classes)

  @cached_property
  def _test_idx(self) -> NDArray[Any, Bool]:
    """Mask for the test split."""
    return np.in1d(self._labels, self.test_classes)
