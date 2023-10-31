"""A `SymbolicDataset` of class exemplars from which to draw sequences."""
from functools import partial
from typing import Self

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from jax import Array

from nets.datasets.base import (
  Dataset,
  DatasetSplit,
  ExemplarLabeling,
  ExemplarType,
  HoldoutClassLabeling,
)


class SymbolicDataset(Dataset):
  """A `SymbolicDataset` of class exemplars from which to draw sequences."""

  _exemplars: np.ndarray
  _labels: np.ndarray

  num_train_classes: int
  prop_train_labels: int
  prop_test_labels: int
  num_test_classes: int
  num_valid_classes: int
  prop_valid_labels: int

  def __init__(
    self: Self,
    key: Array,
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
    exemplar_noise_scale: float = 1e-2,
  ) -> None:
    """A `SymbolicDataset` of class exemplars from which to draw sequences.

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
      exemplar_noise_scale: Scale of the noise to add to exemplars.
    """
    super().__init__(
      key=key,
      split=split,
      exemplar_labeling=exemplar_labeling,
      holdout_class_labeling=holdout_class_labeling,
      num_train_classes=num_train_classes,
      prop_train_labels=prop_train_labels,
      num_test_classes=num_test_classes,
      prop_test_labels=prop_test_labels,
      num_valid_classes=num_valid_classes,
      prop_valid_labels=prop_valid_labels,
      num_exemplars_per_class=num_exemplars_per_class,
    )

    self.exemplar_noise_scale = exemplar_noise_scale

    # Exemplar generation for `SymbolicDataset`.
    labels = np.arange(self.num_classes)

    if num_exemplars_per_class > 1:
      labels = np.repeat(
        labels[:, np.newaxis],
        num_exemplars_per_class,
        axis=-1,
      ).reshape(-1)

      # TODO(eringrant): Use it or lose it.
      del exemplar_labeling

    self._labels = labels

    if self.num_exemplars_per_class > 1:
      self._exemplar_keys = jax.random.split(
        key,
        self.num_classes * num_exemplars_per_class,
      )

      # Compile functions for sampling at `Dataset.__init__`.
      self.generate_exemplar = jax.jit(
        jax.vmap(
          jax.vmap(
            partial(
              jax.random.multivariate_normal,
              # Isotropic with scale a/C to keep noise level in embeddings constant.
              cov=exemplar_noise_scale / self.num_classes * jnp.eye(self.num_classes),
            ),
          ),
        ),
      )

  @property
  def exemplar_shape(self: Self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    """Shape of an exemplar."""
    return (self.num_classes,)

  def __getitem__(self: Self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""
    labels = self._labels[index]
    onehot_labels = jnn.one_hot(labels, self.num_classes)

    if self.num_exemplars_per_class == 1:
      exemplars = onehot_labels

    else:
      exemplar_key = self._exemplar_keys[index]

      if isinstance(index, int):
        exemplar_key = jnp.expand_dims(exemplar_key, 0)

      exemplars = self.generate_exemplar(
        key=exemplar_key,
        mean=onehot_labels,
      )

    labels = self.wrap_labels(labels)

    if isinstance(index, int):
      if not (len(exemplars) == 1 and len(labels) == 1):
        msg = "Expected single exemplar and label."
        raise ValueError(msg)
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels
