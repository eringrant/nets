from typing import Tuple
from typing import Union
from jax.random import KeyArray

from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
import jax.nn as jnn

from nets.datasets.base import Dataset
from nets.datasets.base import DatasetSplit
from nets.datasets.base import ExemplarLabeling
from nets.datasets.base import ExemplarType
from nets.datasets.base import HoldoutClassLabeling


class SymbolicDataset(Dataset):

  _exemplars: np.ndarray
  _labels: np.ndarray
  num_train_classes: int
  num_valid_classes: int
  num_test_classes: int
  num_train_labels: int
  num_valid_labels: int
  num_test_labels: int

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
    exemplar_noise_scale: float = 1e-2,
  ):
    """A `SymbolicDataset` of class exemplars from which to draw sequences.

    Args:
      ...`Dataset` args...
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
      exemplar_noise_scale=exemplar_noise_scale,
    )

    # Exemplar generation for `SymbolicDataset`.
    labels = np.arange(self.num_classes)

    if num_exemplars_per_class > 1:
      labels = np.repeat(
        labels[:, np.newaxis], num_exemplars_per_class, axis=-1
      ).reshape(-1)

      # TODO(eringrant): Deal with this params.
      del exemplar_labeling

    self._labels = labels

    if self.num_exemplars_per_class > 1:
      self._exemplar_keys = jax.random.split(
        key, self.num_classes * num_exemplars_per_class
      )

      # Compile functions for sampling at `Dataset.__init__`.
      self.generate_exemplar = jax.jit(
        jax.vmap(
          jax.vmap(
            partial(
              jax.random.multivariate_normal,
              # Isotropic with scale a/C to keep noise level in embeddings constant.
              cov=exemplar_noise_scale / self.num_classes * jnp.eye(self.num_classes),
            )
          )
        )
      )

  @property
  def exemplar_shape(self) -> Tuple[int]:
    return (self.num_classes,)

  def __getitem__(self, index: Union[int, slice]) -> ExemplarType:
    labels = self._labels[index]
    onehot_labels = jnn.one_hot(labels, self.num_classes)

    if self.num_exemplars_per_class == 1:
      exemplars = onehot_labels

    else:
      exemplar_key = self._exemplar_keys[index]

      # TODO(eringrant): Deal with other `index` shapes.
      if isinstance(index, int):
        exemplar_key = jnp.expand_dims(exemplar_key, 0)  # type: ignore[arg-type]

      exemplars = self.generate_exemplar(
        key=exemplar_key,
        mean=onehot_labels,
      )

    labels = self.wrap_labels(labels)

    if isinstance(index, int):
      assert len(exemplars) == 1 and len(labels) == 1
      exemplars = exemplars[0]
      labels = labels[0]

    return exemplars, labels
