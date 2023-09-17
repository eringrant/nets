"""A `ParityDataset` that generates parity-labelled examples in `D` dimensions."""
from jax.random import KeyArray

import jax
import jax.numpy as jnp
from jax import random
from functools import partial

from nets.datasets.base import Dataset
from nets.datasets.base import DatasetSplit
from nets.datasets.base import ExemplarType
from nets.datasets.base import ExemplarLabeling
from nets.datasets.base import HoldoutClassLabeling


class ParityDataset(Dataset):
  """Parity-labelled exmaples in many dimensions."""

  def __init__(
    self,
    key: KeyArray,
    num_dimensions: int = 2,  # num classes is c == 2**d
    num_exemplars_per_class: int = 400,
    exemplar_noise_scale: float = 1e-1,
    # TODO(eringrant): Decide whether to use these arguments.
    split: DatasetSplit = DatasetSplit.TRAIN,
    exemplar_labeling: ExemplarLabeling = ExemplarLabeling.STANDARD,
    holdout_class_labeling: HoldoutClassLabeling = HoldoutClassLabeling.STANDARD,
  ):
    """Initializes a `ParityDataset` instance."""
    super().__init__(
      key=key,  # TODO(eringrant): Use a separate key.
      split=split,
      exemplar_labeling=exemplar_labeling,
      holdout_class_labeling=holdout_class_labeling,
      num_train_classes=2,
      prop_train_labels=1.0,
      num_test_classes=2,
      prop_test_labels=1.0,
      num_valid_classes=0,
      prop_valid_labels=1.0,
      num_exemplars_per_class=num_exemplars_per_class,
    )

    self.exemplar_noise_scale = exemplar_noise_scale
    self.num_dimensions = num_dimensions

    # TODO(eringrant): Lazily compute the below.
    labels = jnp.arange(2**num_dimensions)
    # TODO(eringrant): Assert labels are 32-bit integers for this conversion.
    bit_labels = jnp.unpackbits(labels.view("uint8"), bitorder="little").reshape(
      labels.size, 32
    )[:, :num_dimensions]
    parity_labels = jax.lax.reduce(
      bit_labels, init_values=jnp.uint8(0), computation=jnp.bitwise_xor, dimensions=(1,)
    )

    self._exemplars = bit_labels.astype(jnp.int32)
    self._labels = parity_labels.astype(jnp.int32)

    if num_exemplars_per_class > 1:
      # Repeat each exemplar and label `num_exemplars_per_class` times.
      self._exemplars = jnp.repeat(
        self._exemplars[:, jnp.newaxis, :], num_exemplars_per_class, axis=1
      ).reshape(num_exemplars_per_class * self.num_classes, num_dimensions)
      self._labels = jnp.repeat(
        self._labels[:, jnp.newaxis], num_exemplars_per_class, axis=-1
      ).reshape(num_exemplars_per_class * self.num_classes)

      # Produce unique keys for each exemplar.
      self._exemplar_keys = jax.random.split(
        key, self.num_classes * num_exemplars_per_class
      )

      # Compile a function for sampling exemplars at `Dataset.__init__`.
      self.generate_exemplar = jax.jit(
        jax.vmap(
          partial(
            jax.random.multivariate_normal,
            # Isotropic with scale a/C to keep noise scale constant.
            cov=exemplar_noise_scale / self.num_classes * jnp.eye(num_dimensions),
          )
        )
      )

  @property
  def exemplar_shape(self) -> tuple[int]:
    """Returns the shape of an exemplar."""
    return (self.num_dimensions,)

  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Get the exemplar(s) and the corresponding label(s) at `index`."""
    exemplars = self._exemplars[index]
    labels = self._labels[index]

    if self.num_exemplars_per_class > 1:
      exemplar_key = self._exemplar_keys[index]

      if isinstance(index, int):
        exemplars = jnp.expand_dims(exemplars, 0)
        exemplar_key = jnp.expand_dims(exemplar_key, 0)

      exemplars = self.generate_exemplar(
        key=exemplar_key,
        mean=exemplars,
      )

    return exemplars, labels


if __name__ == "__main__":
  # Test the class
  key = random.PRNGKey(0)
  dataset = ParityDataset(key)
  # Only do the below for small datasets...
  exemplars, labels = dataset[:]

  import matplotlib.pyplot as plt

  plt.figure()
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.title("2D XOR Dataset")
  plt.grid(True)

  # Use the labels to distinguish classes and plot
  plt.scatter(
    exemplars[labels == 0, 0], exemplars[labels == 0, 1], c="red", label="Class 0"
  )
  plt.scatter(
    exemplars[labels == 1, 0], exemplars[labels == 1, 1], c="blue", label="Class 1"
  )

  # Add legend and show plot
  plt.legend()
  plt.show()
