"""Configs to pass as `kwargs` to `nets.simulate.in_context_learning`."""
from typing import Any
from collections.abc import Mapping
from itertools import count

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields

from nets.launch.hparams import Param
from nets.launch.hparams import FixedParam
from nets.launch.hparams import EnumParam

from nets import datasets
from nets import samplers

import numpy as np
import jax
import optax


# TODO(eringrant): Re-enable grid search if all hyperparameters are enums.
# TODO(eringrant): Idea: Make `num_configs` required iff hparams are not enums.
@dataclass(frozen=True, kw_only=True)
class Config:
  """Abstract base class for configs."""

  key: jax.random.PRNGKey  # Randomness for hyperparameter sampling.
  num_configs: int  # Number of hyperparameter configurations.

  def __post_init__(self):
    """Check that types are as expected after initialization."""
    num_unique_configs = 1

    for field_ in fields(self):
      if field_.name == "key" or field_.name == "num_configs":
        continue
      current_field_value = asdict(self)[field_.name]

      if not isinstance(current_field_value, field_.type):
        current_field_type = type(current_field_value)
        raise TypeError(
          f"The field `{field_.name}` has type "
          f"`{current_field_type}` instead of `{field_.type}`."
        )

      try:
        num_unique_configs *= len(current_field_value)
      except TypeError:
        num_unique_configs *= np.inf

    if self.num_configs > num_unique_configs:
      raise ValueError(
        f"Requested {self.num_configs} hyperparameter "
        f"configurations, but only {num_unique_configs} "
        "unique configuration(s) specified."
      )

  def __len__(self) -> int:
    """Return the total number of hyperparameter configurations."""
    return self.num_configs

  def __iter__(self):
    """Iterate over `self`; needed to appear as `Iterable` to `mypy`."""
    try:
      for i in count():
        yield self[i]
    except IndexError:
      return

  def __getitem__(self, index: int) -> Mapping[str, Any]:
    """Get the hyperparameter configuration at `index`."""
    try:
      index = range(len(self))[index]
    except IndexError as e:
      raise IndexError(
        f"Index out of bounds for `Config` of length {len(self)}."
      ) from e

    hparam_dict = asdict(self)
    del hparam_dict["key"]
    del hparam_dict["num_configs"]
    hparam_names, hparams = zip(*hparam_dict.items(), strict=True)

    hparam_samples = tuple(
      (
        hparam.generate(key)
        for key, hparam in zip(
          jax.random.split(jax.random.fold_in(self.key, index), len(hparams)),
          hparams,
          strict=True,
        )
      ),
    )

    return dict(zip(hparam_names, hparam_samples, strict=True))


@dataclass(frozen=True, kw_only=True)
class SearchConfig(Config):
  """Generic config for a hyperparameter search.

  This config and its subclasses can be iterated over to pass keyword arguments
  to `nets.simulate.simulate` as follows:

  ```
  cfg = next(iter(configs.SearchConfig(...)))
  nets.simulate.simulate(**cfg)
  ```

  or:

  ```
  for cfg in configs.SearchConfig(...):
    nets.simulate.simulate(**cfg)
  ```

  See also `scripts/launcher_...` for examples of using these configs with
  `submitit.Executor.map_array`.
  """

  seed: Param = field(default_factory=lambda: EnumParam(range(0, 3)))

  # Model params.
  embed_dim: Param = field(init=False)
  num_heads: Param = field(init=False)
  depth: Param = field(init=False)
  mlp_ratio: Param = field(init=False)
  causal: Param = field(init=False)

  # Training and evaluation params.
  optimizer_fn: Param = field(default_factory=lambda: FixedParam(optax.adam))
  learning_rate: Param = field(default_factory=lambda: FixedParam(1e-3))
  train_batch_size: Param = field(default_factory=lambda: FixedParam(32))
  eval_batch_size: Param = field(default_factory=lambda: FixedParam(32))
  num_epochs: Param = field(default_factory=lambda: FixedParam(1))
  evaluations_per_epoch: Param = field(default_factory=lambda: FixedParam(100))
  evaluate_on_test_split: Param = field(default_factory=lambda: FixedParam(False))

  # Dataset params.
  num_train_classes: Param = field(init=False)  # `init=False` toavoid init
  num_valid_classes: Param = field(init=False)  # of to-be-overridden value.
  num_test_classes: Param = field(init=False)
  prop_train_labels: Param = field(init=False)
  prop_valid_labels: Param = field(init=False)
  prop_test_labels: Param = field(init=False)
  dataset_cls: Param = field(init=False)
  exemplar_labeling: Param = field(init=False)
  holdout_class_labeling: Param = field(init=False)
  num_exemplars_per_class: Param = field(init=False)
  exemplar_noise_scale: Param = field(init=False)

  # Sampler params.
  num_train_seqs: Param = field(init=False)
  num_eval_seqs: Param = field(init=False)
  train_sampler_cls: Param = field(init=False)
  eval_sampler_cls: Param = field(init=False)
  train_query_type: Param = field(init=False)
  train_context_len: Param = field(init=False)
  train_zipf_exponent: Param = field(init=False)
  train_relabeling: Param = field(init=False)


@dataclass(frozen=True, kw_only=True)
class DebugSearchConfig(SearchConfig):
  """Singleton config for debugging."""

  seed: Param = field(default_factory=lambda: FixedParam(0))

  num_epochs: Param = field(default_factory=lambda: FixedParam(0))  # No training.
  evaluations_per_epoch: Param = field(default_factory=lambda: FixedParam(1))

  embed_dim: Param = field(default_factory=lambda: FixedParam(8))  # Teeny tiny model.
  num_heads: Param = field(default_factory=lambda: FixedParam(8))
  depth: Param = field(default_factory=lambda: FixedParam(2))
  mlp_ratio: Param = field(default_factory=lambda: FixedParam(4.0))
  causal: Param = field(default_factory=lambda: FixedParam(True))

  num_train_classes: Param = field(default_factory=lambda: FixedParam(80))
  num_valid_classes: Param = field(default_factory=lambda: FixedParam(20))
  num_test_classes: Param = field(default_factory=lambda: FixedParam(16))
  prop_train_labels: Param = field(default_factory=lambda: FixedParam(0.8))
  prop_valid_labels: Param = field(default_factory=lambda: FixedParam(0.7))
  prop_test_labels: Param = field(default_factory=lambda: FixedParam(0.3))
  dataset_cls: Param = field(
    default_factory=lambda: FixedParam(datasets.SymbolicDataset)
  )
  exemplar_labeling: Param = field(
    default_factory=lambda: FixedParam(datasets.ExemplarLabeling.STANDARD)
  )
  holdout_class_labeling: Param = field(
    default_factory=lambda: FixedParam(datasets.HoldoutClassLabeling.STANDARD)
  )
  num_exemplars_per_class: Param = field(default_factory=lambda: FixedParam(20))
  exemplar_noise_scale: Param = field(default_factory=lambda: FixedParam(1.0))

  num_train_seqs: Param = field(default_factory=lambda: FixedParam(int(1e3)))
  num_eval_seqs: Param = field(default_factory=lambda: FixedParam(int(1e2)))
  train_sampler_cls: Param = field(
    default_factory=lambda: FixedParam(samplers.DirichletMultinomialSampler)
  )
  eval_sampler_cls: Param = field(
    default_factory=lambda: FixedParam(samplers.DirichletMultinomialSampler)
  )
  train_query_type: Param = field(
    default_factory=lambda: FixedParam(samplers.QueryType.SUPPORTED)
  )
  train_context_len: Param = field(default_factory=lambda: FixedParam(2))
  train_zipf_exponent: Param = field(default_factory=lambda: FixedParam(1.0))
  train_relabeling: Param = field(default_factory=lambda: FixedParam(False))


@dataclass(frozen=True, kw_only=True)
class SymbolicSearchConfig(SearchConfig):
  """Singleton hyperparameter search for the symbolic dataset."""

  evaluations_per_epoch: Param = field(default_factory=lambda: FixedParam(100))

  embed_dim: Param = field(default_factory=lambda: FixedParam(64))
  num_heads: Param = field(default_factory=lambda: FixedParam(8))
  depth: Param = field(default_factory=lambda: FixedParam(2))
  mlp_ratio: Param = field(default_factory=lambda: FixedParam(4.0))
  causal: Param = field(default_factory=lambda: FixedParam(True))

  num_train_classes: Param = field(default_factory=lambda: FixedParam(1600))
  num_valid_classes: Param = field(default_factory=lambda: FixedParam(2))
  num_test_classes: Param = field(default_factory=lambda: FixedParam(2))
  prop_train_labels: Param = field(default_factory=lambda: FixedParam(1.0))
  prop_valid_labels: Param = field(default_factory=lambda: FixedParam(1.0))
  prop_test_labels: Param = field(default_factory=lambda: FixedParam(1.0))
  dataset_cls: Param = field(
    default_factory=lambda: FixedParam(datasets.SymbolicDataset)
  )
  exemplar_labeling: Param = field(
    default_factory=lambda: FixedParam(datasets.ExemplarLabeling.STANDARD)
  )
  holdout_class_labeling: Param = field(
    default_factory=lambda: FixedParam(datasets.HoldoutClassLabeling.TRAIN_LABELS)
  )
  num_exemplars_per_class: Param = field(default_factory=lambda: FixedParam(20))
  exemplar_noise_scale: Param = field(default_factory=lambda: FixedParam(0.1))

  num_train_seqs: Param = field(default_factory=lambda: FixedParam(int(1e5 * 32)))
  num_eval_seqs: Param = field(default_factory=lambda: FixedParam(int(1e2 * 32)))
  train_sampler_cls: Param = field(
    default_factory=lambda: FixedParam(samplers.DirichletMultinomialSampler)
  )
  eval_sampler_cls: Param = field(
    default_factory=lambda: FixedParam(samplers.DirichletMultinomialSampler)
  )
  train_query_type: Param = field(
    default_factory=lambda: FixedParam(samplers.QueryType.SUPPORTED)
  )
  train_context_len: Param = field(default_factory=lambda: FixedParam(2))
  train_zipf_exponent: Param = field(default_factory=lambda: FixedParam(1.0))
  train_relabeling: Param = field(default_factory=lambda: FixedParam(True))
