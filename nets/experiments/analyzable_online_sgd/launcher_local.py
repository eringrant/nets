"""Launcher for local runs of analyzable online SGD."""
import logging
from pathlib import Path

from dataclasses import dataclass
from dataclasses import field

import jax
import optax

import nets
from nets.launch import configs
from nets.launch import submit
from nets.launch.hparams import Param
from nets.launch.hparams import EnumParam
from nets.launch.hparams import FixedParam

from nets.simulators import in_context_learning

from nets import datasets
from nets import samplers


@dataclass(frozen=True, kw_only=True)
class SearchConfig(configs.Config):
  """Generic config for a hyperparameter search."""

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

  # No training.
  num_epochs: Param = field(default_factory=lambda: FixedParam(0))
  evaluations_per_epoch: Param = field(default_factory=lambda: FixedParam(1))

  # Teeny tiny model.
  embed_dim: Param = field(default_factory=lambda: FixedParam(8))
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


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  executor = submit.get_submitit_executor(
    cluster="local",
    log_dir=Path(
      nets.SCRATCH_DIR,
      "osgd",
      submit.get_timestamp(),
    ),
  )

  jobs = executor.map_array(
    lambda kwargs: in_context_learning.simulate(
      **kwargs,
    ),
    DebugSearchConfig(
      key=jax.random.PRNGKey(0),
      num_configs=1,
    ),
  )
  result = jobs[0].result()
  print(result)
