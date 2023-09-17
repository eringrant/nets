"""Launcher for local runs of in-context learning simulations."""
import logging
from pathlib import Path

from dataclasses import dataclass
from dataclasses import field

import jax
import optax

import nets
from nets import datasets
from nets import samplers
from nets.launch import configs
from nets.launch import submit
from nets.launch.hparams import Param
from nets.launch.hparams import EnumParam
from nets.launch.hparams import FixedParam

from nets.simulators.online_sgd import simulate


@dataclass(frozen=True, kw_only=True)
class SearchConfig(configs.Config):
  """Generic config for a hyperparameter search."""

  seed: Param = field(default_factory=lambda: EnumParam(range(0, 3)))

  # Model params.
  num_hiddens: Param = field(init=False)
  init_scale: Param = field(init=False)

  # Training and evaluation params.
  optimizer_fn: Param = field(default_factory=lambda: FixedParam(optax.adam))
  learning_rate: Param = field(default_factory=lambda: FixedParam(1e-3))
  train_batch_size: Param = field(default_factory=lambda: FixedParam(32))
  eval_batch_size: Param = field(default_factory=lambda: FixedParam(32))
  num_epochs: Param = field(default_factory=lambda: FixedParam(1))
  evaluations_per_epoch: Param = field(default_factory=lambda: FixedParam(100))
  evaluate_on_test_split: Param = field(default_factory=lambda: FixedParam(False))

  # Dataset params.
  dataset_cls: Param = field(init=False)
  num_dimensions: Param = field(init=False)
  num_exemplars_per_class: Param = field(init=False)
  exemplar_noise_scale: Param = field(init=False)

  # Sampler params.
  sampler_cls: Param = field(init=False)


@dataclass(frozen=True, kw_only=True)
class DebugSearchConfig(SearchConfig):
  """Singleton config for debugging."""

  seed: Param = field(default_factory=lambda: FixedParam(0))

  # No training.
  num_epochs: Param = field(default_factory=lambda: FixedParam(0))
  evaluations_per_epoch: Param = field(default_factory=lambda: FixedParam(1))

  # Teeny tiny model.
  num_hiddens: Param = field(default_factory=lambda: FixedParam(8))
  init_scale: Param = field(default_factory=lambda: FixedParam(1.0))

  dataset_cls: Param = field(default_factory=lambda: FixedParam(datasets.ParityDataset))
  num_dimensions: Param = field(default_factory=lambda: FixedParam(2))
  num_exemplars_per_class: Param = field(default_factory=lambda: FixedParam(16))
  exemplar_noise_scale: Param = field(default_factory=lambda: FixedParam(0.1))
  sampler_cls: Param = field(default_factory=lambda: FixedParam(samplers.EpochSampler))


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  executor = submit.get_submitit_executor(
    cluster="debug",
    log_dir=Path(
      nets.SCRATCH_DIR,
      "osgd",
      submit.get_timestamp(),
    ),
    gpus_per_node=0,
  )

  jobs = submit.submit_jobs(
    executor=executor,
    fn=simulate,
    cfg=DebugSearchConfig(
      num_epochs=FixedParam(10),
      key=jax.random.PRNGKey(0),
      num_configs=1,
    ),
  )

  result = jobs[0].result()
  print(result)
