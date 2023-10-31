"""Launcher for SLURM runs of in-context learning simulations."""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import jax
import optax

import nets
from nets import datasets, samplers
from nets.launch import configs, submit
from nets.launch.hparams import (
  EnumParam,
  FixedParam,
  LogUniformParam,
  Param,
  UniformParam,
)
from nets.simulators.in_context_learning import simulate


@dataclass(frozen=True, kw_only=True)
class SearchConfig(configs.Config):
  """Generic config for a hyperparameter search."""

  seed: Param = field(default_factory=lambda: EnumParam(range(3)))

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
  evaluate_on_test_split: Param = field(
    default_factory=lambda: FixedParam(False),  # noqa: FBT003
  )

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
class SymbolicSearchConfig(SearchConfig):
  """Singleton hyperparameter search for the symbolic dataset."""

  evaluations_per_epoch: Param = field(default_factory=lambda: FixedParam(100))

  embed_dim: Param = field(default_factory=lambda: FixedParam(64))
  num_heads: Param = field(default_factory=lambda: FixedParam(8))
  depth: Param = field(default_factory=lambda: FixedParam(2))
  mlp_ratio: Param = field(default_factory=lambda: FixedParam(4.0))
  causal: Param = field(default_factory=lambda: FixedParam(True))  # noqa: FBT003

  num_train_classes: Param = field(default_factory=lambda: FixedParam(1600))
  num_valid_classes: Param = field(default_factory=lambda: FixedParam(2))
  num_test_classes: Param = field(default_factory=lambda: FixedParam(2))
  prop_train_labels: Param = field(default_factory=lambda: FixedParam(1.0))
  prop_valid_labels: Param = field(default_factory=lambda: FixedParam(1.0))
  prop_test_labels: Param = field(default_factory=lambda: FixedParam(1.0))
  dataset_cls: Param = field(
    default_factory=lambda: FixedParam(datasets.SymbolicDataset),
  )
  exemplar_labeling: Param = field(
    default_factory=lambda: FixedParam(datasets.ExemplarLabeling.STANDARD),
  )
  holdout_class_labeling: Param = field(
    default_factory=lambda: FixedParam(datasets.HoldoutClassLabeling.TRAIN_LABELS),
  )
  num_exemplars_per_class: Param = field(default_factory=lambda: FixedParam(20))
  exemplar_noise_scale: Param = field(default_factory=lambda: FixedParam(0.1))

  num_train_seqs: Param = field(default_factory=lambda: FixedParam(int(1e5 * 32)))
  num_eval_seqs: Param = field(default_factory=lambda: FixedParam(int(1e2 * 32)))
  train_sampler_cls: Param = field(
    default_factory=lambda: FixedParam(samplers.DirichletMultinomialSampler),
  )
  eval_sampler_cls: Param = field(
    default_factory=lambda: FixedParam(samplers.DirichletMultinomialSampler),
  )
  train_query_type: Param = field(
    default_factory=lambda: FixedParam(samplers.QueryType.SUPPORTED),
  )
  train_context_len: Param = field(default_factory=lambda: FixedParam(2))
  train_zipf_exponent: Param = field(default_factory=lambda: FixedParam(1.0))
  train_relabeling: Param = field(
    default_factory=lambda: FixedParam(True),  # noqa: FBT003
  )


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  executor = submit.get_submitit_executor(
    cluster="slurm",
    #
    ### Output directory. ###
    log_dir=Path(
      nets.SCRATCH_DIR,
      "in-ctx",
      submit.get_timestamp(),
    ),
    #
    ### GPU mode. ###
    slurm_partition="gpu",
    slurm_parallelism=50,
    #
    ### CPU mode. ###
    #
    # 24-hour time limit per job.
    timeout_min=60 * 24,
  )

  # Change config here.
  cfg = SymbolicSearchConfig(
    key=jax.random.PRNGKey(0),
    num_configs=500,
    seed=UniformParam(0, (1 << 15) - 1),
    embed_dim=EnumParam((16, 32, 64)),
    num_train_classes=LogUniformParam(20, 2000, base=10),
    prop_train_labels=UniformParam(0.25, 1.0),
    num_exemplars_per_class=LogUniformParam(1, 1000, base=10),
    exemplar_noise_scale=LogUniformParam(1e-1, 1e3, base=10),
  )

  jobs = submit.submit_jobs(executor, simulate, cfg)
