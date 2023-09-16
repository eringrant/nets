"""Simulate in-context learning of classification tasks."""

# Pandas before JAX or JAXtyping.
import pandas as pd
from pandas.api.types import CategoricalDtype

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Generator
from jaxtyping import Array
from jax.random import KeyArray

import itertools
from functools import partial
import pprint
import time

import numpy as np

import jax
import jax.numpy as jnp

import equinox as eqx
import optax

from nets import datasets
from nets import samplers
from nets import models


def accuracy(pred_y: Array, y: Array) -> Array:
  """Compute elementwise accuracy."""
  predicted_class = jnp.argmax(pred_y, axis=-1)
  return predicted_class == y


def ce(pred_y: Array, y: Array) -> Array:
  """Compute elementwise cross-entropy."""
  pred_y = jax.nn.log_softmax(pred_y, axis=-1)
  num_classes = pred_y.shape[-1]
  onehot_y = jax.nn.one_hot(y, num_classes)
  return -jnp.sum(pred_y * onehot_y, axis=-1)


def batcher(sampler: Sequence, batch_size: int) -> Generator[Sequence, None, None]:
  """Batch a sequence of examples."""
  n = len(sampler)
  for i in range(0, n, batch_size):
    yield sampler[i : min(i + batch_size, n)]


@eqx.filter_value_and_grad
def compute_loss(model: eqx.Module, x: Array, y: Array, key: KeyArray) -> Array:
  """Compute cross-entropy loss on a single sequence."""
  keys = jax.random.split(key, x.shape[0])
  pred_y = jax.vmap(model)(x, y, key=keys)
  query_ce = ce(pred_y[:, -1, :], y[:, -1])
  return query_ce.mean()


@eqx.filter_jit
def train_step(
  model: eqx.Module,
  optimizer: optax.GradientTransformation,
  opt_state: Array,
  x: Array,
  y: Array,
  key: KeyArray,
) -> tuple[Array, eqx.Module, Array]:
  """Train the model on a single sequence."""
  loss, grads = compute_loss(model, x, y, key)
  updates, opt_state = optimizer.update(grads, opt_state)
  model = eqx.apply_updates(model, updates)
  return loss, model, opt_state


@eqx.filter_jit
def eval_step(
  x: Array,
  y: Array,
  key: KeyArray,
  model: eqx.Module,
) -> Mapping[str, Array]:
  """Evaluate the model on a single sequence."""
  pred_y = model(x, y, key=key)

  # Standard metrics.
  elementwise_acc = accuracy(pred_y, y)
  elementwise_loss = ce(pred_y, y)

  # Top-k accuracy.
  pred_top_5 = jax.lax.top_k(pred_y, 5)[1]
  top_5_acc = jax.vmap(jax.numpy.isin)(pred_top_5, y).max(-1)

  # Closed class (classes in sequence) accuracy.
  closed_class_pred_y = jnp.full_like(pred_y, -jnp.inf)
  closed_class_pred_y = closed_class_pred_y.at[:, y].set(pred_y[:, y])
  closed_class_acc = accuracy(closed_class_pred_y, y)

  # Random baseline.
  c = pred_y.shape[-1]
  random_baseline = 1.0 / c

  # Use most frequent class in context as query prediction.
  support_mode = jax.nn.one_hot(jax.nn.one_hot(y[:-1], c).sum(0).argmax(), c)
  support_mode_baseline = accuracy(support_mode, y[-1])

  return {
    "loss": elementwise_loss,
    "accuracy": elementwise_acc,
    "context-closed class query accuracy": closed_class_acc,
    "top-5 accuracy": top_5_acc,
    "rank-1 prediction": pred_top_5[:, 0],
    "rank-2 prediction": pred_top_5[:, 1],
    "rank-3 prediction": pred_top_5[:, 2],
    "rank-4 prediction": pred_top_5[:, 3],
    "rank-5 prediction": pred_top_5[:, 4],
    "random baseline query accuracy": random_baseline,
    "context frequency baseline query accuracy": support_mode_baseline,
    "ground truth label": y,
  }


def summarize_metrics(metrics):
  """Summarize metrics output from `eval_step` for printing."""
  with np.printoptions(precision=2):
    return (
      (
        "\n\tloss:"
        f"\t\t\t{metrics['loss'].mean(0)}"
        "\n\taccuracy:"
        f"\t\t{metrics['accuracy'].mean(0)}"
        "\n\ttop-5 accuracy:"
        f"\t\t{metrics['top-5 accuracy'].mean(0)}"
        "\n\tclosed acc.:"
        f"\t\t{metrics['context-closed class query accuracy'].mean(0)}"
        "\n"
        "\n\tBASELINE (random) query accuracy:"
        f"\t\t{metrics['random baseline query accuracy'].mean() * 100:.2f}%"
        "\n\tBASELINE (context frequency) query accuracy:"
        f"\t{metrics['context frequency baseline query accuracy'].mean() * 100:.2f}%"
        "\n"
        f"\n\tlabel sequences (context + query):\n{metrics['ground truth label'][:5]}"
        "\n"
        "\n\ttop-5 query predictions:\n"
      )
      + str(
        np.stack(
          tuple(metrics[f"rank-{i + 1} prediction"][:5, -1] for i in range(5)),
          axis=-1,
        )
      )
      + (
        "\n"
        # f"\n\tlast layer weight norm:\t\t{metrics['last layer weight norm']}"
        # f"\n\tlast layer bias norm:\t\t{metrics['last layer bias norm']}"
      )
    )


def metrics_to_df(metrics: Mapping[str, Array]) -> pd.DataFrame:
  """Pandas-ify metrics from `eval_step` for later analysis."""
  df = pd.DataFrame.from_dict(metrics, orient="index").transpose()

  # Probe to get shape.
  num_iters = len(df)
  num_seqs, seq_len = df["loss"][0].shape

  # Determine metric structures.
  # TODO(eringrant): Replace this logic with `Metric` class with knowledge of dtypes.
  def has_shape(col, shape):
    a = df[col][0]
    return hasattr(a, "shape") and a.shape == shape

  def has_no_shape(col):
    a = df[col][0]
    return not hasattr(a, "shape") or has_shape(col, (1,))

  elementwise_metrics = tuple(
    filter(partial(has_shape, shape=(num_seqs, seq_len)), df.columns)
  )
  seqwise_metrics = tuple(filter(partial(has_shape, shape=(num_seqs,)), df.columns))
  iterationwise_metrics = tuple(filter(has_no_shape, df.columns))

  # Accounted for all and only these metrics?
  valid_metrics = elementwise_metrics + seqwise_metrics + iterationwise_metrics
  assert len(valid_metrics) == len(df.columns)
  assert len(set(valid_metrics)) == len(valid_metrics)

  # Flatten arrays of sequences.
  df = df.explode(list(seqwise_metrics + elementwise_metrics)).reset_index(drop=True)
  df.insert(1, "sequence", list(range(num_seqs)) * num_iters)

  # Flatten arrays of elements.
  df = df.explode(list(elementwise_metrics)).reset_index(drop=True)
  df.insert(2, "element", list(range(seq_len)) * num_seqs * num_iters)

  # Try to infer datatypes. Disasllow object types for compression.
  df = df.infer_objects()
  if np.dtype("object") in set(df.dtypes):
    raise ValueError(f"`object` data type in DataFrame:\n{df.dtypes}")

  # Optimize data types.
  for col in df.select_dtypes(("int32", "int64")):
    if col == "training iteration":
      df[col] = df[col].astype("int32")
    else:
      df[col] = df[col].astype("int16")
  for col in df.select_dtypes(("float32", "float64")):
    df[col] = df[col].astype("float16")

  return df


def evaluate(
  iteration: int,
  dataset_split: str,
  key: KeyArray,
  model: eqx.Module,
  sampler: Sequence,
  batch_size: int,
) -> pd.DataFrame:
  """Convenience function to evaluate `model` on batches from `sampler`."""
  metrics = {}

  ### Metrics metadata.
  metrics["training iteration"] = iteration

  ### Behavioral metrics.
  # TODO(eringrant): Figure out the right syntax for `eqx.filter_vmap`.
  _eval_step = jax.vmap(partial(eval_step, model=model), (0, 0, 0))

  # Probing metric shapes.
  num_examples = len(sampler)
  incremental_metrics = dict(
    (
      metric_name,
      np.repeat(np.empty_like(metric_value), repeats=num_examples, axis=0),
    )
    for metric_name, metric_value in _eval_step(
      sampler[:1][0], sampler[:1][1], key[jnp.newaxis]
    ).items()
  )

  print("Starting evaluation...")
  start = time.time()

  for i, (x, y) in enumerate(batcher(sampler, batch_size)):
    (key,) = jax.random.split(key, 1)
    batch_metrics = _eval_step(x, y, jax.random.split(key, x.shape[0]))
    for metric_name in incremental_metrics.keys():
      incremental_metrics[metric_name][
        i * batch_size : min((i + 1) * batch_size, num_examples)
      ] = batch_metrics[metric_name]

  metrics.update(incremental_metrics)

  ### Model / parameter metrics.
  # metrics["last layer weight norm"] = float(jnp.linalg.norm(model.unembed.weight))
  # metrics["last layer bias norm"] = float(jnp.linalg.norm(model.unembed.bias))

  end = time.time()
  print(f"Completed evaluation over {num_examples} examples in {end - start:.2f} secs.")

  print("####")
  print(f"ITERATION {iteration}")
  print(f"{dataset_split} set:")
  print(f"{summarize_metrics(metrics)}")

  return metrics_to_df(metrics)


def simulate(
  seed: int,
  # Model params.
  embed_dim: int,
  num_heads: int,
  depth: int,
  causal: bool,
  mlp_ratio: float,
  # Training and evaluation params.
  optimizer_fn: Callable,  # TODO(eringrant): Define interface.
  learning_rate: float | Callable,  # TODO(eringrant): Define interface.
  train_batch_size: int,
  eval_batch_size: int,
  num_epochs: int,
  evaluations_per_epoch: int,
  evaluate_on_test_split: bool,
  # Dataset params.
  # TODO(eringrant): Generalize to `datasets.Dataset`.
  dataset_cls: type[datasets.SymbolicDataset],
  exemplar_labeling: datasets.ExemplarLabeling,
  holdout_class_labeling: datasets.HoldoutClassLabeling,
  num_train_classes: int,
  prop_train_labels: float,
  num_test_classes: int,
  prop_test_labels: float,
  num_valid_classes: int,
  prop_valid_labels: float,
  num_exemplars_per_class: int,
  exemplar_noise_scale: float,
  # Sampler params.
  # TODO(eringrant): Generalize to `samplers.ClassificationSequenceSampler`.
  train_sampler_cls: type[samplers.DirichletMultinomialSampler],
  eval_sampler_cls: type[samplers.DirichletMultinomialSampler],
  train_query_type: samplers.QueryType,
  train_relabeling: bool,
  train_context_len: int,
  train_zipf_exponent: float,
  num_train_seqs: int,
  num_eval_seqs: int,
) -> tuple[pd.DataFrame, ...]:
  """Simulate in-context learning of classification tasks."""
  print(f"Using JAX backend: {jax.default_backend()}\n")

  print("Using configuration:")
  pprint.pprint(locals())
  print()

  # Single source of randomness.
  data_key, model_key, train_key, eval_key = jax.random.split(
    jax.random.PRNGKey(seed), 4
  )

  #########
  # Data setup.
  dataset_key, sampler_key = jax.random.split(data_key)

  dataset = dataset_cls(
    split=datasets.DatasetSplit.ALL,  # Ignore any native class splits.
    exemplar_labeling=exemplar_labeling,
    holdout_class_labeling=holdout_class_labeling,
    num_train_classes=num_train_classes,
    num_test_classes=num_test_classes,
    num_valid_classes=num_valid_classes,
    prop_train_labels=prop_train_labels,
    prop_test_labels=prop_test_labels,
    prop_valid_labels=prop_valid_labels,
    num_exemplars_per_class=num_exemplars_per_class,
    exemplar_noise_scale=exemplar_noise_scale,
    key=dataset_key,
  )

  train_sampler_key, *eval_sampler_keys = jax.random.split(sampler_key, 4)
  train_sampler = train_sampler_cls(
    dataset=dataset,
    class_split=datasets.DatasetSplit.TRAIN,
    exemplar_split=datasets.DatasetSplit.ALL,
    key=train_sampler_key,
    relabel_sequences=train_relabeling,
    num_seqs=num_train_seqs,
    context_len=train_context_len,
    query_type=train_query_type,
    zipf_exponent=train_zipf_exponent,
  )

  eval_samplers = {
    "train_set_prefix": train_sampler.take(num_eval_seqs),
    "train_classes_holdout_sequences": eval_sampler_cls(
      dataset=dataset,
      class_split=datasets.DatasetSplit.TRAIN,
      exemplar_split=datasets.DatasetSplit.ALL,
      key=eval_sampler_keys[0],
      num_seqs=num_eval_seqs,
      context_len=train_context_len,
      query_type=samplers.QueryType.SUPPORTED,
      zipf_exponent=0.0,  # Classes appear with equal probability.
    ),
    "train_classes_relabelled": eval_sampler_cls(
      dataset=dataset,
      class_split=datasets.DatasetSplit.TRAIN,
      exemplar_split=datasets.DatasetSplit.ALL,
      key=eval_sampler_keys[1],
      relabel_sequences=True,
      num_seqs=num_eval_seqs,
      context_len=train_context_len,
      query_type=samplers.QueryType.SUPPORTED,
      zipf_exponent=0.0,  # Classes appear with equal probability.
    ),
    "holdout_classes": eval_sampler_cls(
      dataset=dataset,
      class_split=(
        datasets.DatasetSplit.TEST
        if evaluate_on_test_split
        else datasets.DatasetSplit.VALID
      ),
      exemplar_split=datasets.DatasetSplit.ALL,
      key=eval_sampler_keys[2],
      num_seqs=num_eval_seqs,
      context_len=train_context_len,
      query_type=samplers.QueryType.SUPPORTED,
      zipf_exponent=0.0,  # Classes appear with equal probability.
    ),
  }

  print("Dataset sequences...")
  for sampler_name, eval_sampler in eval_samplers.items():
    print(f"{sampler_name}:\t\n{eval_sampler[:5][1]}")
    print()

  # `None` batch size implies full-batch optimization.
  if train_batch_size is None:
    # TODO(eringrant): Deal with infinite samplers.
    train_batch_size = len(train_sampler)

  #########
  # Model setup.
  model = models.SequenceClassifier(
    example_shape=dataset.exemplar_shape,
    num_classes=dataset.num_observed_classes,
    embed_dim=embed_dim,
    num_heads=num_heads,
    depth=depth,
    mlp_ratio=mlp_ratio,
    causal=causal,
    key=model_key,
  )

  print(f"Model:\n{model}\n")

  #########
  # Training loop.
  optimizer = optimizer_fn(learning_rate=learning_rate)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  # Bookkeeping.
  metrics: dict = dict.fromkeys(eval_samplers.keys(), [])
  itercount = itertools.count()

  # Evaluate before starting training.
  for sampler_name, eval_sampler in eval_samplers.items():
    (eval_key,) = jax.random.split(eval_key, 1)
    metrics[sampler_name].append(
      evaluate(
        iteration=0,
        dataset_split=sampler_name,
        sampler=eval_sampler,
        model=eqx.tree_inference(model, True),
        key=eval_key,
        batch_size=eval_batch_size,
      )
    )

  # Training starts at iteration 1.
  next(itercount)
  evaluation_interval = len(train_sampler) // train_batch_size // evaluations_per_epoch
  if evaluation_interval == 0:
    raise ValueError("Too many `evaluations_per_epoch`.")

  print("\nStarting training...")
  for epoch in range(num_epochs):
    start_time = time.time()

    for i, (x, y) in enumerate(batcher(train_sampler, train_batch_size)):
      (train_key,) = jax.random.split(train_key, 1)
      train_step_num = int(next(itercount))
      train_loss, model, opt_state = train_step(
        model, optimizer, opt_state, x, y, train_key
      )

      if train_step_num % evaluation_interval == 0 or i + 1 == len(train_sampler):
        for sampler_name, eval_sampler in eval_samplers.items():
          (eval_key,) = jax.random.split(eval_key, 1)
          metrics[sampler_name].append(
            evaluate(
              iteration=train_step_num,
              dataset_split=sampler_name,
              sampler=eval_sampler,
              model=eqx.tree_inference(model, True),
              key=eval_key,
              batch_size=eval_batch_size,
            )
          )

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} in {epoch_time:0.2f} seconds.")

  print("Training finished.")

  df = pd.concat(
    pd.concat(sampler_metrics).assign(dataset=sampler_name)
    for sampler_name, sampler_metrics in metrics.items()
  )
  df = df[df.columns[[-1, *list(range(0, len(df.columns) - 1))]]]
  df["dataset"] = df["dataset"].astype(
    CategoricalDtype(categories=metrics.keys(), ordered=True)
  )

  return df


if __name__ == "__main__":
  from nets.launch import configs

  cfg = next(
    iter(
      configs.DebugSearchConfig(
        key=jax.random.PRNGKey(0),
        num_configs=1,
      )
    )
  )
  simulate(**cfg)
