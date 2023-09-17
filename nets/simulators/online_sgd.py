"""Simulate online stochastic gradient descent learning of a simple task."""

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
  """Compute cross-entropy loss on a single example."""
  keys = jax.random.split(key, x.shape[0])
  pred_y = jax.vmap(model)(x, key=keys)
  loss = ce(pred_y, y)
  return loss.mean()


@eqx.filter_jit
def train_step(
  model: eqx.Module,
  optimizer: optax.GradientTransformation,
  opt_state: Array,
  x: Array,
  y: Array,
  key: KeyArray,
) -> tuple[Array, eqx.Module, Array]:
  """Train the model on a single example."""
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
  """Evaluate the model on a single example-label pair."""
  pred_y = model(x, key=key)

  # Standard metrics.
  elementwise_acc = accuracy(pred_y, y)
  elementwise_loss = ce(pred_y, y)

  # Random baseline.
  c = pred_y.shape[-1]
  random_baseline = 1.0 / c

  return {
    "loss": elementwise_loss.mean(),
    "accuracy": elementwise_acc.mean(),
    "random baseline accuracy": random_baseline,
    "ground truth label": y,
  }


def summarize_metrics(metrics):
  """Summarize metrics output from `eval_step` for printing."""
  with np.printoptions(precision=2):
    return (
      "\n\tloss:"
      f"\t\t\t{metrics['loss'].mean(0)}"
      "\n\taccuracy:"
      f"\t\t{metrics['accuracy'].mean(0)}"
      "\n\tBASELINE:"
      f"\t\t{metrics['random baseline accuracy'].mean() * 100:.2f}%"
      f"\n\tGT labels:\t\t{metrics['ground truth label']}"
    )


def metrics_to_df(metrics: Mapping[str, Array]) -> pd.DataFrame:
  """Pandas-ify metrics from `eval_step` for later analysis."""
  df = pd.DataFrame.from_dict(metrics, orient="index").transpose()

  # Probe to get shape.
  num_iters = len(df)
  num_examples = df["loss"][0].size

  # Determine metric structures.
  # TODO(eringrant): Replace this logic with `Metric` class with knowledge of dtypes.
  def has_shape(col, shape):
    a = df[col][0]
    return hasattr(a, "shape") and a.shape == shape

  def has_no_shape(col):
    a = df[col][0]
    return not hasattr(a, "shape") or has_shape(col, (1,))

  elementwise_metrics = tuple(
    filter(partial(has_shape, shape=(num_examples,)), df.columns)
  )
  iterationwise_metrics = tuple(filter(has_no_shape, df.columns))

  # Accounted for all and only these metrics?
  valid_metrics = elementwise_metrics + iterationwise_metrics
  assert len(valid_metrics) == len(df.columns)
  assert len(set(valid_metrics)) == len(valid_metrics)

  # Flatten arrays of elements.
  df = df.explode(list(elementwise_metrics)).reset_index(drop=True)
  df.insert(1, "element", list(range(num_examples)) * num_iters)

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
  num_hiddens: int,
  init_scale: float,
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
  dataset_cls: type[datasets.ParityDataset],
  num_dimensions: int,
  num_exemplars_per_class: int,
  exemplar_noise_scale: float,
  # Sampler params.
  # TODO(eringrant): Generalize to `samplers.SingletonSampler`.
  sampler_cls: type[samplers.EpochSampler],
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

  train_dataset_key, eval_dataset_key = jax.random.split(dataset_key, 2)
  train_dataset = dataset_cls(
    key=train_dataset_key,
    split=datasets.DatasetSplit.TRAIN,
    num_exemplars_per_class=num_exemplars_per_class,
    exemplar_noise_scale=exemplar_noise_scale,
    num_dimensions=num_dimensions,
  )
  eval_dataset = dataset_cls(
    key=eval_dataset_key,
    split=datasets.DatasetSplit.VALID,
    num_exemplars_per_class=num_exemplars_per_class,
    exemplar_noise_scale=exemplar_noise_scale,
    num_dimensions=num_dimensions,
  )

  # `None` batch size implies full-batch optimization.
  if train_batch_size is None:
    train_batch_size = len(train_dataset)

  if len(train_dataset) % train_batch_size != 0:
    raise ValueError("Batch size must evenly divide the number of training examples.")
  if len(eval_dataset) % eval_batch_size != 0:
    raise ValueError("Batch size must evenly divide the number of evaluation examples.")

  train_sampler_key, eval_sampler_key = jax.random.split(sampler_key, 2)
  train_sampler = sampler_cls(
    key=train_sampler_key,
    dataset=eval_dataset,
    num_epochs=max(num_epochs, 1),
  )
  eval_sampler = sampler_cls(
    key=eval_sampler_key,
    dataset=eval_dataset,
    num_epochs=1,
  )

  #########
  # Model setup.
  model = models.MLP(
    in_features=2,
    hidden_features=num_hiddens,
    out_features=2,
    act=jax.nn.relu,
    key=model_key,
    init_scale=init_scale,
  )

  print(f"Model:\n{model}\n")

  #########
  # Training loop.
  optimizer = optimizer_fn(learning_rate=learning_rate)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  # Bookkeeping.
  metrics = []
  itercount = itertools.count()

  # Evaluate before starting training.
  metrics.append(
    evaluate(
      iteration=0,
      dataset_split="eval",
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
      i += epoch * len(train_sampler) // train_batch_size

      (train_key,) = jax.random.split(train_key, 1)
      train_step_num = int(next(itercount))
      train_loss, model, opt_state = train_step(
        model, optimizer, opt_state, x, y, train_key
      )

      if train_step_num % evaluation_interval == 0 or i + 1 == len(train_sampler):
        metrics.append(
          evaluate(
            iteration=train_step_num,
            dataset_split="eval",
            sampler=eval_sampler,
            model=eqx.tree_inference(model, True),
            key=eval_key,
            batch_size=eval_batch_size,
          )
        )

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} in {epoch_time:0.2f} seconds.")

  print("Training finished.")

  # TODO(eringrant): Simplify given only a single eval set.
  df = pd.concat(
    pd.concat(sampler_metrics).assign(dataset=sampler_name)
    for sampler_name, sampler_metrics in {"eval": metrics}.items()
  )
  df = df[df.columns[[-1, *list(range(0, len(df.columns) - 1))]]]
  # df["dataset"] = df["dataset"].astype(
  #  CategoricalDtype(categories=metrics.keys(), ordered=True)
  # )
  df["dataset"] = df["dataset"].astype(
    CategoricalDtype(categories=("train", "eval"), ordered=True)
  )

  return model, df


def simulate_return_df(**kwargs):
  """Simulate and return only the DataFrame."""
  model, df = simulate(**kwargs)
  del model
  return df
