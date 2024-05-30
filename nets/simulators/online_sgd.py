"""Simulate online stochastic gradient descent learning of a simple task."""
import itertools
import logging
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax import Array
from pandas.api.types import CategoricalDtype

from nets import datasets, models, samplers


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
def compute_loss(model: eqx.Module, x: Array, y: Array, key: Array) -> Array:
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
  key: Array,
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
  key: Array,
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


def summarize_metrics(metrics: Mapping[str, Array]) -> str:
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
  metrics_df = pd.DataFrame.from_dict(metrics, orient="index").transpose()

  # Probe to get shape.
  num_iters = len(metrics_df)
  num_examples = metrics_df["loss"][0].size

  # Determine metric structures.
  # TODO(eringrant): Replace this logic with `Metric` class with knowledge of dtypes.
  def has_shape(col: str, shape: tuple[int, ...]) -> bool:
    a = metrics_df[col][0]
    return hasattr(a, "shape") and a.shape == shape

  def has_no_shape(col: str) -> bool:
    a = metrics_df[col][0]
    return not hasattr(a, "shape") or has_shape(col, (1,))

  elementwise_metrics = tuple(
    filter(partial(has_shape, shape=(num_examples,)), metrics_df.columns),
  )
  iterationwise_metrics = tuple(filter(has_no_shape, metrics_df.columns))

  # Accounted for all and only these metrics?
  valid_metrics = elementwise_metrics + iterationwise_metrics
  if len(valid_metrics) != len(metrics_df.columns):
    msg = f"Invalid metrics: {set(metrics_df.columns) - set(valid_metrics)}"
    raise ValueError(msg)
  if len(set(valid_metrics)) != len(valid_metrics):
    msg = f"Duplicate metrics: {valid_metrics}"
    raise ValueError(msg)

  # Flatten arrays of elements.
  metrics_df = metrics_df.explode(list(elementwise_metrics)).reset_index(drop=True)
  metrics_df.insert(1, "element", list(range(num_examples)) * num_iters)

  # Try to infer datatypes. Disasllow object types for compression.
  metrics_df = metrics_df.infer_objects()
  if np.dtype("object") in set(metrics_df.dtypes):
    msg = f"`object` data type in DataFrame:\n{metrics_df.dtypes}"
    raise ValueError(msg)

  # Optimize data types.
  for col in metrics_df.select_dtypes(("int32", "int64")):
    if col == "training iteration":
      metrics_df[col] = metrics_df[col].astype("int32")
    else:
      metrics_df[col] = metrics_df[col].astype("int16")
  for col in metrics_df.select_dtypes(("float32", "float64")):
    metrics_df[col] = metrics_df[col].astype("float16")

  return metrics_df


def evaluate(
  iteration: int,
  dataset_split: str,
  key: Array,
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
  incremental_metrics = {
    metric_name: np.repeat(np.empty_like(metric_value), repeats=num_examples, axis=0)
    for metric_name, metric_value in _eval_step(
      sampler[:1][0],
      sampler[:1][1],
      key[jnp.newaxis],
    ).items()
  }

  logging.info("Starting evaluation...")
  start = time.time()

  for i, (x, y) in enumerate(batcher(sampler, batch_size)):
    (key,) = jax.random.split(key, 1)
    batch_metrics = _eval_step(x, y, jax.random.split(key, x.shape[0]))
    for metric_name in incremental_metrics:
      incremental_metrics[metric_name][
        i * batch_size : min((i + 1) * batch_size, num_examples)
      ] = batch_metrics[metric_name]

  metrics.update(incremental_metrics)

  ### Model / parameter metrics.

  end = time.time()
  logging.info(
    f"Completed evaluation over {num_examples} examples in {end - start:.2f} secs.",
  )

  logging.info("####")
  logging.info(f"ITERATION {iteration}")
  logging.info(f"{dataset_split} set:")
  logging.info(f"{summarize_metrics(metrics)}")

  return metrics_to_df(metrics)


def simulate(
  *,
  seed: int,
  # Model params.
  num_hiddens: int,
  init_scale: float,
  # Training and evaluation params.
  optimizer_fn: Callable,  # TODO(eringrant): Define interface.
  learning_rate: float | Callable,  # TODO(eringrant): Define interface.
  train_batch_size: int,
  eval_batch_size: int,
  num_epochs: int | None,
  evaluations_per_epoch: int,
  # Dataset params.
  dataset_cls: type[datasets.ParityDataset],  # TODO(eringrant): Use `Dataset`.
  num_dimensions: int,
  num_exemplars_per_class: int,
  exemplar_noise_scale: float,
  # Sampler params.
  sampler_cls: type[samplers.EpochSampler],  # TODO(eringrant): Use `SingletonSampler`.
) -> tuple[eqx.Module, pd.DataFrame]:
  """Simulate in-context learning of classification tasks."""
  logging.info(f"Using JAX backend: {jax.default_backend()}\n")

  logging.info("Using configuration: pprint.pformat(locals())")

  # Single source of randomness.
  data_key, model_key, train_key, eval_key = jax.random.split(
    jax.random.PRNGKey(seed),
    4,
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
    msg = "Batch size must evenly divide the number of training examples."
    raise ValueError(msg)
  if len(eval_dataset) % eval_batch_size != 0:
    msg = "Batch size must evenly divide the number of evaluation examples."
    raise ValueError(msg)

  train_sampler_key, eval_sampler_key = jax.random.split(sampler_key, 2)
  train_sampler = sampler_cls(
    key=train_sampler_key,
    dataset=train_dataset,
    num_epochs=max(num_epochs, 1) if num_epochs is not None else 1,
  )
  eval_sampler = sampler_cls(
    key=eval_sampler_key,
    dataset=eval_dataset,
    num_epochs=1,
  )

  #########
  # Model setup.
  model = models.MLP(
    in_features=num_dimensions,
    hidden_features=(num_hiddens, num_hiddens),
    out_features=2,
    activation=jax.nn.relu,
    key=model_key,
    init_scale=init_scale,
  )

  logging.info(f"Model:\n{model}\n")

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
      model=eqx.nn.inference_mode(model, value=True),
      key=eval_key,
      batch_size=eval_batch_size,
    ),
  )

  # Training starts at iteration 1.
  next(itercount)
  evaluation_interval = len(train_sampler) // train_batch_size // evaluations_per_epoch
  if evaluation_interval == 0:
    msg = "Too many `evaluations_per_epoch`."
    raise ValueError(msg)

  logging.info("\nStarting training...")
  # for epoch in range(num_epochs):
  loss = np.inf
  epoch = 0
  while loss > 9e-2:
    start_time = time.time()

    for i, (x, y) in enumerate(batcher(train_sampler, train_batch_size)):
      _ = i + epoch * len(train_sampler) // train_batch_size

      (train_key,) = jax.random.split(train_key, 1)
      train_step_num = int(next(itercount))
      train_loss, model, opt_state = train_step(
        model,
        optimizer,
        opt_state,
        x,
        y,
        train_key,
      )

      if train_step_num % evaluation_interval == 0 or i + 1 == len(train_sampler):
        metrics.append(
          evaluate(
            iteration=train_step_num,
            dataset_split="eval",
            sampler=eval_sampler,
            model=eqx.nn.inference_mode(model, value=True),
            key=eval_key,
            batch_size=eval_batch_size,
          ),
        )

    loss = metrics[-1]["loss"].mean()
    epoch_time = time.time() - start_time
    logging.info(f"Epoch {epoch} in {epoch_time:0.2f} seconds.")
    epoch += 1
    if epoch > 1000:
      break

  logging.info("Training finished.")

  # TODO(eringrant): Simplify given only a single eval set.
  metrics_df = pd.concat(
    pd.concat(sampler_metrics).assign(dataset=sampler_name)
    for sampler_name, sampler_metrics in {"eval": metrics}.items()
  )
  metrics_df = metrics_df[
    metrics_df.columns[[-1, *list(range(len(metrics_df.columns) - 1))]]
  ]
  # metrics_df["dataset"] = metrics_df["dataset"].astype(
  metrics_df["dataset"] = metrics_df["dataset"].astype(
    CategoricalDtype(categories=("train", "eval"), ordered=True),
  )

  return model, metrics_df
