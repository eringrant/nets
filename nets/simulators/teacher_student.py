"""Teacher-student learning dynamics."""

import logging
import pprint
import time
from collections.abc import Callable, Generator, Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax import Array

from nets import models


def batcher(sampler: Sequence, batch_size: int) -> Generator[Sequence, None, None]:
  """Batch a sequence of examples."""
  n = len(sampler)
  for i in range(0, n, batch_size):
    yield sampler[i : min(i + batch_size, n)]


def l2(pred_y: Array, y: Array) -> Array:
  """Compute squared error."""
  return jnp.sum(jnp.square(pred_y - y))


@eqx.filter_value_and_grad
def batch_objective(
  model: eqx.Module,
  x: Array,
  y: Array,
  key: Array,
) -> Array:
  """Evaluate objective on a batch of examples."""
  pred_y = jax.vmap(model)(x, key)
  loss = jax.vmap(l2)(pred_y, y)
  return loss.mean()


@eqx.filter_jit
def batch_train_step(
  key: Array,
  batch_size: int,
  teacher: eqx.Module,
  student: eqx.Module,
  optimizer: optax.GradientTransformation,
  opt_state: Array,
) -> tuple[Array, eqx.Module, Array]:
  """Update the student model on a batch of example-target pairs."""
  student_key, teacher_key = jax.random.split(key)

  x, y = jax.vmap(teacher)(jax.random.split(teacher_key, batch_size))
  loss, grads = batch_objective(
    student,
    x,
    y,
    jax.random.split(student_key, batch_size),
  )

  updates, opt_state = optimizer.update(grads, opt_state)
  student = eqx.apply_updates(student, updates)

  return loss, student, opt_state


@eqx.filter_vmap(in_axes=(0, None, None))
@eqx.filter_jit
def eval_step(
  key: Array,
  teacher: eqx.Module,
  student: eqx.Module,
) -> Mapping[str, Array]:
  """Evaluate the student model on a single example-target pair."""
  teacher_key, student_key = jax.random.split(key)

  x, y = teacher(key=teacher_key)
  y_hat = student(x, key=student_key)

  # Standard metrics.
  loss = l2(y_hat, y)

  return {
    "loss": loss,
    "accuracy @ 0.1": loss < 0.1,
    "accuracy @ 0.01": loss < 0.01,
    "accuracy @ 0.001": loss < 0.001,
    "ground truth input": x,
    "ground truth target": y,
    "predicted target": y_hat,
  }


def metrics_to_df(metrics: Mapping[str, Array]) -> pd.DataFrame:
  """Pandas-ify metrics from `eval_step` for later analysis."""
  metrics_df = pd.DataFrame.from_dict(metrics, orient="index").transpose()

  # Probe to get shape.
  num_iters = len(metrics_df)
  num_examples = metrics_df["ground truth target"][0].size

  # Determine metric structures.
  def has_shape(col: str, shape: tuple | tuple[int]) -> bool:
    a = metrics_df[col][0]
    return hasattr(a, "shape") and a.shape == shape

  def has_no_shape(col: str) -> bool:
    a = metrics_df[col][0]
    return not hasattr(a, "shape") or has_shape(col, ()) or has_shape(col, (1,))

  iterationwise_metrics = tuple(filter(has_no_shape, metrics_df.columns))
  elementwise_metrics = tuple(set(metrics_df.columns) - set(iterationwise_metrics))

  # Flatten arrays of elements.
  metrics_df = metrics_df.explode(list(elementwise_metrics)).reset_index(drop=True)
  metrics_df.insert(1, "element", list(range(num_examples)) * num_iters)

  # Try to infer datatypes.
  metrics_df = metrics_df.infer_objects()

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
  key: Array,
  teacher: eqx.Module,
  student: eqx.Module,
  batch_size: int,
  num_examples: int,
  epoch: int,
  iteration: int,
) -> pd.DataFrame:
  """Evaluate the model on `num_examples` example-target pairs."""
  metrics = {}

  # Metrics metadata.
  metrics["training epoch"] = np.full(
    shape=(num_examples,),
    fill_value=epoch,
  )
  metrics["training iteration"] = np.full(
    shape=(num_examples,),
    fill_value=iteration,
  )

  # Probing metric shapes.
  incremental_metrics = {
    metric_name: np.repeat(np.empty_like(metric_value), repeats=num_examples, axis=0)
    for metric_name, metric_value in eval_step(
      key[jnp.newaxis],
      teacher,
      student,
    ).items()
  }

  # Incremental evaluation.
  for i, eval_keys in enumerate(
    batcher(jax.random.split(key, num_examples), batch_size),
  ):
    batch_metrics = eval_step(
      eval_keys,
      teacher,
      student,
    )

    for metric_name in incremental_metrics:
      incremental_metrics[metric_name][
        i * batch_size : min((i + 1) * batch_size, num_examples)
      ] = batch_metrics[metric_name]

  metrics.update(incremental_metrics)
  return metrics_to_df(metrics)


def simulate(
  seed: int,
  # Data params.
  input_num_dimensions: int,
  output_num_dimensions: int,
  input_noise_scale: float,
  # Model params.
  teacher_num_hiddens: tuple[int, ...],
  teacher_activation_fn: Callable,
  teacher_init_scale: float,
  student_num_hiddens: tuple[int, ...],
  student_activation_fn: Callable,
  student_init_scale: float,
  # Training params.
  optimizer_fn: Callable,
  learning_rate: float,
  train_num_epochs: int,
  train_batch_size: int,
  train_num_examples: int,
  # Evaluation params.
  eval_interval: int,
  eval_batch_size: int,
  eval_num_examples: int,
) -> tuple[pd.DataFrame, ...]:
  """Simulate teacher-student learning."""
  logging.info(f"Using JAX backend: {jax.default_backend()}\n")
  logging.info(f"Using configuration: {pprint.pformat(locals())}")

  # Single source of randomness.
  teacher_key, student_key, train_key, eval_key = jax.random.split(
    jax.random.PRNGKey(seed),
    4,
  )

  #########
  # Data model setup.
  teacher = models.CanonicalTeacher(
    in_features=input_num_dimensions,
    hidden_features=teacher_num_hiddens,
    out_features=output_num_dimensions,
    activation=teacher_activation_fn,
    init_scale=teacher_init_scale,
    key=teacher_key,
  )
  teacher = eqx.nn.inference_mode(teacher, value=True)

  #########
  # Learner model setup.
  student = models.MLP(
    in_features=input_num_dimensions,
    hidden_features=student_num_hiddens,
    out_features=output_num_dimensions,
    activation=student_activation_fn,
    init_scale=student_init_scale,
    key=student_key,
  )

  logging.info(f"Teacher:\n{teacher}\n")
  logging.info(f"Student:\n{student}\n")

  #########
  # Training loop.
  optimizer = optimizer_fn(learning_rate=learning_rate)
  opt_state = optimizer.init(eqx.filter(student, eqx.is_array))

  # Bookkeeping.
  metrics = []

  last_batch_size = train_num_examples % train_batch_size
  ragged_last_batch = bool(last_batch_size)
  num_training_batches = train_num_examples // train_batch_size + ragged_last_batch

  logging.info(f"{num_training_batches} batches per epoch...")
  logging.info("\nStarting training...")
  training_start_time = time.time()

  for epoch_num in range(1, train_num_epochs + 1):
    # Reset data generation.
    iter_train_key = train_key
    for step_num in range(1, num_training_batches + 1):
      # Evaluate at the start of each epoch.
      metrics.append(
        evaluate(
          key=eval_key,
          teacher=teacher,
          student=eqx.nn.inference_mode(student, value=True),
          batch_size=eval_batch_size,
          num_examples=eval_num_examples,
          epoch=0,
          iteration=0,
        ),
      )

      # Mutate key.
      (iter_train_key,) = jax.random.split(iter_train_key, 1)

      train_loss, student, opt_state = batch_train_step(
        iter_train_key,
        train_batch_size,
        teacher,
        student,
        optimizer,
        opt_state,
      )

      if step_num % eval_interval == 0:
        metrics.append(
          evaluate(
            key=eval_key,
            teacher=teacher,
            student=eqx.nn.inference_mode(student, value=True),
            batch_size=eval_batch_size,
            num_examples=eval_num_examples,
            epoch=epoch_num,
            iteration=step_num,
          ),
        )

        loss = metrics[-1]["loss"].mean()
        logging.info(
          f"\titeration:\t{(epoch_num - 1) * num_training_batches + step_num}"
          f"\tloss:\t{loss:.4f}",
        )

  training_time = time.time() - training_start_time
  logging.info(f"Finished training in {training_time:0.2f} seconds.")

  # Evaluate at the end of training.
  if step_num % eval_interval != 0:
    metrics.append(
      evaluate(
        key=eval_key,
        teacher=teacher,
        student=eqx.nn.inference_mode(student, value=True),
        batch_size=eval_batch_size,
        num_examples=eval_num_examples,
        epoch=epoch_num,
        iteration=step_num,
      ),
    )

  metrics_df = pd.concat(metrics)

  # TODO(eringrant): Robust handling of hyperparameters.
  metrics_df["init scale"] = student_init_scale

  return student, metrics_df


# TODO(eringrant): Allow varying multiple dimensions.
def batched_metrics_to_df(
  metrics: Mapping[str, Array],
  batched_key: str,
  batched_values: tuple[float, ...],
) -> pd.DataFrame:
  """Wrapper function calling `metrics_to_df` with a leading batch dimension."""
  singleton_dfs = []

  for i, singleton_value in enumerate(batched_values):
    singleton_metrics = {
      key: value[i] if value is not None else None for key, value in metrics.items()
    }

    singleton_df = metrics_to_df(singleton_metrics)
    singleton_df[batched_key] = singleton_value
    singleton_dfs.append(singleton_df)

  return pd.concat(singleton_dfs, ignore_index=True)


batched_student_eval_step = eqx.filter_vmap(
  fun=eval_step,
  in_axes=(None, None, eqx.if_array(0)),
)


batched_student_batch_train_step = eqx.filter_vmap(
  fun=batch_train_step,
  in_axes=(None, None, None, eqx.if_array(0), None, None),
)


def batched_student_evaluate(
  key: Array,
  teacher: eqx.Module,
  students: eqx.Module,
  batch_size: int,
  num_examples: int,
  epoch: int,
  iteration: int,
  batched_key: str,
  batched_values: tuple[float, ...],
) -> pd.DataFrame:
  """Evaluate the batched student models on `num_examples` example-target pairs."""
  metrics = {}

  # Probing metric shapes.
  incremental_metrics = {
    metric_name: np.repeat(np.empty_like(metric_value), repeats=num_examples, axis=1)
    for metric_name, metric_value in batched_student_eval_step(
      key[jnp.newaxis],
      teacher,
      students,
    ).items()
  }

  # Incremental evaluation.
  for i, eval_keys in enumerate(
    batcher(jax.random.split(key, num_examples), batch_size),
  ):
    batch_metrics = batched_student_eval_step(
      eval_keys,
      teacher,
      students,
    )

    for metric_name in incremental_metrics:
      incremental_metrics[metric_name][
        :,
        i * batch_size : min((i + 1) * batch_size, num_examples),
        ...,
      ] = batch_metrics[metric_name]

  metrics.update(incremental_metrics)
  num_models = incremental_metrics["loss"].shape[0]

  # Metrics metadata.
  metrics["training epoch"] = np.full(
    shape=(num_models, num_examples),
    fill_value=epoch,
  )
  metrics["training iteration"] = np.full(
    shape=(num_models, num_examples),
    fill_value=iteration,
  )

  return batched_metrics_to_df(
    metrics,
    batched_key=batched_key,
    batched_values=batched_values,
  )


def batch_student_init_scale_simulate(
  seed: int,
  # Data params.
  input_num_dimensions: int,
  output_num_dimensions: int,
  input_noise_scale: float,
  # Model params.
  teacher_num_hiddens: tuple[int, ...],
  teacher_activation_fn: Callable,
  teacher_init_scale: float,
  student_num_hiddens: tuple[int, ...],
  student_activation_fn: Callable,
  student_init_scale: tuple[float, ...],
  # Training params.
  optimizer_fn: Callable,
  learning_rate: float,
  train_num_epochs: int,
  train_batch_size: int,
  train_num_examples: int,
  # Evaluation params.
  eval_interval: int,
  eval_batch_size: int,
  eval_num_examples: int,
) -> tuple[pd.DataFrame, ...]:
  """Simulate teacher-student learning."""
  if eval_interval > train_num_examples // train_batch_size:
    msg = "Evaluation interval must be no more than the number of training batches."
    raise ValueError(msg)

  logging.info(f"Using JAX backend: {jax.default_backend()}\n")
  logging.info(f"Using configuration: {pprint.pformat(locals())}")

  # Single source of randomness.
  teacher_key, student_key, train_key, eval_key = jax.random.split(
    jax.random.PRNGKey(seed),
    4,
  )

  #########
  # Data model setup.
  teacher = models.CanonicalTeacher(
    in_features=input_num_dimensions,
    hidden_features=teacher_num_hiddens,
    out_features=output_num_dimensions,
    activation=teacher_activation_fn,
    init_scale=teacher_init_scale,
    key=teacher_key,
  )
  teacher = eqx.nn.inference_mode(teacher, value=True)

  #########
  # Learner model setup.
  student_keys = jax.random.split(student_key, len(student_init_scale))
  student_init_scales = jnp.asarray(student_init_scale)

  @eqx.filter_vmap
  def make_students(init_scale: float, key: Array) -> eqx.Module:
    return models.MLP(
      in_features=input_num_dimensions,
      hidden_features=student_num_hiddens,
      out_features=output_num_dimensions,
      activation=student_activation_fn,
      init_scale=init_scale,
      key=key,
    )

  students = make_students(student_init_scales, student_keys)

  logging.info(f"Teacher:\n{teacher}\n")
  logging.info(f"Student:\n{students}\n")

  #########
  # Training loop.
  optimizer = optimizer_fn(learning_rate=learning_rate)
  opt_state = optimizer.init(eqx.filter(students, eqx.is_array))

  # Bookkeeping.
  metrics = []

  last_batch_size = train_num_examples % train_batch_size
  ragged_last_batch = bool(last_batch_size)
  num_training_batches = train_num_examples // train_batch_size + ragged_last_batch

  logging.info(f"{num_training_batches} batches per epoch...")
  logging.info("\nStarting training...")
  training_start_time = time.time()
  for epoch_num in range(1, train_num_epochs + 1):
    # Evaluate at the start of each epoch.
    metrics.append(
      batched_student_evaluate(
        key=eval_key,
        teacher=teacher,
        students=eqx.nn.inference_mode(students, value=True),
        batch_size=eval_batch_size,
        num_examples=eval_num_examples,
        epoch=epoch_num,
        iteration=0,
        batched_key="init scale",
        batched_values=student_init_scale,
      ),
    )

    # Reset data generation.
    iter_train_key = train_key
    for step_num in range(1, num_training_batches + 1):
      # Mutate key.
      (iter_train_key,) = jax.random.split(iter_train_key, 1)

      # Last batch may be smaller.
      if ragged_last_batch and step_num == num_training_batches:
        iter_batch_size = last_batch_size
      else:
        iter_batch_size = train_batch_size

      train_loss, students, opt_state = batched_student_batch_train_step(
        iter_train_key,
        iter_batch_size,
        teacher,
        students,
        optimizer,
        opt_state,
      )

      if step_num % eval_interval == 0:
        metrics.append(
          batched_student_evaluate(
            key=eval_key,
            teacher=teacher,
            students=eqx.nn.inference_mode(students, value=True),
            batch_size=eval_batch_size,
            num_examples=eval_num_examples,
            epoch=epoch_num,
            iteration=step_num,
            batched_key="init scale",
            batched_values=student_init_scale,
          ),
        )

        logging.info(
          f"\titeration:\t{(epoch_num - 1) * num_training_batches + step_num}",
        )
        for init_scale, loss in zip(
          student_init_scales.tolist(),
          metrics[-1].groupby("init scale").loss.mean(),
          strict=True,
        ):
          logging.info(f"\t\t\t\tinit scale:\t{init_scale:.4f}\tloss:\t{loss:.4f}")

  training_time = time.time() - training_start_time
  logging.info(f"Finished training in {training_time:0.2f} seconds.")

  # Evaluate at the end of training.
  if step_num % eval_interval != 0:
    metrics.append(
      batched_student_evaluate(
        key=eval_key,
        teacher=teacher,
        students=eqx.nn.inference_mode(students, value=True),
        batch_size=eval_batch_size,
        num_examples=eval_num_examples,
        epoch=epoch_num,
        iteration=step_num,
        batched_key="init scale",
        batched_values=student_init_scale,
      ),
    )

  metrics_df = pd.concat(metrics)
  return students, metrics_df


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  _, metrics_df = batch_student_init_scale_simulate(
    seed=0,
    input_noise_scale=1e-1,
    input_num_dimensions=5,
    output_num_dimensions=5,
    teacher_num_hiddens=(100, 100),
    teacher_activation_fn=jax.nn.tanh,
    teacher_init_scale=1e-1,
    student_num_hiddens=(100, 100),
    student_activation_fn=jax.nn.tanh,
    student_init_scale=np.logspace(-4, 1, num=20).tolist(),
    optimizer_fn=optax.sgd,
    learning_rate=1e-2,
    train_num_epochs=int(1e1),
    train_batch_size=2 * int(1e2),
    train_num_examples=int(1e4),
    eval_interval=int(1e1),
    eval_batch_size=4 * int(1e2),
    eval_num_examples=int(1e4),
  )

  import matplotlib.pyplot as plt
  import seaborn as sns

  sns.lineplot(
    data=metrics_df.round(4),
    x="training epoch",
    y="accuracy @ 0.001",
    errorbar=None,
    hue="init scale",
    markers=True,
    palette=sns.cubehelix_palette(n_colors=metrics_df["init scale"].nunique()),
  )

  plt.legend(title="student init scale", bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.tight_layout()
  plt.show()
