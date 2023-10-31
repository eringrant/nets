"""Utilities for analyzing results."""
import logging
from dataclasses import fields
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from nets.launch import configs, hparams

# Identifiers that are neither hyperparameters nor metrics.
ELEMENT_IDENTIFIER_HPARAMS = (
  "dataset",
  "sequence",
  "element",
  "training iteration",
)


def compress_df(df: pd.DataFrame) -> pd.DataFrame:
  """Compress `df` by optimizing data types."""
  for col in df.select_dtypes("category"):
    if df[col].dtype.categories.dtype is np.dtype("object"):
      df[col] = df[col].map(str)

  for col in df.select_dtypes(("int32", "int64")):
    if (
      np.iinfo(np.int16).min <= df[col].min()
      and np.iinfo(np.int16).max >= df[col].max()
    ):
      df[col] = df[col].astype("int16")
    elif (
      np.iinfo(np.int32).min <= df[col].min()
      and np.iinfo(np.int32).max >= df[col].max()
    ):
      df[col] = df[col].astype("int32")

  for col in df.select_dtypes(("float32", "float64")):
    if (
      np.finfo(np.float16).min <= df[col].min()
      and np.finfo(np.float16).max >= df[col].max()
      and df[col].abs().min() >= np.finfo(np.float16).resolution
    ):
      df[col] = df[col].astype("float16")
    elif (
      np.finfo(np.float32).min <= df[col].min()
      and np.finfo(np.float32).max >= df[col].max()
      and df[col].abs().min() >= np.finfo(np.float32).resolution
    ):
      df[col] = df[col].astype("float32")

  return df


def postprocess_result(result: pd.DataFrame, cfg: configs.Config) -> pd.DataFrame:
  """Postprocess `result` using hyperparameter search space in `cfg`."""
  categories = {}
  for field in fields(cfg):
    param = getattr(cfg, field.name)
    if field.name in {"key", "num_configs"}:
      continue
    if isinstance(param, hparams.EnumParam | hparams.FixedParam):
      categories[field.name] = CategoricalDtype(categories=param, ordered=True)

  # Optimize data types.
  for field_name in categories:
    result[field_name] = result[field_name].astype(categories[field_name])
  result = compress_df(result)
  if result.isna().to_numpy.any():
    msg = "Failed to cast."
    raise ValueError(msg)

  # Drop constant columns to save memory.
  for col in result.select_dtypes("category"):
    if len(result[col].cat.categories) == 1:
      result = result.drop(col, axis=1)

  return result


def truncate(df: pd.DataFrame, col: str, n: int = int(1e2 * 32)) -> pd.DataFrame:
  """Utility to compress a data frame by truncation."""
  return df[df[col] < n]


def load_result_from_pkl(filepath: Path) -> pd.DataFrame | None:
  """Load result from `filepath`."""
  if filepath is None:
    return None

  with Path.open(filepath, "rb") as f:
    _, result = pickle.load(f)  # noqa: S301
  if isinstance(result, pd.DataFrame):
    return result

  logging.info(f"Ignored result from {filepath}:\n{result}")
  return None


def save_result(
  job_path: Path,
  results: pd.DataFrame,
  result_filename: str = "result",
) -> Path:
  """Save `result` at `job_path`."""
  logging.info("Saving results...")
  results_path = Path(job_path, f"{result_filename}.h5")
  results.to_hdf(
    results_path,
    key=f"{result_filename}",
    mode="w",
    format="table",  # For category data types.
    complevel=9,
    index=False,
    # TODO(eringrant): Do not hardcode based on position of "element".
    data_columns=results.columns[: results.columns.tolist().index("element") + 1],
  )
  logging.info("Finished saving results at:")
  logging.info(results_path)

  logging.info(
    f"""Read like:
```
import pandas as pd
df = pd.read_hdf('{results_path}')
```
""",
  )
  return results_path


# TODO(eringrant): Generalize to arbitrary #s.
def pd_categorical_concat(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
  """DFs must have common columns."""
  for col1, col2 in zip(
    df1.select_dtypes("category"),
    df2.select_dtypes("category"),
    strict=True,
  ):
    if col1 != col2:
      msg = f"Columns {col1} and {col2} do not match."
      raise ValueError(msg)

    if df1[col1].cat != df2[col2].cat:
      union_cat = pd.Series(
        pd.api.types.union_categoricals([df1[col1], df2[col2]], ignore_order=True),
      )

      df1[col1] = df1[col1].cat.set_categories(union_cat.cat.categories)
      df2[col2] = df2[col2].cat.set_categories(union_cat.cat.categories)

  concat_df = pd.concat((df1, df2))

  if not all(
    concat_df[col].dtype == "category" for col in df1.select_dtypes("category")
  ):
    msg = "Failed to concatenate."
    raise ValueError(msg)

  return concat_df


# TODO(eringrant): Generalize to arbitrary #s.
def read_concat_hdf(f1: Path, f2: Path) -> pd.DataFrame:
  """Read and concatenate two HDFs."""
  ignore_columns = (
    "optimizer_fn",
    "learning_rate",
    "train_batch_size",
    "eval_batch_size",
    "num_epochs",
    "evaluations_per_epoch",
    "evaluate_on_test_split",
  )

  df1 = pd.read_hdf(f1, stop=0)
  df2 = pd.read_hdf(f2, stop=0)

  common_columns = tuple(set(df1.columns).intersection(df2.columns))
  common_columns = tuple(
    x for x in df1.columns if x in common_columns and x not in ignore_columns
  )

  df1 = pd.read_hdf(f1, columns=common_columns)
  df2 = pd.read_hdf(f2, columns=common_columns)

  return pd_categorical_concat(df1, df2)
