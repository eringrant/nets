"""Submit jobs locally or to a cluster using `submitit`."""
import asyncio
import datetime
import logging
import pickle
import pprint

# Ignore warnings about invalid column names for PyTables.
import warnings
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Self

import pandas as pd
import submitit
from tables import NaturalNameWarning
from tqdm.asyncio import tqdm

from nets.launch import analyze, configs

warnings.filterwarnings("ignore", category=NaturalNameWarning)


def augment_df_with_kwargs(func: Callable) -> Callable:
  """Return a function that augments a `pd.DataFrame` with keyword arguments."""

  def wrapped(**kwargs: Mapping) -> pd.DataFrame:
    results_df = func(**kwargs)
    kwargs_df = pd.DataFrame(kwargs, index=(0,))
    return kwargs_df.merge(results_df, how="cross")

  return wrapped


class Executor(submitit.AutoExecutor):
  """A `submitit.AutoExecutor` with a custom `starmap_array` method."""

  def starmap_array(self: Self, func: Callable, iterable: Iterable) -> list[Any]:
    """A distributed equivalent of the `itertools.starmap` function."""
    submissions = [
      submitit.core.utils.DelayedSubmission(func, **kwargs) for kwargs in iterable
    ]
    if len(submissions) == 0:
      logging.info("Received an empty job array.")
      return []
    return self._internal_process_submissions(submissions)


class IndexedAsyncJobProxy(submitit.core.core.AsyncJobProxy):
  """Return the job and the result."""

  async def result(self: Self, poll_interval: float = 1) -> tuple[submitit.Job, Any]:
    """Return the job and the result."""
    await self.wait(poll_interval)
    return self.job, self.job.result()


def get_timestamp() -> str:
  """Return a date and time `str` timestamp."""
  return datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d-%H:%M")


def get_submitit_executor(
  cluster: Literal["slurm", "local", "debug"],
  log_dir: str | Path,
  timeout_min: int = 60,
  gpus_per_node: int = 1,
  cpus_per_task: int = 1,
  nodes: int = 1,
  mem_gb: int = 16,
  slurm_partition: Literal["debug", "cpu", "gpu"] | None = None,
  slurm_parallelism: int | None = None,
) -> submitit.Executor:
  """Return a `submitit.Executor` with the given parameters."""
  if gpus_per_node > 4:
    msg = "The cluster has no more than 4 GPUs per node."
    raise ValueError(msg)

  if gpus_per_node > 0:
    if slurm_partition != "gpu":
      msg = "GPUs requested but GPU partition not specified."
      raise ValueError(msg)

    slurm_setup = [
      "nvidia-smi",
      "echo",
      "printenv | grep LD_LIBRARY_PATH",
      "echo",
    ]
  else:
    slurm_setup = None

  executor = Executor(
    folder=Path(log_dir, "%j"),
    cluster=cluster,
  )

  executor.update_parameters(
    timeout_min=timeout_min,
    gpus_per_node=gpus_per_node,
    cpus_per_task=cpus_per_task,
    nodes=nodes,
    mem_gb=mem_gb,
    slurm_partition=slurm_partition,
    slurm_array_parallelism=slurm_parallelism,
    slurm_setup=slurm_setup,
  )

  return executor


def submit_jobs(
  executor: Executor,
  func: Callable,
  cfg: configs.Config,
) -> list[submitit.Job]:
  """Submit jobs via `executor` by mapping `func` over `kwargs_array`."""
  # Launch jobs.
  logging.info("Launching jobs...")
  jobs = executor.starmap_array(
    func,
    cfg,
  )
  logging.info(f"Waiting for {len(jobs)} jobs to terminate...")
  tuple(job.result() for job in jobs)
  logging.info("All jobs terminated.")

  return jobs


def submit_and_annotate_jobs(
  executor: Executor,
  func: Callable,
  cfg: configs.Config,
) -> list[submitit.Job]:
  """Submit jobs to the cluster and annotate results as they come in."""
  logging.info(f"Using config {pprint.pformat(cfg)}.")

  # Launch jobs.
  logging.info("Launching jobs...")
  jobs = executor.starmap_array(
    augment_df_with_kwargs(func),
    cfg,
  )
  logging.info(f"Waiting for {len(jobs)} jobs to terminate...")

  # Dump the config at root.
  job_root = executor.folder.parent
  if executor.cluster != "debug":
    with Path.open(Path(job_root, "config.pkl"), "wb") as f:
      pickle.dump(cfg, f)

  async def async_annotate() -> tuple[Path, ...]:
    # Annotate results as they become available.
    results_paths = []
    for aws in tqdm.as_completed([IndexedAsyncJobProxy(j).result() for j in jobs]):
      try:
        job, result = await aws
        result = analyze.postprocess_result(result, cfg)
        results_paths += [analyze.save_result(job.paths.folder, result)]
      except submitit.core.utils.UncompletedJobError as e:
        logging.info("A job failed to produce a result:")
        logging.info(f"{e}")
    return tuple(results_paths)

  results_paths = asyncio.run(async_annotate())
  logging.info("All jobs terminated.")

  if executor.cluster == "debug":
    with Path.open(Path(job_root, "config.pkl"), "wb") as f:
      pickle.dump(cfg, f)

  # Last step: Try to concatenate all results into a single HDF file.
  # This might error out depending on the joint size of results and the
  # available memory on the current machine.
  analyze.save_result(job_root, pd.concat(map(pd.read_hdf, results_paths)))

  return jobs
