from typing import Any
from collections.abc import Callable
from collections.abc import Iterable
from typing import Literal

import asyncio
import datetime
import pickle
import logging
import os
import pandas as pd
from pathlib import Path
import pprint
from tqdm.asyncio import tqdm

import submitit

from nets import simulate
from nets.launch import analyze
from nets.launch import configs


# Ignore warnings about invalid column names for PyTables.
import warnings
from tables import NaturalNameWarning

warnings.filterwarnings("ignore", category=NaturalNameWarning)


def augment_df_with_kwargs(func):
  def wrapped(**kwargs):
    results_df = func(**kwargs)
    kwargs_df = pd.DataFrame(kwargs, index=(0,))
    return kwargs_df.merge(results_df, how="cross")

  return wrapped


class Executor(submitit.AutoExecutor):
  def starmap_array(self, fn: Callable, iterable: Iterable[Any]) -> list[Any]:
    submissions = [
      submitit.core.utils.DelayedSubmission(fn, **kwargs) for kwargs in iterable
    ]
    if len(submissions) == 0:
      print("Received an empty job array")
      return []
    return self._internal_process_submissions(submissions)


class IndexedAsyncJobProxy(submitit.core.core.AsyncJobProxy):
  """Return the job and the result."""

  async def result(self, poll_interval: int | float = 1):
    await self.wait(poll_interval)
    return self.job, self.job.result()


def get_timestamp():
  """Return a date and time `str` timestamp."""
  return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")


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
  slurm_exclude: int | None = None,
) -> submitit.Executor:
  """Return a `submitit.Executor` with the given parameters."""
  if gpus_per_node > 4:
    raise ValueError("The cluster has no more than 4 GPUs per node.")

  if gpus_per_node > 0:
    if slurm_partition != "gpu":
      raise ValueError("GPUs requested but GPU partition not specified.")

    slurm_setup = [
      "nvidia-smi",
      "echo",
      "printenv | grep LD_LIBRARY_PATH",
      "echo",
    ]
  else:
    slurm_setup = None

  executor = Executor(
    folder=os.path.join(log_dir, "%j"),
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


def submit_jobs(executor: Executor, cfg: configs.Config):
  logging.info(f"Using config {pprint.pformat(cfg)}.")

  # Launch jobs.
  logging.info("Launching jobs...")
  jobs = executor.starmap_array(
    augment_df_with_kwargs(simulate.simulate),
    cfg,
  )
  logging.info(f"Waiting for {len(jobs)} jobs to terminate...")

  # Dump the config at root.
  job_root = executor.folder.parent
  with open(os.path.join(job_root, "config.pkl"), "wb") as f:
    pickle.dump(cfg, f)

  async def async_annotate():
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
    return results_paths

  results_paths = asyncio.run(async_annotate())
  logging.info("All jobs terminated.")

  # Last step: Try to concatenate all results into a single HDF file.
  # This might error out depending on the joint size of results and the
  # available memory on the current machine.
  analyze.save_result(job_root, pd.concat(map(pd.read_hdf, results_paths)))

  return jobs
