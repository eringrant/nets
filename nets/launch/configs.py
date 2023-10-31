"""Configs to pass as `kwargs` to simulators within `nets.simulators`.

This config and its subclasses can be iterated over to pass keyword arguments
to a function, say `simulate`, as follows:

To simulate a single configuration:
```
cfg = next(iter(Config(...)))
simulate(**cfg)
```

To simulate all configurations in sequence:
```
for cfg in Config(...):
  simulate(**cfg)
```

To use `submitit` to submit a job array of simulations:
```
from nets.launch.submit import Executor
executor = Executor(...)
executor.starmap_array(simulate, Config(...))
```
"""
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, fields
from itertools import count
from typing import Any, Self

import jax
import numpy as np

from nets.launch.hparams import Param


# TODO(eringrant): Re-enable grid search if all hyperparameters are enums.
# TODO(eringrant): Idea: Make `num_configs` required iff hparams are not enums.
@dataclass(frozen=True, kw_only=True)
class Config:
  """Abstract base class for configs."""

  key: jax.random.PRNGKey  # Randomness for hyperparameter sampling.
  num_configs: int  # Number of hyperparameter configurations.

  # Dataclass fields shared by all subclasses.
  seed: Param = field(init=False)

  def __post_init__(self: Self) -> None:
    """Check that types are as expected after initialization."""
    num_unique_configs = 1

    for field_ in fields(self):
      if field_.name in {"key", "num_configs"}:
        continue
      current_field_value = asdict(self)[field_.name]

      if not isinstance(current_field_value, field_.type):
        current_field_type = type(current_field_value)
        msg = (
          f"The field `{field_.name}` has type `{current_field_type}` "
          f"instead of `{field_.type}`."
        )
        raise TypeError(msg)

      try:
        num_unique_configs *= len(current_field_value)
      except TypeError:
        num_unique_configs *= np.inf

    if self.num_configs > num_unique_configs:
      msg = (
        f"Requested {self.num_configs} hyperparameter configurations, but only "
        f"{num_unique_configs} unique configuration(s) specified."
      )
      raise ValueError(msg)

  def __len__(self: Self) -> int:
    """Return the total number of hyperparameter configurations."""
    return self.num_configs

  def __iter__(self: Self) -> Iterator[Mapping[str, Any]]:
    """Iterate over `self`; needed to appear as `Iterable` to `mypy`."""
    try:
      for i in count():
        yield self[i]
    except IndexError:
      return

  def __getitem__(self: Self, index: int) -> Mapping[str, Any]:
    """Get the hyperparameter configuration at `index`."""
    try:
      index = range(len(self))[index]
    except IndexError as e:
      msg = f"Index out of bounds for `Config` of length {len(self)}."
      raise IndexError(msg) from e

    hparam_dict = asdict(self)
    del hparam_dict["key"]
    del hparam_dict["num_configs"]
    hparam_names, hparams = zip(*hparam_dict.items(), strict=True)

    hparam_samples = tuple(
      (
        hparam.generate(key)
        for key, hparam in zip(
          jax.random.split(jax.random.fold_in(self.key, index), len(hparams)),
          hparams,
          strict=True,
        )
      ),
    )

    return dict(zip(hparam_names, hparam_samples, strict=True))
