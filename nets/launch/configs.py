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
from typing import Any
from collections.abc import Mapping
from itertools import count

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields

from nets.launch.hparams import Param


import numpy as np
import jax


# TODO(eringrant): Re-enable grid search if all hyperparameters are enums.
# TODO(eringrant): Idea: Make `num_configs` required iff hparams are not enums.
@dataclass(frozen=True, kw_only=True)
class Config:
  """Abstract base class for configs."""

  key: jax.random.PRNGKey  # Randomness for hyperparameter sampling.
  num_configs: int  # Number of hyperparameter configurations.

  # Dataclass fields shared by all subclasses.
  seed: Param = field(init=False)

  def __post_init__(self):
    """Check that types are as expected after initialization."""
    num_unique_configs = 1

    for field_ in fields(self):
      if field_.name == "key" or field_.name == "num_configs":
        continue
      current_field_value = asdict(self)[field_.name]

      if not isinstance(current_field_value, field_.type):
        current_field_type = type(current_field_value)
        raise TypeError(
          f"The field `{field_.name}` has type "
          f"`{current_field_type}` instead of `{field_.type}`."
        )

      try:
        num_unique_configs *= len(current_field_value)
      except TypeError:
        num_unique_configs *= np.inf

    if self.num_configs > num_unique_configs:
      raise ValueError(
        f"Requested {self.num_configs} hyperparameter "
        f"configurations, but only {num_unique_configs} "
        "unique configuration(s) specified."
      )

  def __len__(self) -> int:
    """Return the total number of hyperparameter configurations."""
    return self.num_configs

  def __iter__(self):
    """Iterate over `self`; needed to appear as `Iterable` to `mypy`."""
    try:
      for i in count():
        yield self[i]
    except IndexError:
      return

  def __getitem__(self, index: int) -> Mapping[str, Any]:
    """Get the hyperparameter configuration at `index`."""
    try:
      index = range(len(self))[index]
    except IndexError as e:
      raise IndexError(
        f"Index out of bounds for `Config` of length {len(self)}."
      ) from e

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
