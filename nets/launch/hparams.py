"""Hyperparameter searches."""
from typing import Any
from typing import Iterator
from typing import Sequence
from typing import Union
from jax.random import KeyArray

import abc
import math
import numpy as np

import jax


class Param(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def generate(self, key: KeyArray):
    """Return a value for the hyperparameter."""
    return


class FixedParam(Param):
  def __init__(self, value: Any):
    super().__init__()
    self._value = value

  def __repr__(self):
    cls = self.__class__.__name__
    return f"{cls}({self._value})"

  def generate(self, key: KeyArray) -> Any:
    del key
    return self._value

  def __len__(self) -> int:
    return 1

  def __iter__(self) -> Iterator:
    yield from [self._value]


class EnumParam(Param):
  def __init__(self, possible_values: Sequence):
    super().__init__()
    self.possible_values = possible_values

  def __repr__(self):
    cls = self.__class__.__name__
    return f"{cls}({self.possible_values})"

  def generate(self, key: KeyArray) -> Any:
    return self.possible_values[jax.random.choice(key, len(self.possible_values))]

  def __len__(self) -> int:
    return len(self.possible_values)

  def __iter__(self) -> Iterator:
    yield from self.possible_values


class UniformParam(Param):
  def __init__(self, min_value: Union[float, int], max_value: Union[float, int]):
    super().__init__()
    if min_value >= max_value:
      raise ValueError(
        "Invalid minimum or maximum values: " f"{min_value} >= {max_value}"
      )
    if type(min_value) != type(max_value):
      raise ValueError(
        "Conflicting data types: " f"{type(min_value)} != {type(max_value)}"
      )

    self.min_value = min_value
    self.max_value = max_value

  def __repr__(self):
    cls = self.__class__.__name__
    return f"{cls}({self.min_value, self.max_value})"

  def generate(self, key: KeyArray) -> Union[float, int]:
    return (
      jax.random.randint if isinstance(self.min_value, int) else jax.random.uniform
    )(
      key,
      shape=(),
      minval=self.min_value,
      maxval=self.max_value,
      dtype=type(self.min_value),
    ).item()


class LogUniformParam(UniformParam):
  def __init__(self, min_value: float, max_value: float, base: float = math.e):
    if type(min_value) != type(max_value):
      raise ValueError(
        "Received conflicting data types: " f"{type(min_value)} != {type(max_value)}"
      )

    if 0.0 < base < 1.0:
      super().__init__(
        min_value=np.log(max_value) / np.log(base),
        max_value=np.log(min_value) / np.log(base),
      )
    elif base > 1.0:
      super().__init__(
        min_value=np.log(min_value) / np.log(base),
        max_value=np.log(max_value) / np.log(base),
      )
    else:
      raise ValueError(f"Invalid base: {base}")

    self.exp_min_value = min_value
    self.exp_max_value = max_value
    self.base = base

  def __repr__(self):
    cls = self.__class__.__name__
    return f"{cls}({self.exp_min_value, self.exp_max_value, self.base})"

  def generate(self, key: KeyArray) -> float:
    return (
      np.array(self.base ** super().generate(key))
      .astype(type(self.exp_min_value))
      .item()
    )


if __name__ == "__main__":
  print(FixedParam(0))
  print(EnumParam(range(10)))
  print(UniformParam(0.0, 1.0))
  print(UniformParam(0, 10))
  print(LogUniformParam(1, 10, 10))
  print(LogUniformParam(1.0, 10.0, math.e))
