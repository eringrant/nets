"""Hyperparameter searches."""
from typing import Any
from collections.abc import Iterator
from collections.abc import Sequence
from jax.random import KeyArray

import abc
import math
import numpy as np

import jax


class Param(metaclass=abc.ABCMeta):
  """Abstract class for hyperparameters."""

  @abc.abstractmethod
  def generate(self, key: KeyArray):
    """Return a value for the hyperparameter."""
    return


class FixedParam(Param):
  """A hyperparameter with a fixed value."""

  def __init__(self, value: Any):
    """Initialize a fixed hyperparameter."""
    super().__init__()
    self._value = value

  def __repr__(self):
    """Return a string representation of the hyperparameter."""
    cls = self.__class__.__name__
    return f"{cls}({self._value})"

  def generate(self, key: KeyArray) -> Any:
    """Return the fixed value of the hyperparameter."""
    del key
    return self._value

  def __len__(self) -> int:
    """Return the length of the hyperparameter."""
    return 1

  def __iter__(self) -> Iterator:
    """Return an iterator over the hyperparameter."""
    yield from [self._value]


class EnumParam(Param):
  """A hyperparameter with a fixed set of values."""

  def __init__(self, possible_values: Sequence):
    """Initialize an enumerated hyperparameter."""
    super().__init__()
    self.possible_values = possible_values

  def __repr__(self):
    """Return a string representation of the hyperparameter."""
    cls = self.__class__.__name__
    return f"{cls}({self.possible_values})"

  def generate(self, key: KeyArray) -> Any:
    """Return a random value from the set of possible values."""
    return self.possible_values[jax.random.choice(key, len(self.possible_values))]

  def __len__(self) -> int:
    """Return the length of the hyperparameter."""
    return len(self.possible_values)

  def __iter__(self) -> Iterator:
    """Return an iterator over the hyperparameter."""
    yield from self.possible_values


class UniformParam(Param):
  """A hyperparameter with a uniform distribution."""

  def __init__(self, min_value: float | int, max_value: float | int):
    """Initialize a uniform hyperparameter."""
    super().__init__()
    if min_value >= max_value:
      raise ValueError(f"Invalid minimum or maximum values: {min_value} >= {max_value}")
    if type(min_value) != type(max_value):
      raise ValueError(
        f"Conflicting data types: {type(min_value)} != {type(max_value)}"
      )

    self.min_value = min_value
    self.max_value = max_value

  def __repr__(self):
    """Return a string representation of the hyperparameter."""
    cls = self.__class__.__name__
    return f"{cls}({self.min_value, self.max_value})"

  def generate(self, key: KeyArray) -> float | int:
    """Return a random value from the uniform distribution."""
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
  """A hyperparameter with a log-uniform distribution."""

  def __init__(self, min_value: float, max_value: float, base: float = math.e):
    """Initialize a log-uniform hyperparameter."""
    if type(min_value) != type(max_value):
      raise ValueError(
        f"Received conflicting data types: {type(min_value)} != {type(max_value)}"
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
    """Return a string representation of the hyperparameter."""
    cls = self.__class__.__name__
    return f"{cls}({self.exp_min_value, self.exp_max_value, self.base})"

  def generate(self, key: KeyArray) -> float:
    """Return a random value from the log-uniform distribution."""
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
