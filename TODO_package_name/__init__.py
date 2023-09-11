"""Module-wide constants for `TODO_package_name`."""
# ruff: noqa: N999
import TODO_package_name
import os

__version__ = "0.0.1"

# Module-wide path constants as in
# https://stackoverflow.com/a/59597733.
__package_path = os.path.split(TODO_package_name.__path__[0])[0]
DATA_DIR = os.path.join(__package_path, "data")
TMP_DIR = os.path.join("/tmp", "TODO_package_name")
os.makedirs(TMP_DIR, exist_ok=True)

del TODO_package_name
del os
del __package_path
