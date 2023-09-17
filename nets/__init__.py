"""Module-wide constants for `nets`."""
import os
from pathlib import Path
import nets

__version__ = "0.0.1"

# Module-wide path constants as in
# https://stackoverflow.com/a/59597733.
__package_path = os.path.split(nets.__path__[0])[0]
DATA_DIR = Path(__package_path, "data")
TMP_DIR = Path("/tmp", "nets")
os.makedirs(TMP_DIR, exist_ok=True)

scratch_home = os.environ.get("SCRATCH_HOME")
if scratch_home is not None:
  SCRATCH_DIR = Path(scratch_home, "nets")
else:
  SCRATCH_DIR = TMP_DIR
os.makedirs(SCRATCH_DIR, exist_ok=True)

del nets
del os
del __package_path
