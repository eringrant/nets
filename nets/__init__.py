import nets
import os

__version__ = "0.0.1"

# Module-wide path constants as in
# https://stackoverflow.com/a/59597733.
__package_path = os.path.split(nets.__path__[0])[0]
DATA_DIR = os.path.join(__package_path, "data")
TMP_DIR = os.path.join("/tmp", "nets")
os.makedirs(TMP_DIR, exist_ok=True)

del nets
del os
del __package_path
