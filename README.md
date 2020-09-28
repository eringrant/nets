# coding-project-template

This repository template is useful for developing Python packages with pre-commit autoformatting.

## Development setup

1. (Optional:) Create a repo-specific Anaconda environment via `conda create -n $(basename -s .git `git config --get remote.origin.url)`.

1. (Optional:) Install the package requirements via `pip install -r requirements.txt`.

1. Install the development requirements via `pip install -r requirements_dev.txt`.

1. Run `pre-commit install` to install all pre-commit tools.

1. Run `pre-commit autoupdate` to update all pre-commit tools.

1. Run `pre-commit run` to install the pre-commit environments into the cache.
