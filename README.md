# coding-project-template

This repository template is useful for developing Python packages with pre-commit autoformatting.

## Development setup

1. (*Optional*:) Create a repo-specific Anaconda environment via `conda create -n $(basename -s .git $(git config --get remote.origin.url)) python=3.9`.

1. Install the development requirements via `pip install -r requirements_dev.txt`.

1. (*Optional*:) Run `pre-commit install && pre-commit run` to install, update, and cache all pre-commit tools.

## Execution setup

1. Install the package requirements via `pip install -r requirements.txt`.

1. Execute the setup script via `source setup_env.sh`. (If Anaconda is not in use, the last command will fail, but can be safely ignored.)
