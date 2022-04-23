# coding-project-template

This repository template is useful for developing Python packages with pre-commit autoformatting.

To cite the work that this code is associated with, use:

```
@inproceedings{TODO_citation_key,
  title={TODO},
  author={TODO},
  booktitle={TODO},
  year={TODO}
}
```

## Installation

### Option #1: Conda install

To install via [Conda](https://docs.conda.io/), do:

```bash
git clone git@github.com:eringrant/TODO_package_name.git
cd TODO_package_name
conda env create --file environment.yml
```

The Conda environment can then be activated via `conda activate TODO_package_name`.

### Option: pip install

To install via [pip](https://pip.pypa.io/), do:

```bash
git clone git@github.com:eringrant/TODO_package_name.git
cd TODO_package_name
pip install -e .
```

## Devtools

### Pre-commit

[`.pre-commit-config.yaml`](/.pre-commit-config.yaml) has been configured to run several autoformatters,
including the [Black](https://black.readthedocs.io/) autoformatter as well as [Flake8](https://flake8.pycqa.org/).
Run the following to install, update, and cache all pre-commit tools:

```bash
pre-commit install && pre-commit run
```
