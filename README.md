# nets

Equinox (JAX) neural nets.

## Installation

### Command-line

```bash
python -m pip install git+https://github.com/eringrant/nets
```

### In a requirements file

```
nets @ git+https://github.com/eringrant/nets
```

### Locally & as an editable package

```bash
git clone git@github.com:eringrant/nets.git
cd nets/
python -m pip install -e .
```

## Devtools

### Pre-commit

[`.pre-commit-config.yaml`](/.pre-commit-config.yaml) has been configured to run several autoformatters,
including the [Black](https://black.readthedocs.io/) autoformatter as well as [Flake8](https://flake8.pycqa.org/).
Run the following to install, update, and cache all pre-commit tools:

```bash
pre-commit install && pre-commit run
```
