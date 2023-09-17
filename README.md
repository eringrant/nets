# nets

Equinox (JAX) neural nets.

## Quickstart

Clone the repo:

```sh
git clone git@github.com:eringrant/nets.git
cd nets/
```

To install a Conda environment with the requisite packages on CPU:

```sh
conda env create --file environment-cpu.yml
```

To test the code a quick debug run:

```sh
python -m nets.experiments.in_context_learning.launcher_local
```

## Installation

Optionally, define a few environment variables
by adding the following to a shell configuration file such as
`~/.bashrc`, `~/.bash_profile`, `~/.bash_login`, or `~/.profile`:

```sh
export SCRATCH_HOME="..."
```

Then, follow one of two methods below to install `nets`.

### Method #1: via Conda

Use Conda to develop `nets` code directly.

#### Option #1.1: CPU-only

To install via [Mamba](https://mamba.readthedocs.io/) (recommended)
or [Conda](https://docs.conda.io/), do:

```sh
conda env create --file environment-cpu.yml
```

#### Option #1.2: On GPU on a local machine

To install via [Mamba](https://mamba.readthedocs.io/) (recommended)
or [Conda](https://docs.conda.io/) with GPU support, do:

```sh
conda env create --file environment-gpu.yml
```

#### Option #1.3: On GPU on a SLURM cluster

If working on the head node of a SLURM cluster, you will need
to create a GPU-compatible environment on a compute node with an available GPU via:

```sh
srun --partition=gpu --gres=gpu:1 conda env create -f environment-gpu.yml
```

Note that you may have to adapt the `partition` name to the available partitions on your cluster;
run `sinfo -s` to display details about partitions.

### Method #2: via Pip

Use Pip in order to install this code as a package in another project.

#### Option #2.1: Command-line

```bash
python -m pip install git+https://github.com/eringrant/nets
```

#### Option #2.2: In a requirements file

```
nets @ git+https://github.com/eringrant/nets
```

#### Option #2.3: Locally & as an editable package

```bash
git clone git@github.com:eringrant/nets.git
cd nets/
python -m pip install -e .
```

## Devtools

### Pre-commit

[`.pre-commit-config.yaml`](/.pre-commit-config.yaml) has been configured to run several autoformatters.
Run the following to install, update, and cache all pre-commit tools:

```bash
pre-commit install && pre-commit run
```
