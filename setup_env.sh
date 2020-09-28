#!/bin/bash
# Execute with `source setup_env.sh`.

# Set shell variable PYTHONPATH.
export PYTHONPATH=$(pwd):$PYTHONPATH  # add path for current module
SUBMODULES=  # space-separateed list: "SUBMODULE1 SUBMODULE2 ..."
for submodule in $SUBMODULES
do
    export PYTHONPATH=$(pwd)/${submodule}:$PYTHONPATH  # add paths for submodules
done

# Activate the conda environment.
REPO_NAME=$(basename -s .git `git config --get remote.origin.url`)
conda activate $REPO_NAME
