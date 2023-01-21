#!/bin/sh

source /home/dreherk/.bashrc

export RUN_BY_BASH="True"

export EXPERIMENT_NAME="$1"

export DATA_BASE_PATH="/dkfz/cluster/gpu/data/OE0176/dreherk/domain_adaptation_data"

export SAVE_DATA_PATH="/dkfz/cluster/gpu/checkpoints/OE0176/dreherk/domain_adaptation_results"

export PYTHON_PATH="$PWD"

python3 train.py "${@:2}"
