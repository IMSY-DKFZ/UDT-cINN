#!/bin/sh

export RUN_BY_BASH="True"

export EXPERIMENT_NAME="$1"

export DATA_BASE_PATH="/home/kris/Work/Data/domain_adaptation_simulations"

export SAVE_DATA_PATH="/home/kris/Work/Data/DA_results"

export PYTHON_PATH="$PWD"

python3 train.py "${@:2}"
