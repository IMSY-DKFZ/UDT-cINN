#!/bin/sh

export RUN_BY_BASH="True"

export EXPERIMENT_NAME="$1"

export DATA_BASE_PATH="/path/to/publication_data/simulated_data/PAT/"

export SAVE_DATA_PATH="/path/to/save/folder/"

export HSI_DATA_PATH="/path/to/publication_data/simulated_data/HSI_Data/"
export UDT_cINN_PROJECT_PATH="/path/to/publication_data/"

export PYTHON_PATH="$PWD"

python3 train.py "${@:2}"
