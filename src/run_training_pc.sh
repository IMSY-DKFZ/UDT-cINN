#!/bin/sh

export RUN_BY_BASH="True"

export EXPERIMENT_NAME="$1"

export DATA_BASE_PATH="/home/kris/Work/Data/domain_adaptation_simulations"

export SAVE_DATA_PATH="/home/kris/Work/Data/Test/"

export PATH_MICCAI_23_INTERMEDIATES="/home/kris/Work/Data/domain_adaptation_simulations/HSI_Data/organ_data/intermediates"
export PATH_MICCAI_23_PROJECT="/home/kris/Work/Data/domain_adaptation_simulations/HSI_Data/organ_data"
export PATH_MICCAI_23_SEMANTIC_DATASET="/home/kris/Work/Data/domain_adaptation_simulations/HSI_Data/organ_data/intermediates/semantic"

export PYTHON_PATH="$PWD"

python3 train.py "${@:2}"
