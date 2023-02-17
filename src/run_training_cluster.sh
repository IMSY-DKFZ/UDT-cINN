#!/bin/sh

source /home/dreherk/.bashrc

export RUN_BY_BASH="True"

export EXPERIMENT_NAME="$1"

export DATA_BASE_PATH="/dkfz/cluster/gpu/data/OE0176/dreherk/domain_adaptation_data"

export SAVE_DATA_PATH="/dkfz/cluster/gpu/checkpoints/OE0176/dreherk/domain_adaptation_results"

export PATH_MICCAI_23_INTERMEDIATES="/dkfz/cluster/gpu/data/OE0176/dreherk/domain_adaptation_data/HSI_Data/organ_data/intermediates"
export PATH_MICCAI_23_PROJECT="/dkfz/cluster/gpu/data/OE0176/dreherk/domain_adaptation_data/HSI_Data/organ_data"
export PATH_MICCAI_23_SEMANTIC_DATASET="/dkfz/cluster/gpu/data/OE0176/dreherk/domain_adaptation_data/HSI_Data/organ_data/intermediates/semantic"

export PYTHON_PATH="$PWD"

python3 train.py "${@:2}"
