# Unsupervised Domain Transfer with Conditional Invertible Neural Networks
 This repository contains code for the paper "Unsupervised Domain Transfer with Conditional Invertible Neural Networks", see 
 [MICCAI proceedings](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_73) that was published at MICCAI 2023.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── configs        <- Config files for training models
    │   ├── data           <- Scripts to download or generate data
    │   ├── evluation      <- Scripts for model evaluation
    │   ├── features       <- Scripts to calculate features of data
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   └── trainers       <- pytorch lightning trainer classes
    │   └── utils          <- util functions
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

* [Getting started](#installation-and-setup)
* [Use pretrained models](#use-pretrained-models)
* [Run experiments](#run-experiments)

# Installation and setup
To install this repository and its dependencies, you need to run the following commands:
1. Make sure that you have your preferred virtual environment activated with one of the following commands:
    * `virtualenv udt` and then `source udt/bin/activate`
    * `conda create -n udt`
2. Install the project by running `pip install .` in the `UDT-cINN` directory.

To run the experiments you will have to define the environment variables indicated below, this can be done via 
`export` commands from the terminal or by populating these variables in an `.env` file at the root directory of this 
repository (do not commit `.env` file); the latter option is the recommended one.

```dotenv
UDT_cINN_PROJECT_PATH="<path to where you put the downloaded data, e.g. /home/user/data>"
SAVE_DATA_PATH="<path to where you want to save the data>"
HSI_DATA_PATH="<path to project path plus path to HSI data, e.g /home/user/data/simulated_data/HSI_Data>"
```

# Use pretrained models

We uploaded the pretrained models and all the simulated data used for the paper on [Zenodo](https://doi.org/10.5281/zenodo.8419563).
Download the .zip file, unpack it and note the path to where you saved it in the `.env` file.
For inference, run the `inference.py` script. 
This will use the pretrained models for both PAT and HSI in order to generate realistic-looking PAT and HSI data based on the simulations.
The output will be stored where the pretrained models are saved.
For HSI, however, you need to first create dummy real data with the `generate_dummy_hsi_data.py` in the `src/data` folder as follows:
```
python3 generate_dummy_hsi_data.py --generate --output /path/to/publication_data/simulated_data/HSI_Data/intermediates/semantic_v2/
```

Afterwards you can run:

```
python3 inference.py
```

# Run experiments
In order to fully run all the experiments in the paper, real data is needed which will be released in the future.
However, it is possible to run trainings in case you provide your own data.
For PAT, the data is expected to in "/path/to/publication_data/simulated/data/PAT/real_images".
For HSI, the data is expected to in "/path/to/home/publication_data/simulated_data/HSI_Data/intermediates/semantic_v2/".

The easiest way of training any model in this repo is by using the `run_training_pc.sh` script.
For example, running a training with the gan conditional cINN would work like this:

`bash run_training_pc.sh gan_cinn`

## Models

In the src/models folder, the basic building blocks (models) and helper functions for all the networks used in this repo can be found.
These are usually `torch.nn.Module`'s. based on these, fully connected and convolutional VAE's, UNIT's, GAN's and (c)INN's can be constructed.

## Trainers

In the src/trainers folder, the `pytorch_lightning` trainer's can be found for both HSI and PAT.
They are written in an inheritance scheme that follows: `DomainAdaptationTrainerBase` (-> `DAInnBase`) -> final trainer.
The parent classes usually contain loss or data loading functionalities as well as visualization functions. 
Every final trainer imports its neural network from the src/models folder either in the `__init__` or the `build_model` method.  

These models are configured with a config.yaml.

## Config

Each trainer has its own config in the src/configs with the naming convention {model_name}_{data_set_type}_conf.yaml.
The first line of each config contains the `experiment_name`. 
This keyword can be used in combination with the `run_training_pc.sh` script for training every model.
The configs contain all necessary information for each training.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
