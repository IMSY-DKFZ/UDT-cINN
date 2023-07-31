import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from datetime import datetime

from src.trainers import get_model
from src.data import get_data_module
from src.utils.config_io import load_config, get_conf_path
from src.utils.parser import DomainAdaptationParser


try:
    run_by_bash: bool = bool(os.environ["RUN_BY_BASH"])
    print("This runner script is invoked in a bash script!")
except KeyError:
    run_by_bash: bool = False

if run_by_bash:
    EXPERIMENT_NAME = os.environ['EXPERIMENT_NAME']
    SAVE_DATA_PATH = os.environ["SAVE_DATA_PATH"]
    DATA_BASE_PATH = os.environ["DATA_BASE_PATH"]
    PYTHON_PATH = os.environ["PYTHON_PATH"]

else:
    EXPERIMENT_NAME = "gan_cinn"
    SAVE_DATA_PATH = "/home/kris/Work/Data/DA_results"
    DATA_BASE_PATH = "/home/kris/Work/Data/domain_adaptation_simulations"
    PYTHON_PATH = "/home/kris/Work/Repositories/miccai23/src"


config_path = "/home/kris/Work/Data/DA_results/gan_cinn_hsi/2023_03_01_21_58_36/version_0/hparams.yaml"
checkpoint = "/home/kris/Work/Data/DA_results/gan_cinn_hsi/2023_03_01_21_58_36/version_0/checkpoints/epoch=299-step=274800.ckpt"
config = load_config(config_path)
time_stamp = datetime.now()
time_stamp = time_stamp.strftime("%Y_%m_%d_%H_%M_%S")

save_path = os.path.join(SAVE_DATA_PATH, EXPERIMENT_NAME)
config["save_path"] = os.path.join(save_path, time_stamp)
config["data_base_path"] = DATA_BASE_PATH

parser = DomainAdaptationParser(config=config)
config = parser.get_new_config()

config.checkpoint = checkpoint
# config.noise_aug = True
# config.noise_aug_level = 0.2
# config.label_noise_level = 0.8
# config.real_labels = "random_choice"
# config.data.balance_classes = False

config.test_run = True

pl.seed_everything(config.seed)

data_module = get_data_module(experiment_name=EXPERIMENT_NAME)

if isinstance(data_module, tuple):
    test_data_manager = data_module[1]
    data_module = data_module[0]

model = get_model(experiment_name=EXPERIMENT_NAME)

data_module = data_module(experiment_config=config)
enable_test_data = True
if isinstance(data_module, tuple):
    enable_test_data = True
    test_data_manager = data_module[1]
    data_module = data_module[0]
model = model(experiment_config=config)
logger = TensorBoardLogger(save_dir=save_path, name=time_stamp)
logger.log_hyperparams(config)

trainer = pl.trainer.Trainer(accelerator='gpu', devices=1, max_epochs=config.epochs, logger=logger,
                             callbacks=[],
                             num_sanity_val_steps=0, check_val_every_n_epoch=1,
                             limit_val_batches=1,
                             gradient_clip_val=0.1, gradient_clip_algorithm="value",
                             deterministic=False)

trainer.test(model, datamodule=data_module)
