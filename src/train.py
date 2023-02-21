import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.backends.cudnn as cudnn
import os
from datetime import datetime

from src.trainers import get_model
from src.data import get_data_module
from src.utils.config_io import load_config, get_conf_path
from src.utils.parser import DomainAdaptationParser

import matplotlib
matplotlib.use('Agg')

cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
# torch.set_float32_matmul_precision("high")


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
    EXPERIMENT_NAME = "waic_hsi"
    SAVE_DATA_PATH = "/home/kris/Work/Data/DA_results"
    DATA_BASE_PATH = "/home/kris/Work/Data/domain_adaptation_simulations"
    PYTHON_PATH = "/home/kris/Work/Repositories/miccai23/src"


config_path = get_conf_path(PYTHON_PATH, EXPERIMENT_NAME)
config = load_config(config_path)
time_stamp = datetime.now()
time_stamp = time_stamp.strftime("%Y_%m_%d_%H_%M_%S")

save_path = os.path.join(SAVE_DATA_PATH, EXPERIMENT_NAME)
config["save_path"] = os.path.join(save_path, time_stamp)
config["data_base_path"] = DATA_BASE_PATH

parser = DomainAdaptationParser(config=config)
config = parser.get_new_config()

pl.seed_everything(config.seed)

data_module = get_data_module(experiment_name=EXPERIMENT_NAME)
enable_test_data = False
if isinstance(data_module, tuple):
    enable_test_data = True
    test_data_manager = data_module[1]
    data_module = data_module[0]
model = get_model(experiment_name=EXPERIMENT_NAME)

data_module = data_module(experiment_config=config)
model = model(experiment_config=config)
logger = TensorBoardLogger(save_dir=save_path, name=time_stamp)
logger.log_hyperparams(config)

trainer = pl.trainer.Trainer(accelerator='gpu', devices=1, max_epochs=config.epochs, logger=logger,
                             callbacks=[ModelCheckpoint(save_top_k=-1, every_n_epochs=50)],
                             num_sanity_val_steps=0, check_val_every_n_epoch=1,
                             limit_val_batches=1,
                             gradient_clip_val=0.1, gradient_clip_algorithm="value",
                             deterministic=False)
trainer.fit(model, datamodule=data_module)

# trainer.predict(model, data_module.val_dataloader())

if enable_test_data and config.get("test_run"):
    with test_data_manager(data_module):
        trainer.test(model, datamodule=data_module)
else:
    trainer.test(model, datamodule=data_module)
