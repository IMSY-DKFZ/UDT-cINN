import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import os
from datetime import datetime

from src.trainers import CondinitionalDomainAdaptationINN, GANDomainAdaptationInn, UNIT, GanCondinitionalDomainAdaptationINN
from src.data import DomainAdaptationDataModule
from src.utils.config_io import load_config


try:
    run_by_bash: bool = bool(os.environ["RUN_BY_BASH"])
    print("This runner script is invoked in a bash script!")
except KeyError:
    run_by_bash: bool = False

if run_by_bash:
    CONFIG_PATH = os.environ['CONFIG_PATH']
    SAVE_DATA_PATH = os.environ["SAVE_DATA_PATH"]
    EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
    DATA_DIR_A = os.environ["DATA_DIR_A"]
    DATA_DIR_B = os.environ["DATA_DIR_B"]

else:
    # CONFIG_PATH = "/home/kris/Work/Repositories/dreherk/DomainAdaptation/domain_adaptation/configs/unit_conf.yaml"
    CONFIG_PATH = "/home/kris/Work/Repositories/dreherk/DomainAdaptation/domain_adaptation/configs/unit_mnist_conf.yaml"
    SAVE_DATA_PATH = "/home/kris/Work/Data/Test"
    EXPERIMENT_NAME = "Domain_Adaptation_Run"
    # DATA_DIR_A = "/home/kris/Work/Data/domain_adaptation_simulations/preprocessed_data/DAS"
    DATA_DIR_A = "/home/kris/Work/Data/MNIST"
    # DATA_DIR_B = "/home/kris/Work/Data/domain_adaptation_simulations/preprocessed_data/TR"
    DATA_DIR_B = "/home/kris/Work/Data/USPS"

config = load_config(CONFIG_PATH)
time_stamp = datetime.now()
time_stamp = time_stamp.strftime("%Y_%m_%d_%H_%M_%S")

save_path = os.path.join(SAVE_DATA_PATH, EXPERIMENT_NAME)
config["save_path"] = save_path
config["data"]["data_dir_a"] = DATA_DIR_A
config["data"]["data_dir_b"] = DATA_DIR_B

pl.seed_everything(config.seed)

data_module = DomainAdaptationDataModule(config)
model = UNIT(experiment_config=config)
logger = TensorBoardLogger(save_dir=save_path, name=time_stamp)
logger.log_hyperparams(config)

trainer = pl.trainer.Trainer(gpus=1, max_epochs=config.epochs, logger=logger, callbacks=[],
                             num_sanity_val_steps=0, check_val_every_n_epoch=1,
                             limit_val_batches=1, gradient_clip_val=0.5,
                             deterministic=False)
trainer.test(model=model,
             ckpt_path="/home/kris/Work/Data/DA_results/unit_mnist/2022_09_07_13_18_18/version_0/checkpoints/epoch=999-step=131999.ckpt",
             datamodule=data_module)
