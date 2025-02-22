import torch
import datetime
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import math
import argparse
import shutil

import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.loggers as plog
from pytorch_lightning.callbacks import Callback
from model import MInterface
from data import DInterface
import logging
import wandb
import sys
sys.path.append('..')
print(sys.path)
LOG = logging.getLogger(__name__)




def load_callbacks(conf):
    callback_list = []
    # Checkpoint Callback
    callback_list.append(plc.ModelCheckpoint(
        monitor= conf.experiment.monitor,
        filename='best-{epoch:02d}-{train_loss:.3f}',
        save_top_k=2,
        mode='min',
        save_last=True,
        every_n_epochs=conf.experiment.ckpt_freq,
        dirpath=f'./checkpoints'
    ))
    # Learning Rate Callback
    if conf.experiment.lr_scheduler:
        callback_list.append(plc.LearningRateMonitor(
            logging_interval=None))
    # Epoch callback
    # callback_list.append(MethodCallback(conf.method_name))
    return callback_list


@hydra.main(version_base=None, config_path="config", config_name="train")
def run(conf: DictConfig) -> None:
    pl_logger = None
    if conf.experiment.use_wandb:
        # Change wandb working dir to hydra chdir
        os.environ["WANDB_DIR"] = os.path.abspath(os.getcwd())
        wandb.login(key="7878871205533d1968d0e0736c7a47eb50d4ac69")
        pl_logger = WandbLogger(project=f"Lit-ProteinDF", log_model='all')


    pl.seed_everything(conf.experiment.seed)
    data_interface = DInterface(conf)
    data_interface.datamodule.setup()
    model_interface = MInterface(conf)



    trainer_config = {
        'devices': -1,  # Use all available GPUs
        # 'precision': 'bf16',  # Use 32-bit floating point precision
        'precision': '32',
        'max_epochs': conf.experiment.num_epoch,  # Maximum number of epochs to train for
        'num_nodes': 1,  # Number of nodes to use for distributed training
        "strategy": conf.experiment.strategy,
        "accumulate_grad_batches": 1,
        'accelerator': 'cuda',
        'callbacks': load_callbacks(conf),
        'use_distributed_sampler': conf.experiment.use_distributed_sampler,
        'check_val_every_n_epoch': conf.experiment.check_val_every_n_epoch,
        'logger': pl_logger
    }

    trainer = Trainer(**trainer_config)

    trainer.fit(model_interface.model, data_interface.datamodule)
    print(trainer_config)


if __name__ == '__main__':
    run()