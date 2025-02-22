import os
import time
import numpy as np
import hydra
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from lightning.sampler.frameflow import utils as su
from lightning.model.frameflow.lightning_model import frameflow_Lightning_Model


torch.set_float32_matmul_precision('high')
log = su.get_pylogger(__name__)

class frameflow_Sampler:
    def __init__(self, conf: DictConfig):
        self.conf = conf
        self.exp_conf = conf.experiment
        self.infer_conf = conf.inference
        self.samples_conf = self.infer_conf.samples
        self.rng = np.random.default_rng(self.infer_conf.seed)

        ckpt_path = self.infer_conf.ckpt_path
        self.flow_module = frameflow_Lightning_Model.load_from_checkpoint(
            checkpoint_path=ckpt_path
        )
        self.flow_module.eval()
        self.flow_module.infer_conf = self.infer_conf
        self.flow_module.samples_conf = self.samples_conf
        
    def run_sampling(self):
        log.info(f'Evaluating {self.infer_conf.task}')
        if self.infer_conf.task == 'unconditional':
            eval_dataset = su.LengthDataset(self.samples_conf)
        elif self.infer_conf.task == 'scaffolding':
            eval_dataset = su.ScaffoldingDataset(self.samples_conf)
        else:
            raise ValueError(f'Unknown task {self.infer_conf.task}')
        dataloader = data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices= -1,  # Use all available GPUs
        )
        trainer.predict(self.flow_module, dataloaders=dataloader)

