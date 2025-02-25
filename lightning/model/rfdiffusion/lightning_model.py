import hydra.utils
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from lightning.model.rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
from lightning.model.rfdiffusion.kinematics import get_init_xyz, xyz_to_t2d
from lightning.data.rfdiffusion.diffusion import Diffuser
from preprocess.tools.chemical import seq2chars
from lightning.model.rfdiffusion.util_module import ComputeAllAtomCoords
from lightning.model.rfdiffusion.contigs import ContigMap
from lightning.sampler.rfdiffusion import symmetry
import lightning.data.rfdiffusion.denoiser as iu
from lightning.model.rfdiffusion.potentials.manager import PotentialManager
import logging
import random
import torch.nn.functional as nn
from lightning.model.rfdiffusion import util
from hydra.core.hydra_config import HydraConfig
import os
import pytorch_lightning as pl
import time
HYDRA_DIR=hydra.utils.get_original_cwd()

TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles
class rfdiffusion_Lightning_Model(pl.LightningModule):

    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()
        self._log = logging.getLogger(__name__)
        self.conf = conf
        self.exp_conf = conf.experiment
        self.model_conf = conf.model
        self.data_conf = conf.dataset
        self.diffuser_conf = conf.diffuser
        self.infer_conf = conf.inference


        self.model = self.initialize_model()



        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self._checkpoint_dir = None
        self._inference_dir = None



    def initialize_model(self):
        model_directory = f"{HYDRA_DIR}/resource/rfdiffusion/official_ckpts"
        print("WARNING: Official checkpoints will be loaded for fine-tuning.")
        self.ckpt_path = f'{model_directory}/Base_ckpt.pt'

        # Load checkpoint, so that we can assemble the config
        self.load_checkpoint()
        self.assemble_config_from_chk()
        # Now actually load the model weights into RF
        return self.load_model()

    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f'Reading checkpoint from {self.ckpt_path}')
        # print('This is inf_conf.ckpt_path')
        # print(self.ckpt_path)
        self.ckpt = torch.load(self.ckpt_path)



    def assemble_config_from_chk(self) -> None:
        """
        Function for loading model config from checkpoint directly.

        Takes:
            - config file

        Actions:
            - Replaces all -model and -diffuser items
            - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint

        This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
        This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.

        """
        # get overrides to re-apply after building the config from the checkpoint
        overrides = []
        if HydraConfig.initialized():
            overrides = HydraConfig.get().overrides.task
        print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

        for cat in ['model', 'diffuser', 'preprocess']:
            for key in self.conf[cat]:
                try:
                    print(f"USING MODEL CONFIG: self.conf[{cat}][{key}] = {self.ckpt['config_dict'][cat][key]}")
                    self.conf[cat][key] = self.ckpt['config_dict'][cat][key]
                except:
                    pass

        # add overrides back in again
        for override in overrides:
            if override.split(".")[0] in ['model', 'diffuser', 'preprocess']:
                print(
                    f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?')
                mytype = type(self.conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
                self.conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(
                    override.split("=")[1])

    def load_model(self):
        """Create RosettaFold model from preloaded checkpoint."""
        # Read input dimensions from checkpoint.
        self.d_t1d = self.conf.preprocess.d_t1d
        self.d_t2d = self.conf.preprocess.d_t2d
        model = RoseTTAFoldModule(**self.conf.model, d_t1d=self.d_t1d, d_t2d=self.d_t2d, T=self.conf.diffuser.T)
        model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            **self.exp_conf.optimizer
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.95)
        return {'optimizer':optimizer, 'lr_scheduler':scheduler}


    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()



    def training_step(self, batch, batch_idx, **kwargs):
        batch_t = batch['t'].squeeze() - 1

        noisy_xyz = torch.stack([batch['fa_stack'][idx, t] for idx,t in enumerate(batch_t)])


        pass



