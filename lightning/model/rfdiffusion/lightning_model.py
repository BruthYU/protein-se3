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

    def _preprocess(self, seq, xyz_t, t, motif_mask, repack=False):

        """
        Function to prepare inputs to diffusion model

            seq (L,22) one-hot sequence 

            msa_masked (1,1,L,48)

            msa_full (1,1,L,25)

            xyz_t (L,14,3) template crds (diffused) 

            t1d (1,L,28) this is the t1d before tacking on the chi angles:
                - seq + unknown/mask (21)
                - global timestep (1-t/T if not motif else 1) (1)

                MODEL SPECIFIC:
                - contacting residues: for ppi. Target residues in contact with binder (1)
                - empty feature (legacy) (1)
                - ss (H, E, L, MASK) (4)

            t2d (1, L, L, 45)
                - last plane is block adjacency
    """
        batch_size = seq.shape[0]
        L = seq.shape[1]

        ##################
        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((batch_size, 1, L, 48))
        msa_masked[:, :, :, :22] = seq[:, None]
        msa_masked[:, :, :, 22:44] = seq[:, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0

        ################
        ### msa_full ###
        ################
        msa_full = torch.zeros((batch_size, 1, L, 25))
        msa_full[:, :, :, :22] = seq[:, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0

        ###########
        ### t1d ###
        ########### 

        # Here we need to go from one hot with 22 classes to one hot with 21 classes (last plane is missing token)
        t1d = torch.zeros((batch_size, 1, L, 21))

        seqt1d = torch.clone(seq)
        for batch_idx in range(batch_size):
            for idx in range(L):
                if seqt1d[batch_idx, idx, 21] == 1:
                    seqt1d[batch_idx, idx, 20] = 1
                    seqt1d[batch_idx, idx, 21] = 0

        t1d[:, :, :, :21] = seqt1d[:, None, :, :21]

        # Set timestep feature to 1 where diffusion mask is True, else 1-t/T
        timefeature = 1 - t[:, None].repeat(1, L).float() / self.diffuser_conf.T
        timefeature[motif_mask] = 1
        timefeature = timefeature[:, None, :, None]

        t1d = torch.cat((t1d, timefeature.cpu()), dim=-1).float()

        #############
        ### xyz_t ###
        #############
        xyz_t[~motif_mask][:, 3:, :] = float('nan')
        xyz_t = xyz_t[:, None].cpu()


        ###########
        ### t2d ###
        ###########
        t2d = xyz_to_t2d(xyz_t)

        ###########      
        ### idx ###
        ###########
        idx = torch.tensor(list(range(L)))[None,:].repeat(batch_size, 1)

        ###############
        ### alpha_t ###
        ###############
        seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
        alpha, _, alpha_mask, _ = util.get_torsions(xyz_t.reshape(-1, L, 27, 3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP,
                                                    REF_ANGLES)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[..., 0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(-1, 1, L, 10, 2)
        alpha_mask = alpha_mask.reshape(-1, 1, L, 10, 1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(-1, 1, L, 30)

        # put tensors on device
        msa_masked = msa_masked.to(self.device)
        msa_full = msa_full.to(self.device)
        seq = seq.to(self.device)
        xyz_t = xyz_t.to(self.device)
        idx = idx.to(self.device)
        t1d = t1d.to(self.device)
        t2d = t2d.to(self.device)
        alpha_t = alpha_t.to(self.device)

        processed_info = {'msa_masked':msa_masked, 'msa_full': msa_full, 'seq_in': seq,
                          'xt_in': torch.squeeze(xyz_t, dim=1), 'idx_pdb':idx,
                          't1d': t1d, 't2d':t2d, 'xyz_t':xyz_t, 'alpha_t': alpha_t}


        # return msa_masked, msa_full, seq, torch.squeeze(xyz_t, dim=1), idx, t1d, t2d, xyz_t, alpha_t
        return processed_info


    def loss_fn_normal(self, batch):
        seq_init = batch['input_seq_onehot'].float()
        motif_mask = batch['motif_mask'].bool()
        # plain mode, t ranges from 1 to T, index from 0 to T-1
        t = batch['t'].squeeze() - 1
        x_t = torch.stack([batch['fa_stack'][idx, t_idx] for idx, t_idx in enumerate(t)])
        processed_info = self._preprocess(seq_init, x_t, t, motif_mask)

        processed_info['xyz_t'] = torch.zeros_like(processed_info['xyz_t'])
        t2d_44 = torch.zeros_like(processed_info['t2d'])
        # No effect if t2d is only dim 44
        processed_info['t2d'][...,:44] = t2d_44

        logits, logits_aa, logits_exp, xyz, alpha_s, lddt \
            = self.model(processed_info['msa_masked'],
                         processed_info['msa_full'],
                         processed_info['seq_in'],
                         processed_info['xt_in'],
                         processed_info['idx_pdb'],
                         t1d=processed_info['t1d'],
                         t2d=processed_info['t2d'],
                         xyz_t=processed_info['xyz_t'],
                         alpha_t=processed_info['alpha_t'],
                         msa_prev=None,
                         pair_prev=None,
                         state_prev=None,
                         t=torch.tensor(t),
                         motif_mask=motif_mask)

    def loss_fn_self_conditioning(self, batch):
        seq_init = batch['input_seq_onehot'].float()
        motif_mask = batch['motif_mask'].bool()
        # train with self-conditioning, t+1 ranges from 1 to T
        t_plus_1 = batch['t'].squeeze()
        # modify: t+1 ranges from 2 to T, index from 1 to T-1
        t_plus_1 = torch.where(t_plus_1 == 1, 2, t_plus_1) - 1
        x_t_plus_1 = torch.stack([batch['fa_stack'][idx, t_idx] for idx, t_idx in enumerate(t_plus_1)])
        # t ranges from 1 to T-1, index from 0 to T-2
        t = t_plus_1 - 1


        # get_px0_prev
        plus_processed_info = self._preprocess(seq_init, x_t_plus_1, t_plus_1, motif_mask)
        with torch.no_grad():
            msa_prev, pair_prev, px0, state_prev, alpha_prev \
                = self.model(plus_processed_info['msa_masked'],
                             plus_processed_info['msa_full'],
                             plus_processed_info['seq_in'],
                             plus_processed_info['xt_in'],
                             plus_processed_info['idx_pdb'],
                             t1d=plus_processed_info['t1d'],
                             t2d=plus_processed_info['t2d'],
                             xyz_t=plus_processed_info['xyz_t'],
                             alpha_t=plus_processed_info['alpha_t'],
                             msa_prev=None,
                             pair_prev=None,
                             state_prev=None,
                             t=torch.tensor(t_plus_1),
                             return_raw=True,
                             motif_mask=motif_mask)

        prev_pred = torch.clone(px0)

        # get x_t from denoiser
        _, px0 = self.allatom(torch.argmax(plus_processed_info['seq_in'], dim=-1), px0, alpha_prev)
        px0 = px0.squeeze()[:, :14]
        # Generate Next Input
        x_t, px0 = self.denoiser.get_next_pose(
            xt=x_t_plus_1,
            px0=px0,
            t=t_plus_1,
            diffusion_mask=self.mask_str.squeeze(),
            align_motif=self.inf_conf.align_motif,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input
        )

        # Forward Pass
        processed_info = self._preprocess(seq_init, x_t, t, motif_mask)
        B, N, L = processed_info['xyz_t'].shape[:3]
        zeros = torch.zeros(B, N, L, 24, 3).float().to(processed_info['xyz_t'].device)
        processed_info['xyz_t'] = torch.cat((prev_pred.unsqueeze(1), zeros), dim=-2)  # [B,T,L,27,3]
        t2d_44 = xyz_to_t2d(processed_info['xyz_t'])  # [B,T,L,L,44]
        # No effect if t2d is only dim 44
        processed_info['t2d'][...,:44] = t2d_44

        logits, logits_aa, logits_exp, xyz, alpha_s, lddt \
            = self.model(processed_info['msa_masked'],
                         processed_info['msa_full'],
                         processed_info['seq_in'],
                         processed_info['xt_in'],
                         processed_info['idx_pdb'],
                         t1d=processed_info['t1d'],
                         t2d=processed_info['t2d'],
                         xyz_t=processed_info['xyz_t'],
                         alpha_t=processed_info['alpha_t'],
                         msa_prev=None,
                         pair_prev=None,
                         state_prev=None,
                         t=torch.tensor(t),
                         motif_mask=motif_mask)


    def training_step(self, batch, batch_idx, **kwargs):
        if self.exp_conf.self_conditioning_percent < random.random():
            self.loss_fn_normal(batch)
        else:
            self.loss_fn_self_conditioning(batch)







        pass



