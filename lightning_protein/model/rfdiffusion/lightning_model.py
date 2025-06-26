import hydra.utils
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from lightning_protein.model.rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
from lightning_protein.model.rfdiffusion.kinematics import *
from lightning_protein.data.rfdiffusion.diffusion import Diffuser
from preprocess.tools.chemical import seq2chars
from lightning_protein.model.rfdiffusion.util_module import ComputeAllAtomCoords
from lightning_protein.model.rfdiffusion.contigs import ContigMap
from lightning_protein.sampler.rfdiffusion import symmetry
import lightning_protein.data.rfdiffusion.denoiser as iu
from lightning_protein.model.rfdiffusion.potentials.manager import PotentialManager
import logging
import random
import torch.nn.functional as F
import torch.nn as nn
from lightning_protein.model.rfdiffusion import util
from hydra.core.hydra_config import HydraConfig
import os
import pytorch_lightning as pl
import time
import math
from lightning_protein.model.rfdiffusion.util import rigid_from_3_points
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
        self.denoiser_conf = conf.denoiser
        self.infer_conf = conf.inference
        self.preprocess_conf = conf.preprocess


        self.model = self.initialize_model()
        self.allatom = ComputeAllAtomCoords()
        self.diffuser = Diffuser(**self.diffuser_conf)



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
        # self.assemble_config_from_chk()
        # Now actually load the model weights into RF
        return self.load_model()

    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f'Reading checkpoint from {self.ckpt_path}')
        # print('This is inf_conf.ckpt_path')
        # print(self.ckpt_path)
        self.ckpt = torch.load(self.ckpt_path)

    def construct_denoiser(self, L):
        """Make length-specific denoiser."""
        denoise_kwargs = OmegaConf.to_container(self.diffuser_conf)
        denoise_kwargs.update(OmegaConf.to_container(self.denoiser_conf))
        denoise_kwargs.update({
            'L': L,
            'diffuser': self.diffuser,
        })
        return iu.Denoise(**denoise_kwargs)



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

    def get_next_batch_poses(self, denoiser, x_t_plus_1, px0, t_plus_1, motif_mask):
        x_t_list = []
        for X, P, T, M in zip(x_t_plus_1, px0, t_plus_1, motif_mask):
            x_t, px0 = denoiser.get_next_pose(xt=X.cpu(), px0=P, t=int(T.cpu()), diffusion_mask=M.cpu(),
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)
            x_t_list.append(x_t)
        return torch.stack(x_t_list)

    def rigid_from_xyz(self, xyz):
        '''
         xyz: [B,L,3,3] or [B,L,14,3]
        '''
        xyz = xyz
        N = xyz[:, :, 0, :]
        Ca = xyz[:, :, 1, :]
        C = xyz[:, :, 2, :]
        # scipy rotation object for true coordinates
        R_true, Ca = rigid_from_3_points(N, Ca, C)
        return R_true, Ca

    def d_frame(self, xyz, xyz_0):
        B, L, N = xyz.shape[:3]
        d_clamp = self.exp_conf.d_clamp
        w_rot = self.exp_conf.w_rot
        w_trans = self.exp_conf.w_trans

        R_xyz, R_Ca = self.rigid_from_xyz(xyz)
        R_xyz_0, R_Ca_0 = self.rigid_from_xyz(xyz_0)

        # R_Ca [B*L,3] R_xyz[B*L,3,3]
        R_Ca, R_Ca_0 = R_Ca.view(B*L,-1), R_Ca_0.view(B*L,-1)
        R_xyz, R_xyz_0 = R_xyz.view(B*L,N,-1), R_xyz_0.view(B*L,N,-1)

        # Ca_Distance.shape [B*L]
        Ca_Distance_Norm = torch.norm(R_Ca - R_Ca_0, dim=-1, p=2)
        Ca_Distance = torch.where(Ca_Distance_Norm > d_clamp, d_clamp, Ca_Distance_Norm)

        # R_Distance.shape [B*L]
        batch_identity = torch.eye(3)[None,:].repeat(B*L,1,1).to(self.device)
        # I - r^(0).T r(0)
        R_Diff = batch_identity - torch.bmm(R_xyz.transpose(1,2),R_xyz_0) # (B*L,3,3)
        R_Distance_Norm = torch.linalg.matrix_norm(R_Diff)
        R_Distance = R_Distance_Norm**2

        frame_distance = torch.sqrt(w_trans*Ca_Distance + w_rot*R_Distance).mean()

        return frame_distance

    def metric_frame(self, xyz, xyz_0):
        block_num = xyz.shape[0]
        gamma = self.exp_conf.block_gamma
        gamma_list = [math.pow(gamma, i) for i in range(block_num)]
        gamma_factor = np.sum(gamma_list)
        frame_loss = 0
        for i in range(block_num):
            frame_loss += gamma_list[block_num - i - 1] * self.d_frame(xyz[i], xyz_0)
        return frame_loss / gamma_factor

    def metric_c6d(self, c6d_pred_logits, xyz_0):
        c6d_0, c6d_0_mask = xyz_to_c6d(xyz_0)
        bins_0 = c6d_to_bins(c6d_0)
        c6d_loss = 0
        for i in range(len(c6d_pred_logits)):
            logits = c6d_pred_logits[i]  # [B,37,L,L] or [B,19,L,L]
            target = bins_0[:, :, :, i]  # [B,L,L]
            criterion = torch.nn.CrossEntropyLoss()
            c6d_loss += criterion(logits, target.long())
        return c6d_loss






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

        loss_frame = self.metric_frame(xyz, batch['xyz'])
        loss_c6d = self.metric_c6d(logits, batch['xyz'])
        final_loss = loss_frame + self.exp_conf.w_2D * loss_c6d

        loss_dict = {'metric_frame': loss_frame,
                     'metric_c6d': loss_c6d,
                     'final_loss': final_loss}

        return loss_dict

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
        B, N, L = plus_processed_info['xyz_t'].shape[:3]
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
        denoiser = self.construct_denoiser(L)
        _, px0 = self.allatom(torch.argmax(plus_processed_info['seq_in'], dim=-1), px0, alpha_prev)
        x_t = self.get_next_batch_poses(denoiser, x_t_plus_1, px0, t_plus_1, motif_mask).to(self.device)
        zeros = torch.zeros(B, L, 13, 3).float().to(self.device)
        x_t = torch.cat((x_t, zeros), dim=-2)



        # Forward Pass
        processed_info = self._preprocess(seq_init, x_t, t, motif_mask)
        B, N, L = processed_info['xyz_t'].shape[:3]
        zeros = torch.zeros(B, N, L, 24, 3).float().to(self.device)
        prev_xyz_t = torch.cat((prev_pred.unsqueeze(1), zeros), dim=-2)  # [B,T,L,27,3]
        t2d_44 = xyz_to_t2d(prev_xyz_t)  # [B,T,L,L,44]
        # No effect if t2d is only dim 44
        prev_t2d = processed_info['t2d']
        prev_t2d[...,:44] = t2d_44


        logits, logits_aa, logits_exp, xyz, alpha_s, lddt \
            = self.model(processed_info['msa_masked'],
                         processed_info['msa_full'],
                         processed_info['seq_in'],
                         processed_info['xt_in'],
                         processed_info['idx_pdb'],
                         t1d=processed_info['t1d'],
                         t2d=prev_t2d,
                         xyz_t=prev_xyz_t,
                         alpha_t=processed_info['alpha_t'],
                         msa_prev=None,
                         pair_prev=None,
                         state_prev=None,
                         t=torch.tensor(t),
                         motif_mask=motif_mask)

        loss_frame = self.metric_frame(xyz, batch['xyz'])
        loss_c6d = self.metric_c6d(logits, batch['xyz'])
        final_loss = loss_frame + self.exp_conf.w_2D * loss_c6d

        loss_dict = {'metric_frame': loss_frame,
                     'metric_c6d': loss_c6d,
                     'final_loss': final_loss}

        return loss_dict


    def training_step(self, batch, batch_idx, **kwargs):
        if self.exp_conf.self_conditioning_percent < random.random():
             step_loss_dict =  self.loss_fn_normal(batch)
        else:
             step_loss_dict = self.loss_fn_self_conditioning(batch)

        self.log("train_loss", step_loss_dict['final_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log("metric_frame", step_loss_dict['metric_frame'], on_step=True, on_epoch=True, prog_bar=True)
        self.log("metric_c6d", step_loss_dict['metric_c6d'], on_step=True, on_epoch=True, prog_bar=True)

        return step_loss_dict['final_loss']










