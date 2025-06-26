import torch
from torch.optim import Adam
from abc import ABC, abstractmethod
from pytorch_lightning.core import LightningModule
import math
from lightning_protein.model.genie1.denoiser import Denoiser
import pytorch_lightning as pl

from lightning_protein.model.genie1.schedule import get_betas
from lightning_protein.data.genie1.loss import rmsd
from lightning_protein.data.genie1.affine_utils import T
from lightning_protein.data.genie1.geo_utils import compute_frenet_frames
from tqdm import tqdm

class genie1_Lightning_Model(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.model_conf = conf.model
        self.data_conf = conf.dataset
        self.diff_conf = conf.diffusion
        self.exp_conf = conf.experiment

        self.model = Denoiser(
            **self.model_conf,
            n_timestep=self.diff_conf.n_timestep,
        )

        # Flag for lazy setup and same device requirements
        self.setup = False

    def setup_schedule(self):

        self.betas = get_betas(self.diff_conf.n_timestep, self.diff_conf.schedule).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([torch.Tensor([1.]).to(self.device), self.alphas_cumprod[:-1]])
        self.one_minus_alphas_cumprod = 1. - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1. - self.alphas_cumprod_prev

        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = 1. / self.sqrt_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * self.sqrt_alphas_cumprod_prev / self.one_minus_alphas_cumprod
        self.posterior_mean_coef2 = self.one_minus_alphas_cumprod_prev * self.sqrt_alphas / self.one_minus_alphas_cumprod
        self.posterior_variance = self.betas * self.one_minus_alphas_cumprod_prev / self.one_minus_alphas_cumprod

    def transform(self, batch):

        coords, mask = batch
        ca_coords = coords.float()

        mask = mask.float()

        trans = ca_coords - torch.mean(ca_coords, dim=1, keepdim=True)
        rots = compute_frenet_frames(trans, mask)

        return T(rots, trans), mask

    def sample_timesteps(self, num_samples):
        return torch.randint(0, self.diff_conf.n_timestep, size=(num_samples,)).to(self.device)

    def sample_frames(self, mask):
        trans = torch.randn((mask.shape[0], mask.shape[1], 3)).to(self.device)
        trans = trans * mask.unsqueeze(-1)
        rots = compute_frenet_frames(trans, mask)
        return T(rots, trans)

    def q(self, t0, s, mask):

        # [b, n_res, 3]
        trans_noise = torch.randn_like(t0.trans) * mask.unsqueeze(-1)
        rots_noise = torch.eye(3).view(1, 1, 3, 3).repeat(t0.shape[0], t0.shape[1], 1, 1).to(self.device)

        trans = self.sqrt_alphas_cumprod[s].view(-1, 1, 1).to(self.device) * t0.trans + \
                self.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1).to(self.device) * trans_noise
        rots = compute_frenet_frames(trans, mask)

        return T(rots, trans), T(rots_noise, trans_noise)

    def p(self, ts, s, mask, noise_scale):

        # [b, 1, 1]
        w_noise = ((1. - self.alphas[s].to(self.device)) / self.sqrt_one_minus_alphas_cumprod[s].to(self.device)).view(
            -1, 1, 1)

        # [b, n_res]
        noise_pred_trans = ts.trans - self.model(ts, s, mask).trans
        noise_pred_rots = torch.eye(3).view(1, 1, 3, 3).repeat(ts.shape[0], ts.shape[1], 1, 1)
        noise_pred = T(noise_pred_rots, noise_pred_trans)

        # [b, n_res, 3]
        trans_mean = (1. / self.sqrt_alphas[s]).view(-1, 1, 1).to(self.device) * (ts.trans - w_noise * noise_pred.trans)
        trans_mean = trans_mean * mask.unsqueeze(-1)

        if (s == 0.0).all():
            rots_mean = compute_frenet_frames(trans_mean, mask)
            return T(rots_mean.detach(), trans_mean.detach())
        else:

            # [b, n_res, 3]
            trans_z = torch.randn_like(ts.trans).to(self.device)

            # [b, 1, 1]
            trans_sigma = self.sqrt_betas[s].view(-1, 1, 1).to(self.device)

            # [b, n_res, 3]
            trans = trans_mean + noise_scale * trans_sigma * trans_z
            trans = trans * mask.unsqueeze(-1)

            # [b, n_res, 3, 3]
            rots = compute_frenet_frames(trans, mask)

            return T(rots.detach(), trans.detach())

    def loss_fn(self, tnoise, ts, s, mask):

        noise_pred_trans = ts.trans - self.model(ts, s, mask).trans

        trans_loss = rmsd(
            noise_pred_trans,
            tnoise.trans,
            mask
        )

        return trans_loss

    def configure_optimizers(self):
        return Adam(
            self.model.parameters(),
            lr=self.exp_conf.lr
        )

    def training_step(self, batch, batch_idx):
        '''
        Training iteration.

        Input:
            batch     - coordinates from data pipeline (shape: b x (n_res * 3))
            batch_idx - batch index (shape: b)

        Output: Either a single loss value or a dictionary of losses, containing
            one key as 'loss' (loss value for optimization)
        '''
        if not self.setup:
            self.setup_schedule()
            self.setup = True
        t0, mask = self.transform(batch)
        s = self.sample_timesteps(t0.shape[0])
        ts, tnoise = self.q(t0, s, mask)
        loss = self.loss_fn(tnoise, ts, s, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True)
        return loss

    def p_sample_loop(self, mask, noise_scale, verbose=True):
        if not self.setup:
            self.setup_schedule()
            self.setup = True
        ts = self.sample_frames(mask)
        ts_seq = [ts]
        for i in tqdm(reversed(range(self.diff_conf.n_timestep)), desc='sampling loop time step',
                      total=self.diff_conf.n_timestep, disable=not verbose):
            s = torch.Tensor([i] * mask.shape[0]).long().to(self.device)
            ts = self.p(ts, s, mask, noise_scale)
            ts_seq.append(ts)
        return ts_seq

