import torch
from torch.optim import Adam
from abc import ABC, abstractmethod
from pytorch_lightning.core import LightningModule
import math
from lightning.model.genie2.denoiser import Denoiser
import pytorch_lightning as pl
from lightning.data.genie2.affine_utils import T
from lightning.data.genie2.geo_utils import compute_frenet_frames

from lightning.data.genie2.feat_utils import prepare_tensor_features

class genie2_Lightning_Model(pl.LightningModule):
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
            max_n_res=self.data_conf.max_n_res,
            max_n_chain=self.data_conf.max_n_chain
        )

        # Flag for lazy setup and same device requirements
        self.setup = False

    def setup_schedule(self):
        """
        Set up variance schedule and precompute its corresponding terms.
        """
        self.betas = self.get_betas(
            self.diff_conf.n_timestep,
            self.diff_conf.schedule
        ).to(self.device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([
            torch.Tensor([1.]).to(self.device),
            self.alphas_cumprod[:-1]
        ])
        self.one_minus_alphas_cumprod = 1. - self.alphas_cumprod

        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1. - self.alphas_cumprod_prev)
        self.sqrt_recip_alphas_cumprod = 1. / self.sqrt_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

    def configure_optimizers(self):
        return Adam(
            self.model.parameters(),
            lr=self.exp_conf.lr
        )

    def training_step(self, batch, batch_idx):
        # Perform setup in the first run
        if not self.setup:
            self.setup_schedule()
            self.setup = True


        train_loss, aux_data = self.loss_fn(batch)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)

        batch_mask, condition_losses, infill_losses = \
            aux_data['batch_mask'], aux_data['condition_losses'], aux_data['infill_losses']
        for i in range(batch_mask.shape[0]):
            if batch_mask[i]:
                self.log('motif_mse_loss', condition_losses[i], on_step=True, on_epoch=True)
                self.log('scaffold_mse_loss', infill_losses[i], on_step=True, on_epoch=True)
            else:
                self.log('unconditional_mse_loss', infill_losses[i], on_step=True, on_epoch=True)

        return train_loss


    def loss_fn(self, batch):
        """
        Training iteration.

        Args:
            batch:
                A batched feature dictionary with a batch size B, where each
                structure is padded to the maximum sequence length N. It contains
                the following information
                    -	aatype:
                            [B, N, 20] one-hot encoding on amino acid types
                    -	num_chains:
                            [B, 1] number of chains in the structure
                    -	num_residues:
                            [B, 1] number of residues in the structure
                    -	num_residues_per_chain:
                            [B, 1] an array of number of residues by chain
                    -	atom_positions:
                            [B, N, 3] an array of Ca atom positions
                    -	residue_mask:
                            [B, N] residue mask to indicate which residue position is masked
                    -	residue_index:
                            [B, N] residue index (started from 0)
                    -	chain_index:
                            [B, N] chain index (started from 0)
                    -	fixed_sequence_mask:
                            [B, N] mask to indicate which residue contains conditional
                            sequence information
                    -	fixed_structure_mask:
                            [B, N, N] mask to indicate which pair of residues contains
                            conditional structural information
                    -	fixed_group:
                            [B, N] group index to indicate which group the residue belongs to
                            (useful for specifying multiple functional motifs)
                    -	interface_mask:
                            [B, N] deprecated and set to all zeros.
            batch_idx:
                [1] Index of this training batch.

        Returns:
            loss:
                [1] Motif-weighted mean of per-residue mean squared error between the predicted
                noise and the groundtruth noise, averaged across all structures in the batch
        """
        # Define features
        features = prepare_tensor_features(batch)

        # Sample time step
        s = torch.randint(
            self.diff_conf.n_timestep,
            size=(features['atom_positions'].shape[0],)
        ).to(self.device) + 1

        # Sample noise
        z = torch.randn_like(features['atom_positions']) * features['residue_mask'].unsqueeze(-1)

        # Apply noise
        trans_s = self.sqrt_alphas_cumprod[s].view(-1, 1, 1) * features['atom_positions'] + \
                  self.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1) * z
        rots_s = compute_frenet_frames(
            trans_s,
            features['chain_index'],
            features['residue_mask']
        )
        ts = T(rots_s, trans_s)

        # Predict noise
        output = self.model(ts, s, features)

        # Compute masks
        condition_mask = features['residue_mask'] * features['fixed_sequence_mask']
        infill_mask = features['residue_mask'] * ~features['fixed_sequence_mask']

        # Compute condition and infill losses
        condition_losses = self.mse(output['z'], z, condition_mask, aggregate='sum')
        infill_losses = self.mse(output['z'], z, infill_mask, aggregate='sum')

        # Compute weighted losses
        unweighted_losses = (condition_losses + infill_losses) / features['num_residues']
        weighted_losses = (self.exp_conf.condition_loss_weight * condition_losses + infill_losses) / \
                          (self.exp_conf.condition_loss_weight * torch.sum(condition_mask,
                                                                                     dim=-1) + torch.sum(infill_mask,
                                                                                                         dim=-1))

        # Aggregate
        unweighted_loss = torch.mean(unweighted_losses)
        weighted_loss = torch.mean(weighted_losses)
        self.log('unweighted_loss', unweighted_loss, on_step=True, on_epoch=True)
        self.log('weighted_loss', weighted_loss, on_step=True, on_epoch=True)

        # Log
        batch_mask = torch.sum(condition_mask, dim=-1) > 0
        condition_losses = condition_losses / torch.sum(condition_mask, dim=-1)
        infill_losses = infill_losses / torch.sum(infill_mask, dim=-1)


        aux_data = {
            'unweighted_loss': unweighted_loss,
            'condition_losses': condition_losses,
            'infill_losses': infill_losses,
            'batch_mask': batch_mask
        }

        return weighted_loss, aux_data

    def mse(self, x_pred, x, mask, aggregate=None, eps=1e-10):
        """
        Compute mean squared error.

        Args:
            x_pred:
                [B, N, D] Predicted values.
            x:
                [B, N, D] Groundtruth values.
            mask:
                [B, N] Mask.
            aggregation:
                Aggregation method within each sample, including
                    -   None: no aggregation (default)
                    -   mean: aggregation by computing mean along second dimension
                    -   sum: aggregation by computing sum along second dimension.
            eps:
                Epsilon for computational stability. Default to 1e-10.

        Returns:
            A tensor of mean squared errors, with a shape of [B, N] if no
            aggregation, or a shape of [B] if using 'mean' or 'sum' aggregation.
        """
        errors = (eps + torch.sum((x_pred - x) ** 2, dim=-1)) ** 0.5
        if aggregate is None:
            return errors * mask
        elif aggregate == 'mean':
            return torch.sum(errors * mask, dim=-1) / torch.sum(mask, dim=-1)
        elif aggregate == 'sum':
            return torch.sum(errors * mask, dim=-1)
        else:
            print('Invalid aggregate method: {}'.format(aggregate))
            exit(0)





    def sinusoidal_encoding(self, v, N, D):
        # v: [*]

        # [D]
        k = torch.arange(1, D + 1).to(v.device)

        # [*, D]
        sin_div_term = N ** (2 * k / D)
        sin_div_term = sin_div_term.view(*((1,) * len(v.shape) + (len(sin_div_term),)))
        sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)

        # [*, D]
        cos_div_term = N ** (2 * (k - 1) / D)
        cos_div_term = cos_div_term.view(*((1,) * len(v.shape) + (len(cos_div_term),)))
        cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)

        # [*, D]
        enc = torch.zeros_like(sin_enc).to(v.device)
        enc[..., 0::2] = cos_enc[..., 0::2]
        enc[..., 1::2] = sin_enc[..., 1::2]

        return enc

    def cosine_beta_schedule(self, n_timestep):
        """
        Set up a cosine variance schedule.

        Args:
            n_timestep:
                Number of diffusion timesteps (denoted as N).

        Returns:
            A sequence of variances (with a length of N + 1), where the
            i-th element denotes the variance at diffusion step i. Note
            that diffusion step is one-indexed and i = 0 indicates the
            un-noised stage.
        """
        steps = n_timestep + 1
        x = torch.linspace(0, n_timestep, steps)
        alphas_cumprod = torch.cos((x / steps) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.concat([
            torch.zeros((1,)),
            torch.clip(betas, 0, 0.999)
        ])

    def get_betas(self, n_timestep, schedule):
        """
        Set up a variance schedule.

        Args:
            n_timestep:
                Number of diffusion timesteps (denoted as N).
            schedule:
                Name of variance schedule. Currently support 'cosine'.

        Returns:
            A sequence of variances (with a length of N + 1), where the
            i-th element denotes the variance at diffusion step i. Note
            that diffusion step is one-indexed and i = 0 indicates the
            un-noised stage.
        """
        if schedule == 'cosine':
            return self.cosine_beta_schedule(n_timestep)
        else:
            print('Invalid schedule: {}'.format(schedule))
            exit(0)






