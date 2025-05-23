import torch
import numpy as np
import math

class r3_ddpm_scheduler:
    def __init__(self, diff_conf):
        self.diff_conf = diff_conf
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.setup_schedule()


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


