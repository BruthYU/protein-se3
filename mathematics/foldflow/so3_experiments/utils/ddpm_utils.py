import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation



def quaternion_to_rotation_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    quaternions = F.normalize(quaternions, dim=-1)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def log_rotation(R):
    trace = R[..., range(3), range(3)].sum(-1)
    if torch.is_grad_enabled():
        # The derivative of acos at -1.0 is -inf, so to stablize the gradient, we use -0.9999
        min_cos = -0.999
    else:
        min_cos = -1.0
    cos_theta = ((trace - 1) / 2).clamp_min(min=min_cos)
    sin_theta = torch.sqrt(1 - cos_theta ** 2)
    theta = torch.acos(cos_theta)
    coef = ((theta + 1e-8) / (2 * sin_theta + 2e-8))[..., None, None]
    logR = coef * (R - R.transpose(-1, -2))
    return logR


def skewsym_to_so3vec(S):
    x = S[..., 1, 2]
    y = S[..., 2, 0]
    z = S[..., 0, 1]
    w = torch.stack([x, y, z], dim=-1)
    return w


def so3vec_to_skewsym(w):
    x, y, z = torch.unbind(w, dim=-1)
    o = torch.zeros_like(x)
    S = torch.stack([
        o, z, -y,
        -z, o, x,
        y, -x, o,
    ], dim=-1).reshape(w.shape[:-1] + (3, 3))
    return S


def exp_skewsym(S):
    x = torch.linalg.norm(skewsym_to_so3vec(S), dim=-1)
    I = torch.eye(3).to(S).view([1 for _ in range(S.dim() - 2)] + [3, 3])

    sinx, cosx = torch.sin(x), torch.cos(x)
    b = (sinx + 1e-8) / (x + 1e-8)
    c = (1 - cosx + 1e-8) / (x ** 2 + 2e-8)  # lim_{x->0} (1-cosx)/(x^2) = 0.5

    S2 = S @ S
    return I + b[..., None, None] * S + c[..., None, None] * S2


def so3vec_to_rotation(w):
    return exp_skewsym(so3vec_to_skewsym(w))


def rotation_to_so3vec(R):
    logR = log_rotation(R)
    w = skewsym_to_so3vec(logR)
    return w


def random_uniform_so3(size, device='cpu'):
    q = F.normalize(torch.randn(list(size) + [4, ], device=device), dim=-1)  # (..., 4)
    return rotation_to_so3vec(quaternion_to_rotation_matrix(q))


class ApproxAngularDistribution(nn.Module):

    def __init__(self, stddevs, std_threshold=0.1, num_bins=8192, num_iters=1024):
        super().__init__()
        self.std_threshold = std_threshold
        self.num_bins = num_bins
        self.num_iters = num_iters
        self.register_buffer('stddevs', torch.FloatTensor(stddevs))
        self.register_buffer('approx_flag', self.stddevs <= std_threshold)
        self._precompute_histograms()

    @staticmethod
    def _pdf(x, e, L):
        """
        Args:
            x:  (N, )
            e:  Float
            L:  Integer
        """
        x = x[:, None]  # (N, *)
        c = ((1 - torch.cos(x)) / math.pi)  # (N, *)
        l = torch.arange(0, L)[None, :]  # (*, L)
        a = (2 * l + 1) * torch.exp(-l * (l + 1) * (e ** 2))  # (*, L)
        b = (torch.sin((l + 0.5) * x) + 1e-6) / (torch.sin(x / 2) + 1e-6)  # (N, L)

        f = (c * a * b).sum(dim=1)
        return f

    def _precompute_histograms(self):
        X, Y = [], []
        for std in self.stddevs:
            std = std.item()
            x = torch.linspace(0, math.pi, self.num_bins)  # (n_bins,)
            y = self._pdf(x, std, self.num_iters)  # (n_bins,)
            y = torch.nan_to_num(y).clamp_min(0)
            X.append(x)
            Y.append(y)
        self.register_buffer('X', torch.stack(X, dim=0))  # (n_stddevs, n_bins)
        self.register_buffer('Y', torch.stack(Y, dim=0))  # (n_stddevs, n_bins)

    def sample(self, std_idx):
        """
        Args:
            std_idx:  Indices of standard deviation.
        Returns:
            samples:  Angular samples [0, PI), same size as std.
        """
        size = std_idx.size()
        std_idx = std_idx.flatten()  # (N,)

        # Samples from histogram
        prob = self.Y[std_idx]  # (N, n_bins)
        bin_idx = torch.multinomial(prob[:, :-1], num_samples=1).squeeze(-1)  # (N,)
        bin_start = self.X[std_idx, bin_idx]  # (N,)
        bin_width = self.X[std_idx, bin_idx + 1] - self.X[std_idx, bin_idx]
        samples_hist = bin_start + torch.rand_like(bin_start) * bin_width  # (N,)

        # Samples from Gaussian approximation
        mean_gaussian = self.stddevs[std_idx] * 2
        std_gaussian = self.stddevs[std_idx]
        samples_gaussian = mean_gaussian + torch.randn_like(mean_gaussian) * std_gaussian
        samples_gaussian = samples_gaussian.abs() % math.pi

        # Choose from histogram or Gaussian
        gaussian_flag = self.approx_flag[std_idx]
        samples = torch.where(gaussian_flag, samples_gaussian, samples_hist)

        return samples.reshape(size)


def random_normal_so3(std_idx, angular_distrib, device='cpu'):
    size = std_idx.size()
    u = F.normalize(torch.randn(list(size) + [3, ], device=device), dim=-1)
    theta = angular_distrib.sample(std_idx)
    w = u * theta[..., None]
    return w


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)

class RotationTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

        # Forward (perturb)
        c1 = torch.sqrt(1 - self.var_sched.alpha_bars) # (T,).
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)

        # Inverse (generate)
        sigma = self.var_sched.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist(), **angular_distrib_inv_opt)

        self.register_buffer('_dummy', torch.empty([0, ]))


    def add_noise(self, v_0, t):
        """
        Args:
            v_0:    (N, L, 3).
            t:  (N,).
        """
        N, L = v_0.shape[:2]
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        # Noise rotation
        e_scaled = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_fwd, device=self._dummy.device)    # (N, L, 3)
        e_normal = e_scaled / (c1 + 1e-8)
        E_scaled = so3vec_to_rotation(e_scaled)   # (N, L, 3, 3)

        # Scaled true rotation
        R0_scaled = so3vec_to_rotation(c0 * v_0)  # (N, L, 3, 3)

        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)


        return v_noisy, e_scaled

    def denoise(self, v_next, t):
        N, L = v_next.shape[:2]
        e = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_inv, device=self._dummy.device) # (N, L, 3)
        e = torch.where(
            (t > 1)[:, None, None].expand(N, L, 3),
            e,
            torch.zeros_like(e) # Simply denoise and don't add noise at the last step
        )
        E = so3vec_to_rotation(e).double()

        R_next = E @ so3vec_to_rotation(v_next.cpu())
        v_next = rotation_to_so3vec(R_next)

        return v_next


