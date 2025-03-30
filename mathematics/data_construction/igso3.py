"""Copyright (c) Dreamfold."""
import torch
from torch import Tensor
import numpy as np
from functorch import vmap
from lightning.data.foldflow.so3_helpers import so3_exp_map
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import math
def f_igso3_small(omega, sigma):
    """Borrowed from: https://github.com/tomato1mule/edf/blob/1dd342e849fcb34d3eb4b6ad2245819abbd6c812/edf/dist.py#L99
    This function implements the approximation of the density function of omega of the isotropic Gaussian distribution.
    """
    # TODO: check for stability and maybe replace by limit in 0 for small values

    pi = torch.Tensor([torch.pi]).to(device=omega.device)
    eps = (sigma / torch.sqrt(torch.tensor([2])).to(device=omega.device)) ** 2


    small_number = 1e-9
    small_num = small_number / 2
    small_dnm = (
        1 - torch.exp(-1.0 * pi**2 / eps) * (2 - 4 * (pi**2) / eps)
    ) * small_number

    return (
        0.5
        * torch.sqrt(pi)
        * (eps**-1.5)
        * torch.exp((eps - (omega**2 / eps)) / 4)
        / (torch.sin(omega / 2) + small_num)
        * (
            small_dnm
            + omega
            - (
                (omega - 2 * pi) * torch.exp(pi * (omega - pi) / eps)
                + (omega + 2 * pi) * torch.exp(-pi * (omega + pi) / eps)
            )
        )
    )


# Marginal density of rotation angle for uniform density on SO(3)
def angle_density_unif(omega):
    return (1 - torch.cos(omega)) / torch.pi


def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])  # slope
    b = fp[:-1] - (m * xp[:-1])  # y-intercept

    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), dim=1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * x + b[indicies]


def _f(omega, eps):
    return f_igso3_small(omega, eps)


def _pdf(omega, eps):
    f_unif = angle_density_unif(omega)
    return _f(omega, eps) * f_unif


def _sample(eps, n):
    # sample n points from IGSO3(I, eps)
    num_omegas = 1024
    omega_grid = torch.linspace(0, torch.pi, num_omegas + 1).to(eps.device)[
        1:
    ]  # skip omega=0
    # numerical integration of (one-cos(omega))/pi*f_igso3(omega, eps) over omega
    pdf = _pdf(omega_grid, eps)
    dx = omega_grid[1] - omega_grid[0]
    cdf = torch.cumsum(pdf, dim=-1) * dx  # cumalative density function

    # sample n points from the distribution
    rand_angle = torch.rand(n).to(eps.device)
    omegas = interp(rand_angle, cdf, omega_grid)
    axes = torch.randn(n, 3).to(eps.device)  # sample axis uniformly
    axis_angle = (
        omegas[..., None] * axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    )
    return axis_angle


def _batch_sample(mu, eps, n):
    aa_samples = (
        vmap(_sample, in_dims=(0, None), randomness="different")(eps, n)
        .squeeze()
        .double()
    )
    return mu @ so3_exp_map(aa_samples)


def rotationMatrixToEulerAngles(R):
    """https://learnopencv.com/rotation-matrix-to-euler-angles/"""

    # assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def plot_so3(x):
    return plot_scatter3D(
        np.array([rotationMatrixToEulerAngles(x[i]) for i in range(len(x))]),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
    )

def plot_scatter3D(xyz, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0)):
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = plt.axes(projection="3d")

    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.5, c=xyz[:, 2], cmap='viridis')



    # # 2 circles
    cycle_list = [0, 90]
    # colors=['orange','purple']
    theta = np.linspace(0, 2 * np.pi, 200)
    y = np.cos(theta)*2.5
    z = np.sin(theta)*2.5

    for i in range(len(cycle_list)):
        degree = cycle_list[i]
        phi = np.deg2rad(degree + 90)
        ax.plot(y * np.cos(phi),
                y * np.sin(phi), z, label=degree,)


    # x = np.cos(theta)*2.5
    # y = np.sin(theta)*2.5
    # for i in range(len(cycle_list)):
    #     degree = cycle_list[i]
    #     phi = np.deg2rad(degree + 90)
    #     ax.plot(x, y * np.sin(phi),
    #             y * np.cos(phi), label=degree)

    ax.axes.set_xlim3d(xlim[0], xlim[1])
    ax.axes.set_ylim3d(ylim[0], ylim[1])
    ax.axes.set_zlim3d(zlim[0], zlim[1])
    return fig


if __name__ == '__main__':
    mu = torch.from_numpy(np.identity(3))
    samples = _batch_sample(mu, torch.tensor(0.5)[None], 2000)

    fig = plot_so3(samples)
    plt.tight_layout()
    plt.savefig('figs/so3_synthetic_data.png', dpi=500)
    plt.show()
    pass