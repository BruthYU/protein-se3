"""Copyright (c) Dreamfold."""
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
from PIL import Image

# import orthogonal_group
import io
import plotly.graph_objects as go


def plot_circular_hist(y, bins=40, fig=None):
    """
    :param y: angle values in [0, 1]
    :param bins: number of bins
    :param fig: matplotlib figure object
    """

    theta = np.linspace(0, 2 * np.pi, num=bins, endpoint=False)
    radii = np.histogram(y, bins, range=(0, 2 * np.pi), density=True)[0]

    # # Display width
    width = (2 * np.pi) / (bins * 1.25)

    # Construct ax with polar projection
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # Set Orientation
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xlim(0, 2 * np.pi)  # workaround for a weird issue
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 8, endpoint=False))

    # Plot bars:
    _ = ax.bar(x=theta, height=radii, width=width, color="gray")

    # Grid settings
    ax.set_rgrids([])

    return fig, ax


def fig2pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.tostring_rgb()
    pil_im = Image.frombytes("RGB", (w, h), buf)
    plt.close("all")
    return pil_im


def plot_scatter3D(xyz, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0), adjust=False,title=None):
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = plt.axes(projection="3d")

    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], cmap="plasma")
    if adjust:
        ax.view_init(azim=20, elev=20)
    # ax.axes.set_xlim3d(xlim[0], xlim[1])
    # ax.axes.set_ylim3d(ylim[0], ylim[1])
    # ax.axes.set_zlim3d(zlim[0], zlim[1])
    if title is not None:
        ax.set_title(title, fontsize=18, y=1.05)
    return fig


def plot_r3(x, adjust=False,title=None):
    return plot_scatter3D(
        x,
        adjust = adjust,
        title = title
    )





