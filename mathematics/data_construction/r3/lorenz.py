import matplotlib.pyplot as plt
import numpy as np


def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


dt = 0.02
num_steps = 1500

xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
xyzs[0] = (0., 1., 1.05)  # Set initial values
# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

xyzs = xyzs[:1500]
xyzs = xyzs.T
xyzs = xyzs / 20
xyzs[-1] = xyzs[-1]/2-0.7
xyzs[1] = xyzs[1]/1.5

# Plot
ax = plt.figure().add_subplot(projection='3d')

ax.scatter3D(*xyzs, lw=0.5,c=xyzs[-1], cmap="plasma")

plt.show()

xyzs = xyzs.T
np.save('lorenz.npy', xyzs)