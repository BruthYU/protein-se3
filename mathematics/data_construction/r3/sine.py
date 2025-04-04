import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-1, 1, 0.05)
Y = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(X, Y)
X = X.flatten()
Y = Y.flatten()
R = np.sqrt(X**2 + Y**2)
Z = np.sin(5*R)

# Plot the surface.
surf = ax.scatter3D(X, Y, Z,c=Z, cmap="plasma")

points = np.stack([X,Y,Z]).T

np.save('sine.npy', points)



plt.show()