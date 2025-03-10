import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Number of points per cluster
num_points = 5000

# Define grid size
grid_size = 3
theta = np.linspace(0, 5 * np.pi, num_points)
r = theta
# x = r * np.cos(theta)
# y = r * np.sin(theta)
z = np.linspace(0 , 2, num_points )
x = z * np.cos(10*z) + 0.3 * np.random.rand(num_points)
y = z * np.sin(10*z) + 0.3 * np.random.rand(num_points)
points = np.stack((x,y,z), axis=1)


def normalization1(data):
    for i in range(data.shape[1]):
        data[:,i] -= data[:,i].mean()
        _range = np.max(abs(data[:,i]))
        data[:,i] = data[:,i] / _range * 1.4
    return data



points = normalization1(points)
np.save('spiral.npy', points)

ax = plt.axes(projection="3d")
ax.scatter3D(points[:,0], points[:,1], points[:,2], s=0.5,c=points[:,2],cmap='viridis')

plt.show()