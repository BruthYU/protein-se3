import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the bunny PLY file
bunny = o3d.io.read_triangle_mesh("bun_zipper_res2.ply")

# Convert mesh to a point cloud (if needed)
bunny_pcd = o3d.geometry.PointCloud()
bunny_pcd.points = bunny.vertices

# Convert to numpy array (x, y, z)
num_points = 5000
points = np.asarray(bunny_pcd.points)
points = points[np.random.choice(points.shape[0], num_points, replace=False)]  # Random sample
points = points[:, [2,0,1]]
np.save('bunny.npy', points)
# Visualize using Open3D
# o3d.visualization.draw_geometries([bunny_pcd])

# Optional: Plot with Matplotlib
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 1], cmap="viridis", s=1)
plt.show()