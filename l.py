# %%
import torch
import matplotlib.pyplot as plt

# %%
# 1. Create a range of input values, 0 to 1 with n_stepts:
n_steps = 100
# Get a set of 2D points covering the unit square:
x = torch.linspace(-1, 1, n_steps)
y = torch.linspace(-1, 1, n_steps)
# Create a grid of points:
X, Y = torch.meshgrid(x, y)
# Flatten the grid to get a list of 2D points:
points = torch.stack([X.flatten(), Y.flatten()], dim=1)

# %%
points.shape

# %%
# Define a layer of the network with a randomly initialized weight vector:
weights = torch.randn(2, 1)
# Define biases:
biases = torch.randn(1)

# %%
weights.shape

# %%
plt.figure(figsize=(6, 6))
z = torch.heaviside(torch.matmul(points, weights) + biases, values=torch.tensor([0.0]))
# Plot xy, and z as a color:
plt.contourf(X, Y, z.reshape(X.shape), 50, cmap='jet')

plt.grid()
plt.xlabel('x')
plt.ylabel('y')

# %%
# Plot x,y,z on a 3D plot:
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], z[:, 0])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# Rotate the plot:
ax.view_init(30, 30)
# make interactive:
plt.show()

