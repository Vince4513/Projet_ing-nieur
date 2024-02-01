import numpy as np

nx, ny = (3, 4)

x = np.linspace(0, 1, nx)
print("x",x)
y = np.linspace(0, 3, ny)
print("\ny",y)
xv, yv = np.meshgrid(x, y)

print("\nxv\n",xv)
print("\nyv\n",yv)
