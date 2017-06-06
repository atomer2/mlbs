import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mlbs import mlbs

data = np.loadtxt('data.txt')
di = 2
fdi = 1
start_res = np.array([10, 10, 1])
target_res = np.array([200, 200, 1])
input_range = np.array([[0, 0], [100, 100]])
s = mlbs(data, target_res, start_res,  input_range)
# multilevel B spline approximation
s.approximation()

plot_res = 2

fig = plt.figure()
ax = Axes3D(fig)
z = []
x = np.arange(0, 100, plot_res)
y = np.arange(0, 100, plot_res)
for i in x:
    for j in y:
        r = s.slbs.interpolation(np.array([i, j]))
        z.append(r)

z = np.array(z).reshape(len(x), len(y))

X, Y = np.meshgrid(x, y)

scatx = data[:, 0]
scaty = data[:, 1]
scatz = data[:, 2]

ax.plot_surface(Y, X, z, rstride=1, cstride=1, cmap='rainbow')
ax.scatter(scatx, scaty, scatz, c='r')
plt.show()
