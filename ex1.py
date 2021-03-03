from fenics import *
# Create mesh and define function space
mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, 'P', 1)
# Define boundary condition
u_D = Expression('0', degree=2)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx
# Compute solution
u = Function(V)
solve(a == L, u, bc)
# Plot solution and mesh
plot(u)
plot(mesh)
#interactive()
# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u

p1 = Point(0.5,0.1)
print(u(p1.x(),p1.y()))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

fig = plt.figure(figsize=plt.figaspect(0.5))

#============
# First plot
#============

# Make a mesh in the space of parameterisation variables u and v
x = np.linspace(0, 1, endpoint=True, num=50)
y = np.linspace(0, 1, endpoint=True, num=50)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()
z=[]

# This is the Mobius mapping, taking a u, v pair and returning an x, y, z
# triple
k=0
for i in x:
    u0=u(i,y[k])
    z.append(u0)
    k=k+1

# Triangulate parameter space to determine the triangles
tri = mtri.Triangulation(x, y)

# Plot the surface.  The triangles in parameter space determine which x, y, z
# points are connected by an edge.
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
ax.set_zlim(0, 0.5)

plt.show()
