import numpy as np
from mayavi import mlab

from origami.quadranglearray import QuadrangleArray


def plot_with_mayavi(quads: QuadrangleArray):
    dots, indexes = quads.dots, quads.indexes
    dots = dots.astype('float64')

    rows, cols = indexes.shape

    x = dots[0, :].reshape((rows, cols))
    y = dots[1, :].reshape((rows, cols))
    z = dots[2, :].reshape((rows, cols))

    fig = mlab.figure()
    s = mlab.mesh(x, y, z, color=(0.7, 0, 0))
    mlab.axes(s, nb_labels=5, z_axis_visibility=False)
    xx = yy = zz = np.arange(-0.6, 0.7, 0.1)
    xy = xz = yx = yz = zx = zy = np.zeros_like(xx)
    lensoffset = 2
    mlab.plot3d(yx, yy + lensoffset, yz, line_width=0.01, tube_radius=0.01)
    mlab.plot3d(zx, zy + lensoffset, zz, line_width=0.01, tube_radius=0.01)
    mlab.plot3d(xx, xy + lensoffset, xz, line_width=0.01, tube_radius=0.01)
