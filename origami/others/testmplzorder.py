"""
This module emphasize the bad behavior of 3d plots using matplotlib
when considering the zorder of the panels.
We provide here a minimal example to this bug.
It doesn't seem easy to overcome this in the general case, but manual drawing
of the Miura-Ori panels may solve the problem for a specific view point.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def minimal_example():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', elev=18, azim=123)

    X = [1, 2, 3]
    Y = [1, 2]
    Xs, Ys = np.meshgrid(X, Y)
    Ys[:, 2] += 2
    # Zs = 0.01*np.abs(Xs - 2)
    Zs = 1.0 * np.abs(Xs - 2)
    # Zs[:, 0] *= 0.05

    surf: Poly3DCollection = ax.plot_surface(Xs, Ys, Zs)
    surf.set_facecolor(['r', 'g'])

    original_func = surf.do_3d_projection
    self = surf

    def my_3d():
        """
        Perform the 3D projection for this object.
        """
        if self._A is not None:
            # force update of color mapping because we re-order them
            # below.  If we do not do this here, the 2D draw will call
            # this, but we will never port the color mapped values back
            # to the 3D versions.
            #
            # We hold the 3D versions in a fixed order (the order the user
            # passed in) and sort the 2D version by view depth.
            self.update_scalarmappable()
            if self._face_is_mapped:
                self._facecolor3d = self._facecolors
            if self._edge_is_mapped:
                self._edgecolor3d = self._edgecolors
        txs, tys, tzs = proj3d._proj_transform_vec(self._vec, self.axes.M)
        xyzlist = [(txs[sl], tys[sl], tzs[sl]) for sl in self._segslices]

        print(xyzlist)
        print(self._segslices)

        # This extra fuss is to re-order face / edge colors
        cface = self._facecolor3d
        cedge = self._edgecolor3d
        if len(cface) != len(xyzlist):
            cface = cface.repeat(len(xyzlist), axis=0)
        if len(cedge) != len(xyzlist):
            if len(cedge) == 0:
                cedge = cface
            else:
                cedge = cedge.repeat(len(xyzlist), axis=0)

        if xyzlist:
            # sort by depth (furthest drawn first)
            z_segments_2d = sorted(
                ((self._zsortfunc(zs), np.column_stack([xs, ys]), fc, ec, idx)
                 for idx, ((xs, ys, zs), fc, ec)
                 in enumerate(zip(xyzlist, cface, cedge))),
                key=lambda x: x[0], reverse=True)

            _, segments_2d, self._facecolors2d, self._edgecolors2d, idxs = \
                zip(*z_segments_2d)

            print(segments_2d)
        else:
            segments_2d = []
            self._facecolors2d = np.empty((0, 4))
            self._edgecolors2d = np.empty((0, 4))
            idxs = []

        if self._codes3d is not None:
            codes = [self._codes3d[idx] for idx in idxs]
            PolyCollection.set_verts_and_codes(self, segments_2d, codes)
        else:
            PolyCollection.set_verts(self, segments_2d, self._closed)

        if len(self._edgecolor3d) != len(cface):
            self._edgecolors2d = self._edgecolor3d

        # Return zorder value
        if self._sort_zpos is not None:
            zvec = np.array([[0], [0], [self._sort_zpos], [1]])
            ztrans = proj3d._proj_transform_vec(zvec, self.axes.M)
            return ztrans[2][0]
        elif tzs.size > 0:
            # FIXME: Some results still don't look quite right.
            #        In particular, examine contourf3d_demo2.py
            #        with az = -54 and elev = -45.
            return np.min(tzs)
        else:
            return np.nan

    surf.do_3d_projection = my_3d


def plot_bad_triangle():
    v1 = np.array([0, 1, 0])
    v2 = np.array([np.sqrt(3) / 2, -1 / 2, 0])
    v3 = np.array([-np.sqrt(3) / 2, -1 / 2, 0])

    print(np.linalg.norm(v1 - v2))
    print(np.linalg.norm(v3 - v2))

    v1_perp = np.array([1, 0, 0])
    v2_perp = np.array([-1 / 2, -np.sqrt(3) / 2, 0])
    v3_perp = np.array([1 / 2, -np.sqrt(3) / 2, 0])
    w = 0.3

    dots = np.array([v1 - w * v1_perp, v1 + w * v1_perp,
                     v2 - w * v2_perp, v2 + w * v2_perp,
                     v3 - w * v3_perp, v3 + w * v3_perp])

    xs = np.array([[dots[0, 0], dots[1, 0]],
                   [dots[3, 0], dots[2, 0]],
                   [dots[5, 0], dots[4, 0]],
                   [dots[1, 0], dots[0, 0]]])
    ys = np.array([[dots[0, 1], dots[1, 1]],
                   [dots[3, 1], dots[2, 1]],
                   [dots[5, 1], dots[4, 1]],
                   [dots[1, 1], dots[0, 1]]])
    zs = np.array([[dots[0, 2], dots[1, 2]],
                   [dots[3, 2], dots[2, 2]],
                   [dots[5, 2], dots[4, 2]],
                   [dots[1, 2], dots[0, 2]]])

    z_shift = 0.2
    zs[2, :] += z_shift
    zs[3, :] += z_shift
    # zs[4, :] += z_shift

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xs, ys, zs, lw=3, color='C1', edgecolor='g')
    ax.set_aspect('equal')
    # ax.plot_surface(np.array([[0, 0], [1, 1]]),
    #                 np.array([[0, 1], [0, 1]]), np.array([[0, 0], [0,0]]), lw=1)

    fig, ax = plt.subplots()
    # ax.plot(dots[:, 0], dots[:, 1], '.')
    # ax.set_aspect('equal')

    plt.show()


def test_miura_ori():
    import numpy as np

    from origami import marchingalgorithm, quadranglearray, RFFQMOrigami, origamiplots

    # angle = 0.7 * np.pi
    angle = 1.2
    ls = np.ones(4)
    cs = np.ones(4)
    angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, angle)
    marching = marchingalgorithm.MarchingAlgorithm(angles_left, angles_bottom)
    quads = quadranglearray.dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQMOrigami.RFFQM(quads)
    ori.set_gamma(ori.calc_gamma_by_omega(2))

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d', elev=-154, azim=-120)
    ax:Axes3D = fig.add_subplot(111, projection='3d', elev=24, azim=-146)
    # quadranglearray.plot_panels_manual_zorder(ori.dots, ax, 'C1', 'g')
    ori.dots.plot(ax, 'C1', 'g')
    ori.dots.dots[2, :] += 1.2
    ax.computed_zorder = False
    quadranglearray.plot_panels_manual_zorder(ori.dots, ax, 'C1', 'g')
    plt.show()


def main():
    test_miura_ori()


if __name__ == '__main__':
    main()
