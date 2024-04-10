import fractions
from typing import Optional

import numpy as np
from matplotlib import patches as mpatches, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D, proj3d


def set_pi_ticks(ax, axis, pi_range=(0, 1), divisions=4):
    ticks = np.linspace(pi_range[0], pi_range[1], divisions + 1) * np.pi
    values = [pi_range[0] +
              fractions.Fraction((pi_range[1] - pi_range[0]) * i, divisions) for i in range(divisions + 1)]

    def fraction_to_latex(frac: fractions.Fraction):
        if frac == 1:
            return ''
        if frac.denominator == 1:
            return str(frac.numerator)
        return r'\frac{' + str(frac.numerator) + '}{' + str(frac.denominator) + '}'

    latexs = [fraction_to_latex(f) for f in values]
    labels = ['$ ' + frac_latex + r'\pi $' if frac_latex != '0' else '0' for frac_latex in latexs]

    if 'x' in axis:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    if 'y' in axis:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def set_3D_labels(ax: Axes3D, x_pad: Optional[float] = None,
                  y_pad: Optional[float] = None,
                  z_pad: Optional[float] = None):
    ax.set_xlabel('X', labelpad=x_pad)
    ax.set_ylabel('Y', labelpad=y_pad)
    ax.set_zlabel('Z', labelpad=z_pad)


def set_labels_off(ax: Axes3D):
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')


def remove_tick_labels(ax: Axes | Axes3D):
    # I tried to use ax.set_xticklabels([]) but due to some bug, it made
    # ax.get_tightbbox() return a huge bbox
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    if isinstance(ax, Axes3D):
        ax.zaxis.set_major_formatter(plt.NullFormatter())


def set_axis_scaled(ax: Axes3D):
    ax.set_aspect('equal')


def set_axis_scaled_by_limits(ax: Axes3D):
    """
    This is a workaround for times when ax.set_aspect didn't work well.
    By now, it is preferred to use set_axis_scaled.
    This method reach similar results, but it is not really scaled.
    """
    max_lim = max(ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1])
    min_lim = min(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0])
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    ax.set_zlim(min_lim, max_lim)


def set_zoom_by_limits(ax: Axes | Axes3D, zoom: float):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] / zoom, xlim[1] / zoom)
    ax.set_ylim(ylim[0] / zoom, ylim[1] / zoom)

    if isinstance(ax, Axes3D):
        zlim = ax.get_zlim()
        ax.set_zlim(zlim[0] / zoom, zlim[1] / zoom)


def create_colorbar(fig: Figure, ax: Axes, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.15)
    fig.colorbar(im, cax=cax)


def imshow_with_colorbar(fig: Figure, ax: Axes, data: np.ndarray, ax_title) -> AxesImage:
    if data.dtype == np.float128:
        data = np.array(data, np.float64)
    im = ax.imshow(data)
    create_colorbar(fig, ax, im)
    ax.set_title(ax_title)
    ax.invert_yaxis()
    return im


def save_fig_cropped(fig: Figure, file_path, expand_x, expand_y, pad_x=0.0, pad_y=0.0,
                     translate_x=0.0, translate_y=0.0, **kwargs):
    bbox = fig.get_tightbbox()
    new_bbox = bbox.expanded(expand_x, expand_y)
    new_bbox = new_bbox.padded(pad_x, pad_y)
    new_bbox = new_bbox.translated(translate_x, translate_y)
    fig.savefig(file_path, bbox_inches=new_bbox, **kwargs)


class Arrow3D(FancyArrowPatch):
    """
    A class used to draw arrows in 3D plots.
    The code is taken from
    https://stackoverflow.com/a/74122407
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


# Based on
# https://stackoverflow.com/a/38208040
def draw_elliptic_arrow(ax: Axes, center, width, height, angle, theta_start, theta_end, color='black'):
    # make counterclockwise for arc
    theta1 = min(theta_start, theta_end)
    theta2 = max(theta_start, theta_end)
    arc = mpatches.Arc(center, width, height, angle=angle,
                       theta1=theta1, theta2=theta2, capstyle='round', linestyle='-', lw=2, color=color,
                       zorder=15)
    ax.add_patch(arc)

    end_theta = theta_end

    rad = np.radians
    # cent_x, cent_y = center
    # Create the arrow head
    end_x = (width / 2) * np.cos(rad(end_theta))
    end_y = (height / 2) * np.sin(rad(end_theta))
    cos_a, sin_a = np.cos(rad(angle)), np.sin(rad(angle))
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    end_pos = center + rot_matrix @ np.array([end_x, end_y])

    # Make sure the arrow points outwards
    orientation = theta_end if theta_end > theta_start else np.pi - theta_start

    # Create triangle as arrow head
    head = mpatches.RegularPolygon(
        end_pos,
        3,
        radius=max(width, height) / 6,
        orientation=rad((orientation + angle)),
        color=color
    )
    ax.add_patch(head)

    return arc, head
