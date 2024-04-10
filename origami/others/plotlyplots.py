import plotly.graph_objects as go

from origami.quadranglearray import QuadrangleArray


def plot_with_plotly(quads: QuadrangleArray, alpha=1.0) -> go.Figure:
    dots, indexes = quads.dots, quads.indexes
    dots = dots.astype('float64')

    rows, cols = indexes.shape

    x = dots[0, :].reshape((rows, cols))
    y = dots[1, :].reshape((rows, cols))
    z = dots[2, :].reshape((rows, cols))

    # print(x)
    # print(y)
    # print(z)

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=0.0 * z, opacity=alpha)])
    fig.update_layout(scene_aspectmode='data')
    return fig



def simple_test():
    import numpy as np

    from origami import marchingalgorithm, quadranglearray, RFFQMOrigami, origamiplots

    angle = 0.7 * np.pi
    ls = np.ones(4)
    cs = np.ones(2)
    angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, angle)
    marching = marchingalgorithm.MarchingAlgorithm(angles_left, angles_bottom)
    quads = quadranglearray.dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQMOrigami.RFFQM(quads)
    ori.set_gamma(2)
    fig = plot_with_plotly(ori.dots)
    fig.show()


if __name__ == '__main__':
    simple_test()