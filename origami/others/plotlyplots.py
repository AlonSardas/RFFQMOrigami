import plotly.graph_objects as go

from origami.quadranglearray import QuadrangleArray


def plot_with_plotly(quads: QuadrangleArray, alpha=1.0) -> go.Figure:
    dots, indexes = quads.dots, quads.indexes
    dots = dots.astype('float64')

    rows, cols = indexes.shape

    x = dots[0, :].reshape((rows, cols))
    y = dots[1, :].reshape((rows, cols))
    z = dots[2, :].reshape((rows, cols))

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=0.0 * z, opacity=alpha)])
    fig.update_layout(scene_aspectmode='data')
    return fig
