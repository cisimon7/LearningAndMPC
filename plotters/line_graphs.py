import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Union
from numpy import ndarray
from torch import Tensor


# Maybe set the default custom size for all plots and then individual sizes can be given to each lot

def list_plot(x_data: Union[ndarray, Tensor], y_data: Union[ndarray, Tensor], name="", marker=None) -> go.Scatter:
    if marker is None:
        marker = dict(size=3)
    return go.Scatter(y=y_data, x=x_data, mode="markers", marker=marker, name=name)


def line_plot(x_data: Union[ndarray, Tensor], y_data: Union[ndarray, Tensor], z_data: Union[ndarray, Tensor]=None, name="", marker=None) -> go.Scatter:
    if marker is None:
        marker = dict(size=3)
    return go.Scatter(y=y_data, x=x_data, mode="lines", marker=marker, name=name) if z_data is None else go.Scatter3d(x=x_data, y=y_data, z=z_data, mode="lines", marker=marker, name=name)


def show(graphs: [go.Scatter], layout: go.Layout = None, title: str = "", height=700, width=1_000, x_axis=None,
         y_axis=None):
    if y_axis is None:
        y_axis = dict()
    if x_axis is None:
        x_axis = dict()
    go.Figure(
        data=graphs,
        layout=go.Layout(title=dict(text=title, x=0.5), height=height, width=width, xaxis=x_axis, yaxis=y_axis) if (
            layout is None) else layout
    ).show()


def show_multi(graphs: [[[go.Scatter]]], title=None, subplot_titles: [str] = None, x_axis_label=None,
               y_axis_label=None, show_legend=False):
    row, col = len(graphs), len(graphs[0])
    fig = make_subplots(rows=row, cols=col, subplot_titles=subplot_titles)
    for (i, row_graph) in enumerate(graphs):
        for (j, row_col_graph) in enumerate(row_graph):
            for graph in row_col_graph:
                fig.add_trace(graph, row=i + 1, col=j + 1)
            fig.update_xaxes(title_text=x_axis_label, row=i + 1, col=j + 1)
            fig.update_yaxes(title_text=y_axis_label, row=i + 1, col=j + 1)

    fig.update_layout(title=title, showlegend=show_legend)
    fig.show()


def confusion_graph(labels: [str], enc_labels, actual: ndarray, prediction: ndarray, color_scale="Blues"):
    n, x = len(labels), enc_labels
    y = x.copy()
    xz, yz = np.meshgrid(x, y)
    z = np.zeros_like(xz)
    for (row, col) in zip(xz, yz):
        for (x_label, y_label) in zip(row, col):
            i = np.where(x == x_label)
            j = np.where(y == y_label)
            z[i, j] = (np.where(actual == x_label, 1, 0) + np.where(prediction == y_label, 1, 0) == 2).sum()
    return go.Heatmap(x=labels, y=labels, z=z, colorscale=color_scale, text=z, texttemplate="%{text}", showscale=False)