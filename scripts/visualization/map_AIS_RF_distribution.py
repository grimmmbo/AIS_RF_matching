import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from scripts.PHMM.distributions.AIS_RF_distribution import AIS_RF_probability

import numpy as np
import plotly.graph_objects as go


def plot_AIS_RF_distribution():
    """
    Plots a transition probability surface over time and distance using a specified transition probability function.

    Args:
        transition_func (function): A function that takes (time, distance) and returns a probability [0,1].
        df (pd.DataFrame, optional): Optional dataframe of observed transitions (with 'delta_t_sec' and 'delta_d_km').
        time_range (tuple): Tuple of (min, max) time values (in minutes).
        dist_range (tuple): Tuple of (min, max) distance values (in km).
        resolution (int): Grid resolution (default 100x100)
    """

    x_vals = np.linspace(0, 36000, 100)
    y_vals = np.linspace(0, 600, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.vectorize(AIS_RF_probability)(X, Y)

    colors = ["#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    plotly_colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    
    fig = go.Figure()

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=plotly_colorscale, showscale=True))

    fig.update_layout(
        title="Transition Probability Surface",
        scene=dict(
            xaxis_title='Time (min)',
            yaxis_title='Distance (km)',
            zaxis_title='Probability'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()