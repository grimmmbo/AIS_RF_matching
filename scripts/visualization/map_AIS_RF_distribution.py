from scripts.PHMM.distributions.AIS_RF_distribution import * 

import numpy as np
import plotly.graph_objects as go

def plot_AIS_RF_distribution():
    """
    Plots a transition probability surface over time and distance using a specified transition probability function.
    """

    x_vals = np.linspace(0, 551, 50)
    y_vals = np.linspace(0, 7, 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.vectorize(AIS_RF_probability)(X, Y)

    colors = ["#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    plotly_colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    
    fig = go.Figure()

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=plotly_colorscale, showscale=True))

    fig.update_layout(
        title="Transition Probability Surface",
        scene=dict(
            xaxis_title='Time (sec)',
            yaxis_title='Distance (km)',
            zaxis_title='Probability'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
    
def plot_extreme_AIS_RF_distribution():
    """
    Visualizes the match probability surface with only extreme values ((0,0), (2,0), (120, 0), (2, 120)) shown as reference points
    """
    x_vals = np.linspace(0, 551, 50)
    y_vals = np.linspace(0, 7, 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.vectorize(AIS_RF_probability)(X, Y)

    colors = ["#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    plotly_colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    
    # Define extreme points
    extreme_points = [
        (0,0),
        (551,0),
        (0,7),
        (551,7)
    ]
    
    # Plot surface 
    fig = go.Figure()

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=plotly_colorscale, showscale=True))
    
    for (sec_extreme, dist_extreme), label in zip(extreme_points, labels):
        prob_extreme = AIS_RF_probability(sec_extreme, dist_extreme)
        fig.add_trace(go.Scatter3d(
            x=[sec_extreme],
            y=[dist_extreme],
            z=[prob_extreme],
            mode='markers+text',
            marker=dict(size=6, color="black"),
            text=[label]        
        ))

    fig.update_layout(
        title="Match Probability Surface (Saddle-Based)",
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Distance (km)',
            zaxis_title='Probability'),
        width=900,
        height=700,
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=40))

    fig.show()