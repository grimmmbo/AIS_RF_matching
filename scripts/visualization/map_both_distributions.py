import numpy as np
import plotly.graph_objects as go

from scripts.PHMM.distributions.AIS_RF_distribution import * 
from scripts.PHMM.distributions.match_distribution import * 

def plot_combined_probability_surfaces():
    """
    Visualizes the combined probability surface
    """
    x_vals = np.linspace(0, 120, 100)
    y_vals = np.linspace(0, 2.0, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z_match = np.vectorize(match_probability)(X, Y)
    Z_transition = np.vectorize(AIS_RF_probability)(X, Y)

    fig = go.Figure()
    
    colors = ["#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    plotly_colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]

    # Match-surface
    fig.add_trace(go.Surface(
        z=Z_match, x=X, y=Y,
        colorscale=plotly_colorscale,
        name="Match Probability",
        showscale=False,
        opacity=0.8
    ))

    # AIS-RF transition-surface 
    fig.add_trace(go.Surface(
        z=Z_transition, x=X, y=Y,
        colorscale="Viridis",
        name="Transition Probability",
        showscale=False,
        opacity=0.5
    ))

    fig.update_layout(
        title="Overlayed Match & Transition Probability Surfaces",
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
