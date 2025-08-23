import numpy as np
import plotly.graph_objects as go
from scripts.modeling.transition_probabilities.AIS_RF_distribution import *
from scripts.modeling.transition_probabilities.match_distribution import *

def plot_combined_probability_surfaces():
    """
    Visualize overlaid AIS–RF probability surfaces: match vs. transition.
    The surface is computed via (1) match_probability(time_sec, distance_km) and (2) AIS_RF_probability(time_sec, distance_km)
    """
    # Define time (0–120 s) and distance (0–2 km) grid
    x_vals = np.linspace(0, 120, 100)
    y_vals = np.linspace(0, 2.0, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute probability surface on grid (both match and transition)
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
