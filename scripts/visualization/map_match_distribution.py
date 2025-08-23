import plotly.graph_objects as go
import numpy as np
from scripts.modeling.transition_probabilities.match_distribution import *

def plot_empirical_match_distribution():
    """
    Visualize the AIS–RF match probability surface over time and distance
    The surface is computed via match_probability(time_sec, distance_km)
    """
    # Define time (0–120 s) and distance (0–2 km) grid
    x_vals = np.linspace(0, 120, 100)
    y_vals = np.linspace(0, 2.0, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute probability surface on grid
    Z = np.vectorize(match_probability)(X, Y)

    colors = ["#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    plotly_colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    
    # Create 3D surface plot
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=plotly_colorscale, showscale=True))
    fig.update_layout(
        title="Match Probability Surface (Saddle-Based)",
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Distance (km)',
            zaxis_title='Probability'),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40))

    fig.show()
    
def plot_extreme_match_distribution():
    """
    Visualize the AIS–RF match probability surface and annotate extreme points ((0,0), (0,2), (120, 0), (120, 2))
    """
    # Define time (0–120 s) and distance (0–2 km) grid
    x_vals = np.linspace(0, 120, 100)
    y_vals = np.linspace(0, 2.0, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute probability surface
    Z = np.vectorize(match_probability)(X, Y)

    colors = ["#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    colors_extreme_points = ["#482966", "#427a86", "#31947d", '#b6c839']
    labels = ["(a)", "(b)", "(c)", "(d)"]
    plotly_colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    
    # Define extreme points
    extreme_points = [(0,0),(120,0),(0,2),(120,2)]
    
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=plotly_colorscale, showscale=True))
    
    for (sec_extreme, dist_extreme), color, label in zip(extreme_points, colors_extreme_points, labels):
        prob_extreme = match_probability(sec_extreme, dist_extreme)
        fig.add_trace(go.Scatter3d(
            x=[sec_extreme],
            y=[dist_extreme],
            z=[prob_extreme],
            mode='markers+text',
            marker=dict(size=6, color=color),
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