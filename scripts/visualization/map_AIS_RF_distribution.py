import numpy as np
import plotly.graph_objects as go
from scripts.modeling.transition_probabilities.AIS_RF_distribution import *

def plot_AIS_RF_distribution():
    """
    Visualize the AIS–RF transition probability surface over time and distance
    Computes the probability surface via AIS_RF_probability(time_sec, distance_km)
    """
    # Define time (0–551 s, using 551 as the trajectory split threshold) and distance (0–7 km) grid
    x_vals = np.linspace(0, 551, 50)
    y_vals = np.linspace(0, 7, 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute probability surface on grid
    Z = np.vectorize(AIS_RF_probability)(X, Y)

    colors = ["#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    plotly_colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    
    # Create 3D surface plot
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
    Visualize the AIS–RF transition probability surface and annotate extreme points ((0,0), (551,0), (0, 7), (551, 7))
    """
    # Define time (0–551 s) and distance (0–7 km) grid
    x_vals = np.linspace(0, 551, 50)
    y_vals = np.linspace(0, 7, 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute probability surface
    Z = np.vectorize(AIS_RF_probability)(X, Y)

    colors = ["#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    plotly_colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    
    # Define extreme points
    extreme_points = [(0,0),(551,0),(0,7),(551,7)]
    
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