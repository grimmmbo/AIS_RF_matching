import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from scripts.PHMM.distributions.match_distribution import match_probability

import plotly.graph_objects as go
import numpy as np
from haversine import haversine, Unit

def plot_empirical_match_distribution(df):
    """
    Visualizes the match probability surface between AIS and RF data points using a saddle-shaped distribution

    Args:
        df (pd.DataFrame): Dataframe containing AIS and RF data points
    """
    # Get matched AIS and RF pairs 
    df = df[df["State"] == "M"].copy()
    
    # Compute absolute time difference
    df['delta_t_sec'] = (df['AIS_Timestamp'] - df['RF_Timestamp']).abs().dt.total_seconds()
    
    # Compute distance between AIS and RF coordinates 
    df['delta_d_km'] = df.apply(lambda row: haversine(row['AIS'], row['RF'], unit=Unit.KILOMETERS), axis=1)
    
    # Calculate the match probability for each point 
    df = df[['delta_t_sec', 'delta_d_km']].copy()
    df["prob"] = df.apply(lambda row: match_probability(row["delta_t_sec"], row["delta_d_km"]), axis=1)

    x_vals = np.linspace(0, 120, 100)
    y_vals = np.linspace(0, 2.0, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.vectorize(match_probability)(X, Y)

    colors = ["#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    plotly_colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    
    # Plot surface and matched points
    fig = go.Figure()

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=plotly_colorscale, showscale=True))
    fig.add_trace(go.Scatter3d(
        x=df["delta_t_sec"],
        y=df["delta_d_km"],
        z=df["prob"],
        mode='markers+text',
        marker=dict(size=6, color='#e50da2'),
        name="AIS & RF Match"))

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
    Visualizes the match probability surface with only extreme values ((0,0), (2,0), (120, 0), (2, 120)) shown as reference points
    """

    x_vals = np.linspace(0, 120, 100)
    y_vals = np.linspace(0, 2.0, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.vectorize(match_probability)(X, Y)

    colors = ["#d53e4f", "#f46d43", "#fee08b", "#feffb2"]
    colors_extreme_points = ["#d53e4f", "#3288bd", "#66c2a5", "#fdae61"]
    labels = ["(0s, 0km)", "(120s, 0km)", "(0s, 2km)", "(120s, 2km)"]
    plotly_colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    
    # Define extreme points
    extreme_points = [
        (0,0),
        (120,0),
        (0,2),
        (120,2)
    ]
    
    # Plot surface 
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
        margin=dict(l=0, r=0, b=0, t=40))

    fig.show()