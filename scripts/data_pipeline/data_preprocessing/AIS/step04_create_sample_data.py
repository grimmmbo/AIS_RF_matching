import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

def compute_track_characteristics(df):
    """
    Computes descriptive statistics for each vessel track

    Args:
        df (pd.DataFrame): AIS data 

    Returns:
        pd.DataFrame: Summary statistics per (MMSI, track_id)
    """
    track_summary = []
    
    for id, vessel_data in df.groupby("ID"):
        # Compute track duration in minutes
        timestamps = vessel_data["BaseDateTime"].dropna()
        duration_minutes = (timestamps.max() - timestamps.min()).total_seconds() / 60
        
        # Count number of observations (AIS points) in this track
        num_observations = len(vessel_data)
        
        # Compute geographic span (how far the vessel moved in latitude and longitude)
        lat_span = vessel_data["LAT"].max() - vessel_data["LAT"].min()
        lon_span = vessel_data["LON"].max() - vessel_data["LON"].min()
        
        # Calculate the average speed
        avg_speed = vessel_data["speed_kph"].mean()
        
        # Compute the average time between consecutive AIS points (in seconds)
        mean_intra_time_diff = (
            timestamps.diff().dropna().dt.total_seconds().mean() if len(timestamps) > 1 else None
        )

        # Append all computed metrics to the list
        track_summary.append({
            "ID": id, 
            "#points": num_observations,
            "track_duration": duration_minutes,
            "avg_speed_kph": avg_speed,
            "lat_span": lat_span,
            "lon_span": lon_span,
            "mean_intra_time_diff": mean_intra_time_diff,
        })

    return pd.DataFrame(track_summary)

def sample_representative_routes(df, sample_size=5000):
    """
    Draws a representative sample of vessel routes from the full AIS population based on KDE density weighting

    Args:
        df (pd.DataFrame): AIS data 
        sample_size (int): Number of tracks to sample
        bandwidth (float): KDE bandwidth for density estimation
        random_state (int): Seed for reproducibility

    Returns:
        pd.DataFrame: Sampled route metadata
        pd.DataFrame: All route metadata with density and sampling scores
        pd.DataFrame: Full AIS data (subset) for the sampled routes
    """
    # Compute summary statistics per track
    AIS_charateristics = compute_track_characteristics(df)

    # Features to match on
    relevant_columns = ["#points", "track_duration", "avg_speed_kph", "lat_span", "lon_span", "mean_intra_time_diff"]
    df_features = AIS_charateristics[relevant_columns].dropna()

    # Fit KDE on track characteristics
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)
    kde = KernelDensity(kernel="gaussian", bandwidth=0.5)
    kde.fit(scaled_features)

    # Invert the log-density score so that rare tracks get higher sampling probability
    log_density = kde.score_samples(scaled_features)
    AIS_charateristics["density_score"] = -log_density
    AIS_charateristics["sampling_weight"] = np.exp(AIS_charateristics["density_score"] * 0.25)

    # Sample representative set of tracks
    sampled_tracks = AIS_charateristics.sample(
        n=sample_size,
        weights=AIS_charateristics["sampling_weight"],
        random_state=42
    )

    # Retrieve full AIS points for sampled tracks
    sampled_track_ids = sampled_tracks["ID"].unique()
    df_sampled_tracks = df[df["ID"].isin(sampled_track_ids)].copy()

    return sampled_tracks, AIS_charateristics, relevant_columns, df_sampled_tracks
