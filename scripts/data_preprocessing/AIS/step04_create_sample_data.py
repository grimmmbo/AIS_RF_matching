import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from scripts.modeling.transition_probabilities.AIS_RF_distribution import AIS_RF_probability

def compute_track_characteristics(df):
    """
    Computes descriptive statistics for each vessel track
    
    For each track, this function summarizes:
        - Track duration 
        - Number of AIS messages 
        - Spatial span (lat/lon range)
        - Average speed
        - Average time interval (in seconds) between AIS messages within a track 
        - Variation (standard deviation) between those time intervals 
        - Average distance interval (in kilometers) between AIS messages within a track 
        - Variation (standard deviation) between in those differences
        - Average transition strength 

    Args:
        df (pd.DataFrame): AIS dataset 

    Returns:
        pd.DataFrame: Summary statistics per track
    """
    track_summary = []
    
    for id, vessel_data in df.groupby("ID"):
        # Duration of the track (minutes)
        timestamps = vessel_data["BaseDateTime"].dropna()
        duration_minutes = (timestamps.max() - timestamps.min()).total_seconds() / 60
        
        # Number of AIS points in this track
        num_observations = len(vessel_data)
        
        # Spatial span (simple bounding-box extent)
        lat_span = vessel_data["LAT"].max() - vessel_data["LAT"].min()
        lon_span = vessel_data["LON"].max() - vessel_data["LON"].min()
        
        # Average speed across the track
        avg_speed = vessel_data["speed_kph"].mean()
        
        # Intra-track cadence stats (time/distance between consecutive points)
        mean_intra_time_diff = vessel_data['time_diff'].mean()
        std_intra_time_diff = vessel_data['time_diff'].std()
        mean_intra_dist_diff = vessel_data['distance_diff'].mean()
        std_intra_dist_diff = vessel_data['distance_diff'].std()
        
        # Transition strength along the path (point-to-point)
        trans_probs = vessel_data.apply(
                lambda row: AIS_RF_probability(row['time_diff'], row['distance_diff']),
                axis=1
            )
        mean_transition_strength = trans_probs.mean()
        std_transition_strength = trans_probs.std()
        
        # Compress extremes (exp 0.5)
        trans_probs_corrected = np.array([prob**0.5 for prob in trans_probs])
        mean_transition_strength_corr = trans_probs_corrected.mean()
        std_transition_strength_corr = trans_probs_corrected.std()
        
        # Log transformation
        trans_probs_log = np.array([np.log(prob) for prob in trans_probs_corrected])
        mean_transition_strength_log = trans_probs_log.mean()
        std_transition_strength_log = trans_probs_log.std()
        
        track_summary.append({
            "ID": id, 
            "#points": num_observations,
            "track_duration": duration_minutes,
            "avg_speed_kph": avg_speed,
            "lat_span": lat_span,
            "lon_span": lon_span,
            "mean_intra_time_diff": mean_intra_time_diff,
            "std_intra_time_diff" : std_intra_time_diff,
            "mean_intra_dist_diff" : mean_intra_dist_diff,
            "std_intra_dist_diff" : std_intra_dist_diff,
            "mean_transition_strength" : mean_transition_strength,
            "std_transition_strength" : std_transition_strength,
            "mean_transition_strength_corr" : mean_transition_strength_corr,
            "std_transition_strength_corr" : std_transition_strength_corr,
            "mean_transition_strength_log":mean_transition_strength_log,
            "std_transition_strength_log" :std_transition_strength_log
        })

    return pd.DataFrame(track_summary)

def sample_representative_routes(df, sample_size=5000):
    """
    Draws a representative sample of vessel routes from the full AIS population based on KDE density weighting

    Args:
        df (pd.DataFrame): AIS data 
        sample_size (int): Number of tracks to sample. Defaults to 5000

    Returns:
        tuple: 
            pd.DataFrame: Sampled route metadata
            pd.DataFrame: All route metadata with density and sampling scores
            list: Feature names used as input to KDE
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
