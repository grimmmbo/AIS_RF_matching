import pandas as pd
import numpy as np

def split_into_continuous_tracks(df, time_gap_pct=0.99, min_points=5, min_avg_speed=5):
    """
    Splits vessel tracks into continuous subtracks based on time gaps,
    and removes short or slow subtracks

    Returns:
        pd.DataFrame: Cleaned dataframe with continuous TrackIDs
    """
    gap_threshold = df["time_diff"].quantile(time_gap_pct)

    # Assign subtrack IDs per vessel
    def assign_track_ids(df):
        def split_group(vessel_data):
            vessel_data = vessel_data.sort_values("BaseDateTime")
            vessel_data["track_id"] = (vessel_data["time_diff"] > gap_threshold).cumsum()
            return vessel_data

        return pd.concat([split_group(vessel_data) for _, vessel_data in df.groupby("MMSI")])
    
    df = assign_track_ids(df).reset_index(drop=True)

    # Remove subtracks with fewer than `min_points`
    vessel_obs_count = df.groupby(["MMSI", "track_id"]).size()
    valid_tracks = vessel_obs_count[vessel_obs_count >= min_points].reset_index()[["MMSI", "track_id"]]
    df = df.merge(valid_tracks, on=["MMSI", "track_id"])

    # Reset time/distance/speed at start of each subtrack
    first_idx = df.groupby(["MMSI", "track_id"]).head(1).index
    df.loc[first_idx, ["time_diff", "distance_diff", "speed_kph"]] = np.nan

    # Remove subtracks with low average speed
    avg_speeds = df.groupby(["MMSI", "track_id"])["speed_kph"].mean().reset_index()
    valid_mmsis = avg_speeds[avg_speeds["speed_kph"] >= min_avg_speed][["MMSI", "track_id"]]
    df = df.merge(valid_mmsis, on=["MMSI", "track_id"])
    
    # Create unique identifier
    df["ID"] = list(zip(df["MMSI"], df["track_id"]))

    cols = [
        "ID", "MMSI", "IMO", "BaseDateTime", "LAT", "LON", "SOG", 
        "Heading", "time_diff", "distance_diff", "speed_kph", "track_id"
    ]
    return df[cols].sort_values(["ID", "BaseDateTime"]).reset_index(drop=True)
