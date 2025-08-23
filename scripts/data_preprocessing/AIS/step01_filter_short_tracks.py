import pandas as pd

def remove_short_tracks(df, min_observations=5):
    """
    Removes vessel tracks with fewer than `min_observations` AIS points

    Args:
        df (pd.DataFrame): AIS data
        min_observations (int): Minimum number of observations required to keep a track

    Returns:
        pd.DataFrame: Cleaned dataframe with only valid tracks
    """
    # Count observations per vessel
    vessel_obs_count = df.groupby("MMSI").size().reset_index(name="obs_count")
    
    # Identify short (invalid) tracks
    invalid_tracks = vessel_obs_count[vessel_obs_count["obs_count"] <= min_observations]["MMSI"]
    
    # Filter out invalid tracks
    df_cleaned = df[~df["MMSI"].isin(invalid_tracks)].copy()
    
    # Print summary
    print(f"{len(invalid_tracks)} tracks with ≤ {min_observations} observations were removed")
    
    return df_cleaned
