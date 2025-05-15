import pandas as pd

def assign_ids(single_track, max_subtrack_duration = pd.Timedelta(days=1)):
    """
    Assigns unique idenitfiers to individual subtracks that result from splitting a vessel's track

    Args:
        single_track (pd.DataFrame): Track data for a single vessel
        max_subtrack_duration (pd.Timedelta): Maximum allowed cumulative duration per 'subtrack'. If the cumulative time exceeds this value, a new subtrack is started, defaults to 1 day

    Returns:
        pd.DataFrame: Same dataframe as input, but with a new column called 'SubTrackID'
    """
    subtrack_ids = []
    subtrack_id = 0
    cumulative = pd.Timedelta(0)

    for diff in single_track['TimeDiff'].fillna(pd.Timedelta(0)):
        cumulative += diff
        if cumulative > max_subtrack_duration:
            subtrack_id += 1
            cumulative = diff
        subtrack_ids.append(subtrack_id)

    single_track['SubTrackID'] = subtrack_ids
    return single_track

def split_vessel_tracks(df, max_duration = pd.Timedelta(days = 2)):
    """
    To increase the amount of usable data, vessel tracks that exceed a predefined duration are segmented into smaller tracks, also referred to as 'subtracks'

    Args:
        df (pd.DataFrame): AIS data
        max_duration (pd.Timedelta): Duration threshold above which a track is split into subtracks, defaults to 2 days
        
    Returns:
        pd.DataFrame: AIS data with 'SubTrackID' column
    """
    # Calculate time difference between AIS signals
    df = df.sort_values(by=['MMSI', 'BaseDateTime']).copy()
    df['TimeDiff'] = df.groupby('MMSI')['BaseDateTime'].diff()
    
    # To improve data reliability, vessels are filtered such that only those with a maximum time gap of 90 minutes or less between AIS data points/signals are retained
    # This threshold helps exclude highly fragmented tracks, while still allowing for a few outliers to remain included
    max_gaps = df.groupby("MMSI")["TimeDiff"].max()
    reliable_vessels = max_gaps[max_gaps <= pd.Timedelta(minutes=90)].index
    df = df[df["MMSI"].isin(reliable_vessels)]
    
    # Determine track durations
    track_durations = df.groupby("MMSI")["BaseDateTime"].agg(["min", "max"])
    track_durations["TrackDuration"] = track_durations["max"] - track_durations["min"]
    track_durations["TrackDays"] = track_durations["TrackDuration"].dt.days
    
    short_tracks = track_durations[track_durations["TrackDays"] < max_duration.days].index
    long_tracks = track_durations[track_durations["TrackDays"] >= max_duration.days].index

    # Assign subtrack IDs
    short_track_df = df[df["MMSI"].isin(short_tracks)].copy()
    short_track_df["SubTrackID"] = 0  

    long_track_df = df[df["MMSI"].isin(long_tracks)].copy()
    long_track_df = long_track_df.groupby("MMSI", group_keys=False).apply(assign_ids)

    return pd.concat([short_track_df, long_track_df]).sort_values(by=['MMSI', 'BaseDateTime'])

if __name__ == "__main__":
    print("Segmenting vessel tracks has started...")
    
    SOURCE_PATH = "../../../../data/processed/cargo_vessels.parquet"
    DESTINATION_PATH = "data/processed/segmented_tracks.parquet"

    df = pd.read_parquet(SOURCE_PATH)
    split_df = split_vessel_tracks(df)
    split_df.to_parquet(DESTINATION_PATH)
    
    print("Data is saved...")
