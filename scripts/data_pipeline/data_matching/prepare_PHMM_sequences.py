import pandas as pd

def split_RF_data(RF_data, time_tolerance_minutes=1):
    """
    Splits RF data into matched and unmatched records based on timestamp proximity to AIS data

    Args:
        RF_data (pd.DataFrame):  A DataFrame containing RF and AIS data
        time_tolerance_minutes (int): The time difference threshold (in minutes) used to determine matches, defaults to 1

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            matched: DataFrame containing RF-AIS pairs where timestamps are within the defined tolerance
            unmatched: DataFrame containing RF points without a corresponding AIS match within tolerance
    """
    threshold = pd.Timedelta(minutes=time_tolerance_minutes)
    RF_data['TimeDifference'] = (RF_data['TimeStamp_RF'] - RF_data['TimeStamp_AIS']).abs()

    matched = RF_data[RF_data['TimeDifference'] <= threshold].copy()
    unmatched = RF_data[RF_data['TimeDifference'] > threshold].copy()

    return matched, unmatched

def assign_state_labels(matched, unmatched, columns):
    """
    Adds state labels to matched and unmatched DataFrames

    Args:
        matched (pd.DataFrame): DataFrame containing RF-AIS pairs where timestamps are within the defined tolerance
        unmatched (pd.DataFrame): DataFrame containing RF points without a corresponding AIS match within tolerance
        columns (list): List containing relevant columns

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            matched: Updated DataFrame with 'State' column set to 'M'
            unmatched: Updated DataFrame with 'State' column set to 'RF'
    """
    # Matched pairs get state 'M'
    matched['State'] = 'M'
    matched['Timestamp'] = matched['TimeStamp_AIS']
    matched['AIS_Timestamp'] = matched['TimeStamp_AIS']
    matched['RF_Timestamp'] = matched['TimeStamp_RF']
    matched['AIS'] = matched.apply(lambda row: (row['AIS_LAT'], row['AIS_LON']), axis = 1)
    matched['RF'] = matched.apply(lambda row: (row['RF_LAT'], row['RF_LON']), axis = 1)

    # Unmatched pairs get state 'RF'
    unmatched['State'] = 'RF'
    unmatched['Timestamp'] = unmatched['TimeStamp_RF']
    unmatched['AIS_Timestamp'] = None
    unmatched['RF_Timestamp'] = unmatched['TimeStamp_RF']
    unmatched['AIS'] = None
    unmatched['RF'] = unmatched.apply(lambda row: (row['RF_LAT'], row['RF_LON']), axis = 1)

    return matched[columns], unmatched[columns]

def extract_unmatched_AIS_data(AIS_data, matched, columns):
    """
    Extracts AIS records that weren't matched with RF points

    Args:
        AIS_data (pd.DataFrame): AIS data
        matched_df (pd.DataFrame): Matched RF-AIS data used to identify already paired entries
        columns (List): List containing relevant columns

    Returns:
        pd.DataFrame:  DataFrame containing AIS-only rows, labeled with state 'AIS'
    """
    # Merge AIS data with matched pairs to find unmatched AIS records
    df = AIS_data.merge(
        matched,
        left_on=['MMSI', 'SubTrackID', 'BaseDateTime'],
        right_on=['MMSI', 'SubTrackID', 'Timestamp'],
        how='left',
        indicator=True
    )
    df = df[df['_merge'] == 'left_only'].copy()
    
    df['AIS'] = df.apply(lambda row: (row['LAT'], row['LON']), axis = 1)
    df['RF'] = None
    df['Timestamp'] = df['BaseDateTime']
    df['AIS_Timestamp'] = df['BaseDateTime']
    df['RF_Timestamp'] = None
    df['State'] = "AIS"
    
    return df[columns]

def combine_and_sort_labeled_data(matched, unmatched, AIS_only):
    """
    Combines matched, unmatched, and AIS-only data into a single sorted DataFrame

    Args:
        matched (pd.DataFrame): DataFrame containing time-matched AIS and RF records
        unmatched (pd.DataFrame): DataFrame containing RF records without corresponding AIS matches containing RF records without corresponding AIS matchestion_
        AIS_only (pd.DataFrame): DataFrame containing AIS records without corresponding RF matches

    Returns:
        pd.DataFrame: Combined and sorted DataFrame of all track states
    """ 
    df = pd.concat([matched, unmatched, AIS_only], ignore_index=True)
    df = df.sort_values(by=['MMSI', 'SubTrackID', 'Timestamp']).reset_index(drop=True)
    
    return df

def drop_shortest_10_percent_tracks(df):
    """
    Removes the shortest 10% of vessel tracks based on the length of the dataframe
    
    Args:
        df (pd.DataFrame): DataFrame containing vessel data
        
    Returns:
        pd.DataFrame: Filtered DataFrame excluding the shortest 10% of tracks
    """
    track_counts = df.groupby(['MMSI', 'SubTrackID']).size()
    num_to_drop = int(len(track_counts) * 0.10)
    tracks_to_drop = track_counts.nsmallest(num_to_drop).index
    df = df[~df.set_index(['MMSI', 'SubTrackID']).index.isin(tracks_to_drop)].reset_index(drop=True)

    return df

def assign_unique_track_ids(df):
    """
    Assigns a unique TrackID to each 'MMSI', 'SubTrackID' pairs
    
    Args:
        df (pd.DataFrame): DataFrame containing vessel data
    
    Returns:
        pd.DataFrame: DataFrame with an added 'TrackID' column
    """
    df['TrackID'] = df.groupby(['MMSI', 'SubTrackID']).ngroup()
    df = df[['TrackID', 'MMSI', 'SubTrackID', 'Timestamp', 'AIS_Timestamp', 'RF_Timestamp', 'AIS', 'RF', 'State']]
    df = df.sort_values(by=['TrackID', 'Timestamp']).reset_index(drop=True)
    
    return df

def add_synthetic_begin_end(df):
    """
    Adds synthetic 'begin' and 'end' rows to each vessel track to mark the start and end of a sequence
    
    Args:
        df (pd.DataFrame): DataFrame containing vessel data
    
    Returns:
        pd.DataFrame: DataFrame with added 'begin' and 'end' rows for each track
    """
    begin_end_rows = []
    
    for (mmsi, subtrack_id), group in df.groupby(['MMSI', 'SubTrackID']):
        group = group.sort_values(by='Timestamp')

        begin_row = group.iloc[0].copy()
        begin_row['State'] = 'begin'
        begin_row['AIS'] = None
        begin_row['RF'] = None
        begin_row['Timestamp'] = group['Timestamp'].min() - pd.Timedelta(seconds=1)
        begin_row['AIS_Timestamp'] = None
        begin_row['RF_Timestamp'] = None

        end_row = group.iloc[-1].copy()
        end_row['State'] = 'end'
        end_row['AIS'] = None
        end_row['RF'] = None
        end_row['Timestamp'] = group['Timestamp'].max() + pd.Timedelta(seconds=1)
        end_row['AIS_Timestamp'] = None
        end_row['RF_Timestamp'] = None

        begin_end_rows.extend([begin_row, end_row])

    df = pd.concat([df, pd.DataFrame(begin_end_rows)], ignore_index=True)
    df = df.sort_values(by=['MMSI', 'SubTrackID', 'Timestamp']).reset_index(drop=True)
    
    return df

def build_PHHM_sequence_dataframe(AIS_data, RF_data, time_tolerance_minutes=1):
    """
    Processes AIS and RF data to generate a dataframe with state labels
    
    Args:
        AIS_data (pd.DataFrame): DataFrame containing AIS records
        RF_data (pd.DataFrame): DataFrame containing RF records,
        time_tolerance_minutes (int): The time difference threshold (in minutes) used to determine matches, defaults to 1

    Returns:
        pd.DataFrame: Processed DataFrame containing labeled vessel tracks
    """
    columns = ['MMSI', 'SubTrackID', 'AIS', 'RF', 'State', 'Timestamp', 'AIS_Timestamp', 'RF_Timestamp']
    matched, unmatched = split_RF_data(RF_data, time_tolerance_minutes)
    matched, unmatched = assign_state_labels(matched, unmatched, columns)
    AIS_only = extract_unmatched_AIS_data(AIS_data, matched, columns)
    
    df = combine_and_sort_labeled_data(matched, unmatched, AIS_only)
    df = drop_shortest_10_percent_tracks(df)
    df = assign_unique_track_ids(df)
    df = add_synthetic_begin_end(df)
    
    return df

if __name__ == "__main__":
    print("Starting PHMM sequence generation...")
        
    AIS_SOURCE_PATH = "../../../data/processed/modified_AIS_data.parquet"
    RF_SOURCE_PATH = "../../../data/processed/synthetic_RF_data.parquet"
    DESTINATION_PATH = "../../../data/processed/PHMM_sequence_data.parquet"
    
    AIS_df = pd.read_parquet(AIS_SOURCE_PATH)
    RF_df = pd.read_parquet(RF_SOURCE_PATH)
    PHMM_df = build_PHHM_sequence_dataframe(AIS_df, RF_df)
    PHMM_df.to_parquet(DESTINATION_PATH)
    
    print("Data is saved...")

