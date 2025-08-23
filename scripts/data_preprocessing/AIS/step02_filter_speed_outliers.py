import numpy as np
import pandas as pd
from haversine import haversine

def add_speed_info(df):
    """
    Adds time_diff, distance_diff, and speed_kph columns to the AIS dataframe
    
    Args:
        df (pd.DataFrame): AIS dataframe
    
    Returns:
        pd.DataFrame: Same dataframe with added columns
    """
    # Calculate time between consecutive points
    df["time_diff"] = df.groupby("MMSI")["BaseDateTime"].diff().dt.total_seconds()
    
    # Calculate distance between consecutive points 
    df["prev_lat"] = df.groupby("MMSI")["LAT"].shift(1)
    df["prev_lon"] = df.groupby("MMSI")["LON"].shift(1)

    def compute_distance(row):
        return haversine((row["prev_lat"], row["prev_lon"]), (row["LAT"], row["LON"]))
    
    df["distance_diff"] = df.apply(compute_distance, axis=1)

    # Calculate speed kph
    df["speed_kph"] = np.where(
        (df["time_diff"] > 0) & (df["distance_diff"] > 0),
        (df["distance_diff"] / df["time_diff"]) * 3600,
        0
    )

    return df

def remove_high_speed_outliers(df):
    """
    Removes vessels that contain high-speed outliers based on the IQR method
    Based on this strategy: https://medium.com/@noorfatimaafzalbutt/outliers-detection-and-removal-using-iqr-method-e629aa8089a8  
    
    Args:
        df (pd.DataFrame): AIS dataframe with a "speed_kph" column
    
    Returns:
        pd.DataFrame: Filtered dataframe with outlier vessels removed  
    """
    Q1 = df["speed_kph"].quantile(0.25)
    Q3 = df["speed_kph"].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR

    df["is_high_outlier"] = df["speed_kph"] > upper_limit
    outlier_mmsis = df.loc[df["is_high_outlier"], "MMSI"].unique()
    
    return df[~df["MMSI"].isin(outlier_mmsis)].copy()

def remove_low_speed_outliers(df, min_avg_speed_kph=5):
    """
    Removes vessels with average speed below a minimum threshold
    
    Args:
        df (pd.DataFrame): AIS dataframe with "speed_kph" column
        min_avg_speed_kph (float, optional): Minimum average speed in km/h. Defaults to 5
                                             
    Returns:
        pd.DataFrame: Filtered dataframe with slow vessels removed
    """
    avg_speeds = df.groupby("MMSI")["speed_kph"].mean().reset_index()
    outlier_mmsis = avg_speeds[avg_speeds["speed_kph"] < min_avg_speed_kph]["MMSI"]
    
    return df[~df["MMSI"].isin(outlier_mmsis)].copy()
