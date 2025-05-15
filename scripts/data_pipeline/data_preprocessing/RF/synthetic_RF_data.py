import random
import math
import pandas as pd
import numpy as np 
from datetime import datetime

def generate_time_specific_data(
    time_options = ["01:41:00", "09:41:00", "13:41:00", "21:41:00"], 
    probabilities = [0.1, 0.3, 0.4, 0.2],
    mean = 1000, 
    std_dev = 100, 
    bearing_options = [0, 45, 90, 135, 180, 225, 270, 315]):
    """
    Generates random time variations with associated error distances and bearings
    
    Args:
        time_options (list): A list containing timestamp values values, defaults to ["01:41:00", "09:41:00", "13:41:00", "21:41:00"]
        probabilities (list): The probabilities associated with each choice, defaults to [0.1, 0.3, 0.4, 0.2]
        mean (float): Mean value for error distance distribution, defaults to 1000
        std_dev (float): Standard deviation for error distribution, defaults to 100
        bearing_options (list): A list containing bearing values, defaults to [0, 45, 90, 135, 180, 225, 270, 315]

    Returns:
        dict: Dictionary where keys are selected time strings, and values are [distance, bearing]
    """
    # Randomly select a subset of times (1 to 4) from the provided options 
    num_times = random.choices(range(1, len(time_options)+1), weights = probabilities)[0]
    selected_times = random.sample(time_options, num_times)
    
    # Generate error distance and direction for each selected time
    time_variations = {
        time: [round(np.random.normal(mean, std_dev, 1)[0], 6), random.choice(bearing_options)]
        for time in selected_times}

    return time_variations

def compute_coordinates(latitude, longitude, distance, bearing, bearing_noise_deg):
    """
    Calculates a new geographical position (latitude, longitude) based on a known AIS signal location

    Args:
        latitude (float): Latitude of the AIS data point to which the RF signal will be aligned
        longitude (float): Longitude of the AIS data point to which the RF signal will be aligned
        distance (float): Error distance in meters
        bearing (float): The heading of the AIS data point
        bearing_noise_deg (float): Random noise (±degrees) to add to the bearing
        
    Returns:
        tuple: Latitude and Longitude of RF data point
    """
    # Radius of Earth (https://www.mathworks.com/help/map/ref/earthradius.html)
    R = 6371000  
    
    # Calculate a new geographical point from a known starting point using a given distance and bearing (https://www.quora.com/What-is-the-calculation-for-the-second-latitude-and-longitude-coordinates-when-we-have-bearing-distance-and-one-latitude-and-longitude-coordinates)
    # Add some noise to bearing variable
    noise = np.random.uniform(-bearing_noise_deg, bearing_noise_deg)
    bearing = (bearing + noise) % 360
    bearing = math.radians(bearing)

    # Calculate new latitude and longitude
    latitude, longitude = math.radians(latitude), math.radians(longitude)
    new_latitude = math.asin(
        math.sin(latitude) * math.cos(distance / R) + math.cos(latitude) * math.sin(distance / R) * math.cos(bearing))
    new_longitude = longitude + math.atan2(
        math.sin(bearing) * math.sin(distance / R) * math.cos(latitude),
        math.cos(distance / R) - math.sin(latitude) * math.sin(new_latitude))

    # Convert back to degrees
    new_latitude = math.degrees(new_latitude)
    new_longitude = math.degrees(new_longitude)

    return new_latitude, new_longitude

def create_synthetic_RF_data(df):
    """
    Generates synthetic RF data by selecting AIS data points closest to randomly generated time variations

    Args:
        df (pd.DataFrame): AIS data

    Returns:
        pd.DataFrame: Synthetic RF data 
    """
    RF_records = []
    grouped_tracks = df.groupby(["MMSI", "SubTrackID"])

    for (mmsi, track_id), single_track in grouped_tracks:
        single_track = single_track.copy()
        
        # Generate time variations with associated errors 
        time_variations = generate_time_specific_data()
        
        # Convert timestamps to minutes to simplify time comparisons
        single_track["Minutes"] = (single_track["BaseDateTime"].dt.hour * 60 + single_track["BaseDateTime"].dt.minute)
        time_variations_minutes = {time: datetime.strptime(time, "%H:%M:%S").hour * 60 + datetime.strptime(time, "%H:%M:%S").minute for time in time_variations}

        # Find closest AIS point per generated RF time
        closest_AIS_points = {RF_time: None for RF_time in time_variations_minutes}
        
        for _, row in single_track.iterrows():
            AIS_time_minutes = row["Minutes"]
            
            # Find the generated RF time that is closest to this AIS time
            closest_RF_time = min(time_variations_minutes, key=lambda time: abs(time_variations_minutes[time] - AIS_time_minutes))
    
            # If there is no stored AIS row yet or the current one is a better match, update it 
            best_match = closest_AIS_points[closest_RF_time]
            target_time_minutes = time_variations_minutes[closest_RF_time]
            
            if best_match is None or abs(target_time_minutes - AIS_time_minutes) < abs(target_time_minutes - best_match["Minutes"]):
                closest_AIS_points[closest_RF_time] = row

        # Generate RF records based on matched AIS points 
        for RF_time, matched_AIS_row in closest_AIS_points.items():
            if matched_AIS_row is not None:
                distance, bearing_noise = time_variations[RF_time]
                RF_lat, RF_lon = compute_coordinates(
                    matched_AIS_row["LAT"], 
                    matched_AIS_row["LON"], 
                    distance, 
                    matched_AIS_row["Heading"], 
                    bearing_noise)

                # Create a unique timestamp for RF by combining date and generated time
                RF_timestamp = pd.to_datetime(f"{matched_AIS_row['BaseDateTime'].date()} {str(RF_time).zfill(8)}")

                RF_records.append({
                    "MMSI": mmsi,
                    "SubTrackID": track_id,
                    "TimeStamp_RF": RF_timestamp,
                    "TimeStamp_AIS": matched_AIS_row["BaseDateTime"],
                    "AIS_LAT": matched_AIS_row["LAT"],
                    "AIS_LON": matched_AIS_row["LON"],
                    "RF_LAT": RF_lat,
                    "RF_LON": RF_lon
                })

    return pd.DataFrame(RF_records)

if __name__ == "__main__":
    print("Generating synthetic data has started...")
        
    SOURCE_PATH = "../../../../data/processed/segmented_tracks.parquet"
    DESTINATION_PATH = "data/processed/synthetic_RF_data.parquet"
    
    df = pd.read_parquet(SOURCE_PATH)
    RF_df = create_synthetic_RF_data(df)
    RF_df.to_parquet(DESTINATION_PATH)
    
    print("Data is saved...")