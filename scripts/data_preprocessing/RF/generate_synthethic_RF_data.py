import math 
import random 
from datetime import datetime, timedelta
import pandas as pd
from haversine import haversine

from scripts.modeling.transition_probabilities.AIS_RF_distribution import *
from scripts.modeling.transition_probabilities.match_distribution import *

def generate_random_start_time():
    """
    Generates a random start time between 00:00 and 02:59

    Returns:
        datetime: A random time representing the start time
    """
    # Pick a random hour between 0–2 and a random minute between 0–59
    start_hour = random.randint(0, 2)
    start_minute = random.randint(0, 59)
    start_time = datetime.strptime(f"{start_hour:02d}:{start_minute:02d}", "%H:%M")
    return start_time 

def generate_synthetic_RF_data():
    """
    Simulates RF revisit timestamps by starting from a randomly generated start time 
    and adding time intervals of approximately 3 to 4 hours
    
    Based on strategy: https://earth.esa.int/eogateway/missions/unseenlabs/description 

    Returns:
        list: Timestamps representing sythethic RF revisit times
    """    
    start_time = generate_random_start_time()
    times = [start_time]
    
    while True:
        # Add ~3–4 hours (180–240 minutes) between revisits
        minutes = random.randint(180, 240)
        seconds = random.randint(0,59)
        next_time = times[-1] + timedelta(minutes = minutes, seconds = seconds)
        
        # Stop once the simulated revisit crosses midnight
        if next_time.day != start_time.day:
            break
        
        times.append(next_time)
    return times

def generate_error_distance_and_bearing(timestamps, mean_distance = 2000, std_dev_distance = 100):
    """
    Generates an error distance for each RF timestamp

    Args:
        timestamps (list): Simulated RF timestamps
        mean_distance (int): Mean error distance in meters. Defaults to 2000
        std_dev_distance (int): Standard deviation of the error distance. Defaults to 100

    Returns:
        dict: timestamp: Mapping timestamp -> error_distance (meters)
    """
    return {
        # Sample distance error around the mean
        timestamp: [round(np.random.normal(mean_distance, std_dev_distance), 6)] 
        for timestamp in timestamps
    }

def compute_coordinates(latitude, longitude, heading, error_distance, error_bearing=60):
    """
    Computes a new geographical coordinate from a base position using a error distance and bearing

    Args:
        latitude (float): Latitude of the reference point
        longitude (float): Longitude of the reference point
        heading (float): Direction of the ship at the reference point
        error_distance (float): Distance in meters to offset from the reference point
        error_bearing (float): Maximum angular deviation (degrees) applied uniformly to heading. Defaults to 60 (i.e., ±60°)
        
        Based on strategy: https://stackoverflow.com/questions/18580414/implementation-of-great-circle-destination-formula 

    Returns:
        tuple: (latitude, longitude) of the simulated RF signal location
    """
    R = 6371000  # Earth radius in meters 

    # Add uniform angular noise to vessel heading
    noise = np.random.uniform(-error_bearing, error_bearing)
    bearing = math.radians((heading + noise) % 360)

    # Convert latitude/longitude to radians 
    latitude, longitude = math.radians(latitude), math.radians(longitude)
    
    # Calculate new coordinates
    new_latitude = math.asin(
        math.sin(latitude) * math.cos(error_distance / R) + 
        math.cos(latitude) * math.sin(error_distance / R) * math.cos(bearing)
    )
    new_longitude = longitude + math.atan2(
        math.sin(bearing) * math.sin(error_distance / R) * math.cos(latitude),
        math.cos(error_distance / R) - math.sin(latitude) * math.sin(new_latitude)
    )

    # Convert back to degrees
    new_latitude = math.degrees(new_latitude)
    new_longitude = math.degrees(new_longitude)
    
    return new_latitude, new_longitude

def get_RF_times_for_AIS_seq(RF_timestamps, AIS_seq, time_margin):
    """
    Selects RF timestamps that fall within the time window of an AIS route
    
    Args:
        timestamps (list): Simulated RF timestamps
        AIS_sequence (pd.DataFrame): AIS data
        time_margin (int): Times in minutes to expand the window before and after the AIS sequence

    Returns:
        list: RF timestamps that fall within the expanded AIS window
    """
    # Define extended AIS window
    start_time = AIS_seq["BaseDateTime"].iloc[0] - timedelta(minutes = time_margin)
    end_time = AIS_seq["BaseDateTime"].iloc[-1] + timedelta(minutes = time_margin)
    
    # Align RF times with AIS dates
    unique_dates = AIS_seq['BaseDateTime'].dt.normalize().unique()
    
    return sorted([
        datetime.combine(pd.to_datetime(date).date(), RF_time.time())
        for date in unique_dates for RF_time in RF_timestamps
        if start_time <= datetime.combine(pd.to_datetime(date).date(), RF_time.time()) <= end_time
    ])
    
def generate_training_data(df, time_margin = 5):
    """
    Generates labeled training data by combining AIS trajectories with simulated RF signal observations
    
    Steps:
      1. Generate synthetic RF revisit times
      2. For each AIS trajectory:
         - Select RF times within the extended AIS window
         - Add spatial errors to simulate RF observations
         - Find the nearest AIS point in time
         - Compute time and distance differences
         - Use AIS_RF_probability() and match_probability() to decide whether the RF is a match ("M") or standalone ("RF")
         - Add unmatched AIS points as "AIS"
         - Add "begin" and "end" markers to each sequence
    
    Note:
    For the scope of this study, such detailed labeling is not strictly required,
    since we only compare a single RF signal to an entire AIS trajectory 
    However, the richer labeling scheme makes the dataset reusable for more advanced sequence alignment 
    or decoding (e.g., Viterbi) tasks in future work

    Args:
        df (pd.DataFrame): AIS data
        time_margin (int):  Number of minutes to extend the AIS route (before and after). Defaults to 5 min

    Returns:
        pd.DataFrame: Labeled sequence data
    """
    random.seed(42)
    np.random.seed(42)
    
    # Step 1: Generate synthetic RF revisit times
    RF_revisit_timestamps = generate_synthetic_RF_data()
    train_data = []
    distance_differences = []
    
    for (id, mmsi, track_id), AIS_seq in df.groupby(["ID", "MMSI", "track_id"]):
        AIS_seq["Matched"] = False
        
        # Step 2: Loop over AIS trajectories (grouped by vessel ID and track_id)
        RF_times = get_RF_times_for_AIS_seq(RF_revisit_timestamps, AIS_seq, time_margin)
        RF_erros = generate_error_distance_and_bearing(RF_times)
        
        RF_timestamps_seen = []
        
        # Step 3: For each RF time, find closest AIS point
        for RF_datetime in RF_times:       
            # Find closest AIS information 
            closest_time_idx = (AIS_seq["BaseDateTime"] - RF_datetime).abs().idxmin()
            closest_AIS_time = AIS_seq.loc[closest_time_idx, "BaseDateTime"]
            closest_AIS_lat = AIS_seq.loc[closest_time_idx, "LAT"]
            closest_AIS_lon = AIS_seq.loc[closest_time_idx, "LON"]
            closest_AIS_heading = AIS_seq.loc[closest_time_idx, "Heading"]
            
            # Generate synthetic RF coordinates with noise
            distance = RF_erros[RF_datetime][0]
            synthetic_RF_lat, synthetic_RF_lon = compute_coordinates(closest_AIS_lat, closest_AIS_lon, closest_AIS_heading, distance)
            
            # Compute time and distance differences
            distance_diff = haversine((closest_AIS_lat, closest_AIS_lon), (synthetic_RF_lat, synthetic_RF_lon))
            distance_differences.append(distance_diff)
            time_diff = abs((RF_datetime - closest_AIS_time).total_seconds())
            
            # Use AIS_RF_probability vs match_probability to decide label
            prob_RF = AIS_RF_probability(time_diff, distance_diff)
            prob_match = match_probability(time_diff, distance_diff)
            
            # If match is more likely → label "M"            
            if prob_match > prob_RF:
                state = "M"
                AIS_seq.at[closest_time_idx, "Matched"] = True 
            else:
                state = "RF"

            train_data.append({
                "ID": id,
                "track_id": track_id,
                "MMSI": mmsi,
                "Timestamp": closest_AIS_time if state == "M" else RF_datetime,
                "AIS_Timestamp": closest_AIS_time if state == "M" else pd.NaT,
                "RF_Timestamp": RF_datetime,
                "AIS": [closest_AIS_lat, closest_AIS_lon] if state == "M" else None,
                "RF": [synthetic_RF_lat, synthetic_RF_lon],
                "State": state
            })
            
            RF_timestamps_seen.append(RF_datetime)
        
        # Step 4: Add unmatched AIS points (still important for sequence completeness)            
        for i, row in AIS_seq[~AIS_seq["Matched"]].iterrows():
            train_data.append({
                "ID": id,
                "track_id": track_id,
                "MMSI": mmsi,
                "Timestamp": row["BaseDateTime"],
                "AIS_Timestamp": row["BaseDateTime"],
                "RF_Timestamp": pd.NaT,
                "AIS": [row["LAT"], row["LON"]],
                "RF": None,
                "State": "AIS"
            })
        
        # Step 5: Add artificial "begin" and "end" markers
        first_AIS_time = AIS_seq["BaseDateTime"].min()
        last_AIS_time = AIS_seq["BaseDateTime"].max()
        
        first_RF_time = min(RF_timestamps_seen) if RF_timestamps_seen else first_AIS_time
        last_RF_time = max(RF_timestamps_seen) if RF_timestamps_seen else last_AIS_time
        
        time_begin = min(first_AIS_time, first_RF_time) - timedelta(seconds=1)
        time_end = max(last_AIS_time, last_RF_time) + timedelta(seconds=1)
        
        for time, state in [(time_begin, "begin"), (time_end, "end")]:
            train_data.append({
                "ID": id,
                "track_id": track_id,
                "MMSI": mmsi,
                "Timestamp": time,
                "AIS_Timestamp": pd.NaT,
                "RF_Timestamp": pd.NaT,
                "AIS": None,
                "RF": None,
                "State": state
            })  
    
    df_training = pd.DataFrame(train_data).sort_values(["ID", "Timestamp"])
    
    return df_training, distance_differences