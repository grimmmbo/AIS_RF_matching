import math
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from haversine import haversine

from scripts.modeling.transition_probabilities.AIS_RF_distribution import *
from scripts.modeling.transition_probabilities.match_distribution import *

# RF position-error models supported by generate_training_data()
ERROR_MODELS = ("uniform", "gaussian")

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
    Generates an error distance for each RF timestamp, for the "uniform" error model

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

def generate_gaussian_error_offset(sigma=800):
    """
    Samples an isotropic 2D Gaussian position error: N(0, sigma) on each axis

    Unlike the "uniform" model's error_distance (which carries a fixed
    mean_distance offset plus a small amount of noise), this offset has no
    systematic bias: east/north components are each drawn from Normal(0, sigma),
    so the true position itself is the most likely outcome and error grows
    only with sigma. The resulting distance therefore follows a Rayleigh
    distribution with expected value sigma * sqrt(pi/2) (~1.25 * sigma), and
    the bearing is uniform over the full circle, independent of vessel heading

    Args:
        sigma (float): Standard deviation (meters) of the position error's
            east/north components. Defaults to 800

    Returns:
        tuple: (error_distance in meters, bearing in degrees, compass convention)
    """
    east_offset = np.random.normal(0, sigma)
    north_offset = np.random.normal(0, sigma)
    distance = math.hypot(east_offset, north_offset)
    bearing = math.degrees(math.atan2(east_offset, north_offset)) % 360
    return distance, bearing

def compute_coordinates(latitude, longitude, bearing, error_distance):
    """
    Computes a new geographical coordinate from a base position using a bearing and distance

    Args:
        latitude (float): Latitude of the reference point
        longitude (float): Longitude of the reference point
        bearing (float): Direction (degrees, compass convention) to offset towards
        error_distance (float): Distance in meters to offset from the reference point

        Based on strategy: https://stackoverflow.com/questions/18580414/implementation-of-great-circle-destination-formula

    Returns:
        tuple: (latitude, longitude) of the simulated RF signal location
    """
    R = 6371000  # Earth radius in meters
    bearing = math.radians(bearing % 360)

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
    
def generate_training_data(
    df, time_margin=5, error_model="uniform", error_bearing=60, sigma=800,
):
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
    This study only compares a single RF signal to an entire AIS
    trajectory, so such detailed labeling isn't strictly required --
    it's kept because it makes the dataset reusable for more advanced
    sequence alignment/decoding (e.g. Viterbi) later

    Args:
        df (pd.DataFrame): AIS data
        time_margin (int):  Number of minutes to extend the AIS route (before and after). Defaults to 5 min
        error_model (str): RF position-error model, one of "uniform" or "gaussian"
            (see ERROR_MODELS). Defaults to "uniform"
        error_bearing (float): Maximum angular deviation (degrees) applied uniformly to
            the vessel heading. Only used when error_model="uniform". Defaults to 60 (i.e. ±60°)
        sigma (float): Standard deviation (meters) of the isotropic Normal(0, sigma)
            position error. Only used when error_model="gaussian". Defaults to 800

    Returns:
        pd.DataFrame: Labeled sequence data
    """
    if error_model not in ERROR_MODELS:
        raise ValueError(f"Unknown error_model {error_model!r}, expected one of {ERROR_MODELS}")

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
        # "uniform" distances are pregenerated in bulk (unchanged from the
        # original algorithm) to keep its random-draw sequence, and therefore
        # its output for a fixed seed, identical; "gaussian" needs no heading
        # and is instead sampled per RF point in generate_gaussian_error_offset()
        RF_erros = generate_error_distance_and_bearing(RF_times) if error_model == "uniform" else None

        RF_timestamps_seen = []
        
        # Step 3: For each RF time, find closest AIS point
        for RF_datetime in RF_times:       
            # Find closest AIS information 
            closest_time_idx = (AIS_seq["BaseDateTime"] - RF_datetime).abs().idxmin()
            closest_AIS_time = AIS_seq.loc[closest_time_idx, "BaseDateTime"]
            closest_AIS_lat = AIS_seq.loc[closest_time_idx, "LAT"]
            closest_AIS_lon = AIS_seq.loc[closest_time_idx, "LON"]
            closest_AIS_heading = AIS_seq.loc[closest_time_idx, "Heading"]
            
            # Generate synthetic RF coordinates with noise, using the selected error model
            if error_model == "uniform":
                distance = RF_erros[RF_datetime][0]
                bearing = (closest_AIS_heading + np.random.uniform(-error_bearing, error_bearing)) % 360
            else:  # "gaussian"
                distance, bearing = generate_gaussian_error_offset(sigma=sigma)

            synthetic_RF_lat, synthetic_RF_lon = compute_coordinates(
                closest_AIS_lat, closest_AIS_lon, bearing, distance
            )
            
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