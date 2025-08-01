import math 
import random 
from datetime import datetime, timedelta
import pandas as pd
from haversine import haversine

from scripts.PHMM.distributions.AIS_RF_distribution import * 
from scripts.PHMM.distributions.match_distribution import * 

# genereer willekeurige tijdstippen met ~3 uur intervallen voor RF-signalen (Unseenlabs revisit time ≈ 3 uur)
# https://earth.esa.int/eogateway/missions/unseenlabs/description

def generate_random_start_time():
    """
    Generates a random start time between 00:00 and 02:59

    Returns:
        datetime: A random time representing the start time
    """
    start_hour = random.randint(0, 2)
    start_minute = random.randint(0, 59)
    start_time = datetime.strptime(f"{start_hour:02d}:{start_minute:02d}", "%H:%M")
    return start_time 

def generate_synthetic_RF_data():
    """
    Simulates RF revisit timestamps by starting from a randomly generated start time 
    and adding time intervals of approximately 3 to 4 hours

    Returns:
        list: Timestamps representing sythethic RF revisit times
    """    
    start_time = generate_random_start_time()
    times = [start_time]
    
    while True:
        minutes = random.randint(180, 240)
        seconds = random.randint(0,59)
        next_time = times[-1] + timedelta(minutes = minutes, seconds = seconds)
        
        if next_time.day != start_time.day:
            break
        
        times.append(next_time)
    return times

def generate_error_distance_and_bearing(
    timestamps, 
    mean_distance = 2000, 
    std_dev_distance = 100, 
    bearing_options = [0, 45, 90, 135, 180, 225, 270, 315]
    ):
    """
    Generates an error distance and bearing angle for each RF timestamp

    Args:
        timestamps (list): Simulated RF timestamps
        mean_distance (int): The average error distance in meters between the AIS point and the simulated RF signal, defaults to 2000
        std_dev_distance (int): Indicates how mcuh the generated distances can deviate from the average value. 
                                Given a mean of 2000 and a standard deviation of 100, most generated distances will fall roughly between 1900 and 2100 meters
        bearing_options (list): A list of possible direction (in degrees) that are added as deviations to the original course of the vessel, 
                                defaults to [0, 45, 90, 135, 180, 225, 270, 315]

    Returns:
        dict: timestamp: [error_distance, error_bearing]
    """
    return {
        timestamp: [
            round(np.random.normal(mean_distance, std_dev_distance), 6),
            random.choice(bearing_options)
        ] 
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
        error_bearing (float): Max error to add or subtract from the 'original' heading

    Returns:
        tuple: (latitude, longitude) of the simulated RF signal location
    """
    R = 6371000  

    noise = np.random.uniform(-error_bearing, error_bearing)
    bearing = math.radians((heading + noise) % 360)

    latitude, longitude = math.radians(latitude), math.radians(longitude)
    
    new_latitude = math.asin(
        math.sin(latitude) * math.cos(error_distance / R) + 
        math.cos(latitude) * math.sin(error_distance / R) * math.cos(bearing)
    )
    new_longitude = longitude + math.atan2(
        math.sin(bearing) * math.sin(error_distance / R) * math.cos(latitude),
        math.cos(error_distance / R) - math.sin(latitude) * math.sin(new_latitude)
    )

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
    start_time = AIS_seq["BaseDateTime"].iloc[0] - timedelta(minutes = time_margin)
    end_time = AIS_seq["BaseDateTime"].iloc[-1] + timedelta(minutes = time_margin)
    unique_dates = AIS_seq['BaseDateTime'].dt.normalize().unique()
    
    return sorted([
        datetime.combine(pd.to_datetime(date).date(), RF_time.time())
        for date in unique_dates for RF_time in RF_timestamps
        if start_time <= datetime.combine(pd.to_datetime(date).date(), RF_time.time()) <= end_time
    ])
    
def generate_training_data(df, time_margin = 5):
    """
    Generates labeled training data by combining AIS trajectories with simulated RF signal observations
    
    For each AIS trajectory, and for every simulated RF timestamp that falls within its extended time window, the nearest AIS point is identified.
    A classification decision is then made based on the time and distance between the RF and AIS point.
        - If the probability of a match (based on time and distance) is higher than a standalone RF point, the RF signal is marked as a match ("M")
        - Otherwise, it is treated as an unmatched RF point ("RF")

    Args:
        df (pd.DataFrame): AIS data
        time_margin (int):  Number of minutes to extend the AIS route (before and after), defaults to 5

    Returns:
        pd.DataFrame: Labeled sequence data with the following columns:
            - UniqueRouteID
            - SubRouteID
            - MMSI
            - Timestamp
            - AIS_Timestamp 
            - RF_Timestamp
            - AIS (coordinates)
            - RF (coordinates)
            - State (one of: "M", "AIS", "RF", "begin", "end")
    """
    random.seed(42)
    np.random.seed(42)
    
    RF_revisit_timestamps = generate_synthetic_RF_data()
    train_data = []
    distance_differences = []
    
    for (id, mmsi, track_id), AIS_seq in df.groupby(["ID", "MMSI", "track_id"]):
        AIS_seq["Matched"] = False
        
        RF_times = get_RF_times_for_AIS_seq(RF_revisit_timestamps, AIS_seq, time_margin)
        RF_erros = generate_error_distance_and_bearing(RF_times)
        
        RF_timestamps_seen = []
        
        for RF_datetime in RF_times:       
            # Find closest AIS information 
            closest_time_idx = (AIS_seq["BaseDateTime"] - RF_datetime).abs().idxmin()
            closest_AIS_time = AIS_seq.loc[closest_time_idx, "BaseDateTime"]
            closest_AIS_lat = AIS_seq.loc[closest_time_idx, "LAT"]
            closest_AIS_lon = AIS_seq.loc[closest_time_idx, "LON"]
            closest_AIS_heading = AIS_seq.loc[closest_time_idx, "Heading"]
            
            # Calculate the coordinates for RF signal
            distance, bearing = RF_erros[RF_datetime]
            synthetic_RF_lat, synthetic_RF_lon = compute_coordinates(closest_AIS_lat, closest_AIS_lon, closest_AIS_heading, distance)
            
            # Calculate the difference in distance and time between AIS and RF point
            distance_diff = haversine((closest_AIS_lat, closest_AIS_lon), (synthetic_RF_lat, synthetic_RF_lon))
            distance_differences.append(distance_diff)
            time_diff = abs((RF_datetime - closest_AIS_time).total_seconds())
            
            # Compute the likelihood that this RF signal is a match with an AIS point, or just a standalone RF signal
            prob_RF = AIS_RF_probability(time_diff, distance_diff)
            prob_match = match_probability(time_diff, distance_diff)
            
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
        
        # Determine earliest and latest time from AIS and RF
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