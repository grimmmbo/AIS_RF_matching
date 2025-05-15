from haversine import haversine, Unit

def compute_abs_time_and_distance(curr_point, prev_point):
    """
    Calculates the absolute time difference and distance between two points

    Args:
        curr_point (tuple[datetime, tuple[float, float]]): A tuple containing a timestamp and a coordinate (latitude, longitude) from the AIS dataset
        prev_point (tuple[datetime, tuple[float, float]]): A tuple containing a timestamp and a coordinate (latitude, longitude) from the RF dataset

    Returns:
        tuple[float, float]:
            Absolute time difference in seconds (float)
            Geographic distance in kilometers (float)
    """
    time_1 = curr_point[0]
    time_2 = prev_point[0]
    time_delta = time_1 - time_2
    time_in_sec = abs(time_delta.total_seconds())

    coordinates_1 = curr_point[1]
    coordinates_2 = prev_point[1]
    distance_in_km = haversine(coordinates_1, coordinates_2, unit=Unit.KILOMETERS)

    return time_in_sec, distance_in_km

def compute_time_and_distance(curr_point, prev_point):
    """
    Calculates time difference and distance between two points

    Args:
        curr_point (tuple[datetime, tuple[float, float]]): A tuple containing a timestamp and a coordinate (latitude, longitude) from the AIS dataset
        prev_point (tuple[datetime, tuple[float, float]]): A tuple containing a timestamp and a coordinate (latitude, longitude) from the RF dataset

    Returns:
        tuple[float, float]: 
            Time difference in seconds (float)
            Geographic distance in kilometers (float)
    """
    time_1 = curr_point[0]
    time_2 = prev_point[0]
    time_delta = time_1 - time_2
    time_in_sec = time_delta.total_seconds()

    coordinates_1 = curr_point[1]
    coordinates_2 = prev_point[1]
    distance_in_km = haversine(coordinates_1, coordinates_2, unit=Unit.KILOMETERS)

    return time_in_sec, distance_in_km

def compute_average_time_and_distance(curr_point, prev_point_AIS, prev_point_RF):
    """
    Computes the average of time differences and distances from current point to two previous points
    
    Args:
        current_point (tuple[datetime, tuple[float, float]]): A tuple containing a timestamp and a coordinate (latitude, longitude) for the current point
        previous_point_AIS (tuple[datetime, tuple[float, float]]): A tuple containing a timestamp and a coordinate (latitude, longitude) from the AIS dataset
        previous_point_RF (tuple[datetime, tuple[float, float]]): A tuple containing a timestamp and a coordinate (latitude, longitude) from the RF dataset

    Returns:
        tuple[float, float]: 
            Average time difference in seconds (float)
            Average geographic distance in kilometers (float)
    """
    time_1, distance_1 = compute_time_and_distance(curr_point, prev_point_AIS)
    time_2, distance_2 = compute_time_and_distance(curr_point, prev_point_RF)
    return ((time_1 + time_2) / 2, (distance_1 + distance_2) / 2)