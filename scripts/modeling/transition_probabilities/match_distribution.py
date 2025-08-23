import numpy as np 

def match_probability(diff_time_sec, diff_dist_km, max_time=120, max_dist=2.0):
    """
    Computes the probability that an AIS observation and an RF observation are a valid match,
    based on their time and distance differences
    
    Unlike AIS_RF_probability (which is a simple exponential decay),
    this function builds a shaped probability surface using multiple peaks,
    penalties, and bumps to capture more realistic matching behavior
    
    Args:
        time_diff_sec (float): Time difference between AIS and RF points (in seconds)
        diff_dist_km (float): Distance between AIS and RF points (in kilometers)
        max_time (int): Maximum allowed time difference. Defaults to 120 seconds. Beyond this → probability = 0.
        max_dist (float): Maximum allowed spatial difference. Defaults to 2 km. Beyond this → probability = 0. 

    Returns:
        float: Probability ∈ [0, 1] that AIS and RF belong to the same match
    """
    # Hard cutoff: if differences are too large, no match possible
    if diff_time_sec > max_time or diff_dist_km > max_dist:
        return 0.0
    
    # Tuning parameters for shaping the surface
    amp_peak = (1.0, 0.0002)
    amp_penalty = (0.6, 0.6)
    sigma = (6, 10, 15)

    # Tuning parameters for shaping the surface
    x_norm = diff_time_sec / max_time
    y_norm = diff_dist_km / max_dist

    # Boosts probability
    peak1 = amp_peak[0] * np.exp(-sigma[0] * ((x_norm + 0.2)**2 + (y_norm + 0.2)**2))
    peak2 = amp_peak[1] * np.exp(-sigma[1] * ((x_norm - 1)**2 + (y_norm - 1)**2))
    center_boost = 0.06 * np.exp(-100 * (x_norm**2 + y_norm**2))
    
    # Reduces probability
    penalty1 = amp_penalty[0] * np.exp(-sigma[2] * ((x_norm - 1)**2 + y_norm**2))
    penalty2 = amp_penalty[1] * np.exp(-sigma[2] * (x_norm**2 + (y_norm - 1)**2))

    bumps = (
        0.08 * np.exp(-20 * ((x_norm - 0.4)**2 + (y_norm - 0.4)**2)) +  
        0.12 * np.exp(-25 * ((x_norm - 0.25)**2 + (y_norm - 0.25)**2)) +
        0.10 * np.exp(-20 * ((x_norm - 0.1)**2 + (y_norm - 0.1)**2)))

    prob = 0.57 + peak1 + peak2 + center_boost + bumps - penalty1 - penalty2
    
    return float(np.clip(prob, 0, 1))

