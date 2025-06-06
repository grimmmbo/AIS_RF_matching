import numpy as np 

def match_probability(diff_time_sec, diff_dist_km, max_time=120, max_dist=2.0):
    """
    Computes a matching probability between an AIS and RF observations based on their time and spatial difference 

    Args:
        time_diff_sec (float): Time difference between AIS and RF points in seconds
        diff_dist_km (float): Distance between AIS and RF points in kilometers
        max_time (int): Maximum time difference, defaults to 120 seconds 
        max_dist (float): Maximum distance, defaults to 2.0 kilometers 

    Returns:
        float: A probability between 0 and 1 representing the likelihood of a match 
    """
    if diff_time_sec > max_time or diff_dist_km > max_dist:
        return 0.0
    
    amp_peak = (1.0, 0.0002)
    amp_penalty = (0.6, 0.6)
    sigma = (6, 10, 15)

    x_norm = diff_time_sec / max_time
    y_norm = diff_dist_km / max_dist

    peak1 = amp_peak[0] * np.exp(-sigma[0] * ((x_norm + 0.2)**2 + (y_norm + 0.2)**2))
    peak2 = amp_peak[1] * np.exp(-sigma[1] * ((x_norm - 1)**2 + (y_norm - 1)**2))
    center_boost = 0.06 * np.exp(-100 * (x_norm**2 + y_norm**2))
    penalty1 = amp_penalty[0] * np.exp(-sigma[2] * ((x_norm - 1)**2 + y_norm**2))
    penalty2 = amp_penalty[1] * np.exp(-sigma[2] * (x_norm**2 + (y_norm - 1)**2))

    bumps = (
        0.08 * np.exp(-20 * ((x_norm - 0.4)**2 + (y_norm - 0.4)**2)) +  
        0.12 * np.exp(-25 * ((x_norm - 0.25)**2 + (y_norm - 0.25)**2)) +
        0.10 * np.exp(-20 * ((x_norm - 0.1)**2 + (y_norm - 0.1)**2)))

    prob = 0.55 + peak1 + peak2 + center_boost + bumps - penalty1 - penalty2
    
    return float(np.clip(prob, 0, 1))