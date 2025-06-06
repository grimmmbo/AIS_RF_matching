import numpy as np

def AIS_RF_probability(time_diff_sec, diff_dist_km, max_time=39840, max_dist=160, alpha=9, beta=9):
    """
    Computes the transition probability of two points (e.g., AIS -> AIS, AIS -> RF, etc.) based on their time and spatial difference 

    Args:
        time_diff_sec (float): Time difference between two points in seconds
        diff_dist_km (float): Distance between two points in kilometers
        max_time (int): Maximum time difference, defaults to 36000 seconds (10 hours)
        max_dist (float): Maximum distance, defaults to 600 kilometers
        
    Returns:
        float: Probability between 0 and 1 representing the likelihood of a transition
    """
    if time_diff_sec < 0:
        return 0.0
    
    x = time_diff_sec / max_time
    y = diff_dist_km / max_dist
    prob = np.exp(-alpha * x - beta * y)
    
    return max(0.0, min(1.0, prob))