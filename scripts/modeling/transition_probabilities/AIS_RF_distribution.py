import numpy as np

def AIS_RF_probability(time_diff_sec, diff_dist_km, max_time=551, max_dist=7, alpha=2, beta=2):
    """
    Computes the transition probability between two observations (e.g., AIS → AIS, AIS → RF)
    based on their temporal and spatial differences
    
    The probability decreases exponentially as:
        - the time difference increases
        - the spatial distance increases
    
    Args:
        time_diff_sec (float): Time difference between two points in seconds
        diff_dist_km (float): Distance between two points in kilometers
        max_time (int): Maximum considered time difference, defaults to 551 sec
        max_dist (int): Maximum considered spatial distance, defaults to 7 km
        alpha (int): Weight controlling how strongly time difference affects probability
        beta (int): Weight controlling how strongly spatial distance affects probability
        
    Returns:
        float: Transition probability ∈ [0, 1], representing the likelihood of a transiton 
    """
    # Negative time differences are invalid (future cannot connect back to past)
    if time_diff_sec < 0:
        return 0.0
    
    # Normalize time and distance so they are relative (0 to ~1 scale)
    x = time_diff_sec / max_time
    y = diff_dist_km / max_dist
    
    # Exponential decay: probability drops as time or distance grows
    prob = np.exp(-alpha * x - beta * y)
    
    # Ensure probability stays in [0, 1]
    return max(0.0, min(1.0, prob))