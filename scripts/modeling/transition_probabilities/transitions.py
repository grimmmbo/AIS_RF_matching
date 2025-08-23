from scripts.utils.geo_utils import *
from scripts.modeling.transition_probabilities.AIS_RF_distribution import AIS_RF_probability
from scripts.modeling.transition_probabilities.match_distribution import match_probability

class TransitionCalculator:
    """
    Computes transition probabilities between AIS, RF, and M states
    
    Each method calculates the likelihood of moving from one state to another, based on:
        - Previous position (prev_i, prev_j) in the gamma (γ) matrix,
        - AIS and RF observation sequences,
        - Time and distance between the relevant observations.
    
    """
    @staticmethod
    def AIS_to_AIS(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Transition from AIS → AIS
        Uses consecutive AIS observations
        """
        # Subtract 1 to get the correct position in AIS_seq and RF_seq
        seq_i, seq_j = prev_i - 1, prev_j - 1
        
        if seq_i < 0 or seq_i + 1 >= len(AIS_seq):
            return 0.0
        
        # Calculate the time and distance difference between current AIS and previous AIS point 
        curr_point, prev_point = AIS_seq[seq_i + 1], AIS_seq[seq_i]
        time, dist = compute_time_and_distance(curr_point, prev_point)
        
        # Get transition probability based on time and distance difference
        return AIS_RF_probability(time, dist)

    @staticmethod
    def AIS_to_RF(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Transition from AIS → RF
        Uses last AIS and next RF observation to compute transition probability
        """
        # Subtract 1 to get the correct position in AIS_seq and RF_seq
        seq_i, seq_j = prev_i - 1, prev_j - 1  
        if seq_i < 0 or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        # Calculate the absolute time and distance difference between current RF and previous AIS point 
        curr_point, prev_point = RF_seq[seq_j + 1], AIS_seq[seq_i]
        time, dist = compute_time_and_distance(curr_point, prev_point)

        # Get transition probability based on time and distance difference
        return AIS_RF_probability(time, dist)

    @staticmethod
    def AIS_to_M(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Transition from AIS →  M
        Pairs the next AIS and RF observations to compute match likelihood
        """
        # Subtract 1 to get the correct position in AIS_seq and RF_seq
        seq_i, seq_j = prev_i - 1, prev_j - 1
        
        if seq_i + 1 >= len(AIS_seq) or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        # Calculate the absolute time and distance difference between current AIS and current RF point 
        curr_AIS, curr_RF = AIS_seq[seq_i + 1], RF_seq[seq_j + 1]
        time, dist = compute_abs_time_and_distance(curr_AIS, curr_RF)

        # Get transition probability based on time and distance difference
        return match_probability(time, dist)

    @staticmethod
    def RF_to_AIS(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Transition from RF → AIS
        Uses last RF and next AIS observation to compute transition probability
        """
        # Subtract 1 to get the correct position in AIS_seq and RF_seq
        seq_i, seq_j = prev_i - 1, prev_j - 1
        if seq_j < 0 or seq_i + 1 >= len(AIS_seq):
            return 0.0
        
        # Calculate the time and distance difference between current AIS and previous RF point
        curr_point, prev_point = AIS_seq[seq_i + 1], RF_seq[seq_j]
        time, dist = compute_time_and_distance(curr_point, prev_point)
        
        # Get transition probability based on time and distance difference
        return AIS_RF_probability(time, dist)

    @staticmethod
    def RF_to_RF(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Transition from RF → RF
        Uses two consecutive RF observations
        """
        # Subtract 1 to get the correct position in AIS_seq and RF_seq
        seq_i, seq_j = prev_i - 1, prev_j - 1
        if seq_j < 0 or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        # Calculate the time and distance difference between current RF and previous RF point
        curr_point, prev_point = RF_seq[seq_j + 1], RF_seq[seq_j]
        time, dist = compute_time_and_distance(curr_point, prev_point)
        
        # Get transition probability based on time and distance difference
        return AIS_RF_probability(time, dist)

    @staticmethod
    def RF_to_M(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Transition from AIS → M
        Pairs the next AIS and RF observations
        """
        # Subtract 1 to get the correct position in AIS_seq and RF_seq
        seq_i, seq_j = prev_i - 1, prev_j - 1 
        if seq_i + 1 >= len(AIS_seq) or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        # Calculate the absolute time and distance difference between current AIS and current RF point 
        curr_AIS, curr_RF = AIS_seq[seq_i + 1], RF_seq[seq_j + 1]
        time, dist = compute_abs_time_and_distance(curr_AIS, curr_RF)
        
        # Get transition probability based on time and distance difference
        return match_probability(time, dist)

    @staticmethod
    def M_to_AIS(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Transition from M → AIS
        Chooses whether to compare new AIS point against the last AIS or last RF, depending on which came later in time
        """
        # Subtract 1 to get the correct position in AIS_seq and RF_seq
        seq_i, seq_j = prev_i - 1, prev_j - 1
        if seq_i < 0 or seq_j < 0 or seq_i + 1 >= len(AIS_seq):
            return 0.0
        
        # Pick the most recent reference point in time to calculate time and distance difference
        curr_point = AIS_seq[seq_i + 1]
        prev_AIS, prev_RF = AIS_seq[seq_i], RF_seq[seq_j]
        if prev_AIS[0] > prev_RF[0]:
            time, dist = compute_time_and_distance(curr_point, prev_AIS)
        else:
            time, dist = compute_time_and_distance(curr_point, prev_RF)
                
        if (time, dist) == (0,0):
            return 0.0
        
        # Get transition probability based on time and distance difference    
        return AIS_RF_probability(time, dist)

    @staticmethod
    def M_to_RF(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Transition from M → RF
        Chooses whether to compare new AIS point against the last AIS or last RF, depending on which came later in time
        """
        # Subtract 1 to get the correct position in AIS_seq and RF_seq
        seq_i, seq_j = prev_i - 1, prev_j - 1
        if seq_i < 0 or seq_j < 0 or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        # Pick the most recent reference point in time to calculate time and distance difference
        curr_point = RF_seq[seq_j + 1]
        prev_AIS, prev_RF = AIS_seq[seq_i], RF_seq[seq_j]
        if prev_AIS[0] > prev_RF[0]:
            time, dist = compute_time_and_distance(curr_point, prev_AIS)
        else:
            time, dist = compute_time_and_distance(curr_point, prev_RF)
                    
        if (time, dist) == (0,0):
            return 0.0
      
        # Get transition probability based on time and distance difference           
        return AIS_RF_probability(time, dist)

    @staticmethod
    def M_to_M(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Transition from M → M
        Uses the next AIS and RF pair directly
        """
        # Subtract 1 to get the correct position in AIS_seq and RF_seq
        seq_i, seq_j = prev_i - 1, prev_j - 1
        if seq_i + 1 >= len(AIS_seq) or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        # Calculate the absolute time and distance difference between current AIS and current RF point 
        curr_AIS, curr_RF = AIS_seq[seq_i + 1], RF_seq[seq_j + 1]
        time, dist = compute_abs_time_and_distance(curr_AIS, curr_RF)
        
        # Get transition probability based on time and distance difference    
        return match_probability(time, dist)

def get_transition_dict(prev_state, prev_i, prev_j, AIS_seq, RF_seq):
    """
    Given a previous state, compute transition probabilities into all valid next states ('AIS', 'RF', 'M')
    
    Args:
        prev_state (str): Name of the previous state ('AIS', 'RF', 'M')
        prev_i (int): Index in the γ matrix corresponding to the AIS sequence
        prev_j (int): Index in the γ matrix corresponding to the RF sequence
        AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
        RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

    Returns:
        dict: {next_state_name: probability}
    """
    transitions = {
        "AIS": {
            "AIS": TransitionCalculator.AIS_to_AIS,
            "RF": TransitionCalculator.AIS_to_RF,
            "M": TransitionCalculator.AIS_to_M
        },
        "RF": {
            "AIS": TransitionCalculator.RF_to_AIS,
            "RF": TransitionCalculator.RF_to_RF,
            "M": TransitionCalculator.RF_to_M
        },
        "M": {
            "AIS": TransitionCalculator.M_to_AIS,
            "RF": TransitionCalculator.M_to_RF,
            "M": TransitionCalculator.M_to_M
        }
    }
    
    # Evaluate each possible outgoing transition
    return {curr_state: transition(prev_i, prev_j, AIS_seq, RF_seq) 
            for curr_state, transition in transitions[prev_state].items()}
    
def normalize(probs_dict, target_key, reserved_prob=0.05):
    """
    Normalize transition probabilities so that all outgoing transitions from a state sum to 1
    A small fixed amount (reserved_prob) is kept aside for transition into the 'end' state

    Args:
        probs_dict (dict): Dictionary containing transition probabilities
        target_key (str): The name of the current state to normalize for ('AIS', 'RF', 'M')
        reserved_prob (float): Fixed probability reserved for transition to 'end', defaults to 0.05

    Returns:
        float:The name of the current state to normalize for ('AIS', 'RF', 'M')
    """
    total = sum(probs_dict.values())
    
    if total == 0:
        return 0.0  
    
    # Scale probabilities so non-end states sum to (1 - reserved_prob)
    normalized_total = 1.0 - reserved_prob
    
    return (probs_dict.get(target_key, 0.0) / total) * normalized_total