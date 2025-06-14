import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from scripts.PHMM.distributions.match_distribution import match_probability
from scripts.PHMM.distributions.AIS_RF_distribution import AIS_RF_probability
from scripts.utils.geo_utils import *

class TransitionCalculator:
    @staticmethod
    def AIS_to_AIS(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Computes the transition probability from an AIS state to another AIS state

        Args:
            prev_i (int): Index in the γ matrix corresponding to the AIS sequence
            prev_j (int): Index in the γ matrix corresponding to the RF sequence
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

        Returns:
            float: Transition probability based on time and distance between two points
        """
        seq_i, seq_j = prev_i - 1, prev_j - 1
        
        if seq_i < 0 or seq_i + 1 >= len(AIS_seq):
            return 0.0
        
        curr_point, prev_point = AIS_seq[seq_i + 1], AIS_seq[seq_i]
        time, dist = compute_time_and_distance(curr_point, prev_point)
        
        return AIS_RF_probability(time, dist)

    @staticmethod
    def AIS_to_RF(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Computes the transition probability from an AIS state to RF state

        Args:
            prev_i (int): Index in the γ matrix corresponding to the AIS sequence
            prev_j (int): Index in the γ matrix corresponding to the RF sequence
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

        Returns:
            float: Transition probability based on time and distance between two points
        """
        seq_i, seq_j = prev_i - 1, prev_j - 1
        
        if seq_i < 0 or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        curr_point, prev_point = RF_seq[seq_j + 1], AIS_seq[seq_i]
        time, dist = compute_time_and_distance(curr_point, prev_point)

        return AIS_RF_probability(time, dist)

    @staticmethod
    def AIS_to_M(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Computes the transition probability from an AIS state to M state

        Args:
            prev_i (int): Index in the γ matrix corresponding to the AIS sequence
            prev_j (int): Index in the γ matrix corresponding to the RF sequence
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

        Returns:
            float: Transition probability based on time and distance between two points
        """
        seq_i, seq_j = prev_i - 1, prev_j - 1
        
        if seq_i + 1 >= len(AIS_seq) or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        curr_AIS, curr_RF = AIS_seq[seq_i + 1], RF_seq[seq_j + 1]
        time, dist = compute_abs_time_and_distance(curr_AIS, curr_RF)

        return match_probability(time, dist)

    @staticmethod
    def RF_to_AIS(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Computes the transition probability from an RF state to AIS state

        Args:
            prev_i (int): Index in the γ matrix corresponding to the AIS sequence
            prev_j (int): Index in the γ matrix corresponding to the RF sequence
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

        Returns:
            float: Transition probability based on time and distance between two points
        """
        seq_i, seq_j = prev_i - 1, prev_j - 1
        if seq_j < 0 or seq_i + 1 >= len(AIS_seq):
            return 0.0
        
        curr_point, prev_point = AIS_seq[seq_i + 1], RF_seq[seq_j]
        time, dist = compute_time_and_distance(curr_point, prev_point)
        
        return AIS_RF_probability(time, dist)

    @staticmethod
    def RF_to_RF(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Computes the transition probability from an RF state to another RF state

        Args:
            prev_i (int): Index in the γ matrix corresponding to the AIS sequence
            prev_j (int): Index in the γ matrix corresponding to the RF sequence
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

        Returns:
            float: Transition probability based on time and distance between two points
        """
        seq_i, seq_j = prev_i - 1, prev_j - 1
        if seq_j < 0 or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        curr_point, prev_point = RF_seq[seq_j + 1], RF_seq[seq_j]
        time, dist = compute_time_and_distance(curr_point, prev_point)
        
        return AIS_RF_probability(time, dist)

    @staticmethod
    def RF_to_M(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Computes the transition probability from an RF state to M state

        Args:
            prev_i (int): Index in the γ matrix corresponding to the AIS sequence
            prev_j (int): Index in the γ matrix corresponding to the RF sequence
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

        Returns:
            float: Transition probability based on time and distance between two points
        """
        seq_i, seq_j = prev_i - 1, prev_j - 1
        
        if seq_i + 1 >= len(AIS_seq) or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        curr_AIS, curr_RF = AIS_seq[seq_i + 1], RF_seq[seq_j + 1]
        time, dist = compute_abs_time_and_distance(curr_AIS, curr_RF)
        
        return match_probability(time, dist)

    @staticmethod
    def M_to_AIS(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Computes the transition probability from an M state to AIS state

        Args:
            prev_i (int): Index in the γ matrix corresponding to the AIS sequence
            prev_j (int): Index in the γ matrix corresponding to the RF sequence
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

        Returns:
            float: Transition probability based on time and distance between two points
        """
        seq_i, seq_j = prev_i - 1, prev_j - 1
        
        if seq_i < 0 or seq_j < 0 or seq_i + 1 >= len(AIS_seq):
            return 0.0
        
        curr_point = AIS_seq[seq_i + 1]
        prev_AIS, prev_RF = AIS_seq[seq_i], RF_seq[seq_j]
        
        if prev_AIS[0] > prev_RF[0]:
            time, dist = compute_time_and_distance(curr_point, prev_AIS)
        else:
            time, dist = compute_time_and_distance(curr_point, prev_RF)
        
        # time, dist = compute_average_time_and_distance(curr_point, prev_AIS, prev_RF)
        
        if (time, dist) == (0,0):
            return 0.0
                
        return AIS_RF_probability(time, dist)

    @staticmethod
    def M_to_RF(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Computes the transition probability from an M state to RF state

        Args:
            prev_i (int): Index in the γ matrix corresponding to the AIS sequence
            prev_j (int): Index in the γ matrix corresponding to the RF sequence
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

        Returns:
            float: Transition probability based on time and distance between two points
        """
        seq_i, seq_j = prev_i - 1, prev_j - 1
        
        if seq_i < 0 or seq_j < 0 or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        curr_point = RF_seq[seq_j + 1]
        prev_AIS, prev_RF = AIS_seq[seq_i], RF_seq[seq_j]
        
        if prev_AIS[0] > prev_RF[0]:
            time, dist = compute_time_and_distance(curr_point, prev_AIS)
        else:
            time, dist = compute_time_and_distance(curr_point, prev_RF)
            
        # time, dist = compute_average_time_and_distance(curr_point, prev_AIS, prev_RF)
        
        if (time, dist) == (0,0):
            return 0.0
        
        return AIS_RF_probability(time, dist)

    @staticmethod
    def M_to_M(prev_i, prev_j, AIS_seq, RF_seq):
        """
        Computes the transition probability from an M state to another M state

        Args:
            prev_i (int): Index in the γ matrix corresponding to the AIS sequence
            prev_j (int): Index in the γ matrix corresponding to the RF sequence
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

        Returns:
            float: Transition probability based on time and distance between two points
        """
        seq_i, seq_j = prev_i - 1, prev_j - 1
        
        if seq_i + 1 >= len(AIS_seq) or seq_j + 1 >= len(RF_seq):
            return 0.0
        
        curr_AIS, curr_RF = AIS_seq[seq_i + 1], RF_seq[seq_j + 1]
        time, dist = compute_abs_time_and_distance(curr_AIS, curr_RF)
        
        return match_probability(time, dist)

def get_transition_dict(prev_state, prev_i, prev_j, AIS_seq, RF_seq):
    """
    Computes the transition probabilities from a given previous state to all possible current states ('AIS', 'RF', 'M')

    Args:
        prev_state (str): Name of the previous state ('AIS', 'RF', 'M')
        prev_i (int): Index in the γ matrix corresponding to the AIS sequence
        prev_j (int): Index in the γ matrix corresponding to the RF sequence
        AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations
        RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations

    Returns:
        dict: A dictionary mapping current state name to their corresponding transition probabilities
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
    
    return {curr_state: transition(prev_i, prev_j, AIS_seq, RF_seq) 
            for curr_state, transition in transitions[prev_state].items()}
    
def normalize(probs_dict, target_key, reserved_prob=0.05):
    """
    Normalizes the transition probability for a target state.

    Args:
        probs_dict (dict): Dictionary containing transition probabilities
        target_key (str): The name of the state to normalize for ('AIS', 'RF', 'M')
        reserved_prob (float): Fixed probability reserved for transition to 'end', defaults to 0.05

    Returns:
        float: Normalized transition probability
    """
    total = sum(probs_dict.values())
    
    if total == 0:
        return 0.0  
    
    normalized_total = 1.0 - reserved_prob
    
    return (probs_dict.get(target_key, 0.0) / total) * normalized_total