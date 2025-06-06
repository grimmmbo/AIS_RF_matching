import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from scripts.PHMM.viterbi_algorithm.transitions import *
from scripts.utils.geo_utils import *
from abc import ABC, abstractmethod

class State(ABC):
    def __init__(self, name: str, id: int):
        """
        Initialize the state with a name and ID

        Args:
            name (str): Name of the state (e.g., 'begin', 'AIS', 'RF', 'M', 'end')
            id (int): Unique identifier for the state, used for referencing the state in matrices
        """
        self.name = name
        self.id = id

    @abstractmethod
    def emission(self, observation):
        """
        This function computes the emission probability (likelihood) of observing a particular observation from this state

        Args:
            observation (tuple): A data point in the format (timestamp, (longitude, latitude))
                observation for AIS state: ((timestamp, (longitude, latitude)), None)
                observation for RF state: (None, (timestamp, (longitude, latitude)))
                observation for M state: ((timestamp, (longitude, latitude)), (timestamp, (longitude, latitude)))
            
        Returns:
            float: Emission probability of the observation given a state
        """
        pass

    @abstractmethod
    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        """
        This function computes the transition probability (likelihood) of moving from 'previous state' to the 'current state'

        Args:
            prev_state (State): The previous state
            prev_i (int): The row index (AIS sequence position) before transitioning
            prev_j (int): The row index (RF sequence position) before transitioning
            AIS_seq (List): List of AIS observations
            RF_seq (List): List of RF observations
        
        Returns:
            float: Transition probability from the previous state to current state
        """
        pass

    @abstractmethod
    def Δ(self):
        """
        Defines how the position in the matrix changes when entering this state
        
        Returns:
            tuple[int, int]: A pair (Δi, Δj) indicating how the matrix indices changes
                (1,0), vertical movement (AIS state)
                (0, 1), horizontal movement (RF state) 
                (1,1), diagonal movement (M state)
                (0,0): no movement (begin/end state)
        """
        pass

    @abstractmethod
    def get_predecessors(self, i, j):
        """
        Determine which previous states (predecessors) could have led to the current state at position i, j

        Args:
            i (int): Current row AIS position
            j (int): Current row RF position
        
        Returns:
            list: A list containing predecessors that could transition into (i, j)
        """
        pass

class BeginState(State):
    def __init__(self):
        super().__init__("begin", 0)

    def emission(self, observation):
        return 0.0

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        return 1.0

    def Δ(self):
        return (0, 0)

    def get_predecessors(self, i, j):
        return []

class AISState(State):
    def __init__(self):
        super().__init__("AIS", 1)

    def emission(self, observation):
        AIS_obs, _ = observation
        return 1.0 if AIS_obs is not None else 0.0

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        if prev_state == "begin":
            return 1.0 / 3.0
        transitions = get_transition_dict(prev_state, prev_i, prev_j, AIS_seq, RF_seq)
        # return normalize(transitions, self.name)
        
        # test
        base_prob = normalize(transitions, self.name)
        square_root = base_prob ** 0.5
        return square_root

    def Δ(self):
        return (1, 0)

    def get_predecessors(self, i, j):
        predecessors = []
        if i > 0 and j >= 0:
            predecessors.append("AIS")
        if j > 0 and i >= 0:
            predecessors.append("RF")
        if i > 0 and j > 0:
            predecessors.append("M")
        if i == 0 and j == 0:
            predecessors.append("begin")
        return predecessors

class RFState(State):
    def __init__(self):
        super().__init__("RF", 2)

    def emission(self, observation):
        _, RF_obs = observation
        return 1.0 if RF_obs is not None else 0.0

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        if prev_state == "begin":
            return 1.0 / 3.0
        transitions = get_transition_dict(prev_state, prev_i, prev_j, AIS_seq, RF_seq)
        # return normalize(transitions, self.name)
        
        # test
        base_prob = normalize(transitions, self.name)
        exponent = 2
        return base_prob ** exponent
            
    def Δ(self):
        return (0, 1)

    def get_predecessors(self, i, j):
        predecessors = []
        if i > 0 and j >= 0:
            predecessors.append("AIS")
        if j > 0 and i >= 0:
            predecessors.append("RF")
        if i > 0 and j > 0:
            predecessors.append("M")
        if i == 0 and j == 0:
            predecessors.append("begin")
        return predecessors

class MState(State):
    def __init__(self):
        super().__init__("M", 3)

    def emission(self, observation):
        AIS_obs, RF_obs = observation
        return match_probability(*compute_abs_time_and_distance(AIS_obs, RF_obs))

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        if prev_state == "begin":
            return 1.0 / 3.0
        transitions = get_transition_dict(prev_state, prev_i, prev_j, AIS_seq, RF_seq)
        # return normalize(transitions, self.name)
        
        # test
        base_prob = normalize(transitions, self.name)
        exponent = 2
        return base_prob ** exponent

    def Δ(self):
        return (1, 1)

    def get_predecessors(self, i, j):
        predecessors = []
        if i > 0 and j >= 0:
            predecessors.append("AIS")
        if j > 0 and i >= 0:
            predecessors.append("RF")
        if i > 0 and j > 0:
            predecessors.append("M")
        if i == 0 and j == 0:
            predecessors.append("begin")
        return predecessors

class EndState(State):
    def __init__(self):
        super().__init__("end", 4)

    def emission(self, observation):
        return 0.0

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        if prev_state in self.get_predecessors(prev_i, prev_j):
            return 0.05
        return 0.0

    def Δ(self):
        return (0, 0)

    def get_predecessors(self, i, j):
        predecessors = []
        if i > 0:
            predecessors.append("AIS")
        if j > 0:
            predecessors.append("RF")
        if i > 0 and j > 0:
            predecessors.append("M")
        return predecessors

