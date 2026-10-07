from abc import ABC, abstractmethod
from scripts.utils.geo_utils import *
from scripts.modeling.transition_probabilities.transitions import *

class State(ABC):
    """
    Abstract base class for a PHMM state
    
    Each state must define: 
        - emission(): 
        Defines how likely it is that this state generates a given observation
            - AIS state emits a single AIS observation 
            - RF state emits a single RF observation 
            - M state emits a paired AIS-RF observation, with probability based on time and distance similarity 
            - begin/end state do not emit any observations
        - transition(): 
        Specifies the probability of moving into this state from a given predecessor state
        - Δ(): 
        Defines how many steps to move forward in the AIS and RF sequence when entering a given state
            - AIS state (Δi=1, Δj=0) move one row (consume on AIS observation), but stay in the same column
            - RF state (Δi=0, Δj=1) move one column (consume one RF observation), but stay in the same row
            - M state (Δi=1, Δj=1) move diagonally (consume one AIS and one RF observation together)
            - begin/end state (Δi=0, Δj=0) do not move in either direction 
        - get_predecessors(): 
        Determines which states are allowed to transition into this state at a given position in DP matrx 
    
    """
    def __init__(self, name: str, id: int):
        """
        Initialize a state with name and ID
        
        Args:
            name (str): Name of the state (e.g., 'begin', 'AIS', 'RF', 'M', 'end')
            id (int): Unique identifier for the state, used for DP matrix indexing 
        """
        self.name = name
        self.id = id

    @abstractmethod
    def emission(self, observation):
        """
        Compute the emission probability of producing the given observation

        Args:
            observation (tuple): 
                For AIS state: ((timestamp, (lat, lon)), None)
                For RF state: (None, (timestamp, (lat, lon)))
                For M state: ((timestamp, (lat, lon)), (timestamp, (lat, lon)))
            
        Returns:
            float: P(observation | current state)
        """
        pass

    @abstractmethod
    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        """
        Compute the transition probability from a previous state.

        Args:
            prev_state (str): Name of the previous state
            prev_i (int): AIS index before the transition 
            prev_j (int): RF index before the transition
            AIS_seq (List): Sequence of AIS observations
            RF_seq (List): Sequence of single RF observation
        
        Returns:
            float: P(current state | previous state)
        """
        pass

    @abstractmethod
    def Δ(self):
        """
        Return the index consumption (steps) when entering this state
        
        Returns:
            tuple[int, int]: (Δi, Δj) meaning:
                (1,0): consume one AIS observation
                (0,1): consume one RF observation
                (1,1): consume both AIS and RF
                (0,0): consume nothing (begin/end states)
        """
        pass

    @abstractmethod
    def get_predecessors(self, i, j):
        """
        Return which states can lead into this state at position (i, j)

        Args:
            i (int): Current row AIS position
            j (int): Current row RF position
        
        Returns:
            list: Names of valid predecessor states
        """
        pass

class BeginState(State):
    """
    Start state
    """
    def __init__(self):
        super().__init__("begin", 0)

    def emission(self, observation):
        # No observation is emitted from the start
        return 0.0

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        # Beginning the sequence has probability 1
        return 1.0

    def Δ(self):
        # Consumes no observations
        return (0, 0)

    def get_predecessors(self, i, j):
        # Has no predecessors 
        return []

class AISState(State):
    """
    AIS state
    """
    def __init__(self, exponent: float = 0.5):
        super().__init__("AIS", 1)
        # kappa exponent smoothing/sharpening the row-scaled transition
        # probability into this state (default 0.5, see RFState/MState)
        self.exponent = exponent

    def emission(self, observation):
        AIS_obs, _ = observation
        # If an AIS observation exists, probability is 1, otherwise 0
        return 1.0 if AIS_obs is not None else 0.0

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        if prev_state == "begin":
            # Equal 1/3 probability of starting in AIS, RF, or M
            return 1.0 / 3.0

        transitions = get_transition_dict(prev_state, prev_i, prev_j, AIS_seq, RF_seq)
        base_prob = normalize(transitions, self.name)

        # Smooth transition probabilities to AIS state
        return base_prob ** self.exponent

    def Δ(self):
        # Consumes one AIS observation
        return (1, 0)

    def get_predecessors(self, i, j):
        # A predecessor state is any state that could plausibly precede
        # AIS here: another AIS/RF/M step, or 'begin' at the origin
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
    def __init__(self, exponent: float = 2):
        super().__init__("RF", 2)
        # kappa exponent sharpening the row-scaled transition
        # probability into this state (default 2, matches MState's)
        self.exponent = exponent

    def emission(self, observation):
        _, RF_obs = observation
        # If an RF observation exists, probability is 1, otherwise 0
        return 1.0 if RF_obs is not None else 0.0

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        if prev_state == "begin":
            # Equal 1/3 probability of starting in AIS, RF, or M
            return 1.0 / 3.0

        transitions = get_transition_dict(prev_state, prev_i, prev_j, AIS_seq, RF_seq)
        base_prob = normalize(transitions, self.name)

        # Sharpen transition probabilities to RF state
        return base_prob ** self.exponent

    def Δ(self):
        # Consumes one RF observation
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
    def __init__(self, exponent: float = 2):
        super().__init__("M", 3)
        # kappa exponent sharpening the row-scaled transition
        # probability into this state (default 2, matches RFState's)
        self.exponent = exponent

    def emission(self, observation):
        AIS_obs, RF_obs = observation
        # The probability is computed using time and distance similarity between AIS and RF observation
        return match_probability(*compute_abs_time_and_distance(AIS_obs, RF_obs))

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        if prev_state == "begin":
            # Equal 1/3 probability of starting in AIS, RF, or M
            return 1.0 / 3.0

        transitions = get_transition_dict(prev_state, prev_i, prev_j, AIS_seq, RF_seq)
        base_prob = normalize(transitions, self.name)

        # Sharpen transition probabilities to M state
        return base_prob ** self.exponent

    def Δ(self):
        # Consumes one AIS and one RF observation simultaneously
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

class NullMState(MState):
    """
    M state for the PHMM null/background model used to compute the
    log-odds score (score_lo = log P_full - log P_null)

    Identical row-scaling, tau, and kappa as MState, but every
    transition into M (from "begin", "AIS", "RF", or "M" itself) is
    forced to zero probability -- forward.py's induction already
    applies np.log(trans_prob + 1e-300), so a hard 0.0 here becomes
    the log-space floor (~-690), i.e. "log 0", with no special-casing
    needed elsewhere. There is nothing left to compute once the
    result is forced to zero, so this simply returns 0.0
    unconditionally, covering both the MState.transition() "begin"
    case and its general (normalize + kappa) case in one line.

    AISState and RFState each compute their own normalize() total
    independently of MState's value, so using them unchanged alongside
    this state leaves their AIS/RF probability mass exactly as in the
    full model -- i.e. the removed M mass is NOT redistributed
    (renormalize=False, the default this module's null-model state
    list uses). renormalize=True is accepted for documentation/API
    completeness (the option the task asks for) but is not wired to
    any behavior here: redistributing M's mass onto AIS/RF would
    require AIS/RF-row variants that exclude "M" from their own
    normalize() total before applying kappa, which no caller in this
    pipeline currently needs.
    """
    def __init__(self, renormalize: bool = False, exponent: float = 2):
        # exponent is accepted (and stored via MState.__init__) only for
        # API symmetry with MState/kappa sweeps -- transition() below
        # never reaches the base_prob ** self.exponent line, so it has
        # no effect on behavior.
        super().__init__(exponent=exponent)
        self.renormalize = renormalize

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        return 0.0

class EndState(State):
    def __init__(self):
        super().__init__("end", 4)

    def emission(self, observation):
        # No observation is emitted from the end
        return 0.0

    def transition(self, prev_state, prev_i, prev_j, AIS_seq, RF_seq):
        # End state can only be reached if the previous state
        # is one of its valid predecessors.
        if prev_state in self.get_predecessors(prev_i, prev_j):
            # Assign a small constant probability to transition into End
            return 0.05
        return 0.0

    def Δ(self):
        # Consumes no observations
        return (0, 0)

    def get_predecessors(self, i, j):
        # Any state is a valid predecessor once it has consumed its
        # corresponding observation(s)
        predecessors = []
        if i > 0:
            predecessors.append("AIS")
        if j > 0:
            predecessors.append("RF")
        if i > 0 and j > 0:
            predecessors.append("M")
        return predecessors

