import numpy as np 
from scipy.special import logsumexp

class PHMM_forward: 
    """
    A Pair Hidden Markov Model (PHMM) for aligning two observation sequences
    
    This class implements the Forward algorithm in log space to avoid numerical underflow when multiplying many small probabilities
    """
    def __init__(self, states, end_state):
        """
        Initializes the PHMM model 

        Args:
            states (list): All states used by the model, each providing:
                - name (str): state label, e.g., "begin", "M", "AIS", "RF", "end"
                - id (int): state id, e.g., begin (0), AIS (1), RF (2), M (3) and end (4)
                - Δ() tuple[int, int]: step sizes (Δ_i, Δ_j) for AIS and RF.
                - emission(obs) -> float: emission probability for the observation.
                - transition(prev_state_name, i, j, AIS_seq, RF_seq) -> float:
                    transition probability from prev_state_name to this state at (i, j).
                - get_predecessors(i, j) -> Iterable[str]: allowed predecessor names.
            end_state (State): Terminal state object used to finalize the sequence. 
        """
        self.states = states
        self.end_state = end_state
    
    def forward(self, AIS_seq, RF_seq):
        """
        Executes the Forward algorithm.

        Args:
            AIS_seq (list[tuple]): AIS observations [(time, (lon, lat)), ...]
            RF_seq (list[tuple]): One single RF observation [(time, (lon, lat)), ...]

        Returns:
            float: Log-probability of the full alignment (sum over all possible paths).
        """
        self.AIS_seq = AIS_seq
        self.RF_seq = RF_seq # Consisting of one RF signal/observation in this study
        self.num_AIS = len(AIS_seq) + 1
        self.num_RF = len(RF_seq) + 1
        self.num_states = len(self.states)
        
        # Initialize dynamic programming table γ in log-space  
        self.γ = np.full((self.num_states, self.num_AIS, self.num_RF), -np.inf)
            
        # Precompute normalized emission probabilities for Match state        
        self.match_emission_matrix = self.compute_normalized_match_emissions()

        # Run forward pass (dynamic programming)
        self._induction()
        
        # Compute final probability of ending in 'end' state
        return self._final_probability()    
    
    def compute_normalized_match_emissions(self):
        """
        Normalized probability distribution over all AIS observations
        (summing to one), representing how likely each is to match the
        RF signal
        """
        match_state = next((s for s in self.states if s.name == "M"), None)
        if not match_state:
            raise ValueError("Match state (M) not found")

        E = np.zeros((self.num_AIS, self.num_RF))  # includes dummy index 0

        for i in range(1, self.num_AIS):
            for j in range(1, self.num_RF):
                obs = (self.AIS_seq[i - 1], self.RF_seq[j - 1])
                E[i, j] = match_state.emission(obs)

        # Normalize each column so match probabilities sum to 1
        column_sums = E.sum(axis=0, keepdims=True) + 1e-300
        E = E / column_sums  
        return E        
    
    def _induction(self):
        """
        Forward pass of the algorithm.

        For each position (i, j) and state, compute the log-probability of being there
        by summing (in log-space) over all possible predecessor paths:
        
            γ[state, i, j] = logsumexp(
                γ[prev_state, prev_i, prev_j] 
                + log(transition_prob) 
                + log(emission_prob)
            )
        """
        # Initialize start state ("begin") at (0,0) with log(1) = 0 and all other entries to -infinity
        for state in self.states:
            self.γ[state.id, 0, 0] = 0.0 if state.name == "begin" else  -np.inf
        
        # Iterate over all matrix positions 
        for i in range(self.num_AIS):
            for j in range(self.num_RF):
                if i == 0 and j == 0:
                    continue 
    
                for curr_state in self.states:
                    # Determine predecessor position based on Δ
                    Δ_i, Δ_j = curr_state.Δ()
                    prev_i, prev_j = i - Δ_i, j - Δ_j
                    
                    # Out-of-bounds check
                    if prev_i < 0 or prev_j < 0:
                        continue
                    
                    # Build observation for emission probability
                    observation = (
                        self.AIS_seq[prev_i] if Δ_i else None, 
                        self.RF_seq[prev_j] if Δ_j else None
                    )
                    
                    log_probs = []
                    
                    # Sum over all valid predecessors
                    for prev_state in self.states:
                        if prev_state.name in curr_state.get_predecessors(prev_i, prev_j):
                            
                            # Forward log probability
                            prev_log_prob = self.γ[prev_state.id, prev_i, prev_j]
                            
                            # Log transition probability
                            trans_prob = curr_state.transition(prev_state.name, prev_i, prev_j, self.AIS_seq, self.RF_seq)
                            log_trans_prob = np.log(trans_prob + 1e-300)

                            # Log emission probability
                            if curr_state.name == "M":
                                emiss_prob = self.match_emission_matrix[i, j]
                            else:
                                emiss_prob = curr_state.emission(observation)
                            log_emiss_prob = np.log(emiss_prob + 1e-300)
                        
                            # The total log probability of reaching the current state at position (i,j), 
                            # through one specific predecessor path
                            prob = prev_log_prob + log_trans_prob + log_emiss_prob
                            log_probs.append(prob)
                    
                    # Combine all possible paths via log-sum-exp
                    if log_probs:
                        self.γ[curr_state.id, i, j] = logsumexp(log_probs)
                                            
    def _final_probability(self):
        """
        Computes the total log-probability of the alignment by 
        summing (log-sum-exp) over all paths that end in the 'end' state.
        """
        end_i = len(self.AIS_seq)
        end_j = len(self.RF_seq)
        
        log_probs = []
        
        # Collect contributions from all valid predecessors of 'end'
        for prev_state in self.states:
            if prev_state.name in self.end_state.get_predecessors(end_i, end_j):
                trans_prob = self.end_state.transition(prev_state.name, end_i, end_j, self.AIS_seq, self.RF_seq)
                log_trans_prob = np.log(trans_prob + 1e-300)

                log_probs.append(self.γ[prev_state.id, end_i, end_j] + log_trans_prob)
        
        # Combine all end paths        
        if log_probs:
            total_log_probs = logsumexp(log_probs)
        else:
            total_log_probs = -np.inf
            
        return total_log_probs
        