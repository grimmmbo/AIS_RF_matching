import numpy as np 
from scipy.special import logsumexp

class PHMM_forward: 
    """
    A Pair Hidden Markov Model (PHMM) for aligning two sequences
    """
    def __init__(self, states, end_state):
        """
        Initializes the PHMM model 

        Args:
            states (list): A list of all possible states
            end_state (State): End state used to indicate the termination of the sequence 
        """
        self.states = states
        self.end_state = end_state
        
    def forward(self, AIS_seq, RF_seq):
        """
        Executes the Forward algorithm 

        Args:
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations 
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations 

        Returns:
            The probability of a sequence summed over all possible paths 
        """
        self.AIS_seq = AIS_seq
        self.RF_seq = RF_seq
        self.num_AIS = len(AIS_seq) + 1
        self.num_RF = len(RF_seq) + 1
        self.num_states = len(self.states)
                
        self.γ = np.full((self.num_states, self.num_AIS, self.num_RF), -np.inf)
                
        self._induction()
        return self._final_probability()
    
    def _induction(self):
        """
        Performs the forward pass of the Forward algorithm, 
        calculating the probability of being in each state at each position by adding up the probabilities of all possible paths that could get there
        """
        for state in self.states:
            self.γ[state.id, 0, 0] = 0.0 if state.name == "begin" else  -np.inf
            
        for i in range(self.num_AIS):
            for j in range(self.num_RF):
                if i == 0 and j == 0:
                    continue
                
                for curr_state in self.states:
                    Δ_i, Δ_j = curr_state.Δ()
                    prev_i, prev_j = i - Δ_i, j - Δ_j
                    
                    if prev_i < 0 or prev_j < 0:
                        continue
                    
                    observation = (self.AIS_seq[prev_i] if Δ_i else None, self.RF_seq[prev_j] if Δ_j else None)
                    
                    log_probs = []
                    
                    for prev_state in self.states:
                        if prev_state.name in curr_state.get_predecessors(prev_i, prev_j):
                            prev_log_prob = self.γ[prev_state.id, prev_i, prev_j]
                            
                            trans_prob = curr_state.transition(prev_state.name, prev_i, prev_j, self.AIS_seq, self.RF_seq)
                            log_trans_prob = np.log(trans_prob + 1e-300)

                            emiss_prob = curr_state.emission(observation)
                            log_emiss_prob = np.log(emiss_prob + 1e-300)
                            
                            prob = prev_log_prob + log_trans_prob + log_emiss_prob
                            
                            log_probs.append(prob)
                    
                    if log_probs:
                        self.γ[curr_state.id, i, j] = logsumexp(log_probs)
                                            
    def _final_probability(self):
        """
        Calculates the total probability 
        """
        end_i = len(self.AIS_seq)
        end_j = len(self.RF_seq)
        
        log_probs = []
        
        for prev_state in self.states:
            if prev_state.name in self.end_state.get_predecessors(end_i, end_j):
                trans_prob = self.end_state.transition(prev_state.name, end_i, end_j, self.AIS_seq, self.RF_seq)
                log_trans_prob = np.log(trans_prob + 1e-300)

                log_probs.append(self.γ[prev_state.id, end_i, end_j] + log_trans_prob)
                
        if log_probs:
            total_log_probs = logsumexp(log_probs)
        else:
            total_log_probs = -np.inf
            
        return total_log_probs
        