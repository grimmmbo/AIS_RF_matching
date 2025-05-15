import numpy as np 

class PHMM: 
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
        
    def viterbi(self, AIS_seq, RF_seq):
        """
        Executes the Viterbi algorithm 

        Args:
            AIS_seq (list[tuple[datetime, tuple[float, float]]]): List of AIS observations 
            RF_seq (list[tuple[datetime, tuple[float, float]]]): List of RF observations 

        Returns:
            tuple: 
                Most likely state path as a list of state names
                Final path probability
        """
        self.AIS_seq = AIS_seq
        self.RF_seq = RF_seq
        self.num_AIS = len(AIS_seq) + 1
        self.num_RF = len(RF_seq) + 1
        self.num_states = len(self.states)
        
        self.γ = np.full((self.num_states, self.num_AIS, self.num_RF), -np.inf)
        self.π = np.full((self.num_states, self.num_AIS, self.num_RF), -1)
        
        self._induction()
        self._handle_end_state() 
        return self._backtracking()
    
    def _induction(self):
        """
        Performs the forward pass of the Viterbi algorithm, calculating the highest probability paths to each state at each step in the alignment matrix
        """
        for state in self.states:
            self.γ[state.id, 0, 0] = 1.0 if state.name == "begin" else 0.0
            
        for i in range(self.num_AIS):
            for j in range(self.num_RF):
                if i == 0 and j == 0:
                    continue
                
                for curr_state in self.states:
                    best_prob = -np.inf
                    best_state_id = -1
                    
                    Δ_i, Δ_j = curr_state.Δ()
                    prev_i, prev_j = i - Δ_i, j - Δ_j
                    
                    if prev_i < 0 or prev_j < 0:
                        continue
                    
                    observation = (self.AIS_seq[prev_i] if Δ_i else None, self.RF_seq[prev_j] if Δ_j else None)
                    
                    for prev_state in self.states:
                        if prev_state.name in curr_state.get_predecessors(prev_i, prev_j):
                            prev_prob = self.γ[prev_state.id, prev_i, prev_j]
                            trans_prob = curr_state.transition(prev_state.name, prev_i, prev_j, self.AIS_seq, self.RF_seq)
                            emiss_prob = curr_state.emission(observation)
                            prob = prev_prob * trans_prob * emiss_prob
                            
                            if prob > best_prob:
                                best_prob = prob
                                best_state_id = prev_state.id

                    self.γ[curr_state.id, i, j] = best_prob
                    self.π[curr_state.id, i, j] = best_state_id
                    
    def _handle_end_state(self):
        """
        Finalizes the Viterbi matrix by computing the highest-probability transition into the end state from any valid predecessor state
        This is done by multiplying each predecessor's probability at the final cell with the transition probability to the end state, and selecting the maximum result
        """
        end_i = len(self.AIS_seq)
        end_j = len(self.RF_seq)
        
        best_prob = -np.inf
        best_state = -1    
        
        for prev_state in self.states:
            if prev_state.name in self.end_state.get_predecessors(end_i, end_j):
                trans_prob = self.end_state.transition(prev_state.name, end_i, end_j, self.AIS_seq, self.RF_seq)
                prob = self.γ[prev_state.id, end_i, end_j] * trans_prob
                
                if prob > best_prob:
                    best_prob = prob
                    best_state = prev_state.id

        self.γ[self.end_state.id, end_i, end_j] = best_prob
        self.π[self.end_state.id, end_i, end_j] = best_state   
        
    def _backtracking(self):   
        """
        Traces back through the path matrix π to reconstruct the most probable sequence of states
        
        Returns:
            tuple: 
                Most likely state path as a list of state names (from 'begin' to 'end')
                Final path probability
        """    
        i = len(self.AIS_seq)
        j = len(self.RF_seq)

        final_prob = self.γ[self.end_state.id, i, j]
        final_state = self.π[self.end_state.id, i, j]
        curr_state = self.states[final_state]

        path = ["end", curr_state.name]

        while i > 0 or j > 0:
            prev_id = self.π[curr_state.id, i, j]
            if prev_id == -1:
                path.append("begin")
                break

            if curr_state.name == "AIS":
                i -= 1
            elif curr_state.name == "RF":
                j -= 1
            elif curr_state.name == "M":
                i -= 1
                j -= 1

            curr_state = self.states[prev_id]
            path.append(curr_state.name)
        
        path.reverse()
        return path, final_prob 