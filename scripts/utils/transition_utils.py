from collections import defaultdict

def count_state_transitions(df):
    """
    Count the number of transitions between consecutive states within each vessel track

    Args:
        df (pd.DataFrame): Input data containing time-ordered sequences of vessel states
        
    Returns:
        dict: A dictionary with keys in the format "StateA_to_StateB" and integer values representing the count of each transition
    """
    count_dict = defaultdict(int)

    for id, data in df.groupby("TrackID"):
        states = data['State']
        # Iterate over consecutive state pairs within the track
        for current_state, next_state in zip(states[:-1], states[1:]):   
            # Increment the count for this transition
            count_dict[f"{current_state}_to_{next_state}"] += 1
            
    return count_dict



