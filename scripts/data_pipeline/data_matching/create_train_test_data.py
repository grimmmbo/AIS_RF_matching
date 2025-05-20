import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(df, path_train_set, path_test_set, test_size=0.2, random_state=42):
    """
    Splits the input DataFrame into training and testing sets 

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset to be split
        test_size (float): Proportion of the dataset to include in the test split, defaults to 0.2
        random_state (int): Random seed for reproducibility, default to 42
    """
    unique_track_ids = df['TrackID'].unique()
    train_ids, test_ids = train_test_split(unique_track_ids, test_size=test_size, random_state=random_state)

    train_df = df[df['TrackID'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['TrackID'].isin(test_ids)].reset_index(drop=True)

    train_df.to_pickle(path_train_set)
    test_df.to_pickle(path_test_set)

if __name__ == "__main__":
    print("Splitting dataset into train and test sets...")
    
    SOURCE_PATH = "../../../../data/processed/phmm_sequence.parquet"
    DESTINATION_PATH_TRAIN_SET = "../../../../data/train_data/train_set.pkl"
    DESTINATION_PATH_TEST_SET = "../../../../data/test_data/test_set.pkl"

    PHMM_df = pd.read_parquet(SOURCE_PATH)

    split_train_test(PHMM_df, DESTINATION_PATH_TRAIN_SET, DESTINATION_PATH_TEST_SET)
    
    print("Data is saved...")