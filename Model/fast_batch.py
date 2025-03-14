import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle  # For caching

data_folder = "/mnt/data/"  # Update with actual dataset path
cache_path = "cached_data.pkl"  # Cache to avoid redundant preprocessing

def load_multi_season_data(data_folder, cache_path):
    """Load and process multi-season NBA data efficiently, with caching."""
    if os.path.exists(cache_path):
        print("✅ Loading cached dataset...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    all_games = []
    
    for player_folder in os.listdir(data_folder):
        player_path = os.path.join(data_folder, player_folder)
        if os.path.isdir(player_path):
            for file_name in os.listdir(player_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(player_path, file_name)

                    # **Load only necessary columns**
                    cols_to_use = ['Date', 'oppDfRtg'] + [col for col in pd.read_csv(file_path, nrows=1).columns if col.startswith('OPP_')]
                    df = pd.read_csv(file_path, usecols=cols_to_use)

                    # **Efficiently handle missing data**
                    df.fillna(0, inplace=True)

                    # **Flag missing players**
                    df['Did_Not_Play'] = df.iloc[:, 2:].apply(lambda row: int((row == 0).all()), axis=1)

                    # **Add to dataset**
                    all_games.append(df)

    combined_df = pd.concat(all_games, ignore_index=True)

    # **Normalize data efficiently**
    scaler = StandardScaler()
    combined_df.iloc[:, 1:] = scaler.fit_transform(combined_df.iloc[:, 1:])

    # **Cache processed data**
    with open(cache_path, "wb") as f:
        pickle.dump(combined_df, f)

    print("✅ Dataset successfully processed and cached!")
    return combined_df

df = load_multi_season_data(data_folder, cache_path)
