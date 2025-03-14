import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

def preprocess_player_data(data_folder):
    # Iterate through each player's folder
    for player_folder in os.listdir(data_folder):
        player_path = os.path.join(data_folder, player_folder)
        
        # Ensure it's a directory
        if os.path.isdir(player_path):
            
            # Iterate through all CSV files (each representing a season)
            for file_name in os.listdir(player_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(player_path, file_name)
                    
                    # Load data
                    df = pd.read_csv(file_path)
                    
                    # Convert Date to game sequence number
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values(by='Date').reset_index(drop=True)
                    df['Game_Num'] = df.index + 1  # Ensure game order for LSTM
                    
                    # Convert 'MP' (Minutes Played) from MM:SS to rounded integer minutes
                    def convert_minutes(mp):
                        if isinstance(mp, str) and ':' in mp:
                            mins, secs = map(int, mp.split(':'))
                            return round(mins + secs / 60.0)  # Rounded integer minutes
                        return 0 if pd.isna(mp) else round(float(mp))  # Handle NaN by setting to 0
                    
                    df['MP'] = df['MP'].apply(convert_minutes)
                    
                    # Handle Home/Away (1 for Home, 0 for Away)
                    df['Home'] = df['Home/Away'].apply(lambda x: 1 if pd.isna(x) else 0)
                    df.drop(columns=['Home/Away'], inplace=True)
                    
                    # Create an Injury/Rest indicator (1 if NaN in performance stats, else 0)
                    stat_columns = ['PTS', 'TRB', 'AST', 'STL', 'TOV']  # Whole number stats
                    fractional_columns = ['FG%', '3P%', 'FT%']  # Fractional stats that should keep decimals
                    
                    df['Injured/Rest'] = df[stat_columns + fractional_columns].isna().any(axis=1).astype(int)
                    
                    # Fill missing performance stats with 0 (Ensure no empty columns)
                    df[stat_columns + fractional_columns] = df[stat_columns + fractional_columns].fillna(0)
                    df = df.fillna(0)  # Ensure all other NaN values are replaced with 0
                    
                    # Ensure whole-number stats are integers (remove decimals)
                    df[stat_columns] = df[stat_columns].astype(int)
                    
                    # One-hot encode `TM` (Team) and `OPP` (Opponent Team)
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') if sklearn.__version__ >= '1.2' else OneHotEncoder(sparse=False, handle_unknown='ignore')
                    
                    for col in ['TM', 'OPP']:
                        encoded = encoder.fit_transform(df[[col]])
                        encoded_columns = [f'{col}_{team}' for team in encoder.categories_[0]]
                        df_encoded = pd.DataFrame(encoded, columns=encoded_columns)
                        df = pd.concat([df, df_encoded], axis=1)
                        df.drop(columns=[col], inplace=True)
                    
                    # Compute 10-game rolling averages (ONLY APPEND NEW COLUMNS, DO NOT ALTER EXISTING ONES)
                    rolling_columns = ['PTS', 'TRB', 'AST', 'STL', 'TOV', 'FG%', '3P%', 'FT%', 'USG%', 'ORTG', 'DRTG', 'GmSc']
                    for col in rolling_columns:
                        df[f'{col}_rolling_avg'] = df[col].rolling(window=10, min_periods=1).mean()
                    
                    # Overwrite the original file with processed data (keeping original values intact)
                    df.to_csv(file_path, index=False)
                    
                    print(f'Processed: {file_path}')

# Example usage
data_folder = 'Data'  # Update with the actual path containing player folders
preprocess_player_data(data_folder)
