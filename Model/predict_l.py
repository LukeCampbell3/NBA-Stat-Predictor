import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# **Check GPU availability**
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Using device: {device}")

### **🔹 Load Pre-Trained LSTM Model**
model_path = "/mnt/data/trained_lstm_model"  # Update with actual model path
model = load_model(model_path)

### **🔹 Load Player Data for Reference**
data_folder = "/mnt/data/"  # Update this to match actual data location

def load_data(data_folder):
    all_data = []
    
    for player_folder in os.listdir(data_folder):
        player_path = os.path.join(data_folder, player_folder)
        if os.path.isdir(player_path):
            for file_name in os.listdir(player_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(player_path, file_name)
                    df = pd.read_csv(file_path)

                    # Ensure 'oppDfRtg' and opponent encoding exist
                    if 'oppDfRtg' not in df.columns:
                        df['oppDfRtg'] = 0  # Default to 0 if missing
                    opp_columns = [col for col in df.columns if col.startswith('OPP_')]

                    # Fill missing values with 0
                    df.fillna(0, inplace=True)

                    # Append to dataset
                    all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

df = load_data(data_folder)

### **🔹 Define Features & Target Variables**
feature_columns = ['oppDfRtg'] + [col for col in df.columns if col.startswith('OPP_')]  # Defensive rating + Opponent one-hot
target_columns = ['PTS', 'TRB', 'AST', 'STL', 'TOV', 'FG%', '3P%', 'FT%']  # Predicted statline

# Scale input features using the same scaler from training
scaler = StandardScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

### **🔹 Find Similar Games Based on Defensive Rating**
def find_similar_games(new_game, df, num_matches=5):
    """Find past games with similar opponent defensive rating"""
    df['rating_diff'] = abs(df['oppDfRtg'] - new_game['oppDfRtg'])
    df_sorted = df.sort_values(by='rating_diff').head(num_matches)
    return df_sorted[target_columns]

### **🔹 Predict Player Statlines Given Opponent Defensive Rating (LSTM-Specific)**
def predict_statline(new_game):
    """Predict a player's statline given an opponent's defensive rating"""
    new_game_scaled = scaler.transform([new_game[feature_columns]])

    # Ensure input shape matches LSTM (batch_size=1, sequence_length=10, num_features)
    new_game_tensor = np.array(new_game_scaled).reshape(1, 10, len(feature_columns))

    prediction = model.predict(new_game_tensor)[0]
    return dict(zip(target_columns, prediction))

### **🔹 Example Prediction**
new_game = {
    'oppDfRtg': 110.5,  # New opponent's defensive rating
    **{col: 0 for col in feature_columns if col.startswith('OPP_')}  # Default one-hot encoding for opponent
}
new_game['OPP_LAL'] = 1  # Assume opponent is Lakers

# **Find similar matchups**
similar_games = find_similar_games(new_game, df)
print("\nSimilar Past Matchups:")
print(similar_games)

# **Generate Statline Prediction**
predicted_stats = predict_statline(new_game)
print("\nPredicted Statline:")
print(predicted_stats)
