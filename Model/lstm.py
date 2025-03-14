import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import AdamW  # ✅ Correct optimizer
from tensorflow.keras.callbacks import LearningRateScheduler

# **Check GPU availability**
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Using device: {device}")

### 🔹 1. Load and Preprocess Data ###
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

# Load all player data
data_folder = "/mnt/data/"  # Change this to your actual data path
df = load_data(data_folder)

# Define features and target variables
feature_columns = ['oppDfRtg'] + [col for col in df.columns if col.startswith('OPP_')]  # Defensive rating + Opponent one-hot
target_columns = ['PTS', 'TRB', 'AST', 'STL', 'TOV', 'FG%', '3P%', 'FT%']  # Statline prediction

# Scale input features
scaler = StandardScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

### 🔹 2. Prepare Data for Training ###
sequence_length = 10  # How many past games to use
batch_size = 32

# Function to create LSTM sequences
def create_sequences(df, feature_columns, target_columns, sequence_length):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[feature_columns].iloc[i:i + sequence_length].values)
        y.append(df[target_columns].iloc[i + sequence_length].values)
    return np.array(X), np.array(y)

X, y = create_sequences(df, feature_columns, target_columns, sequence_length)

# Split into train and validation sets
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

### 🔹 3. Define LSTM Model ###
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(sequence_length, len(feature_columns))),
    Dense(len(target_columns))
])

# **Optimizer & OneCycle Learning Rate Scheduler**
initial_lr = 1e-3
optimizer = AdamW(learning_rate=initial_lr, weight_decay=1e-2)  # ✅ Now using AdamW

# OneCycleLR function for TensorFlow
def one_cycle_lr(epoch, lr):
    max_lr = initial_lr
    num_epochs = 20
    return max_lr * (0.1 ** (epoch / num_epochs))

lr_scheduler = LearningRateScheduler(one_cycle_lr)

# Compile Model
model.compile(optimizer=optimizer, loss="mse")

### 🔹 4. Train the Model ###
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=20,
    callbacks=[lr_scheduler],
    verbose=1
)

### 🔹 5. Find Similar Matchups and Predict ###
def find_similar_games(new_game, df, num_matches=5):
    """Find past games with similar opponent defensive rating"""
    df['rating_diff'] = abs(df['oppDfRtg'] - new_game['oppDfRtg'])
    df_sorted = df.sort_values(by='rating_diff').head(num_matches)
    return df_sorted[target_columns]

def predict_statline(new_game):
    """Predict a player's statline given an opponent's defensive rating"""
    new_game_scaled = scaler.transform([new_game[feature_columns]])
    new_game_tensor = np.array(new_game_scaled).reshape(1, sequence_length, len(feature_columns))

    prediction = model.predict(new_game_tensor)[0]
    return dict(zip(target_columns, prediction))

# Example input for a new game
new_game = {
    'oppDfRtg': 110.5,  # New opponent's defensive rating
    **{col: 0 for col in feature_columns if col.startswith('OPP_')}  # One-hot encoding default
}
new_game['OPP_LAL'] = 1  # Assume opponent is Lakers

# Find similar games
similar_games = find_similar_games(new_game, df)
print("\nSimilar past matchups:")
print(similar_games)

# Predict statline
predicted_stats = predict_statline(new_game)
print("\nPredicted Statline:")
print(predicted_stats)
