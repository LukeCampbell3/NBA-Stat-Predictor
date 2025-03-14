import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention

# **Check GPU availability**
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Using device: {device}")

### **🔹 Load Full Team Game Data**
data_folder = "/mnt/data/"  # Update with actual data location

def load_team_data(data_folder):
    all_games = []
    
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

                    # Handle Missing Players (if statline is all 0s, the player did not play)
                    df['Did_Not_Play'] = df.iloc[:, 3:].apply(lambda row: int((row==0).all()), axis=1)

                    # Fill missing values with 0 (for compatibility)
                    df.fillna(0, inplace=True)

                    # Append to dataset
                    all_games.append(df)

    return pd.concat(all_games, ignore_index=True)

df = load_team_data(data_folder)

### **🔹 Define Features & Target Variables**
feature_columns = ['oppDfRtg'] + [col for col in df.columns if col.startswith('OPP_')]  # Defensive rating + Opponent encoding
target_columns = ['PTS', 'TRB', 'AST', 'STL', 'TOV', 'FG%', '3P%', 'FT%']  # Player statlines

# Scale input features using StandardScaler
scaler = StandardScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

### **🔹 Create Sequences for Transformer Training**
sequence_length = 10  # Use last 10 games for context
batch_size = 32

def create_team_sequences(df, feature_columns, target_columns, sequence_length):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[feature_columns].iloc[i:i + sequence_length].values)
        y.append(df[target_columns].iloc[i + sequence_length].values)
    return np.array(X), np.array(y)

X, y = create_team_sequences(df, feature_columns, target_columns, sequence_length)

# Split into train and validation sets
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

'''
### **🚀 Updated Transformer Model with Multi-Head Latent Attention**
This new model:
✅ **Processes full team stats at once**  
✅ **Uses self-attention to determine lineup adjustments**  
✅ **Outputs player statlines while considering missing teammates** 
'''

class MultiHeadLatentAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, latent_dim, dropout_rate=0.1):
        super(MultiHeadLatentAttention, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.latent_dense = Dense(latent_dim, activation="relu")
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs, inputs)
        latent_output = self.latent_dense(attn_output)
        return self.dropout(latent_output, training=training)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, latent_dim, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.mla = MultiHeadLatentAttention(num_heads, key_dim, latent_dim, dropout_rate)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(key_dim * num_heads),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.mla(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(input_shape, num_layers, num_heads, key_dim, latent_dim, ff_dim, output_dim):
    inputs = Input(shape=input_shape)
    
    x = Dense(key_dim * num_heads)(inputs)
    
    for _ in range(num_layers):
        x = TransformerBlock(num_heads, key_dim, latent_dim, ff_dim)(x)

    x = Dense(output_dim)(x)

    model = Model(inputs=inputs, outputs=x)
    return model

# **Model Parameters**
num_layers = 4
num_heads = 8
key_dim = 64
latent_dim = 32
ff_dim = 512
output_dim = len(target_columns)

# Build & Compile Model
model = build_transformer_model((sequence_length, len(feature_columns)), num_layers, num_heads, key_dim, latent_dim, ff_dim, output_dim)
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-2), loss="mse", metrics=["mae"])

# Train Model
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=20,
    verbose=1
)
