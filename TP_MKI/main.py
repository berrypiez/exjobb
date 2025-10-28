import os
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only errors
# Suppress Python warnings
import warnings
warnings.filterwarnings('ignore')

# Optional: Disable oneDNN floating point warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from sklearn.model_selection import train_test_split
from data_processing import DataProcessing, DataSetup
from models import Models, ade_fde
from pathways import pathways

# --------------------------
# Suppress TensorFlow oneDNN and info logs
# --------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only errors
import tensorflow as tf

# --------------------------
# Config
# --------------------------
MODEL_TO_RUN = "hmm"  # Options: "lstm", "cnn", "hmm"
TAIL = 0.1
TEST_SIZE = 0.2
DATA_PATHS = pathways.home_videos()
VERBOSE = 1  # For model.fit prints

# --------------------------
# Step 1: Load and prepare data
# --------------------------
datasetup = DataSetup()
dataprocessing = DataProcessing()

print("Packaging data...")
pairs = datasetup.data_packager(DATA_PATHS)

if len(pairs) == 0:
    raise RuntimeError("No video/json pairs found.")

print(f"Found {len(pairs)} video/json pairs.")

# --------------------------
# Step 2: Build dataset
# --------------------------
print("Building dataset (bbox only)...")
data = dataprocessing.build_dataset(
    pairs=pairs,
    tail=TAIL,
    use_skeleton=False,
    use_angles=False,
    add_velocity=False
)

Xs = data["Xs"]
ys = data["ys"]
labels = data["labels"]

if len(Xs) == 0:
    raise RuntimeError("No data available for training.")

print(f"Prepared {len(Xs)} sequences for training.")

# --------------------------
# Step 3: Normalize and pad sequences
# --------------------------
scaler = dataprocessing.fit_scaler(Xs)
Xs_scaled = dataprocessing.transform_with_scaler(Xs, scaler)
ys_scaled = dataprocessing.transform_with_scaler(ys, scaler)

X_padded, len_x = dataprocessing.pad_sequences(Xs_scaled)
y_padded, len_y = dataprocessing.pad_sequences(ys_scaled)

print(f"Padded sequences: X shape {X_padded.shape}, y shape {y_padded.shape}")

# --------------------------
# Step 4: Split train/test
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_padded, test_size=TEST_SIZE, random_state=42
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --------------------------
# Step 5: Train the model
# --------------------------
models = Models()

if MODEL_TO_RUN == "lstm":
    print("Training LSTM...")
    model, history = models.train_lstm(X_train, y_train, verbose=VERBOSE)
    preds = models.predict_lstm(X_test, y_test.shape[1])

elif MODEL_TO_RUN == "cnn":
    print("Training CNN...")
    model, history = models.train_cnn(X_train, y_train, verbose=VERBOSE)
    preds = models.predict_cnn(X_test)

elif MODEL_TO_RUN == "hmm":
    print("Training HMM...")
    model = models.train_hmm(X_train)
    preds = models.predict_hmm(X_test[0], n_steps=y_test.shape[1])

else:
    raise ValueError(f"Unknown model type: {MODEL_TO_RUN}")

# --------------------------
# Step 6: Evaluate
# --------------------------
ade, fde = ade_fde(preds, y_test)
print(f"\n{MODEL_TO_RUN.upper()} results:")
print(f"ADE: {ade:.4f}, FDE: {fde:.4f}")

