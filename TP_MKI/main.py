import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_processing import DataProcessing, DataSetup
from models import Models, ade_fde
from pathways import pathways

MODEL_TO_RUN = "knn"  # Options: "lstm", "cnn", "hmm", "knn", "linear"
TAIL = 0.1
TEST_SIZE = 0.2
DATA_PATHS = pathways.home_videos()
# VALIDATION_PATH = pathways.home_videos_no_touch()

# --------------------------
# Step 1: Load and prepare data
# --------------------------

datasetup = DataSetup()
dataprocessing = DataProcessing()

print("Packaging data...")
pairs = datasetup.data_packager(DATA_PATHS)

# Remove validation folder from training pairs
# pairs = [p for p in pairs if VALIDATION_PATH not in p['video']]
# print(f"Found {len(pairs)} video/json pairs for training (validation folder excluded).")

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

# --------------------------
# Step 2: Normalize and pad
# --------------------------

scaler = dataprocessing.fit_scaler(Xs)
Xs_scaled = dataprocessing.transform_with_scaler(Xs, scaler)
ys_scaled = dataprocessing.transform_with_scaler(ys, scaler)

X_padded, len_x = dataprocessing.pad_sequences(Xs_scaled)
y_padded, len_y = dataprocessing.pad_sequences(ys_scaled)

# --------------------------
# Step 3: Split train/test
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_padded, test_size=TEST_SIZE, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --------------------------
# Step 4: Train selected model
# --------------------------

models = Models()

if MODEL_TO_RUN == "lstm":
    print("Training LSTM...")
    model, _ = models.train_lstm(X_train, y_train)
    preds = models.predict_lstm(X_test, y_len=y_test.shape[1])

elif MODEL_TO_RUN == "cnn":
    print("Training CNN...")
    model, _ = models.train_cnn(X_train, y_train)
    preds = models.predict_cnn(X_test)

elif MODEL_TO_RUN == "hmm":
    print("Training HMM...")
    model = models.train_hmm(X_train)
    preds = models.predict_hmm(X_test[0], n_steps=y_test.shape[1])

elif MODEL_TO_RUN == "knn":
    print("Training KNN...")
    model = models.train_knn(X_train, y_train, n_neighbors=5)
    preds = models.predict_knn(X_test)

elif MODEL_TO_RUN == "linear":
    print("Training Linear Regression...")
    model = models.train_linear(X_train, y_train)
    preds = models.predict_linear(X_test)

else:
    raise ValueError(f"Unknown model type: {MODEL_TO_RUN}")

# --------------------------
# Step 5: Evaluate
# --------------------------

ade, fde = ade_fde(preds, y_test)
print(f"{MODEL_TO_RUN.upper()} results:")
print(f"ADE: {ade:.4f}, FDE: {fde:.4f}")
