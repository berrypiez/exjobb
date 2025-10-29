import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from data_processing import DataProcessing, DataSetup
from models import Models, ade_fde
from pathways import pathways


# ============================================
# CONFIG
# ============================================
RAW_DATA_PATHS = pathways.home_videos()
PROCESSED_DIR = "processed_data"
RESULTS_DIR = "results"
TAIL = 0.1
TEST_SIZE = 0.2
N_PLOT_SAMPLES = 3
DOOR_THRESHOLD_X = 0.8  # example threshold for "open door" decision
MODELS_TO_RUN = ["lstm", "cnn", "linear", "knn"] # 

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

datasetup = DataSetup()
dataprocessing = DataProcessing()


# ============================================
# STEP 1: PREPROCESS (if not already done)
# ============================================
if not all(os.path.exists(os.path.join(PROCESSED_DIR, f"{name}.npy")) for name in ["X", "y", "labels"]) \
        or not os.path.exists(os.path.join(PROCESSED_DIR, "scaler.npy")):

    print("Preprocessing data...")
    pairs = datasetup.data_packager(RAW_DATA_PATHS)
    print(f"Found {len(pairs)} video/json pairs.")

    data = dataprocessing.build_dataset(pairs, tail=TAIL)
    Xs, ys, labels = data["Xs"], data["ys"], data["labels"]

    scaler = dataprocessing.fit_scaler(Xs)
    Xs_scaled = dataprocessing.transform_with_scaler(Xs, scaler)
    ys_scaled = dataprocessing.transform_with_scaler(ys, scaler)
    X_padded, _ = dataprocessing.pad_sequences(Xs_scaled)
    y_padded, _ = dataprocessing.pad_sequences(ys_scaled)

    np.save(os.path.join(PROCESSED_DIR, "X.npy"), X_padded)
    np.save(os.path.join(PROCESSED_DIR, "y.npy"), y_padded)
    np.save(os.path.join(PROCESSED_DIR, "labels.npy"), np.array(labels))
    np.save(os.path.join(PROCESSED_DIR, "scaler.npy"), scaler)
else:
    X_padded = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
    y_padded = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
    labels = np.load(os.path.join(PROCESSED_DIR, "labels.npy"), allow_pickle=True)
    scaler = np.load(os.path.join(PROCESSED_DIR, "scaler.npy"), allow_pickle=True).item()
    print("Loaded preprocessed data.")


# ============================================
# STEP 2: TRAIN/TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_padded, test_size=TEST_SIZE, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


# ============================================
# STEP 3: HELPER FUNCTIONS
# ============================================
def plot_trajectories(y_true, y_pred, model_name, n_samples=N_PLOT_SAMPLES):
    cx, cy = 0, 1
    n_samples = min(n_samples, len(y_true))
    for i in range(n_samples):
        plt.figure(figsize=(4, 4))
        plt.plot(y_true[i,:,cx], y_true[i,:,cy], 'o-', label="GT")
        plt.plot(y_pred[i,:,cx], y_pred[i,:,cy], 'x--', label="Pred")
        plt.legend()
        plt.title(f"{model_name.upper()} Sample {i+1}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, "plots", f"{model_name}_sample{i+1}.png")
        plt.savefig(save_path)
        plt.close()


def decide_open_door(trajectory, threshold=DOOR_THRESHOLD_X):
    """Decide if the person should open the door based on reaching X-threshold."""
    final_x = trajectory[-1, 0]  # assuming cx is feature 0
    return final_x > threshold


def evaluate_decisions(y_true, y_pred):
    """Compute accuracy of door decision."""
    true_decisions = [decide_open_door(t) for t in y_true]
    pred_decisions = [decide_open_door(p) for p in y_pred]
    correct = sum(t == p for t, p in zip(true_decisions, pred_decisions))
    return correct / len(y_true)


# ============================================
# STEP 4: RUN ALL MODELS
# ============================================
results = {}
models_obj = Models(feature_size=X_train.shape[2])
models_obj.reset_models()
for model_name in MODELS_TO_RUN:
    print(f"\n=== Training {model_name.upper()} ===")

    if model_name == "lstm":
        model, _ = models_obj.train_lstm(X_train, y_train, verbose=1)
        preds = models_obj.predict_lstm(X_test, y_len=y_test.shape[1])

    elif model_name == "cnn":
        model, _ = models_obj.train_cnn(X_train, y_train, verbose=1)
        preds = models_obj.predict_cnn(X_test)

    elif model_name == "linear":
        model = models_obj.train_linear(X_train, y_train)
        preds = models_obj.predict_linear(X_test)

    elif model_name == "knn":
        model = models_obj.train_knn(X_train, y_train, n_neighbors=5)
        preds = models_obj.predict_knn(X_test)

    else:
        continue

    # Metrics
    ade, fde = ade_fde(preds, y_test)
    acc = evaluate_decisions(y_test, preds)
    print(f"{model_name.upper()} results: ADE={ade:.4f}, FDE={fde:.4f}, ACC={acc:.3f}")

    # Save results
    np.save(os.path.join(RESULTS_DIR, f"preds_{model_name}.npy"), preds)
    metrics = {"ADE": float(ade), "FDE": float(fde), "Accuracy": float(acc)}
    with open(os.path.join(RESULTS_DIR, f"metrics_{model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Trajectory plots
    plot_trajectories(y_test, preds, model_name)

    results[model_name] = metrics


# ============================================
# STEP 5: SUMMARY
# ============================================
print("\n=== Summary of All Models ===")
for m, r in results.items():
    print(f"{m.upper()}: ADE={r['ADE']:.4f}, FDE={r['FDE']:.4f}, ACC={r['Accuracy']:.3f}")
