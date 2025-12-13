import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

# Keras LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_processing import load_master_json, build_dataset, pad_sequences, fit_scaler, transform_with_scaler

# -------------------------
# CONFIG
# -------------------------
MASTER_JSON_PATH = "master_jsons/master_dataset_office_trimmed.json"
BASE_RESULTS_DIR = "results/classification/multi_frame"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

MODE = "sliding" # sliding or trim
UPPER_FRAME_LIMITS = [75]
WINDOW_SIZE = 30
STRIDE = 15
TEST_SPLIT = 0.2

SCENARIOS_TO_RUN =[
    "bbox_only",
    "bbox_skeleton",
    "bbox_skeleton_angles"
]

LABEL_MAP = {"pass": 0, "enter": 1}
CLASS_NAMES = ["pass", "enter"]

# -------------------------
# Helpers (sliding/trim)
# -------------------------
def create_sliding_window(X_seqs, labels, window_size=30, stride=15):
    X_win, y_win = [], []
    for seq, lab in zip(X_seqs, labels):
        n = seq.shape[0]
        if n < window_size:
            continue
        for start in range(0, n-window_size+1, stride):
            X_win.append(seq[start:start+window_size])
            y_win.append(LABEL_MAP[lab])
    return X_win, y_win

def trim_sequences(X_seqs, labels, upper_frame_limit=30):
    X_trimmed, y_trimmed = [], []
    for seq, lab in zip(X_seqs, labels):
        if seq.shape[0] == 0:
            continue
        cut_len = min(seq.shape[0], upper_frame_limit)
        X_trimmed.append(seq[:cut_len])
        y_trimmed.append(LABEL_MAP[lab])
    return X_trimmed, y_trimmed

# -------------------------
# Plots
# -------------------------
def plot_confusion(cm, classes, out_path):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha='center', va='center',
                     color='white' if cm[i,j] > thresh else 'black')
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    r = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={r:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -------------------------
# LSTM builder
# ------------ -------------
def build_lstm_model(timesteps, feat_dim, hidden_units=64, dropout=0.2):
    model = Sequential([
        Masking(mask_value=0., input_shape=(timesteps, feat_dim)),
        LSTM(hidden_units, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -------------------------
# Main
# -------------------------
def main():
    master_json = load_master_json(MASTER_JSON_PATH)
    print(f"Loaded {len(master_json)} videos")

    for scenario in SCENARIOS_TO_RUN:
        flags = {
            "use_bbox": "bbox" in scenario,
            "use_skeleton": "skeleton" in scenario,
            "use_angles": "angles" in scenario,
            "use_reduced_skeleton": True
        }
        for upper_limit in UPPER_FRAME_LIMITS:
            run_name = f"{scenario}_frames(upper_list)_{MODE}_w{WINDOW_SIZE}_s{STRIDE}"
            run_dir = os.path.join(BASE_RESULTS_DIR, run_name)
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
            print(f"\n=== {run_name} ===")

            X_seqs, _, video_labels = build_dataset(master_json, flags)
            if MODE == "sliding":
                X_proc, y_proc = create_sliding_window(X_seqs, video_labels, window_size=WINDOW_SIZE, stride=STRIDE)
            else:
                X_proc, y_proc = trim_sequences(X_seqs, video_labels, upper_frame_limit=upper_limit)

            if len(X_proc) == 0:
                print("No sequences after processing, skipping")
                continue
            
            # Scaling: X_proc is list of (T, F) arrays
            scaler = fit_scaler(X_proc)
            X_scaled_list = transform_with_scaler(X_proc, scaler)

            # Pad sequences to uniform length (T_max)
            X_padded, lengths = pad_sequences(X_scaled_list)
            y_array = np.array(y_proc, dtype=int)

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X_padded, y_array, test_size=TEST_SPLIT, random_state=42, stratify=y_array)

            # ---------- KNN ----------
            # Flatten time+features for KNN
            n_train, t, f = X_train.shape
            X_train_flat = X_train.reshape(n_train, t * f)
            n_test = X_test.shape[0]
            X_test_flat = X_test.reshape(n_test, t * f)

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_flat, y_train)
            y_pred_knn = knn.predict(X_test_flat)
            y_score_knn = knn.predict_proba(X_test_flat)[:,1]

            acc_knn = accuracy_score(y_test, y_pred_knn)
            f1_knn = f1_score(y_test, y_pred_knn)
            cm_knn = confusion_matrix(y_test, y_pred_knn)

            joblib.dump(knn, os.path.join(run_dir, f"knn_model.pkl"))
            joblib.dump(scaler, os.path.join(run_dir, f"scaler.pkl"))

            plot_confusion(cm_knn, CLASS_NAMES, os.path.join(run_dir, "plots", "knn_cm.png"))
            plot_roc(y_test, y_score_knn, os.path.join(run_dir, "plots", "knn_roc.png"))

            # ---------- LSTM ----------
            lstm_model = build_lstm_model(timesteps=X_train.shape[1], feat_dim=X_train.shape[2], hidden_units=64)
            es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
            history = lstm_model.fit(X_train, y_train, validation_split=0.15, epochs=30, batch_size=32, callbacks=[es], verbose=1)

            y_prob_lstm = lstm_model.predict(X_test).ravel()
            y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

            acc_lstm = accuracy_score(y_test, y_pred_lstm)
            f1_lstm = f1_score(y_test, y_pred_lstm)
            cm_lstm = confusion_matrix(y_test, y_pred_lstm)

            lstm_model.save(os.path.join(run_dir, "lstm_model.h5"))

            plot_confusion(cm_lstm, CLASS_NAMES, os.path.join(run_dir, "plots", "lstm_cm.png"))
            plot_roc(y_test, y_prob_lstm, os.path.join(run_dir, "plots", "lstm_roc.png"))

            metrics = {
                "knn": {"accuracy": float(acc_knn), "f1": float(f1_knn), "confusion": cm_knn.tolist()},
                "lstm": {"accuracy": float(acc_lstm), "f1": float(f1_lstm), "confusion": cm_lstm.tolist()}
            }

            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"Saved run to {run_dir} - KNN acc {acc_knn:.3f}, LSTM acc {acc_lstm:.3f}")
        
    
if __name__ == "__main__":
    main()