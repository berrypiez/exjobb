import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

# Keras LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout

from data_processing import load_master_json, pad_sequences, fit_scaler, transform_with_scaler, extract_features

# -------------------------
# CONFIG
# -------------------------
MASTER_JSON_PATH = "master_jsons/master_dataset_office_trimmed.json"
BASE_RESULTS_DIR = "results/classification/all_last_frames"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

SCENARIOS_TO_RUN = [
    "bbox_only",
    "bbox_skeleton",
    "bbox_skeleton_angles"
]

USE_REDUCED_SKELETON = True
LABEL_MAP = {"pass": 0, "enter": 1}
CLASS_NAMES = ["pass", "enter"]
TEST_SPLIT = 0.2
DEFAULT_LAST_K = 10  # default number of last frames

# -------------------------
# Helpers
# -------------------------
def plot_confusion(cm, classes, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

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

def extract_last_k_frames(master_json, scenario_flags, k):
    X_list, y_list = [], []
    for video_name, video_data in master_json.items():
        feats = extract_features(
            video_data,
            use_bbox=scenario_flags["use_bbox"],
            use_skeleton=scenario_flags["use_skeleton"],
            use_reduced_skeleton=scenario_flags["use_reduced_skeleton"],
            use_angles=scenario_flags["use_angles"]
        )
        if feats is None or len(feats) == 0:
            continue
        cut_len = min(k, feats.shape[0])
        X_list.append(feats[-cut_len:])
        y_list.append(LABEL_MAP[video_data["label"]])
    return X_list, y_list

# -------------------------
# MAIN
# -------------------------
def main(last_k=DEFAULT_LAST_K):
    master_json = load_master_json(MASTER_JSON_PATH)
    print(f"Loaded {len(master_json)} videos")

    for scenario in SCENARIOS_TO_RUN:
        flags = {
            "use_bbox": "bbox" in scenario,
            "use_skeleton": "skeleton" in scenario,
            "use_angles": "angles" in scenario,
            "use_reduced_skeleton": USE_REDUCED_SKELETON
        }

        print(f"\n=== SCENARIO: {scenario}, last_k={last_k} ===")
        X_seqs, y_list = extract_last_k_frames(master_json, flags, last_k)

        if len(X_seqs) == 0:
            print("No sequences found, skipping scenario")
            continue

        # Scale features across all frames
        scaler = fit_scaler(X_seqs)
        X_scaled = transform_with_scaler(X_seqs, scaler)

        # Pad sequences to uniform length (last_k frames)
        X_padded, lengths = pad_sequences(X_scaled)
        y_array = np.array(y_list, dtype=int)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y_array, test_size=TEST_SPLIT, stratify=y_array, random_state=42
        )

        run_dir = os.path.join(BASE_RESULTS_DIR, f"{scenario}_last{DEFAULT_LAST_K}_frames")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

        # ---------- KNN ----------
        n_train, t, f = X_train.shape
        X_train_flat = X_train.reshape(n_train, t*f)
        X_test_flat = X_test.reshape(X_test.shape[0], t*f)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_flat, y_train)
        y_pred_knn = knn.predict(X_test_flat)
        y_score_knn = knn.predict_proba(X_test_flat)[:,1]

        acc_knn = accuracy_score(y_test, y_pred_knn)
        f1_knn = f1_score(y_test, y_pred_knn)
        cm_knn = confusion_matrix(y_test, y_pred_knn)

        joblib.dump(knn, os.path.join(run_dir, "knn_model.pkl"))
        joblib.dump(scaler, os.path.join(run_dir, "scaler.pkl"))

        plot_confusion(cm_knn, CLASS_NAMES, os.path.join(run_dir, "plots", "knn_cm.png"))
        plot_roc(y_test, y_score_knn, os.path.join(run_dir, "plots", "knn_roc.png"))

        # ---------- LSTM ----------
        lstm_model = build_lstm_model(timesteps=X_train.shape[1], feat_dim=X_train.shape[2], hidden_units=64)
        lstm_model.fit(X_train, y_train, validation_split=0.15, epochs=30, batch_size=32, verbose=1)

        y_prob_lstm = lstm_model.predict(X_test).ravel()
        y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

        acc_lstm = accuracy_score(y_test, y_pred_lstm)
        f1_lstm = f1_score(y_test, y_pred_lstm)
        cm_lstm = confusion_matrix(y_test, y_pred_lstm)

        lstm_model.save(os.path.join(run_dir, "lstm_model.h5"))

        plot_confusion(cm_lstm, CLASS_NAMES, os.path.join(run_dir, "plots", "lstm_cm.png"))
        plot_roc(y_test, y_prob_lstm, os.path.join(run_dir, "plots", "lstm_roc.png"))

        # Save metrics
        metrics = {
            "knn": {"accuracy": float(acc_knn), "f1": float(f1_knn), "confusion_matrix": cm_knn.tolist()},
            "lstm": {"accuracy": float(acc_lstm), "f1": float(f1_lstm), "confusion_matrix": cm_lstm.tolist()},
            "n_samples": int(len(y_array))
        }

        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved models and metrics to {run_dir}")
        print(f"KNN ACC={acc_knn:.3f}, F1={f1_knn:.3f} | LSTM ACC={acc_lstm:.3f}, F1={f1_lstm:.3f}")

if __name__ == "__main__":
    main(last_k=10)  # you can change 10 to any other number of last frames
