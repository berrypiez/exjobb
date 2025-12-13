import os
import json
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

from data_processing import load_master_json, extract_features

# --------------------
# CONFIG
# --------------------
MASTER_JSON_PATH = "master_jsons/master_dataset_office_trimmed.json"
BASE_RESULTS_DIR = "results/classification/late_stage"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

SCENARIOS = ["bbox_only", "bbox_skeleton", "bbox_skeleton_angles"]

LABEL_MAP = {"pass": 0, "enter": 1}
CLASS_NAMES = ["pass", "enter"]

TRIM_LEN = 10       # keep only last 10 frames
SEQ_LEN = 10        # classifier input length (matches predictor output)
TEST_SPLIT = 0.2

USE_REDUCED_SKELETON = True


# ------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------
def plot_confusion(cm, classes, out_path):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(out_path)
    plt.close()


def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    r = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={r:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------
# Build late-stage dataset (ONE SAMPLE PER VIDEO)
# ------------------------------------------------------
def build_late_dataset(master_json, scenario_flags):
    X_list = []
    y_list = []

    for name, video in master_json.items():
        feats = extract_features(video,
                                 use_bbox=scenario_flags["use_bbox"],
                                 use_skeleton=scenario_flags["use_skeleton"],
                                 use_reduced_skeleton=scenario_flags["use_reduced_skeleton"],
                                 use_angles=scenario_flags["use_angles"])

        if feats is None or len(feats) < SEQ_LEN:
            continue

        # keep only last TRIM_LEN frames
        trimmed = feats[-TRIM_LEN:]

        if trimmed.shape[0] < SEQ_LEN:
            continue

        # classifier input = last 10 frames
        seq_input = trimmed[-SEQ_LEN:]      # shape (10, F)

        X_list.append(seq_input)
        y_list.append(LABEL_MAP[video.get("label")])

    X = np.array(X_list)    # shape (N, 10, F)
    y = np.array(y_list)

    return X, y


# ------------------------------------------------------
# LSTM model
# ------------------------------------------------------
def build_lstm_model(seq_len, feat_dim, hidden=64):
    model = Sequential([
        Masking(mask_value=0., input_shape=(seq_len, feat_dim)),
        LSTM(hidden, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    master_json = load_master_json(MASTER_JSON_PATH)
    print(f"Loaded {len(master_json)} videos")

    for scenario in SCENARIOS:
        print("\n=============================")
        print(f" TRAINING SCENARIO: {scenario}")
        print("=============================\n")

        flags = {
            "use_bbox": "bbox" in scenario,
            "use_skeleton": "skeleton" in scenario,
            "use_angles": "angles" in scenario,
            "use_reduced_skeleton": USE_REDUCED_SKELETON
        }

        # ------------------------
        # Build dataset
        # ------------------------
        X, y = build_late_dataset(master_json, flags)
        if len(X) == 0:
            print(f"Skipping scenario {scenario}: no usable samples")
            continue

        N, L, F = X.shape
        print(f"Dataset: {scenario} → {X.shape}")

        # Flatten for KNN/scaler
        X_flat = X.reshape(N, L * F)

        # ------------------------
        # Split
        # ------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y, test_size=TEST_SPLIT, stratify=y, random_state=42
        )

        # ------------------------
        # Scaling
        # ------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler
        scenario_dir = os.path.join(BASE_RESULTS_DIR, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        os.makedirs(os.path.join(scenario_dir, "plots"), exist_ok=True)
        joblib.dump(scaler, os.path.join(scenario_dir, "scaler.pkl"))

        # ------------------------
        # Train KNN
        # ------------------------
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
        y_pred_knn = (y_prob_knn >= 0.5).astype(int)

        acc_knn = accuracy_score(y_test, y_pred_knn)
        f1_knn  = f1_score(y_test, y_pred_knn)
        cm_knn  = confusion_matrix(y_test, y_pred_knn)

        joblib.dump(knn, os.path.join(scenario_dir, "knn.pkl"))
        plot_confusion(cm_knn, CLASS_NAMES, os.path.join(scenario_dir, "plots", "knn_cm.png"))
        plot_roc(y_test, y_prob_knn, os.path.join(scenario_dir, "plots", "knn_roc.png"))

        # ------------------------
        # Train LSTM (sequence input)
        # ------------------------
        # reshape scaled back to sequence
        X_train_seq = X_train_scaled.reshape(-1, L, F)
        X_test_seq  = X_test_scaled.reshape(-1, L, F)

        lstm = build_lstm_model(L, F)
        lstm.fit(X_train_seq, y_train, epochs=20, batch_size=32,
                 validation_split=0.15, verbose=0)

        y_prob_lstm = lstm.predict(X_test_seq).ravel()
        y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

        acc_lstm = accuracy_score(y_test, y_pred_lstm)
        f1_lstm  = f1_score(y_test, y_pred_lstm)
        cm_lstm  = confusion_matrix(y_test, y_pred_lstm)

        lstm.save(os.path.join(scenario_dir, "lstm.h5"))
        plot_confusion(cm_lstm, CLASS_NAMES, os.path.join(scenario_dir, "plots", "lstm_cm.png"))
        plot_roc(y_test, y_prob_lstm, os.path.join(scenario_dir, "plots", "lstm_roc.png"))

        # ------------------------
        # Save metrics
        # ------------------------
        metrics = {
            "knn": {
                "accuracy": float(acc_knn),
                "f1": float(f1_knn),
                "confusion": cm_knn.tolist()
            },
            "lstm": {
                "accuracy": float(acc_lstm),
                "f1": float(f1_lstm),
                "confusion": cm_lstm.tolist()
            },
            "dataset_shape": [N, L, F]
        }

        with open(os.path.join(scenario_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Finished {scenario}:")
        print(f"  KNN  → acc {acc_knn:.3f}, f1 {f1_knn:.3f}")
        print(f"  LSTM → acc {acc_lstm:.3f}, f1 {f1_lstm:.3f}")


if __name__ == "__main__":
    main()
