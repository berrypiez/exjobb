
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

from data_processing import load_master_json, extract_features
from sklearn.preprocessing import StandardScaler

MASTER_JSON_PATH = "master_jsons/master_dataset_office_trimmed.json"
BASE_RESULTS_DIR = "results/classification/early_frame"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

MODE = "trim"  # not used here except optionally for filtering; kept for consistency
K_VALUES = [0, 5, 10, 15, 20, 25, 30]   # 0 => last frame
TEST_SPLIT = 0.2

SCENARIOS_TO_RUN = [
    "bbox_only",
    "bbox_skeleton",
    "bbox_skeleton_angles"
]

LABEL_MAP = {"pass": 0, "enter": 1}
CLASS_NAMES = ["pass", "enter"]
USE_REDUCED_SKELETON = True

# -------------------------
# plotting helpers
# -------------------------
def plot_confusion(cm, classes, out_path):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest'); plt.colorbar()
    ticks = np.arange(len(classes)); plt.xticks(ticks, classes); plt.yticks(ticks, classes)
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,cm[i,j],ha='center',va='center',
                     color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout(); plt.ylabel("True"); plt.xlabel("Predicted"); plt.savefig(out_path); plt.close()

def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    r = auc(fpr, tpr)
    plt.figure(figsize=(5,4)); plt.plot(fpr,tpr,label=f"AUC={r:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

# -------------------------
# simple LSTM builder for sequence len=1
# -------------------------
def build_lstm_single_frame(feat_dim, hidden_units=32, dropout=0.2):
    model = Sequential([
        Masking(mask_value=0., input_shape=(1, feat_dim)),
        LSTM(hidden_units, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# build dataset for a given offset k
# -------------------------
def build_early_frame_dataset(master_json, scenario_config, k):
    X_list, y_list = [], []
    for name, video in master_json.items():
        feats = extract_features(video,
                                 use_bbox=scenario_config["use_bbox"],
                                 use_skeleton=scenario_config["use_skeleton"],
                                 use_reduced_skeleton=scenario_config["use_reduced_skeleton"],
                                 use_angles=scenario_config["use_angles"])
        if feats is None or len(feats) == 0:
            continue
        idx = feats.shape[0] - 1 - k
        if idx < 0:
            continue
        X_list.append(feats[idx])
        y_list.append(LABEL_MAP[video.get("label")])
    if len(X_list) == 0:
        return np.zeros((0,0)), np.zeros((0,), dtype=int)
    return np.vstack(X_list), np.array(y_list, dtype=int)

# -------------------------
# MAIN
# -------------------------
def main():
    master_json = load_master_json(MASTER_JSON_PATH)
    print(f"Loaded {len(master_json)} videos")

    for scenario in SCENARIOS_TO_RUN:
        flags = {
            "use_bbox": "bbox" in scenario,
            "use_skeleton": "skeleton" in scenario,
            "use_angles": "angles" in scenario,
            "use_reduced_skeleton": USE_REDUCED_SKELETON
        }

        accs_knn = []
        accs_lstm = []
        ks_done = []

        for k in K_VALUES:
            run_dir = os.path.join(BASE_RESULTS_DIR, f"{scenario}_k{k}")
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

            X, y = build_early_frame_dataset(master_json, flags, k)
            if X.size == 0:
                print(f"Scenario {scenario}, k={k}: no samples (skipping)")
                continue

            # Scale (flat)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            joblib.dump(scaler, os.path.join(run_dir, "scaler.pkl"))

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SPLIT, stratify=y, random_state=42)
            # ---------- KNN ----------
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)
            y_pred_knn = knn.predict(X_test)
            y_prob_knn = knn.predict_proba(X_test)[:,1]

            acc_knn = accuracy_score(y_test, y_pred_knn)
            f1_knn = f1_score(y_test, y_pred_knn)
            cm_knn = confusion_matrix(y_test, y_pred_knn)

            joblib.dump(knn, os.path.join(run_dir, "knn.pkl"))
            plot_confusion(cm_knn, CLASS_NAMES, os.path.join(run_dir, "plots", "knn_cm.png"))
            plot_roc(y_test, y_prob_knn, os.path.join(run_dir, "plots", "knn_roc.png"))

            # ---------- LSTM (single-frame as sequence length 1) ----------
            feat_dim = X_train.shape[1]
            X_train_seq = X_train.reshape((X_train.shape[0], 1, feat_dim))
            X_test_seq = X_test.reshape((X_test.shape[0], 1, feat_dim))

            lstm = build_lstm_single_frame(feat_dim, hidden_units=32)
            # train lightly
            lstm.fit(X_train_seq, y_train, validation_split=0.15, epochs=15, batch_size=32, verbose=0)
            y_prob_lstm = lstm.predict(X_test_seq).ravel()
            y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

            acc_lstm = accuracy_score(y_test, y_pred_lstm)
            f1_lstm = f1_score(y_test, y_pred_lstm)
            cm_lstm = confusion_matrix(y_test, y_pred_lstm)

            lstm.save(os.path.join(run_dir, "lstm.h5"))
            plot_confusion(cm_lstm, CLASS_NAMES, os.path.join(run_dir, "plots", "lstm_cm.png"))
            plot_roc(y_test, y_prob_lstm, os.path.join(run_dir, "plots", "lstm_roc.png"))

            # Save per-k metrics
            metrics = {
                "knn": {"accuracy": float(acc_knn), "f1": float(f1_knn), "confusion": cm_knn.tolist()},
                "lstm": {"accuracy": float(acc_lstm), "f1": float(f1_lstm), "confusion": cm_lstm.tolist()},
                "n_samples": int(len(y))
            }
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"{scenario} k={k}: KNN acc {acc_knn:.3f}, LSTM acc {acc_lstm:.3f}")

            accs_knn.append(acc_knn)
            accs_lstm.append(acc_lstm)
            ks_done.append(k)

        # Plot accuracy vs k
        if len(ks_done) > 0:
            plt.figure(figsize=(6,4))
            plt.plot(ks_done, accs_knn, 'o-', label='KNN')
            plt.plot(ks_done, accs_lstm, 's-', label='LSTM')
            plt.xlabel("Frames before end (k)")
            plt.ylabel("Accuracy")
            plt.title(f"Early classification accuracy ({scenario})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_RESULTS_DIR, f"{scenario}_accuracy_vs_k.png"))
            plt.close()

if __name__ == "__main__":
    main()