import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

from tensorflow.keras.models import load_model
from data_processing import load_master_json, extract_features, pad_sequences

# -------------------------
# CONFIG
# -------------------------
MASTER_JSON_PATH = "master_jsons/master_dataset_office_trimmed.json"
RESULTS_DIR = "results/hybrid_classification"
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = ["bbox_only", "bbox_skeleton", "bbox_skeleton_angles"]
CLASS_NAMES = ["pass", "enter"]
LABEL_MAP = {"pass": 0, "enter": 1}

# number of last frames to use
LAST_K_FRAMES = 10  # adjustable

# paths to last-frame classifiers
BASE_LAST_FRAME_PATH = "results/classification/all_last_frames"

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def plot_confusion(cm, classes, out_path):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,cm[i,j],ha='center',va='center',
                     color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout(); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.savefig(out_path)
    plt.close()

def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    r = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr,tpr,label=f"AUC={r:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def load_models(scenario, last_k=LAST_K_FRAMES):
    """Load KNN and LSTM models trained on last-k frames for a scenario"""
    model_dir = os.path.join(BASE_LAST_FRAME_PATH, f"{scenario}_last{last_k}_frames")
    knn_model = joblib.load(os.path.join(model_dir, "knn_model.pkl"))
    lstm_model = load_model(os.path.join(model_dir, "lstm_model.h5"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return knn_model, lstm_model, scaler

# -------------------------
# MAIN
# -------------------------
def main():
    master_json = load_master_json(MASTER_JSON_PATH)
    print(f"Loaded {len(master_json)} videos")

    all_results = {}

    for scenario in SCENARIOS:
        print(f"\n=== Hybrid Evaluation: {scenario} ===")
        # load last-k-frame models
        knn_trained, lstm_trained, scaler = load_models(scenario, LAST_K_FRAMES)

        # Prepare features
        X_list, y_list = [], []
        for vid_name, vid_data in master_json.items():
            feats = extract_features(
                vid_data,
                use_bbox="bbox" in scenario,
                use_skeleton="skeleton" in scenario,
                use_reduced_skeleton=True,
                use_angles="angles" in scenario
            )
            if feats is None or len(feats) == 0:
                continue
            last_feats = feats[-LAST_K_FRAMES:] if feats.shape[0] >= LAST_K_FRAMES else feats
            X_list.append(last_feats)
            y_list.append(LABEL_MAP[vid_data.get("label")])

        if len(X_list) == 0:
            print(f"No samples for {scenario}, skipping")
            continue

        # Scaling
        X_scaled_list = [scaler.transform(x) for x in X_list]
        X_padded, _ = pad_sequences(X_scaled_list)
        y_array = np.array(y_list)

        n_samples, t, f = X_padded.shape
        X_flat = X_padded.reshape(n_samples, t*f)

        # -------------------------
        # CROSS-EVALUATION
        # -------------------------
        results = {}

        # KNN-trained model
        y_pred_knn_knn = knn_trained.predict(X_flat)
        y_prob_knn_knn = knn_trained.predict_proba(X_flat)[:,1]

        y_prob_knn_lstm = lstm_trained.predict(X_padded).ravel()
        y_pred_knn_lstm = (y_prob_knn_lstm >= 0.5).astype(int)

        # LSTM-trained model
        y_pred_lstm_knn = knn_trained.predict(X_flat)
        y_prob_lstm_knn = knn_trained.predict_proba(X_flat)[:,1]

        y_prob_lstm_lstm = lstm_trained.predict(X_padded).ravel()
        y_pred_lstm_lstm = (y_prob_lstm_lstm >= 0.5).astype(int)

        eval_sets = {
            "KNN_model_vs_KNN": (y_array, y_pred_knn_knn, y_prob_knn_knn),
            "KNN_model_vs_LSTM": (y_array, y_pred_knn_lstm, y_prob_knn_lstm),
            "LSTM_model_vs_KNN": (y_array, y_pred_lstm_knn, y_prob_lstm_knn),
            "LSTM_model_vs_LSTM": (y_array, y_pred_lstm_lstm, y_prob_lstm_lstm)
        }

        for key, (y_true, y_pred, y_prob) in eval_sets.items():
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            print(f"{key}: ACC={acc:.3f}, F1={f1:.3f}")
            results[key] = {
                "accuracy": float(acc),
                "f1": float(f1),
                "confusion_matrix": cm.tolist()
            }

            # plots
            plot_confusion(cm, CLASS_NAMES, os.path.join(RESULTS_DIR, f"{scenario}_{key}_cm.png"))
            plot_roc(y_true, y_prob, os.path.join(RESULTS_DIR, f"{scenario}_{key}_roc.png"))

        all_results[scenario] = results

    # Save summary
    with open(os.path.join(RESULTS_DIR, "hybrid_full_cross_eval.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Hybrid cross-evaluation done! Metrics saved ===")

if __name__ == "__main__":
    main()
