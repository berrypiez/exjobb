import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_processing import load_master_json, build_dataset, fit_scaler, transform_with_scaler, pad_sequences
from models import Models, ade_fde

# Silence TensorFlow logging
import warnings
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ============================================
# CONFIG
# ============================================
MASTER_JSON_PATH = "master_json.json"
RESULTS_DIR = "results"

TAIL = 0.1
TEST_SIZE = 0.2
N_PLOT_SAMPLES = 3
DOOR_THRESHOLD_X = 0.8

SCENARIOS_TO_RUN = [
    "bbox_only"
]
# "skeleton_only", "angles_only", "bbox_skeleton", "bbox_angles",
# "skeleton_angles", "bbox_skeleton_angles"

MODELS_TO_RUN = ["knn"]  # "linear", "cnn", "lstm"
USE_REDUCED_SKELETON = True  # if you want to only use reduced landmarks

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
def scenario_flags(name):
    """Parse scenario composition."""
    return {
        "use_bbox": "bbox" in name,
        "use_skeleton": "skeleton" in name,
        "use_angles": "angles" in name
    }

def plot_trajectories(y_true, y_pred, model_name, scenario, n_samples=N_PLOT_SAMPLES):
    """Save side-by-side predicted vs ground truth trajectories."""
    cx, cy = 0, 1
    n_samples = min(n_samples, len(y_true))
    for i in range(n_samples):
        plt.figure(figsize=(4, 4))
        plt.plot(y_true[i, :, cx], y_true[i, :, cy], 'o-', label="GT")
        plt.plot(y_pred[i, :, cx], y_pred[i, :, cy], 'x--', label="Pred")
        plt.legend()
        plt.title(f"{model_name.upper()} ({scenario}) Sample {i+1}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, "plots", f"{scenario}_{model_name}_sample{i+1}.png")
        plt.savefig(save_path)
        plt.close()

def decide_open_door(trajectory, threshold=DOOR_THRESHOLD_X):
    """Simple decision: if final x > threshold, consider 'open door'."""
    final_x = trajectory[-1, 0]
    return final_x > threshold

def evaluate_decisions(y_true, y_pred):
    """Compare final x-positions as a binary decision metric."""
    true_decisions = [decide_open_door(t) for t in y_true]
    pred_decisions = [decide_open_door(p) for p in y_pred]
    correct = sum(t == p for t, p in zip(true_decisions, pred_decisions))
    return correct / len(y_true)

# ============================================
# MAIN LOOP
# ============================================
def main():
    print(f"\nLoading master JSON from {MASTER_JSON_PATH} ...")
    master_json = load_master_json(MASTER_JSON_PATH)

    all_results = {}

    for SCENARIO in SCENARIOS_TO_RUN:
        print(f"\n=== Running scenario: {SCENARIO} ===")

        flags = scenario_flags(SCENARIO)
        Xs, ys, labels = build_dataset(master_json, flags, use_reduced_skeleton=USE_REDUCED_SKELETON)

        # Debug print to confirm structure
        print(f"Dataset built for {SCENARIO}: {len(Xs)} sequences, sample length {len(Xs[0]) if Xs else 0}")

        if len(Xs) == 0:
            print("⚠️ No samples found for this scenario. Skipping.")
            continue

        # ============================================
        # SCALING + PADDING
        # ============================================
        print("Scaling and padding sequences...")
        scaler = fit_scaler(Xs)
        Xs_scaled = transform_with_scaler(Xs, scaler)
        ys_scaled = transform_with_scaler(ys, scaler)
        X_padded, _ = pad_sequences(Xs_scaled)
        y_padded, _ = pad_sequences(ys_scaled)

        # ============================================
        # DEBUG OUTPUT
        # ============================================
        print("\n--- Sample feature vector (first instance) ---")
        print("X[0]:", X_padded[0])
        print("y[0]:", y_padded[0])
        print("Feature vector length:", X_padded[0].shape[0])
        print("--------------------------------------------\n")

        np.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_X.npy"), X_padded)
        np.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_y.npy"), y_padded)
        np.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_labels.npy"), np.array(labels))

        # ============================================
        # TRAIN/TEST SPLIT
        # ============================================
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y_padded, test_size=TEST_SIZE, random_state=42
        )
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

        # ============================================
        # MODEL TRAINING
        # ============================================
        results = {}
        models_obj = Models(feature_size=X_train.shape[2])
        models_obj.reset_models()

        for model_name in MODELS_TO_RUN:
            print(f"\n=== Training {model_name.upper()} ({SCENARIO}) ===")

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

            ade, fde = ade_fde(preds, y_test)
            acc = evaluate_decisions(y_test, preds)
            print(f"{model_name.upper()} ({SCENARIO}) results: ADE={ade:.4f}, FDE={fde:.4f}, ACC={acc:.3f}")

            # Save results
            np.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_preds_{model_name}.npy"), preds)
            with open(os.path.join(RESULTS_DIR, f"{SCENARIO}_metrics_{model_name}.json"), "w") as f:
                json.dump({"ADE": float(ade), "FDE": float(fde), "Accuracy": float(acc)}, f, indent=2)

            plot_trajectories(y_test, preds, model_name, SCENARIO)
            results[model_name] = {"ADE": ade, "FDE": fde, "Accuracy": acc}

        all_results[SCENARIO] = results

    # ============================================
    # SUMMARY
    # ============================================
    print("\n=== Summary of All Scenarios ===")
    for sc, res in all_results.items():
        print(f"\n--- {sc.upper()} ---")
        for m, r in res.items():
            print(f"{m.upper()}: ADE={r['ADE']:.4f}, FDE={r['FDE']:.4f}, ACC={r['Accuracy']:.3f}")

    with open(os.path.join(RESULTS_DIR, "summary_all_scenarios.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
