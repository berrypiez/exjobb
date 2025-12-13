import os
import json
import numpy as np
import matplotlib.pyplot as plt
from models import Models, ade_fde
from data_processing import load_master_json, build_dataset, train_test_split_dataset, pad_sequences, fit_scaler, transform_with_scaler
from pathways import pathways
import joblib


# ===========================
# CONFIGURATION
# ===========================
MASTER_JSON_PATH = "C:/Users/hanna/Documents/Thesis/exjobb/TP_MKI/master_jsons/home_master_dataset.json"
RESULTS_DIR = "results/home_models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

TEST_SIZE = 0.2
N_PLOT_SAMPLES = 3

SCENARIOS_TO_RUN = [
    "bbox_only",
    "skeleton_only",
    "angles_only",
    "bbox_skeleton",
    "bbox_angles",
    "skeleton_angles",
    "bbox_skeleton_angles"
]

MODELS_TO_RUN = [
    "knn",
    "linear",
    "cnn", 
    "lstm"
]

USE_REDUCED_SKELETON = True

# ===========================
# HELPER FUNCTIONS
# ===========================

def scenario_flags(name):
    """Helper to parse scenario composition."""
    return {
        "use_bbox": "bbox" in name,
        "use_skeleton": "skeleton" in name,
        "use_angles": "angles" in name,
        "use_reduced_skeleton": USE_REDUCED_SKELETON
    }

def decide_open_door(trajectory, door_x_range=(0.25, 0.75), door_y_threshold=0.9):
    """
    Decide if the person went through the door based on their final (x, y) position.
    Expects trajectory shape (T, 2) in **original coordinates** (not scaled).
    Door = bottom 10% of frame, center 50% horizontally (these thresholds presume normalized coordinates
    or that you interpret them correctly for your frame coords).
    """
    if trajectory is None:
        return False
    if trajectory.ndim != 2 or trajectory.shape[1] < 2 or trajectory.shape[0] == 0:
        return False
    final_x, final_y = trajectory[-1, 0], trajectory[-1, 1]
    in_door_x = door_x_range[0] <= final_x <= door_x_range[1]
    in_door_y = final_y >= door_y_threshold
    return in_door_x and in_door_y

# def decide_open_door(trajectory, door_x_range=(0.25, 0.75), door_y_threshold=0.9):
#     """
#     Decide if the person went through the door based on their final (x, y) position.
#     Door = bottom 10% of frame, center 50% horizontally.
#     """
#     if trajectory.ndim != 2 or trajectory.shape[1] < 2:
#         return False
#     final_x, final_y = trajectory[-1, 0], trajectory[-1, 1]
#     in_door_x = door_x_range[0] <= final_x <= door_x_range[1]
#     in_door_y = final_y >= door_y_threshold
#     return in_door_x and in_door_y

def evaluate_decisions(y_true, y_pred):
    """
    y_true, y_pred: arrays or lists of (T,2) unscaled center sequences.
    Returns accuracy (fraction where door decision matches).
    """
    if len(y_true) == 0:
        return 0.0
    correct = 0
    for t, p in zip(y_true, y_pred):
        if decide_open_door(np.array(t)) == decide_open_door(np.array(p)):
            correct += 1
    return correct / len(y_true)
# def evaluate_decisions(y_true, y_pred):
#     """
#     Compute accuracy based on whether predicted trajectories pass through the door.
#     """
#     if len(y_true) == 0:
#         return 0.0
#     correct = sum(decide_open_door(t) == decide_open_door(p) for t, p in zip(y_true, y_pred))
#     return correct / len(y_true)

def compute_centroid_ade_fde(preds, targets):
    """
    preds, targets: arrays/lists of shape (N, T, 2) (unscaled coordinates).
    Returns mean ADE and mean FDE.
    """
    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    N = len(preds_arr)
    ade_list, fde_list = [], []
    for i in range(N):
        p = preds_arr[i]
        t = targets_arr[i]
        min_len = min(len(p), len(t))
        if min_len == 0:
            continue
        dist = np.linalg.norm(p[:min_len] - t[:min_len], axis=1)
        ade_list.append(np.mean(dist))
        fde_list.append(dist[-1])
    if len(ade_list) == 0:
        return 0.0, 0.0
    return float(np.mean(ade_list)), float(np.mean(fde_list))

def plot_trajectories(y_true, y_pred, model_name, scenario, n_samples=N_PLOT_SAMPLES):
    """Visualize prediction vs ground truth for qualitative check."""
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


# ===========================
# MAIN TRAINING LOOP
# ===========================
def main():
    print("Loading master dataset...")
    master_json = load_master_json(MASTER_JSON_PATH)
    print(f"Loaded {len(master_json)} videos from master_json.\n")

    all_results = {}

    for SCENARIO in SCENARIOS_TO_RUN:
        print(f"\n=== Running scenario: {SCENARIO} ===")
        flags = scenario_flags(SCENARIO)
        history = None

        # Build dataset
        X, y, labels = build_dataset(master_json, flags)
        print(f"Built dataset for {SCENARIO}: X, y={[len(X), len(y)]}, labels={len(labels)}")

        # Scale
        scaler = fit_scaler(X)
        X_scaled = transform_with_scaler(X, scaler)
        y_scaled = transform_with_scaler(y, scaler)

        # Pad sequences
        X_padded, _ = pad_sequences(X_scaled)
        y_padded, _ = pad_sequences(y_scaled)

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split_dataset(X_padded, y_padded, test_size=TEST_SIZE)
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

        scaler_path = os.path.join(RESULTS_DIR, f"{SCENARIO}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        # Initialize models
        models_obj = Models(feature_size=X_train.shape[2])
        models_obj.reset_models()
        results = {}

        for model_name in MODELS_TO_RUN:
            print(f"\n=== Training {model_name.upper()} ({SCENARIO}) ===")

            if model_name == "lstm":
                model, history = models_obj.train_lstm(X_train, y_train, verbose=1)
                preds = models_obj.predict_lstm(X_test, y_len=y_test.shape[1])
                model.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_model_lstm.h5"), save_format="h5")
            elif model_name == "cnn":
                model, history = models_obj.train_cnn(X_train, y_train, verbose=1)
                preds = models_obj.predict_cnn(X_test)
                model.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_model_cnn.h5"), save_format="h5")
            elif model_name == "linear":
                model = models_obj.train_linear(X_train, y_train)
                preds = models_obj.predict_linear(X_test)
                joblib.dump(model, os.path.join(RESULTS_DIR, f"{SCENARIO}_model_linear.pkl"))
            elif model_name == "knn":
                model = models_obj.train_knn(X_train, y_train, n_neighbors=5)
                preds = models_obj.predict_knn(X_test)
                joblib.dump(model, os.path.join(RESULTS_DIR, f"{SCENARIO}_model_knn.pkl"))
            else:
                continue

            ade, fde = ade_fde(preds, y_test)
            acc = evaluate_decisions(y_test, preds)
            print(f"{model_name.upper()} ({SCENARIO}) results: ADE={ade:.4f}, FDE={fde:.4f}, ACC={acc:.3f}")

            # Save predictions + metrics
            np.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_preds_{model_name}.npy"), preds)
            metrics_dict = {"ADE": float(ade), "FDE": float(fde), "Accuracy": float(acc)}

            # Add loss history if available
            if history is not None:
                metrics_dict["loss"] = [float(l) for l in history.history["loss"]]
                if "val_loss" in history.history:
                    metrics_dict["val_loss"] = [float(l) for l in history.history["val_loss"]]
            
            with open(os.path.join(RESULTS_DIR, f"{SCENARIO}_metrics_{model_name}.json"), "w") as f:
                json.dump(metrics_dict, f, indent=2)

            plot_trajectories(y_test, preds, model_name, SCENARIO)
            results[model_name] = metrics_dict

        all_results[SCENARIO] = results

    # Summary
    print("\n=== Summary of All Scenarios ===")
    for sc, res in all_results.items():
        print(f"\n--- {sc.upper()} ---")
        for m, r in res.items():
            print(f"{m.upper()}: ADE={r['ADE']:.4f}, FDE={r['FDE']:.4f}, ACC={r['Accuracy']:.3f}")

    with open(os.path.join(RESULTS_DIR, "summary_all_scenarios.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
