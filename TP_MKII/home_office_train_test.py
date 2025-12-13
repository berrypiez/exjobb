import os
import json
import numpy as np
import matplotlib.pyplot as plt
from models import Models, ade_fde
from data_processing import (
    load_master_json,
    build_dataset,
    train_test_split_dataset,
    pad_sequences,
    fit_scaler,
    transform_with_scaler
)
import joblib

# ===========================
# CONFIGURATION
# ===========================
HOME_JSON_PATH = "master_jsons/home_master_dataset.json"
OFFICE_JSON_PATH = "master_jsons/office_subset_for_home_test_trimmed.json"

RESULTS_DIR = "results/home_office_models_II"
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
    """Decide if the person went through the door based on final (x, y)."""
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        return False
    final_x, final_y = trajectory[-1, 0], trajectory[-1, 1]
    in_door_x = door_x_range[0] <= final_x <= door_x_range[1]
    in_door_y = final_y >= door_y_threshold
    return in_door_x and in_door_y


def evaluate_decisions(y_true, y_pred):
    true_decisions = [decide_open_door(t) for t in y_true]
    pred_decisions = [decide_open_door(p) for p in y_pred]
    correct = sum(t == p for t, p in zip(true_decisions, pred_decisions))
    return correct / len(y_true) if len(y_true) > 0 else 0.0


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


def match_sequence_length(X, expected_length):
    """Trim or pad sequences to match expected length."""
    current_length = X.shape[1]
    if current_length > expected_length:
        X = X[:, :expected_length, :]
    elif current_length < expected_length:
        pad_width = expected_length - current_length
        X = np.pad(X, ((0, 0), (0, pad_width), (0, 0)), mode="constant")
    return X


# ===========================
# MAIN TRAIN + OFFICE TEST LOOP
# ===========================
def main():
    print("Loading home dataset...")
    home_json = load_master_json(HOME_JSON_PATH)
    print(f"Loaded {len(home_json)} home videos.\n")

    print("Loading office dataset...")
    office_json = load_master_json(OFFICE_JSON_PATH)
    print(f"Loaded {len(office_json)} office videos.\n")

    all_results = {}

    for SCENARIO in SCENARIOS_TO_RUN:
        print(f"\n=== Scenario: {SCENARIO} ===")
        flags = scenario_flags(SCENARIO)
        history = None

        # -------------------
        # Home dataset
        # -------------------
        X_home, y_home, labels_home = build_dataset(home_json, flags)
        scaler = fit_scaler(X_home)
        X_home_scaled = transform_with_scaler(X_home, scaler)
        y_home_scaled = transform_with_scaler(y_home, scaler)
        X_home_padded, _ = pad_sequences(X_home_scaled)
        y_home_padded, _ = pad_sequences(y_home_scaled)
        X_train, X_test_home, y_train, y_test_home = train_test_split_dataset(X_home_padded, y_home_padded, test_size=TEST_SIZE)

        # Save scaler
        scaler_path = os.path.join(RESULTS_DIR, f"{SCENARIO}_scaler.pkl")
        joblib.dump(scaler, scaler_path)

        # -------------------
        # Office dataset
        # -------------------
        X_office, y_office, labels_office = build_dataset(office_json, flags)
        X_office_scaled = transform_with_scaler(X_office, scaler)
        y_office_scaled = transform_with_scaler(y_office, scaler)
        X_office_padded, _ = pad_sequences(X_office_scaled)
        y_office_padded, _ = pad_sequences(y_office_scaled)

        # Align office sequences to home X_test shape
        expected_length = X_test_home.shape[1]
        X_office_adj = match_sequence_length(X_office_padded, expected_length)
        y_office_adj = match_sequence_length(y_office_padded, expected_length)

        # -------------------
        # Initialize models
        # -------------------
        models_obj = Models(feature_size=X_train.shape[2])
        models_obj.reset_models()
        scenario_results = {}

        for model_name in MODELS_TO_RUN:
            print(f"\n--- Training {model_name.upper()} ---")
            if model_name == "lstm":
                model, history = models_obj.train_lstm(X_train, y_train, verbose=1)
                preds_home = models_obj.predict_lstm(X_test_home, y_len=y_test_home.shape[1])
                preds_office = models_obj.predict_lstm(X_office_adj, y_len=y_office_adj.shape[1])
                model.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_model_lstm.h5"), save_format="h5")
            elif model_name == "cnn":
                model, history = models_obj.train_cnn(X_train, y_train, verbose=1)
                preds_home = models_obj.predict_cnn(X_test_home)
                preds_office = models_obj.predict_cnn(X_office_adj)
                model.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_model_cnn.h5"), save_format="h5")
            elif model_name == "linear":
                model = models_obj.train_linear(X_train, y_train)
                preds_home = models_obj.predict_linear(X_test_home)
                preds_office = models_obj.predict_linear(X_office_adj)
                joblib.dump(model, os.path.join(RESULTS_DIR, f"{SCENARIO}_model_linear.pkl"))
            elif model_name == "knn":
                model = models_obj.train_knn(X_train, y_train, n_neighbors=5)
                preds_home = models_obj.predict_knn(X_test_home)
                preds_office = models_obj.predict_knn(X_office_adj)
                joblib.dump(model, os.path.join(RESULTS_DIR, f"{SCENARIO}_model_knn.pkl"))
            else:
                continue

            # -------------------
            # Evaluate
            # -------------------
            ade_home, fde_home = ade_fde(preds_home, y_test_home)
            acc_home = evaluate_decisions(y_test_home, preds_home)
            ade_office, fde_office = ade_fde(preds_office, y_office_adj)
            acc_office = evaluate_decisions(y_office_adj, preds_office)

            print(f"{model_name.upper()} HOME | ADE={ade_home:.4f}, FDE={fde_home:.4f}, ACC={acc_home:.3f}")
            print(f"{model_name.upper()} OFFICE | ADE={ade_office:.4f}, FDE={fde_office:.4f}, ACC={acc_office:.3f}")

            # -------------------
            # Save results
            # -------------------
            np.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_preds_{model_name}_home.npy"), preds_home)
            np.save(os.path.join(RESULTS_DIR, f"{SCENARIO}_preds_{model_name}_office.npy"), preds_office)

            metrics_dict = {
                "ADE_home": float(ade_home),
                "FDE_home": float(fde_home),
                "ACC_home": float(acc_home),
                "ADE_office": float(ade_office),
                "FDE_office": float(fde_office),
                "ACC_office": float(acc_office)
            }

            if history is not None:
                metrics_dict["loss"] = [float(l) for l in history.history["loss"]]
                if "val_loss" in history.history:
                    metrics_dict["val_loss"] = [float(l) for l in history.history["val_loss"]]

            with open(os.path.join(RESULTS_DIR, f"{SCENARIO}_metrics_{model_name}.json"), "w") as f:
                json.dump(metrics_dict, f, indent=2)

            plot_trajectories(y_test_home, preds_home, model_name, SCENARIO)
            plot_trajectories(y_office_adj, preds_office, model_name, SCENARIO)

            scenario_results[model_name] = metrics_dict

        all_results[SCENARIO] = scenario_results

    # ===========================
    # Summary
    # ===========================
    print("\n=== Summary of All Scenarios ===")
    for sc, res in all_results.items():
        print(f"\n--- {sc.upper()} ---")
        for m, r in res.items():
            print(f"{m.upper()} HOME: ADE={r['ADE_home']:.4f}, FDE={r['FDE_home']:.4f}, ACC={r['ACC_home']:.3f}")
            print(f"{m.upper()} OFFICE: ADE={r['ADE_office']:.4f}, FDE={r['FDE_office']:.4f}, ACC={r['ACC_office']:.3f}")

    with open(os.path.join(RESULTS_DIR, "summary_all_scenarios.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
