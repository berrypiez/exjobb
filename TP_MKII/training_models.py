import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from models import Models, compute_centroid_ade_fde
from data_processing import load_master_json, build_dataset, fit_scaler_on_X, train_test_split_dataset, fit_scaler, transform_with_scaler
from model_training import evaluate_decisions, decide_open_door, scenario_flags

EXPERIMENT_NAME = "test"
DATA_DIR = "C:/Users/hanna/Documents/Thesis/exjobb/TP_MKI/master_jsons"
MASTER_JSON_PATH = os.path.join(DATA_DIR, "home_master_dataset.json")
RESULTS_BASE_DIR = os.path.join("results/december_results", EXPERIMENT_NAME)
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

MODELS_TO_RUN = ["lstm"] # , "cnn", "linear", "knn"
RUN_SCENARIOS = [
    "bbox_only",
	"skeleton_only",
	"angles_only",
	"bbox_skeleton",
	"bbox_angles",
	"skeleton_angles",
	"bbox_skeleton_angles"
]

PAST_FRAMES = [30]
FUTURE_FRAMES = [10]
USE_REDUCED_SKELETON = False
TEST_SIZE = 0.2
PLOT_SAMPLES = 5

def plot_trajectories(y_true, y_pred, model_name, scenario, folder, n_samples=PLOT_SAMPLES):
    os.makedirs(folder, exist_ok=True)
    n_samples = min(n_samples, len(y_true))
    cx, cy = 0,1
    for i in range(n_samples):
        plt.figure(figsize=(6,6))
        plt.plot(y_true[i][:, cx], y_true[i][:, cy], 'o-', label="GT")
        plt.plot(y_pred[i][:, cx], y_pred[i][:, cy], 'x--', label="Pred")
        plt.title(f"{model_name.upper()} ({scenario}) Sample {i+1}")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{scenario}_{model_name}_sample{i+1}.png"))
        plt.close()


def main():
    master_json = load_master_json(MASTER_JSON_PATH)

    for past_frame in PAST_FRAMES:
        for future_frame in FUTURE_FRAMES:
            combo_name = f"past{past_frame}_future{future_frame}"
            combo_dir = os.path.join(RESULTS_BASE_DIR, combo_name)
            os.makedirs(combo_dir, exist_ok=True)
            print(f"\n=== Experiment combo: {combo_name} ===")

            for scenario in RUN_SCENARIOS:
                flags = {
                    "use_bbox": "bbox" in scenario,
                    "use_skeleton": "skeleton" in scenario,
                    "use_angles": "angles" in scenario
                }

                scenario_dir = os.path.join(combo_dir, scenario)
                os.makedirs(scenario_dir, exist_ok=True)
                print(f"\n--- Scenario: {scenario} ---")

                Xs, ys, labels = build_dataset(
                    master_json,
                    flags,
                    past_length=past_frame,
                    future_length=future_frame,
                    use_reduced_skeleton=USE_REDUCED_SKELETON,
                    include_center=True,
                    skip_short=True
                )

                if len(Xs) == 0:
                    print("No valid sequences for this scenario, skipping.")
                    continue

                # Train/test split
                N = len(Xs)
                idx = np.arange(N)
                np.random.seed(42)
                np.random.shuffle(idx)
                split = int(N * (1 - TEST_SIZE))
                train_idx, test_idx = idx[:split], idx[split:]

                X_train = [Xs[i] for i in train_idx]
                X_test = [Xs[i] for i in test_idx]
                y_train = [ys[i] for i in train_idx]
                y_test = [ys[i] for i in test_idx]

                # Fit scaler on X_train (which are sequences of shape (T_p, Fx))
                scaler = fit_scaler_on_X(X_train)
                X_train_scaled = transform_with_scaler(X_train, scaler)
                X_test_scaled = transform_with_scaler(X_test, scaler)

                # Convert to numpy arrays for models
                X_train_arr = np.array(X_train_scaled)  # shape (N_train, T_p, Fx)
                X_test_arr = np.array(X_test_scaled)
                y_train_arr = np.array(y_train)        # shape (N_train, T_f, 2)  (UNSCALED)
                y_test_arr = np.array(y_test)          # shape (N_test, T_f, 2)

                # Save scaler
                scaler_dir = os.path.join(scenario_dir, "scalers")
                os.makedirs(scaler_dir, exist_ok=True)
                joblib.dump(scaler, os.path.join(scaler_dir, f"{scenario}_scaler_X.pkl"))

                # Model manager
                Fx = X_train_arr.shape[2]
                models_obj = Models(feature_size=Fx)
                models_obj.reset_models()

                scenario_results = {}

                for model_name in MODELS_TO_RUN:
                    print(f"\nTraining {model_name.upper()}...")

                    # Train and predict
                    if model_name == "lstm":
                        model, _ = models_obj.train_lstm(X_train_arr, y_train_arr, latent_dim=128, epochs=40, batch_size=8)
                        preds = models_obj.predict_lstm(X_test_arr, y_len=y_test_arr.shape[1])
                    elif model_name == "cnn":
                        model, _ = models_obj.train_cnn(X_train_arr, y_train_arr, filters=64, kernel_size=3, epochs=40, batch_size=8)
                        preds = models_obj.predict_cnn(X_test_arr, y_len=y_test_arr.shape[1])
                    elif model_name == "linear":
                        model = models_obj.train_linear(X_train_arr, y_train_arr)
                        preds = models_obj.predict_linear(X_test_arr)
                    elif model_name == "knn":
                        model = models_obj.train_knn(X_train_arr, y_train_arr, n_neighbors=5)
                        preds = models_obj.predict_knn(X_test_arr)
                    else:
                        continue

                    model_dir = os.path.join(scenario_dir, "models")
                    os.makedirs(model_dir, exist_ok=True)
                    if model_name in ["lstm", "cnn"]:
                        model_save_path = os.path.join(model_dir, f"{model_name}.h5")
                        model.save(model_save_path)
                    else:
                        joblib.dump(model, os.path.join(model_dir, f"{model_name}.pkl"))
                    
                    ade, fde = compute_centroid_ade_fde(preds, y_test_arr)
                    acc = evaluate_decisions(y_test_arr, preds)
                    print(f"{model_name.upper()} | ADE={ade:.4f}, FDE={fde:.4f}, ACC={acc:.3f}")

                    # Save predictions (unscaled), metrics
                    preds_dir = os.path.join(scenario_dir, "preds")
                    os.makedirs(preds_dir, exist_ok=True)
                    np.save(os.path.join(preds_dir, f"{model_name}_preds.npy"), preds)

                    metrics_dir = os.path.join(scenario_dir, "metrics")
                    os.makedirs(metrics_dir, exist_ok=True)
                    metrics_dict = {"ADE": float(ade), "FDE": float(fde), "Accuracy": float(acc)}
                    with open(os.path.join(metrics_dir, f"{model_name}_metrics.json"), "w") as f:
                        json.dump(metrics_dict, f, indent=2)

                    # Save sample plots (unscaled coords)
                    plots_dir = os.path.join(scenario_dir, "plots")
                    plot_trajectories(y_test_arr, preds, model_name, scenario, plots_dir)

                    scenario_results[model_name] = metrics_dict

                # Save scenario summary
                with open(os.path.join(scenario_dir, "summary.json"), "w") as f:
                    json.dump(scenario_results, f, indent=2)
    print("\n=== All experiments completed ===")

if __name__ == "__main__":
    main()