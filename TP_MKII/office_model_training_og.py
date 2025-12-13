# train_office_models.py

import os
import numpy as np
from data_processing import load_master_json, build_dataset, pad_sequences, transform_with_scaler, fit_scaler
from model_training import scenario_flags, evaluate_decisions
from models import Models, ade_fde
from joblib import dump
import json

MASTER_JSON_PATH = "master_jsons/master_dataset_office_trimmed.json"
BASE_RESULTS_DIR = "results/office_TTE_pred1"

# === CONFIGURABLE PARAMETERS ===
MODE = "trim"       # options: "trim" or "sliding"
UPPER_FRAME_LIMITS = [25]   # loop through multiple sequence lengths
WINDOW_SIZE = 80         
STRIDE = 40              
TEST_SPLIT = 0.2

SCENARIOS_TO_RUN = [
    "bbox_only",
    "bbox_skeleton",
    "bbox_skeleton_angles"
]

MODELS_TO_RUN = ["knn", "lstm"]

# -------------------------
# Sequence Processing
# -------------------------
def create_sliding_windows(X, y, window_size=80, stride=40):
    X_new, y_new = [], []
    for x_seq, y_seq in zip(X, y):
        if len(x_seq) < window_size:
            continue
        for start in range(0, len(x_seq) - window_size + 1, stride):
            X_new.append(x_seq[start:start+window_size])
            y_new.append(y_seq[start:start+window_size])
    return X_new, y_new

def trim_sequences_based_on_stats(X, y, upper_frame_limit=65):
    X_trimmed, y_trimmed = [], []
    for x_seq, y_seq in zip(X, y):
        if len(x_seq) == 0 or len(y_seq) == 0:
            continue
        cut_len = min(len(x_seq), upper_frame_limit)
        X_trimmed.append(x_seq[:cut_len])
        y_trimmed.append(y_seq[:cut_len])
    return X_trimmed, y_trimmed

# -------------------------
# Main Training Loop
# -------------------------
def main():
    master_json = load_master_json(MASTER_JSON_PATH)
    print(f"Loaded {len(master_json)} videos for office training.")

    for scenario in SCENARIOS_TO_RUN:
        flags = scenario_flags(scenario)
        for upper_limit in UPPER_FRAME_LIMITS:
            run_name = f"{scenario}_frames{upper_limit}"
            run_dir = os.path.join(BASE_RESULTS_DIR, run_name)
            os.makedirs(run_dir, exist_ok=True)
            print(f"\n=== Training scenario: {scenario} | upper frame limit: {upper_limit} ===")

            X, y, labels = build_dataset(master_json, flags)

            # Handle sequence lengths
            if MODE == "sliding":
                X_proc, y_proc = create_sliding_windows(X, y, window_size=WINDOW_SIZE, stride=STRIDE)
            elif MODE == "trim":
                X_proc, y_proc = trim_sequences_based_on_stats(X, y, upper_frame_limit=upper_limit)
            else:
                raise ValueError("MODE must be either 'trim' or 'sliding'.")

            if len(X_proc) == 0:
                print("⚠️ No sequences after processing, skipping this run.")
                continue

            # Scaling
            scaler = fit_scaler(X_proc)
            X_scaled = transform_with_scaler(X_proc, scaler)
            y_scaled = transform_with_scaler(y_proc, scaler)

            # Padding
            X_padded, _ = pad_sequences(X_scaled)
            y_padded, _ = pad_sequences(y_scaled)

            # Train/test split
            split_idx = int((1 - TEST_SPLIT) * len(X_padded))
            X_train, X_test = X_padded[:split_idx], X_padded[split_idx:]
            y_train, y_test = y_padded[:split_idx], y_padded[split_idx:]

            models_obj = Models(feature_size=X_padded.shape[2])

            run_results = {}

            for model_name in MODELS_TO_RUN:
                print(f"\nTraining model: {model_name.upper()}")

                # Train
                history = None
                if model_name == "knn":
                    model = models_obj.train_knn(X_train, y_train)
                elif model_name == "linear":
                    model = models_obj.train_linear(X_train, y_train)
                elif model_name == "lstm":
                    model, history = models_obj.train_lstm(X_train, y_train)
                else:
                    continue

                # Predict
                if model_name == "lstm":
                    preds = models_obj.predict_lstm(X_test, y_len=y_test.shape[1])
                elif model_name == "knn":
                    preds = models_obj.predict_knn(X_test)
                elif model_name == "linear":
                    preds = models_obj.predict_linear(X_test)

                # Evaluate
                ade, fde = ade_fde(preds, y_test)
                acc = evaluate_decisions(y_test, preds)
                print(f"{model_name.upper()} – ADE={ade:.4f}, FDE={fde:.4f}, ACC={acc:.3f}")

                # Save model and scaler
                dump(scaler, os.path.join(run_dir, f"{scenario}_scaler.pkl"))
                if model_name == "lstm":
                    models_obj.lstm_model.save(os.path.join(run_dir, f"{model_name}.h5"))
                else:
                    dump(model, os.path.join(run_dir, f"{model_name}.pkl"))

                # Save predictions and metrics
                np.save(os.path.join(run_dir, f"{model_name}_preds.npy"), preds)
                metrics_dict = {"ADE": float(ade), "FDE": float(fde), "Accuracy": float(acc)}
                if history is not None:
                    metrics_dict["loss"] = [float(l) for l in history.history.get("loss", [])]
                    metrics_dict["val_loss"] = [float(l) for l in history.history.get("val_loss", [])]
                with open(os.path.join(run_dir, f"{model_name}_metrics.json"), "w") as f:
                    json.dump(metrics_dict, f, indent=2)

                run_results[model_name] = metrics_dict

            # Save summary for this run
            with open(os.path.join(run_dir, f"summary.json"), "w") as f:
                json.dump(run_results, f, indent=2)

            # --- NEW: save run info ---
            info = {
                "scenario": scenario,
                "upper_frame_limit": upper_limit,
                "mode": MODE,
                "models_trained": MODELS_TO_RUN,
                "dataset_file": MASTER_JSON_PATH,
                "num_training_samples": len(X_train),
                "num_testing_samples": len(X_test)
            }
            with open(os.path.join(run_dir, "info.json"), "w") as f:
                json.dump(info, f, indent=2)

    print("\nAll office models trained and saved.")


if __name__ == "__main__":
    main()
