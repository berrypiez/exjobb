# office_model_training.py
import os
import json
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split

from data_processing import load_master_json, extract_features, fit_scaler, transform_with_scaler, pad_sequences
from models import Models, ade_fde
from model_training import scenario_flags, evaluate_decisions

# ============================
# CONFIG
# ============================
MASTER_JSON_PATH = "master_jsons/master_dataset_office_trimmed.json"
BASE_RESULTS_DIR = "results/future_predict_office"

PAST_LENGTH = 30        # Number of past frames used as input
FUTURE_LENGTH = 10      # Number of future frames to predict
TEST_SPLIT = 0.2

SCENARIOS_TO_RUN = ["bbox_only", "bbox_skeleton", "bbox_skeleton_angles"]
MODELS_TO_RUN = ["knn", "lstm"]

# ============================
# MAIN TRAINING LOOP
# ============================
def main():
    master_json = load_master_json(MASTER_JSON_PATH)
    print(f"Loaded {len(master_json)} videos for office training.\n")

    for scenario in SCENARIOS_TO_RUN:
        print(f"\n=== Training scenario: {scenario} ===")
        flags = scenario_flags(scenario)

        run_dir = os.path.join(BASE_RESULTS_DIR, scenario)
        os.makedirs(run_dir, exist_ok=True)

        # -----------------------------
        # Build dataset: one sequence per video
        # -----------------------------
        X_list, y_list, labels_list = [], [], []

        for video_name, video_data in master_json.items():
            feats = extract_features(
                video_data,
                use_bbox=flags["use_bbox"],
                use_skeleton=flags["use_skeleton"],
                use_reduced_skeleton=True,
                use_angles=flags["use_angles"]
            )

            # Skip videos too short
            if feats.shape[0] < PAST_LENGTH + FUTURE_LENGTH:
                continue

            # Trim to exact length
            X_list.append(feats[:PAST_LENGTH])
            y_list.append(feats[PAST_LENGTH:PAST_LENGTH + FUTURE_LENGTH])
            labels_list.append(video_data.get("label"))

        if len(X_list) == 0:
            print(f"⚠️ No usable videos for scenario {scenario}, skipping.")
            continue

        print(f"Total videos used: {len(X_list)}")

        # -----------------------------
        # Scaling
        # -----------------------------
        scaler = fit_scaler(X_list)
        X_scaled = transform_with_scaler(X_list, scaler)
        y_scaled = transform_with_scaler(y_list, scaler)

        # Padding (optional, all sequences same length)
        X_padded, _ = pad_sequences(X_scaled)
        y_padded, _ = pad_sequences(y_scaled)

        # -----------------------------
        # Train/test split
        # -----------------------------
        X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
            X_padded, y_padded, labels_list,
            test_size=TEST_SPLIT,
            random_state=42,
            stratify=labels_list
        )

        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # -----------------------------
        # Save test labels for hybrid evaluation
        # -----------------------------
        np.save(os.path.join(run_dir, "labels_test.npy"), np.array(labels_test))

        # -----------------------------
        # Train models
        # -----------------------------
        models_obj = Models(feature_size=X_padded.shape[2])
        run_results = {}

        for model_name in MODELS_TO_RUN:
            print(f"\nTraining model: {model_name.upper()}")
            history = None

            # --- Train ---
            if model_name == "knn":
                model = models_obj.train_knn(X_train, y_train)
            elif model_name == "lstm":
                model, history = models_obj.train_lstm(X_train, y_train)
            else:
                continue

            # --- Predict ---
            if model_name == "lstm":
                preds = models_obj.predict_lstm(X_test, y_len=FUTURE_LENGTH)
            elif model_name == "knn":
                preds = models_obj.predict_knn(X_test)

            # --- Evaluate ---
            ade, fde = ade_fde(preds, y_test)
            static_acc = evaluate_decisions(y_test, preds)
            print(f"{model_name.upper()} – ADE={ade:.4f}, FDE={fde:.4f}, Static_ACC={static_acc:.3f}")

            # --- Save predictions and metrics ---
            np.save(os.path.join(run_dir, f"{model_name}_y_pred.npy"), preds)
            np.save(os.path.join(run_dir, f"{model_name}_y_test.npy"), y_test)

            metrics_dict = {"ADE": float(ade), "FDE": float(fde), "Static_ACC": float(static_acc)}
            if history is not None:
                metrics_dict["loss"] = [float(l) for l in history.history.get("loss", [])]
                metrics_dict["val_loss"] = [float(l) for l in history.history.get("val_loss", [])]

            with open(os.path.join(run_dir, f"{model_name}_metrics.json"), "w") as f:
                json.dump(metrics_dict, f, indent=2)

            # --- Save model and scaler ---
            dump(scaler, os.path.join(run_dir, f"{scenario}_scaler.pkl"))
            if model_name == "lstm":
                models_obj.lstm_model.save(os.path.join(run_dir, f"{model_name}.h5"))
            else:
                dump(model, os.path.join(run_dir, f"{model_name}.pkl"))

            run_results[model_name] = metrics_dict

        # --- Save scenario summary ---
        with open(os.path.join(run_dir, f"summary.json"), "w") as f:
            json.dump(run_results, f, indent=2)

        # --- Save run info ---
        info = {
            "scenario": scenario,
            "past_length": PAST_LENGTH,
            "future_length": FUTURE_LENGTH,
            "models_trained": MODELS_TO_RUN,
            "dataset_file": MASTER_JSON_PATH,
            "num_training_samples": len(X_train),
            "num_testing_samples": len(X_test)
        }
        with open(os.path.join(run_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)

    print("\nAll office models trained and saved with predictions for downstream evaluation.")

if __name__ == "__main__":
    main()
