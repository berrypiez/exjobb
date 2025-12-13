import os
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from data_processing import (
    load_master_json,
    build_dataset,
    pad_sequences,
    transform_with_scaler
)
from tensorflow.keras.utils import custom_object_scope
import tensorflow as tf
from models import Models, ade_fde
from model_training import scenario_flags, evaluate_decisions

# =========================
# CONFIGURATION
# =========================
TEST_JSON = "master_jsons/office_subset_for_home_test.json"
RESULTS_DIR = "results"
MODEL_DIR = os.path.join(RESULTS_DIR, "home_models")
OUTPUT_FILE = "cross_domain_eval_results_office.json"

MODELS_TO_TEST = ["lstm", "cnn", "linear", "knn"]
SCENARIOS_TO_TEST = [
    "bbox_only",
    "skeleton_only",
    "angles_only",
    "bbox_skeleton",
    "bbox_angles",
    "skeleton_angles",
    "bbox_skeleton_angles"
]


# =========================
# HELPER: match input length
# =========================
def match_sequence_length(X, expected_length):
    """
    Trim or pad sequences to match expected length.
    """
    current_length = X.shape[1]
    if current_length > expected_length:
        X = X[:, :expected_length, :]
    elif current_length < expected_length:
        pad_width = expected_length - current_length
        X = np.pad(X, ((0, 0), (0, pad_width), (0, 0)), mode="constant")
    return X


# =========================
# MAIN EVALUATION
# =========================
def evaluate_all_models():
    print(f"Loading test dataset from {TEST_JSON}...")
    master_json = load_master_json(TEST_JSON)
    all_results = {}

    for scenario in SCENARIOS_TO_TEST:
        print(f"\n=== Evaluating scenario: {scenario} ===")
        flags = scenario_flags(scenario)

        # Build dataset
        X, y, labels = build_dataset(master_json, flags)
        if len(X) == 0:
            print(f"⚠️ No data for {scenario}, skipping.")
            continue

        # Load scenario scaler
        scaler_path = os.path.join(MODEL_DIR, f"{scenario}_scaler.pkl")
        if not os.path.exists(scaler_path):
            print(f"⚠️ Scaler not found for {scenario}, skipping scenario.")
            continue
        scaler = joblib.load(scaler_path)

        # Scale and pad
        X_scaled = transform_with_scaler(X, scaler)
        y_scaled = transform_with_scaler(y, scaler)
        X_padded, _ = pad_sequences(X_scaled)
        y_padded, _ = pad_sequences(y_scaled)

        # Initialize wrapper
        models_obj = Models(feature_size=X_padded.shape[2])

        scenario_results = {}

        for model_name in MODELS_TO_TEST:
            # Determine model path
            if model_name in ["cnn", "lstm"]:
                model_path = os.path.join(MODEL_DIR, f"{scenario}_model_{model_name}.h5")
            else:
                model_path = os.path.join(MODEL_DIR, f"{scenario}_model_{model_name}.pkl")

            if not os.path.exists(model_path):
                print(f"⚠️ Model not found: {model_path}")
                continue

            print(f"→ Testing {model_name.upper()} on {scenario}")

            # -----------------------------
            # Load model into wrapper
            # -----------------------------
            if model_name == "lstm":
                model_path = os.path.join(MODEL_DIR, f"{scenario}_model_lstm.h5")

                print(f"→ Testing LSTM on {scenario}")

                try:
                    # Try loading normally first — disable safe_mode to allow legacy ops
                    models_obj.lstm_model = load_model(model_path, compile=False, safe_mode=False)
                except Exception as e:
                    print(f"⚠️ Safe mode load failed, retrying with custom scope: {e}")
                    custom_objects = {
                        # Fallback placeholders for symbolic ops from Masking layers
                        "NotEqual": tf.keras.layers.Layer,
                        "Any": tf.keras.layers.Layer,
                    }
                    with custom_object_scope(custom_objects):
                        models_obj.lstm_model = load_model(model_path, compile=False)

                # -----------------------------
                # Adjust test data to match model’s expected sequence length
                # -----------------------------
                try:
                    expected_length = models_obj.lstm_model.input_shape[0][1]
                except Exception:
                    # Some functional models return shape differently
                    expected_length = models_obj.lstm_model.input_shape[1]

                X_adj = match_sequence_length(X_padded, expected_length)
                y_adj = match_sequence_length(y_padded, expected_length)

                # Run prediction through wrapper
                preds = models_obj.predict_lstm(X_adj, y_len=y_adj.shape[1])

            elif model_name == "cnn":
                models_obj.cnn_model = load_model(model_path)

                expected_length = models_obj.cnn_model.input_shape[1]
                X_adj = match_sequence_length(X_padded, expected_length)
                y_adj = match_sequence_length(y_padded, expected_length)

                preds = models_obj.predict_cnn(X_adj)

            elif model_name == "linear":
                models_obj.lin_model = joblib.load(model_path)
                models_obj.lin_Tf = y_padded.shape[1]
                models_obj.lin_F = y_padded.shape[2]
                preds = models_obj.predict_linear(X_padded)

            elif model_name == "knn":
                models_obj.knn_model = joblib.load(model_path)
                models_obj.knn_Tf = y_padded.shape[1]
                models_obj.knn_F = y_padded.shape[2]
                preds = models_obj.predict_knn(X_padded)

            else:
                continue

            # Evaluate
            ade, fde = ade_fde(preds, y_adj)
            acc = evaluate_decisions(y_adj, preds)

            scenario_results[model_name] = {
                "ADE": float(ade),
                "FDE": float(fde),
                "Accuracy": float(acc)
            }

            print(f"✅ {model_name.upper()} ({scenario}) | ADE={ade:.4f}, FDE={fde:.4f}, ACC={acc:.3f}")

        all_results[scenario] = scenario_results

    # Save all results
    output_path = os.path.join(RESULTS_DIR, OUTPUT_FILE)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n🎯 All evaluations saved to {output_path}")


if __name__ == "__main__":
    evaluate_all_models()
