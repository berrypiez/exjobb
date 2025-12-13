import os
import json
import numpy as np
from models import Models, ade_fde, compute_centroid_ade_fde
from data_processing import load_master_json, build_dataset, train_test_split_dataset, fit_scaler, transform_with_scaler
from model_training import evaluate_decisions, decide_open_door, scenario_flags
import joblib
import matplotlib.pyplot as plt

"""
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
"""

# ===========================
# CONFIGURATION
# ===========================
EXPERIMENT_NAME = "vary_past_frames"
DATA_DIR = "C:/Users/hanna/Documents/Thesis/exjobb/TP_MKI/master_jsons"
MASTER_JSON_PATH = os.path.join(DATA_DIR, "home_master_dataset.json")
RESULTS_BASE_DIR = os.path.join("results/december_results", EXPERIMENT_NAME)
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# ===========================
# Manual settings
# ==========================
MODELS_TP_RUN = ["lstm", "cnn", "linear", "knn"]
RUN_SCENARIOS = [
	"bbox_only",
	"skeleton_only",
	"angles_only",
	"bbox_skeleton",
	"bbox_angles",
	"skeleton_angles",
	"bbox_skeleton_angles"
]

PAST_LENGTH = [30]
FUTURE_LENGTH = [10]
USE_REDUCED_SKELETON = False
TEST_SIZE = 0.2
PLOT_SAMPLES = 5

# ===========================
# HELPER FUNCTIONS
# ===========================
def plot_trajectories(y_true, y_pred, model_name, scenario, folder, n_samples=PLOT_SAMPLES):
    os.makedirs(folder, exist_ok=True)
    n_samples = min(n_samples, len(y_true))
    cx, cy = 0, 1
    for i in range(n_samples):
        plt.figure(figsize=(6, 6))
        plt.plot(y_true[i][:, cx], y_true[i][:, cy], 'o-', label="GT")
        plt.plot(y_pred[i][:, cx], y_pred[i][:, cy], 'x--', label="Pred")
        plt.title(f"{model_name.upper()} ({scenario}) Sample {i+1}")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{scenario}_{model_name}_sample{i+1}.png"))
        plt.close()

# ===========================
# MAIN TRAINING LOOP
# ===========================
def main():
    master_json = load_master_json(MASTER_JSON_PATH)
    for past_len in PAST_LENGTH:
        for future_len in FUTURE_LENGTH:
            combo_name = f"past{past_len}_future{future_len}"
            combo_dir = os.path.join(RESULTS_BASE_DIR, combo_name)
            os.makedirs(combo_dir, exist_ok=True)
            print(f"\n=== Experiment: {combo_name} ===")
            
			for scenario in RUN_SCENARIOS:
                 scenario_flags = scenario_flags(scenario)