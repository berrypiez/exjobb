import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping

from data_processing import load_master_json, extract_features

# ===========================
# CONFIG
# ===========================
MASTER_JSON_PATH = "master_jsons/master_dataset_office_trimmed.json"
BASE_RESULTS_DIR = "results/presentation_classification"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# Sequence parameters
SEQUENCE_LENGTH = 10  # Number of frames in input sequence
DISTANCE_OFFSETS = [5, 10, 15, 20, 25, 30]  # Frames away from end
TEST_SPLIT = 0.2

# Feature scenarios
SCENARIOS_TO_RUN = ["bbox_only", "bbox_skeleton", "bbox_skeleton_angles"]

# Models to train
MODELS_TO_RUN = ["knn", "lstm"]

# Label mapping
LABEL_MAP = {"pass": 0, "enter": 1}
CLASS_NAMES = ["pass", "enter"]

# ============================
# DATASET BUILDING
# ============================
def build_sequence_dataset(master_json, scenario_config, sequence_length, distance_offset):
    """
    Build dataset with sequences of frames at specified distance from video end.
    
    Args:
        master_json: Loaded video data
        scenario_config: Feature flags (bbox, skeleton, angles)
        sequence_length: Number of frames in each sequence
        distance_offset: How many frames before end to start sequence
    
    Returns:
        X_sequences: List of (sequence_length, features) arrays
        y_labels: List of class labels
    """
    X_sequences, y_labels = [], []
    
    for video_name, video_data in master_json.items():
        feats = extract_features(
            video_data,
            use_bbox=scenario_config["use_bbox"],
            use_skeleton=scenario_config["use_skeleton"],
            use_reduced_skeleton=True,
            use_angles=scenario_config["use_angles"],
            include_center=False  # Don't include center for classification
        )
        
        if feats is None or feats.shape[0] == 0:
            continue
            
        total_frames = feats.shape[0]
        
        # Calculate start index: distance_offset frames before end, then go back sequence_length more
        start_idx = total_frames - distance_offset - sequence_length
        
        if start_idx < 0:
            continue  # Video too short
            
        # Extract sequence
        sequence = feats[start_idx:start_idx + sequence_length]
        
        if sequence.shape[0] != sequence_length:
            continue
            
        X_sequences.append(sequence)
        y_labels.append(LABEL_MAP[video_data.get("label", "pass")])
    
    return X_sequences, y_labels

# ============================
# MODEL BUILDERS
# ============================
def build_lstm_classifier(timesteps, feat_dim, hidden_units=64):
    """Build LSTM classifier for sequences."""
    model = Sequential([
        Masking(mask_value=0., input_shape=(timesteps, feat_dim)),
        LSTM(hidden_units, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================
# EVALUATION
# ============================
def evaluate_model(y_true, y_pred):
    """Calculate classification metrics."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }

# ============================
# MAIN TRAINING LOOP
# ============================
def main():
    master_json = load_master_json(MASTER_JSON_PATH)
    print(f"Loaded {len(master_json)} videos")
    
    all_results = {}
    
    for scenario in SCENARIOS_TO_RUN:
        print(f"\n=== Scenario: {scenario} ===")
        
        scenario_config = {
            "use_bbox": "bbox" in scenario,
            "use_skeleton": "skeleton" in scenario,
            "use_angles": "angles" in scenario
        }
        
        scenario_results = {}
        
        for distance in DISTANCE_OFFSETS:
            print(f"\nDistance offset: {distance} frames")
            
            # Build dataset for this distance
            X_sequences, y_labels = build_sequence_dataset(
                master_json, scenario_config, SEQUENCE_LENGTH, distance
            )
            
            if len(X_sequences) == 0:
                print(f"No sequences for distance {distance}, skipping")
                continue
                
            print(f"Built {len(X_sequences)} sequences")
            
            # Scale features
            scaler = StandardScaler()
            X_flat = np.vstack(X_sequences)
            scaler.fit(X_flat)
            
            X_scaled = []
            for seq in X_sequences:
                X_scaled.append(scaler.transform(seq))
            
            # Convert to arrays
            X_array = np.array(X_scaled)
            y_array = np.array(y_labels)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y_array, test_size=TEST_SPLIT, random_state=42, stratify=y_array
            )
            
            distance_results = {}
            
            # Train models
            for model_name in MODELS_TO_RUN:
                print(f"Training {model_name}...")
                
                if model_name == "knn":
                    # Flatten sequences for KNN
                    X_train_flat = X_train.reshape(X_train.shape[0], -1)
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    
                    model = KNeighborsClassifier(n_neighbors=5)
                    model.fit(X_train_flat, y_train)
                    y_pred = model.predict(X_test_flat)
                
                elif model_name == "lstm":
                    model = build_lstm_classifier(X_train.shape[1], X_train.shape[2])
                    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    
                    model.fit(X_train, y_train, validation_split=0.2, epochs=50, 
                             batch_size=32, callbacks=[es], verbose=0)
                    
                    y_prob = model.predict(X_test).ravel()
                    y_pred = (y_prob >= 0.5).astype(int)
                
                # Evaluate
                metrics = evaluate_model(y_test, y_pred)
                distance_results[model_name] = metrics
                
                print(f"{model_name}: Acc={metrics['accuracy']:.3f}, "
                      f"F1={metrics['f1']:.3f}")
            
            scenario_results[f"distance_{distance}"] = distance_results
        
        all_results[scenario] = scenario_results

    # Save results
    results_path = os.path.join(BASE_RESULTS_DIR, "all_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary plots
    create_summary_plots(all_results)
    
    print(f"\nResults saved to {BASE_RESULTS_DIR}")

def create_summary_plots(all_results):
    """Create plots showing how accuracy varies with distance from exit."""
    
    for scenario in all_results.keys():
        plt.figure(figsize=(12, 8))
        
        for model_name in MODELS_TO_RUN:
            distances = []
            accuracies = []
            
            for distance_key, distance_results in all_results[scenario].items():
                if distance_key.startswith("distance_"):
                    distance = int(distance_key.split("_")[1])
                    if model_name in distance_results:
                        distances.append(distance)
                        accuracies.append(distance_results[model_name]['accuracy'])
            
            if distances:
                plt.plot(distances, accuracies, 'o-', label=model_name.upper(), linewidth=2)
        
        plt.xlabel('Distance from Exit (frames)')
        plt.ylabel('Accuracy')
        plt.title(f'Classification Accuracy vs Distance from Exit\n({scenario})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(BASE_RESULTS_DIR, f"{scenario}_accuracy_vs_distance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Summary plots created")

if __name__ == "__main__":
    main()