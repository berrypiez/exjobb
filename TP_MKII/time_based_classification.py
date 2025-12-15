import os
import json
import joblib
import numpy as np
import pandas as pd
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
DATA_SUMMARY_PATH = "../../data_summary.csv"  # Path to fps data
BASE_RESULTS_DIR = "results/time_based_classification"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# Time-based parameters (in seconds)
TIME_HORIZONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Input sequence durations
TIME_OFFSETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # Time before end to start sequence
TEST_SPLIT = 0.2

# Feature scenarios
SCENARIOS_TO_RUN = ["bbox_only", "bbox_skeleton", "bbox_skeleton_angles"]

# Models to train
MODELS_TO_RUN = ["knn", "lstm"]

# Label mapping
LABEL_MAP = {"pass": 0, "enter": 1}
CLASS_NAMES = ["pass", "enter"]

# ============================
# FPS DATA LOADING
# ============================
def load_fps_data(data_summary_path):
    """Load FPS data from CSV file."""
    try:
        df = pd.read_csv(data_summary_path)
        fps_map = {}
        
        for _, row in df.iterrows():
            video_name = row['video_name']
            fps = row['fps']
            
            if pd.isna(fps) or fps <= 0:
                continue
                
            # Store with original name
            fps_map[video_name] = float(fps)
            
            # Also store without extension for better matching
            name_without_ext = os.path.splitext(video_name)[0]
            if name_without_ext != video_name:
                fps_map[name_without_ext] = float(fps)
        
        print(f"Loaded FPS data for {len(set(fps_map.values()))} unique videos")
        return fps_map
    except Exception as e:
        print(f"Warning: Could not load FPS data from {data_summary_path}: {e}")
        return {}

def get_video_fps(video_name, fps_map, default_fps=12.0):
    """Get FPS for a video, with fallback to default."""
    # Try exact match first
    if video_name in fps_map:
        return fps_map[video_name]
    
    # Try without extension
    name_without_ext = os.path.splitext(video_name)[0]
    if name_without_ext in fps_map:
        return fps_map[name_without_ext]
    
    # Try with common extensions
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        if f"{video_name}{ext}" in fps_map:
            return fps_map[f"{video_name}{ext}"]
    
    print(f"Warning: No FPS data found for '{video_name}', using default {default_fps}")
    return default_fps

# ============================
# TIME-BASED DATASET BUILDING
# ============================
def build_time_based_dataset(master_json, fps_map, scenario_config, time_horizon, time_offset):
    """
    Build dataset with time-based sequences instead of fixed frame counts.
    
    Args:
        master_json: Loaded video data
        fps_map: Dictionary mapping video names to fps values
        scenario_config: Feature flags (bbox, skeleton, angles)
        time_horizon: Duration of input sequence in seconds
        time_offset: Time before end to start sequence in seconds
    
    Returns:
        X_sequences: List of variable-length sequences
        y_labels: List of class labels
        sequence_info: List of dicts with metadata about each sequence
    """
    X_sequences, y_labels, sequence_info = [], [], []
    
    for video_name, video_data in master_json.items():
        # Extract features
        feats = extract_features(
            video_data,
            use_bbox=scenario_config["use_bbox"],
            use_skeleton=scenario_config["use_skeleton"],
            use_reduced_skeleton=True,
            use_angles=scenario_config["use_angles"],
            include_center=False
        )
        
        if feats is None or feats.shape[0] == 0:
            continue
            
        total_frames = feats.shape[0]
        fps = get_video_fps(video_name, fps_map)
        
        # Convert time to frames
        horizon_frames = int(time_horizon * fps)
        offset_frames = int(time_offset * fps)
        
        # Calculate start index
        start_idx = total_frames - offset_frames - horizon_frames
        
        if start_idx < 0 or horizon_frames <= 0:
            continue  # Video too short or invalid parameters
            
        # Extract sequence
        end_idx = start_idx + horizon_frames
        sequence = feats[start_idx:end_idx]
        
        if sequence.shape[0] == 0:
            continue
            
        X_sequences.append(sequence)
        y_labels.append(LABEL_MAP[video_data.get("label", "pass")])
        
        # Store metadata
        sequence_info.append({
            'video_name': video_name,
            'fps': fps,
            'total_frames': total_frames,
            'sequence_frames': sequence.shape[0],
            'time_horizon': time_horizon,
            'time_offset': time_offset,
            'actual_duration': sequence.shape[0] / fps
        })
    
    return X_sequences, y_labels, sequence_info

# ============================
# SEQUENCE PADDING AND PROCESSING
# ============================
def pad_sequences_time_based(sequences, max_length=None, padding_value=0.0):
    """Pad sequences to same length for models that require it."""
    if not sequences:
        return np.array([]), 0
    
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    # Ensure minimum length of 1
    max_length = max(1, max_length)
    feat_dim = sequences[0].shape[1]
    
    padded = []
    for seq in sequences:
        if len(seq) == 0:
            # Handle empty sequences
            padded_seq = np.full((max_length, feat_dim), padding_value)
        elif len(seq) < max_length:
            # Pad with specified value
            padding = np.full((max_length - len(seq), feat_dim), padding_value)
            padded_seq = np.vstack([seq, padding])
        else:
            # Truncate if longer
            padded_seq = seq[:max_length]
        
        padded.append(padded_seq)
    
    return np.array(padded), max_length

# ============================
# MODEL BUILDERS
# ============================
def build_lstm_classifier_time_based(max_timesteps, feat_dim, hidden_units=64):
    """Build LSTM classifier for variable-length sequences."""
    model = Sequential([
        Masking(mask_value=0., input_shape=(max_timesteps, feat_dim)),
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

def analyze_sequence_statistics(sequence_info, time_horizon, time_offset):
    """Analyze statistics of the generated sequences."""
    if not sequence_info:
        return {}
    
    df = pd.DataFrame(sequence_info)
    stats = {
        'total_sequences': len(df),
        'mean_fps': df['fps'].mean(),
        'std_fps': df['fps'].std(),
        'min_frames': df['sequence_frames'].min(),
        'max_frames': df['sequence_frames'].max(),
        'mean_frames': df['sequence_frames'].mean(),
        'std_frames': df['sequence_frames'].std(),
        'mean_actual_duration': df['actual_duration'].mean(),
        'std_actual_duration': df['actual_duration'].std(),
        'fps_distribution': df['fps'].value_counts().to_dict()
    }
    
    print(f"\nSequence Statistics for {time_horizon}s horizon, {time_offset}s offset:")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  FPS range: {df['fps'].min():.1f} - {df['fps'].max():.1f} (mean: {stats['mean_fps']:.1f})")
    print(f"  Frame count range: {stats['min_frames']} - {stats['max_frames']} (mean: {stats['mean_frames']:.1f})")
    print(f"  Actual duration range: {df['actual_duration'].min():.2f}s - {df['actual_duration'].max():.2f}s")
    
    return stats

# ============================
# FPS VALIDATION
# ============================
def validate_fps_integration(master_json, fps_map):
    """Validate FPS integration and report statistics."""
    print("\n=== FPS Integration Validation ===")
    
    matched_videos = 0
    unmatched_videos = []
    fps_stats = []
    
    for video_name in master_json.keys():
        fps = get_video_fps(video_name, fps_map)
        if video_name in fps_map or os.path.splitext(video_name)[0] in fps_map:
            matched_videos += 1
            fps_stats.append(fps)
        else:
            unmatched_videos.append(video_name)
    
    print(f"Videos with FPS data: {matched_videos}/{len(master_json)} ({matched_videos/len(master_json)*100:.1f}%)")
    
    if fps_stats:
        print(f"FPS statistics: min={min(fps_stats):.1f}, max={max(fps_stats):.1f}, mean={np.mean(fps_stats):.1f}")
    
    if unmatched_videos:
        print(f"Videos without FPS data ({len(unmatched_videos)}): {unmatched_videos[:5]}{'...' if len(unmatched_videos) > 5 else ''}")
    
    return matched_videos, len(master_json)

# ============================
# PLOTTING FUNCTIONS
# ============================
def create_time_based_plots(all_results):
    """Create plots showing how accuracy varies with time horizons and offsets."""
    
    for scenario in all_results.keys():
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Time-Based Classification Results - {scenario}', fontsize=16)
        
        # Prepare data for plotting
        horizons = TIME_HORIZONS
        offsets = TIME_OFFSETS
        
        for model_idx, model_name in enumerate(MODELS_TO_RUN):
            # Plot 1: Accuracy vs Time Horizon (averaged over offsets)
            ax1 = axes[0, model_idx]
            horizon_accs = []
            horizon_stds = []
            
            for horizon in horizons:
                accs = []
                for offset in offsets:
                    key = f"horizon_{horizon}s_offset_{offset}s"
                    if key in all_results[scenario] and model_name in all_results[scenario][key]:
                        accs.append(all_results[scenario][key][model_name]['accuracy'])
                
                if accs:
                    horizon_accs.append(np.mean(accs))
                    horizon_stds.append(np.std(accs))
                else:
                    horizon_accs.append(0)
                    horizon_stds.append(0)
            
            ax1.errorbar(horizons, horizon_accs, yerr=horizon_stds, 
                        marker='o', linewidth=2, capsize=5)
            ax1.set_xlabel('Time Horizon (seconds)')
            ax1.set_ylabel('Accuracy')
            ax1.set_title(f'{model_name.upper()} - Accuracy vs Time Horizon')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot 2: Accuracy vs Time Offset (averaged over horizons)
            ax2 = axes[1, model_idx]
            offset_accs = []
            offset_stds = []
            
            for offset in offsets:
                accs = []
                for horizon in horizons:
                    key = f"horizon_{horizon}s_offset_{offset}s"
                    if key in all_results[scenario] and model_name in all_results[scenario][key]:
                        accs.append(all_results[scenario][key][model_name]['accuracy'])
                
                if accs:
                    offset_accs.append(np.mean(accs))
                    offset_stds.append(np.std(accs))
                else:
                    offset_accs.append(0)
                    offset_stds.append(0)
            
            ax2.errorbar(offsets, offset_accs, yerr=offset_stds, 
                        marker='s', linewidth=2, capsize=5, color='orange')
            ax2.set_xlabel('Time Offset (seconds)')
            ax2.set_ylabel('Accuracy')
            ax2.set_title(f'{model_name.upper()} - Accuracy vs Time Offset')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plot_path = os.path.join(BASE_RESULTS_DIR, f"{scenario}_time_based_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create heatmap showing frame count variation
    create_frame_count_heatmap(all_results)
    
    print("Time-based analysis plots created")

def create_frame_count_heatmap(all_results):
    """Create heatmap showing average frame counts for different time horizons and offsets."""
    
    for scenario in all_results.keys():
        # Get frame count data
        frame_counts = np.zeros((len(TIME_HORIZONS), len(TIME_OFFSETS)))
        
        for i, horizon in enumerate(TIME_HORIZONS):
            for j, offset in enumerate(TIME_OFFSETS):
                key = f"horizon_{horizon}s_offset_{offset}s"
                if key in all_results[scenario]:
                    # Get frame count from first available model
                    for model_name in MODELS_TO_RUN:
                        if model_name in all_results[scenario][key]:
                            frame_counts[i, j] = all_results[scenario][key][model_name].get('mean_frames', 0)
                            break
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(frame_counts, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Average Frame Count', rotation=270, labelpad=20)
        
        # Set ticks and labels
        plt.xticks(range(len(TIME_OFFSETS)), [f'{o}s' for o in TIME_OFFSETS])
        plt.yticks(range(len(TIME_HORIZONS)), [f'{h}s' for h in TIME_HORIZONS])
        plt.xlabel('Time Offset')
        plt.ylabel('Time Horizon')
        plt.title(f'Average Frame Count per Sequence - {scenario}')
        
        # Add text annotations
        for i in range(len(TIME_HORIZONS)):
            for j in range(len(TIME_OFFSETS)):
                text = plt.text(j, i, f'{frame_counts[i, j]:.1f}',
                               ha="center", va="center", color="white", fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(BASE_RESULTS_DIR, f"{scenario}_frame_count_heatmap.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

# ============================
# MAIN TRAINING LOOP
# ============================
def main():
    # Load data
    master_json = load_master_json(MASTER_JSON_PATH)
    fps_map = load_fps_data(DATA_SUMMARY_PATH)
    
    print(f"Loaded {len(master_json)} videos from master_json")
    print(f"Loaded FPS data for {len(fps_map)} videos")
    
    # Validate FPS integration
    validate_fps_integration(master_json, fps_map)
    
    all_results = {}
    
    for scenario in SCENARIOS_TO_RUN:
        print(f"\n=== Scenario: {scenario} ===")
        
        scenario_config = {
            "use_bbox": "bbox" in scenario,
            "use_skeleton": "skeleton" in scenario,
            "use_angles": "angles" in scenario
        }
        
        scenario_results = {}
        
        for time_horizon in TIME_HORIZONS:
            for time_offset in TIME_OFFSETS:
                print(f"\nTime horizon: {time_horizon}s, Time offset: {time_offset}s")
                
                # Build time-based dataset
                X_sequences, y_labels, sequence_info = build_time_based_dataset(
                    master_json, fps_map, scenario_config, time_horizon, time_offset
                )
                
                if len(X_sequences) == 0:
                    print(f"No sequences for horizon {time_horizon}s, offset {time_offset}s, skipping")
                    continue
                
                # Analyze sequence statistics
                stats = analyze_sequence_statistics(sequence_info, time_horizon, time_offset)
                
                # Scale features
                scaler = StandardScaler()
                X_flat = np.vstack(X_sequences)
                scaler.fit(X_flat)
                
                X_scaled = []
                for seq in X_sequences:
                    X_scaled.append(scaler.transform(seq))
                
                # Pad sequences for models that need fixed input size
                X_padded, max_length = pad_sequences_time_based(X_scaled)
                y_array = np.array(y_labels)
                
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_padded, y_array, test_size=TEST_SPLIT, random_state=42, 
                    stratify=y_array if len(np.unique(y_array)) > 1 else None
                )
                
                time_results = {}
                
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
                        model = build_lstm_classifier_time_based(X_train.shape[1], X_train.shape[2])
                        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        
                        model.fit(X_train, y_train, validation_split=0.2, epochs=50, 
                                 batch_size=32, callbacks=[es], verbose=0)
                        
                        y_prob = model.predict(X_test).ravel()
                        y_pred = (y_prob >= 0.5).astype(int)
                    
                    # Evaluate
                    metrics = evaluate_model(y_test, y_pred)
                    metrics.update(stats)  # Add sequence statistics
                    time_results[model_name] = metrics
                    
                    print(f"{model_name}: Acc={metrics['accuracy']:.3f}, "
                          f"F1={metrics['f1']:.3f}, Frames={stats['mean_frames']:.1f}±{stats['std_frames']:.1f}")
                
                key = f"horizon_{time_horizon}s_offset_{time_offset}s"
                scenario_results[key] = time_results
        
        all_results[scenario] = scenario_results

    # Save results
    results_path = os.path.join(BASE_RESULTS_DIR, "time_based_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary plots
    create_time_based_plots(all_results)
    
    print(f"\nResults saved to {BASE_RESULTS_DIR}")

if __name__ == "__main__":
    main()