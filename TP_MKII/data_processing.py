import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MASTER_JSON_DIR = "master_jsons"
TEST_SIZE = 0.2

# Reduced landmark indices for lightweight skeletons
REDUCED_LANDMARKS = [0, 11, 12, 23, 24]


# ============================================
# LOADING
# ============================================
def load_master_json(master_json_path=None):
    """Load the master dataset JSON file."""
    if master_json_path is None:
        master_json_path = os.path.join(MASTER_JSON_DIR, "master_dataset.json")
    with open(master_json_path, "r") as f:
        data = json.load(f)
    return data

# ============================================
# CUTTING 
# ============================================
# def trim_trailing_empty_frames(feats, empty_threshold)

# ============================================
# FEATURE EXTRACTION
# ============================================
def extract_features(video_data, 
                     use_bbox=True, 
                     use_skeleton=True,
                     use_reduced_skeleton=True, 
                     use_angles=True,
                     include_center=True):
    """Extract per-frame features (bbox, skeleton, angles) from one video entry."""
    frames = video_data.get("frames", [])
    features = []

    for f in frames:
        frame_feats = []

        # BBox features: x_min, y_min, x_max, y_max, center_x, center_y, width, height, area
        if use_bbox:
            bbox = f.get("bbox")
            if bbox:
                x_min, y_min, x_max, y_max = map(float, bbox)
                w_box = max(1.0, x_max - x_min)
                h_box = max(1.0, y_max - y_min)
                area = w_box * h_box
                frame_feats.extend([x_min, y_min, x_max, y_max, area])
            else:
                frame_feats.extend([0.0] * 5)

        # Skeleton features
        if use_skeleton:
            lm = f.get("pose_landmarks") if not use_reduced_skeleton else f.get("pose_landmarks_reduced")
            if lm:
                for point in lm:
                    frame_feats.extend([float(point[0]), float(point[1]), float(point[2])])
            else:
                n_landmarks = 5 if use_reduced_skeleton else 33
                frame_feats.extend([0.0]*(n_landmarks*3))

        # Angle features
        if use_angles:
            ang = f.get("pose_angles")
            if ang:
                frame_feats.extend([float(v) for v in ang.values()])
            else:
                frame_feats.extend([0.0] * 5) # assuming 5 angles

        if include_center:
            center = f.get("center", [0.0, 0.0])
            frame_feats.extend([float(center[0]), float(center[1])])

        features.append(frame_feats)
    
    return np.array(features, dtype=np.float32)


# ============================================
# PADDING
# ============================================
def pad_sequences(sequences, padding="post"):
    """Pad variable-length sequences into uniform 3D array."""
    if len(sequences) == 0:
        return np.zeros((0, 0, 0)), np.array([], dtype=np.int32)
    max_t = max([s.shape[0] for s in sequences])
    feat = sequences[0].shape[1]
    out = np.zeros((len(sequences), max_t, feat), dtype=np.float32)
    lengths = []
    for i, s in enumerate(sequences):
        t = s.shape[0]
        lengths.append(t)
        if padding == "post":
            out[i, :t, :] = s
        else:
            out[i, -t:, :] = s
    return out, np.array(lengths, dtype=np.int32)


# ============================================
# SCALER UTILITIES
# ============================================
def fit_scaler(sequences):
    """Fit StandardScaler on all frames concatenated."""
    all_data = np.vstack([s for s in sequences if s.shape[0] > 0])
    scaler = StandardScaler()
    scaler.fit(all_data)
    return scaler

def fit_scaler_on_X(sequences):
    """Fit StandardScaler on all frames concatenated for X sequences (list of (T, Fx))."""
    # flatten frames
    all_frames = np.vstack([s for s in sequences if s.shape[0] > 0])
    scaler = StandardScaler()
    scaler.fit(all_frames)
    return scaler

def transform_with_scaler(sequences, scaler):
    """Transform list of sequences using fitted scaler."""
    out = []
    for s in sequences:
        if s.shape[0] > 0:
            out.append(scaler.transform(s))
        else:
            out.append(s)
    return out


# ============================================
# DATASET BUILDING
# ============================================
def build_dataset(master_json, 
                  scenario_config, 
                  past_length=30, 
                  future_length=10,
                  use_reduced_skeleton=True, 
                  include_center=True,
                  skip_short=True):
    
    Xs, ys, labels = [], [], []

    for video_name, video_data in master_json.items():
        feats = extract_features(
            video_data,
            use_bbox=scenario_config.get("use_bbox", True),
            use_skeleton=scenario_config.get("use_skeleton", True),
            use_reduced_skeleton=use_reduced_skeleton,
            use_angles=scenario_config.get("use_angles", True),
            include_center=include_center
        )
        if feats is None or feats.shape[0] == 0:
            continue

        n_frames = feats.shape[0]
        if skip_short and n_frames < (past_length + future_length):
            continue
        if include_center:
            centers = feats[:, -2:]  # last two features are center_x, center_y
            X_all = feats[:, :-2]
        else:
            centers = []
            for f in video_data.get("frames", []):
                center = f.get("center", [0.0, 0.0])
                centers.append([float(center[0]), float(center[1])])
            centers = np.array(centers, dtype=np.float32)
            X_all = feats
        
        for start in range(0, n_frames - past_length - future_length + 1):
            X_seg = X_all[start:start+past_length]
            y_seg = centers[start+past_length:start+past_length+future_length]

            if X_seg.shape[0] != past_length or y_seg.shape[0] != future_length:
                continue

            Xs.append(X_seg.copy())
            ys.append(y_seg.copy())
            labels.append(video_data.get("label", "unknown"))
    return Xs, ys, labels


# ============================================
# TRAIN/TEST SPLIT (Optional convenience)
# ============================================
def train_test_split_dataset(X, y, test_size=TEST_SIZE):
    return train_test_split(X, y, test_size=test_size, random_state=42)

"""def build_dataset(master_json, scenario_config, tail=0.1, use_reduced_skeleton=True, skip_short=5):
    '''
    Build dataset (X, y) for training.

    Parameters
    ----------
    master_json : dict
        Loaded master dataset.
    scenario_config : dict
        { use_bbox, use_skeleton, use_angles }
    tail : float
        Fraction of frames reserved as prediction target.
    use_reduced_skeleton : bool
        Whether to reduce skeleton landmarks.
    skip_short : int
        Skip sequences shorter than this.
    '''
    skip_short = 2
    Xs, ys, labels = [], [], []

    for video_name, video_data in master_json.items():

        feats = extract_features(
            video_data,
            use_bbox=scenario_config.get("use_bbox", True),
            use_skeleton=scenario_config.get("use_skeleton", True),
            use_reduced_skeleton=use_reduced_skeleton,
            use_angles=scenario_config.get("use_angles", True)
        )

        n = feats.shape[0]
        if n < skip_short:
            continue

        split_idx = -1
        Xs.append(feats[:split_idx])
        ys.append(feats[split_idx:])
        labels.append(video_data.get("label", "unknown"))
    return Xs, ys, labels"""