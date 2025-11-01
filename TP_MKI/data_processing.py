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
# FEATURE EXTRACTION
# ============================================
def extract_features(video_data, use_bbox=True, use_skeleton=True,
                     use_reduced_skeleton=True, use_angles=True):
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
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w_box = max(1.0, x_max - x_min)
                h_box = max(1.0, y_max - y_min)
                area = w_box * h_box
                frame_feats.extend([x_min, y_min, x_max, y_max, cx, cy, w_box, h_box, area])
            else:
                frame_feats.extend([0.0] * 9)

        # Skeleton features
        if use_skeleton:
            lm = f.get("pose_landmarks")
            if lm:
                coords = []
                if use_reduced_skeleton:
                    for idx in REDUCED_LANDMARKS:
                        if idx < len(lm):
                            coords.extend([float(lm[idx][0]), float(lm[idx][1]), float(lm[idx][2])])
                        else:
                            coords.extend([0.0, 0.0, 0.0])
                else:
                    for i in range(len(lm)):
                        coords.extend([float(lm[i][0]), float(lm[i][1]), float(lm[i][2])])
                frame_feats.extend(coords)
            else:
                frame_feats.extend([0.0] * (len(REDUCED_LANDMARKS) * 3 if use_reduced_skeleton else 33 * 3))

        # Angle features
        if use_angles:
            ang = f.get("pose_angles")
            if ang:
                frame_feats.extend([float(v) for v in ang.values()])
            else:
                frame_feats.extend([0.0] * len(REDUCED_LANDMARKS))

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
def build_dataset(master_json, scenario_config, tail=0.1, use_reduced_skeleton=True, skip_short=5):
    """
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
    """
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

        split_idx = max(1, int(n * (1 - tail)))
        Xs.append(feats[:split_idx])
        ys.append(feats[split_idx:])
        labels.append(video_data.get("label", "unknown"))
    return Xs, ys, labels


# ============================================
# TRAIN/TEST SPLIT (Optional convenience)
# ============================================
def train_test_split_dataset(X, y, test_size=TEST_SIZE):
    return train_test_split(X, y, test_size=test_size, random_state=42)
