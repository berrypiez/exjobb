import json
import os
import random

# ---------- CONFIG ----------
MASTER_JSON_PATH = "master_jsons/office_master_dataset.json"
OUT_DIR = "master_jsons"
TEST_RATIO = 0.1
CROSS_SUBSET_SIZE = 80
RANDOM_SEED = 42
# ----------------------------

def split_master_json(master_json_path, out_dir, test_ratio=0.1, seed=42):
    """Split master JSON into train and holdout test sets."""
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    with open(master_json_path, "r") as f:
        data = json.load(f)

    keys = list(data.keys())
    random.shuffle(keys)

    split_idx = int(len(keys) * (1 - test_ratio))
    train_keys = keys[:split_idx]
    test_keys = keys[split_idx:]

    train_data = {k: data[k] for k in train_keys}
    test_data = {k: data[k] for k in test_keys}

    train_path = os.path.join(out_dir, "master_dataset_office.json")
    test_path = os.path.join(out_dir, "master_dataset_office_holdout.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"✅ Train set: {len(train_data)} videos → {train_path}")
    print(f"✅ Holdout set: {len(test_data)} videos → {test_path}")

    return train_data, test_data


def extract_cross_test_subset(data, out_path, n_samples=80, seed=42):
    """Extract a small balanced subset for cross-scenario testing."""
    random.seed(seed)

    # Try to balance by label if available
    enter_videos = [k for k, v in data.items() if v.get("label") == "enter"]
    pass_videos = [k for k, v in data.items() if v.get("label") == "pass"]

    n_half = n_samples // 2

    chosen = []
    if enter_videos:
        chosen += random.sample(enter_videos, min(n_half, len(enter_videos)))
    if pass_videos:
        chosen += random.sample(pass_videos, min(n_half, len(pass_videos)))

    # If labels are missing or imbalanced, fill with random samples
    if len(chosen) < n_samples:
        remaining = [k for k in data.keys() if k not in chosen]
        chosen += random.sample(remaining, min(n_samples - len(chosen), len(remaining)))

    subset = {k: data[k] for k in chosen}

    with open(out_path, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"✅ Cross-scenario subset: {len(subset)} videos → {out_path}")


if __name__ == "__main__":
    print("Splitting dataset...")
    train_data, _ = split_master_json(MASTER_JSON_PATH, OUT_DIR, TEST_RATIO, RANDOM_SEED)

    cross_out_path = os.path.join(OUT_DIR, "office_subset_for_home_test.json")
    extract_cross_test_subset(train_data, cross_out_path, CROSS_SUBSET_SIZE, RANDOM_SEED)

    print("🎉 Dataset splitting complete.")