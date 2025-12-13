# import numpy as np
# import os

# # path to the predicted sequences
# pred_file = "results/future_predict_office/bbox_only/lstm_y_pred.npy"

# # load the numpy array
# y_pred = np.load(pred_file, allow_pickle=True)  # allow_pickle=True if saved objects, usually not needed

# # print shape and type
# print("Type:", type(y_pred))
# print("Shape:", y_pred.shape)

# # print first sample (or a slice)
# print("First sample:", y_pred[0])

# import numpy as np
# from data_processing import load_master_json, extract_features

# MASTER_JSON_PATH = "master_jsons/master_dataset_office_trimmed.json"

# SCENARIO_FLAGS = {
#     "use_bbox": True,
#     "use_skeleton": False,
#     "use_reduced_skeleton": False,
#     "use_angles": False
# }

# SEQ_LEN = 10  # classifier input length
# TRIM_LEN = 10  # last frames to keep

# def inspect_bbox_only_data():
#     master_json = load_master_json(MASTER_JSON_PATH)
#     print(f"Loaded {len(master_json)} videos")

#     for name, video in master_json.items():
#         feats = extract_features(video,
#                                  use_bbox=SCENARIO_FLAGS["use_bbox"],
#                                  use_skeleton=SCENARIO_FLAGS["use_skeleton"],
#                                  use_reduced_skeleton=SCENARIO_FLAGS["use_reduced_skeleton"],
#                                  use_angles=SCENARIO_FLAGS["use_angles"])
#         if feats is None:
#             continue

#         print(f"\nVideo: {name}")
#         print("Original features shape:", feats.shape)

#         # keep last TRIM_LEN frames
#         trimmed = feats[-TRIM_LEN:]
#         print("Trimmed features shape:", trimmed.shape)

#         if trimmed.shape[0] >= SEQ_LEN:
#             seq_input = trimmed[-SEQ_LEN:]
#             print("Final classifier input shape:", seq_input.shape)
#             print("First frame features:", seq_input[0])

# if __name__ == "__main__":
#     inspect_bbox_only_data()

# import numpy as np

# traj_knn_pred = np.load("results/future_predict_office/bbox_only/knn_labels.npy")
# print(traj_knn_pred.shape)

import numpy as np
import os
from collections import Counter

# ----------------------------
# CONFIG
# ----------------------------
SCENARIOS = ["bbox_only", "bbox_skeleton", "bbox_skeleton_angles"]
BASE_PRED_DIR = "results/future_predict_office"

label_map = {"pass": 0, "enter": 1}

# ----------------------------
# MAIN LOOP
# ----------------------------
for scenario in SCENARIOS:
    print("\n=====================================")
    print(f"SANITY CHECK: {scenario}")
    print("=====================================")

    run_dir = os.path.join(BASE_PRED_DIR, scenario)

    # Load test labels
    labels_file = os.path.join(run_dir, "labels_test.npy")
    if not os.path.exists(labels_file):
        print(f"⚠️ labels_test.npy not found for {scenario}, skipping.")
        continue

    labels_test = np.load(labels_file, allow_pickle=True)
    print("Number of test videos:", len(labels_test))
    if len(labels_test) == 0:
        continue

    # Numeric labels
    y_labels_numeric = np.array([label_map[l] for l in labels_test])
    print("First 5 numeric labels:", y_labels_numeric[:10])

    # # Load predicted future sequences
    # pred_file = os.path.join(run_dir, "lstm_y_pred.npy")
    # if not os.path.exists(pred_file):
    #     print(f"⚠️ lstm_y_pred.npy not found for {scenario}, skipping.")
    #     continue

    # y_pred = np.load(pred_file)
    # print("Predicted sequences shape:", y_pred.shape)

    # # Check that number of predictions matches labels
    # if y_pred.shape[0] != len(labels_test):
    #     print("⚠️ Number of predictions does NOT match number of labels!")
    # else:
    #     print("✅ Number of predictions matches labels")

    # # Class distribution
    # print("Class distribution (test labels):", Counter(labels_test))

    # # Print first predicted sequence
    # print("First predicted sequence (future frames, first video):")
    # print(y_pred[0])

print("\n✅ Sanity check completed for all scenarios.")
