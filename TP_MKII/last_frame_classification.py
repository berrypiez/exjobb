import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

from data_processing import extract_features

# ============================
# CONFIG
# ============================
DATASET_PATH = "master_jsons/master_dataset_office_trimmed.json"
RESULTS_DIR = "results/last_frame_classification"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

SCENARIOS_TO_RUN = [
    "bbox_only",
    "skeleton_only",
    "angles_only",
    "bbox_skeleton",
    "bbox_angles",
    "skeleton_angles",
    "bbox_skeleton_angles"
]

USE_REDUCED_SKELETON = False

# ============================
# HELPERS
# ============================
def scenario_flags(name):
    return {
        "use_bbox": "bbox" in name,
        "use_skeleton": "skeleton" in name,
        "use_angles": "angles" in name,
        "use_reduced_skeleton": USE_REDUCED_SKELETON
    }

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def build_last_frame_dataset(master_json, scenario_config):
    X_list, y_list = [], []
    for video_name, video_data in master_json.items():
        feats = extract_features(
            video_data,
            use_bbox=scenario_config["use_bbox"],
            use_skeleton=scenario_config["use_skeleton"],
            use_reduced_skeleton=scenario_config["use_reduced_skeleton"],
            use_angles=scenario_config["use_angles"]
        )
        if feats is None or len(feats) == 0:
            continue
        last_feats = feats[-1]
        X_list.append(last_feats)
        y_list.append(video_data["label"])
    return np.array(X_list), np.array(y_list)

def plot_confusion(cm, classes, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc_curve(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ============================
# MAIN
# ============================
def main():
    master_json = load_json(DATASET_PATH)
    print(f"Loaded {len(master_json)} videos")

    results_dict = {}

    for scenario in SCENARIOS_TO_RUN:
        print(f"\n=== SCENARIO: {scenario} ===")
        flags = scenario_flags(scenario)

        # Build dataset
        X, y = build_last_frame_dataset(master_json, flags)

        # Encode labels
        label_map = {"pass": 0, "enter": 1}
        y_encoded = np.array([label_map[label] for label in y])
        class_names = ["pass", "enter"]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, os.path.join(RESULTS_DIR, f"{scenario}_scaler.pkl"))
        joblib.dump(label_map, os.path.join(RESULTS_DIR, f"{scenario}_label_map.pkl"))

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )

        scenario_results = {}

        MODELS = {
            "knn": KNeighborsClassifier(n_neighbors=5),
            "logistic": LogisticRegression(max_iter=200),
            "random_forest": RandomForestClassifier(n_estimators=200)
        }

        for model_name, model in MODELS.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Probabilities for ROC
            y_score = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            cm = confusion_matrix(y_test, y_pred)

            print(f"{model_name}: ACC={acc:.3f}, F1={f1:.3f}")

            # Save model
            joblib.dump(model, os.path.join(RESULTS_DIR, f"{scenario}_{model_name}.pkl"))

            # Plots
            plot_confusion(cm, class_names, os.path.join(RESULTS_DIR, "plots", f"{scenario}_{model_name}_cm.png"))
            if y_score is not None:
                plot_roc_curve(y_test, y_score, os.path.join(RESULTS_DIR, "plots", f"{scenario}_{model_name}_roc.png"))

            # Metrics
            scenario_results[model_name] = {
                "accuracy": float(acc),
                "f1": float(f1),
                "confusion_matrix": cm.tolist(),
                "classes": class_names
            }

        results_dict[scenario] = scenario_results

    # Save summary
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(results_dict, f, indent=2)

    print("\n=== Done. Results saved ===")

if __name__ == "__main__":
    main()
