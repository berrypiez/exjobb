import json
import os

# INPUT_JSON = "C:/Users/hanna/Documents/Thesis/exjobb/TP_MKI/master_jsons/office_master_dataset.json"
# OUTPUT_JSON = "C:/Users/hanna/Documents/Thesis/exjobb/TP_MKI/master_jsons/office_data_trimmed.json"

INPUT_JSON = "C:/Users/hanna/Documents/Thesis/exjobb/TP_MKI/master_jsons/master_dataset_office_holdout.json"
OUTPUT_JSON = "C:/Users/hanna/Documents/Thesis/exjobb/TP_MKI/master_jsons/master_dataset_office_holdout_trimmed.json"

def trim_trailing(data):
    trimmed = {}

    for video_name, video_data in data.items():
        frames = video_data.get("frames", [])
        last_valid_index = None
        for i in range(len(frames) -1, -1, -1):
            bbox = frames[i].get("bbox", None)
            if bbox is not None:
                last_valid_index = i
                break

        if last_valid_index is None:
            trimmed_frames = []
        else:
            trimmed_frames = frames[:last_valid_index + 1]
        
        trimmed[video_name] = {
            **video_data,
            "frames": trimmed_frames
        }
    return trimmed

def main():
    print(f"Loading: {INPUT_JSON}...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    print(f"Trimming trailing None bboxes from {len(data)} videos...")
    trimmed_data = trim_trailing(data)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(trimmed_data, f, indent=2)

    print(f"Trimmed data saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
