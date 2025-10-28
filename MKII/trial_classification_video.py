import cv2
import os
import json
import math
import datetime
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Config
VIDEO_PATH = r'C:\Users\hanna\Documents\Thesis\exjobb\MKI\live_output\enter\20250726_183411_enter_1.mp4'
MODEL = "yolo11s.pt"
OUTPUT_DIR = "output"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
ENTER_DIR = os.path.join(OUTPUT_DIR, "enter")
PASS_DIR = os.path.join(OUTPUT_DIR, "pass")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ENTER_DIR, exist_ok=True)
os.makedirs(PASS_DIR, exist_ok=True)

# Door region
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 30


door_center = (
    int((frame_width * 0.4 + frame_width * 0.7) / 2),
    int(frame_height)
)
ellipse_axes = (70, 70)
DOOR_REGION = (
    int(frame_width * 0.4),
    int(frame_height - ellipse_axes[1] * 2),
    int(frame_width * 0.7),
    int(frame_height)
)

def distance_to_door(x, y, door_region):
    door_center_x = (door_region[0] + door_region[2]) // 2
    door_center_y = (door_region[1] + door_region[3]) // 2
    return math.sqrt((x - door_center_x) ** 2 + (y - door_center_y) ** 2)

def is_inside_ellipse(x, y, center, axes):
    dx = x - center[0]
    dy = y - center[1]
    if dy > 0:
        return False
    return (dx * dx) / (axes[0] * axes[0]) + (dy * dy) / (axes[1] * axes[1]) <= 1

# Model
yolo_model = YOLO(MODEL)

# Trackers
writers = {}
json_data = defaultdict(list)
recordings = {}
vanish_counter = {}
tracked_ids = set()
finished_ids = set()
frame_idx = 0
VANISH_THRESHOLD = 50

try:
    # Run YOLO tracking on ideo
    results = yolo_model.track(
        source=VIDEO_PATH,
        stream=True,
        persist=True,
        classes=[0],
        tracker='botsort.yaml'
    )

    for result in results:
        frame = result.orig_img.copy()
        frame_idx += 1

        boxes = result.boxes
        current_ids = set()

        if boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            for i, box in enumerate(boxes):
                track_id = int(ids[i]) if i < len(ids) else None
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = xyxy
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_ids.add(track_id)

                if track_id not in tracked_ids:
                    tracked_ids.add(track_id)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_name = f"{timestamp}_track_{track_id}.mp4"
                    video_path = os.path.join(TEMP_DIR, video_name)
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
                    writers[track_id] = writer
                    recordings[track_id] = {
                        "video_path": video_path,
                        "video_name": video_name,
                        "positions": [],
                    }
                # save frame to person video
                writers[track_id].write(frame)

                # save center of path + JSON
                recordings[track_id]["positions"].append(center)
                json_data[track_id].append({
                    "frame": frame_idx,
                    "bbox": xyxy,
                    "center": center,
                })

        # Handle vanishing tracks
        for track_id in tracked_ids:
            if track_id not in current_ids and track_id not in finished_ids:
                vanish_counter[track_id] = vanish_counter.get(track_id, 0) + 1
                if vanish_counter[track_id] >= VANISH_THRESHOLD:
                    finished_ids.add(track_id)



        for track_id in list(finished_ids):
            if track_id in recordings:
                last_x, last_y = recordings[track_id]["positions"][-1]
                print("checking if inside ellipse")
                if is_inside_ellipse(last_x, last_y, door_center, ellipse_axes):
                    label = "enter"
                    out_dir = ENTER_DIR
                    print("enter")
                else:
                    label = "pass"
                    out_dir = PASS_DIR
                    print("pass")

                old_path = recordings[track_id]["video_path"]
                new_name = recordings[track_id]["video_name"].replace("track", label)
                new_path = os.path.join(out_dir, new_name)

                writer = writers.pop(track_id, None)
                if writer is not None:
                    writer.release()

                os.rename(old_path, new_path)

                json_output = {
                    "id": track_id,
                    "label": label,
                    "trajectory": json_data[track_id]
                }
                json_name = new_name.replace(".mp4", ".json")
                with open(os.path.join(out_dir, json_name), 'w') as jf:
                    json.dump(json_output, jf, indent=2)

                print(f"Saved: {new_path} with label '{label}'")

                del recordings[track_id]

                tracked_ids.remove(track_id)
                finished_ids.remove(track_id)
            else:
                print(f"Track ID {track_id} not found in recordings, skipping.")
finally:
    cap.release()

    for writer in writers.values():
        writer.release()
    writers.clear()

    for track_id in list(tracked_ids):
        if track_id in recordings:
            last_x, last_y = recordings[track_id]["positions"][-1]
            label = "enter" if is_inside_ellipse(last_x, last_y, door_center, ellipse_axes) else "pass"
            out_dir = ENTER_DIR if label == "enter" else PASS_DIR

            # Safely release the writer
            writer = writers.pop(track_id, None)
            if writer is not None:
                writer.release()

            # Rename and save
            old_path = recordings[track_id]["video_path"]
            new_name = recordings[track_id]["video_name"].replace("track", label)
            new_path = os.path.join(out_dir, new_name)

            try:
                os.rename(old_path, new_path)
            except PermissionError:
                print(f"❌ Could not move {old_path} – still in use.")
                continue

            json_output = {
                "id": track_id,
                "label": label,
                "trajectory": json_data[track_id],
                "finalized_by": "video_end"
            }
            json_name = new_name.replace(".mp4", ".json")
            with open(os.path.join(out_dir, json_name), 'w') as jf:
                json.dump(json_output, jf, indent=2)

            print(f"Saved (video end): {new_path} with label '{label}'")

            # cleanup
            del recordings[track_id]
            tracked_ids.remove(track_id)
