import cv2
import os
import datetime
from ultralytics import YOLO
import torch
import math
import numpy as np
import time

prev_time = time.time()
actual_fps = 0

def distance_to_door(x, y, door_region):
    door_center_x = (door_region[0] + door_region[2]) // 2
    door_center_y = (door_region[1] + door_region[3]) // 2
    return math.sqrt((x-door_center_x) ** 2 + (y-door_center_y) ** 2)

def is_inside_ellipse(x, y, center, axes):
    dx = x - center[0]
    dy = y - center[1]
    if dy > 0:
        return False
    return (dx*dx)/(axes[0]**2) + (dy*dy)/(axes[1]**2) <= 1

# === Config ===
OUTPUT_DIR = "live_output_undistort"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
ENTER_DIR = os.path.join(OUTPUT_DIR, "enter")
PASS_DIR = os.path.join(OUTPUT_DIR, "pass")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ENTER_DIR, exist_ok=True)
os.makedirs(PASS_DIR, exist_ok=True)

yolo_model = YOLO("yolo11s.pt")

cap = cv2.VideoCapture(0)
cv2.namedWindow('Live Tracking (Undistorted)', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Live Tracking (Undistorted)', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) # or 30

mtx = np.array([[307.8047385, 0, 355.19676862],
                [0, 302.44366762, 233.22849986],
                [0, 0, 1]])
dist = np.array([-0.290496, 0.07539763, -0.00075077, -0.00159761, -0.00811828])

door_center = (int(frame_width * 0.55), frame_height)
ellipse_axes = (280, 140)
DOOR_REGION = (
    int(frame_width * 0.4),
    int(frame_height - ellipse_axes[1] * 2),
    int(frame_width * 0.7),
    frame_height
)

vanish_counter = {}
VANISH_THRESHOLD = 120
tracked_ids = set()
recordings = {}
frame_count = 0
results = None
last_results = None
vanish_points = []

print("Starting UNDISTORT live tracking. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # === Undistort full frame ===
    frame = cv2.undistort(frame, mtx, dist, None, mtx)

    # === YOLO tracking ===
    if frame_count % 2 == 0:
        results = yolo_model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)
        last_results = results
    elif last_results is not None:
        results = last_results
    else:
        continue

    boxes = results[0].boxes
    current_ids = set()

    if boxes.id is not None:
        for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
            track_id = int(track_id)
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            current_ids.add(track_id)

            if track_id not in tracked_ids:
                tracked_ids.add(track_id)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_name = f"{timestamp}_track_{track_id}.mp4"
                video_path = os.path.join(TEMP_DIR, video_name)
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
                recordings[track_id] = {
                    "writer": writer,
                    "positions": [],
                    "video_path": video_path,
                    "video_name": video_name
                }

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            recordings[track_id]["writer"].write(frame)
            recordings[track_id]["positions"].append(center)

    # === Handle vanished tracks ===
    finished_ids = set()
    for track_id in tracked_ids:
        if track_id not in current_ids:
            vanish_counter[track_id] = vanish_counter.get(track_id, 0) + 1
            if vanish_counter[track_id] >= VANISH_THRESHOLD:
                finished_ids.add(track_id)
        else:
            vanish_counter[track_id] = 0

    for finished_id in finished_ids:
        if finished_id in recordings:
            writer = recordings[finished_id]["writer"]
            writer.release()

            last_x, last_y = recordings[finished_id]["positions"][-1]
            vanish_points.append((int(last_x), int(last_y)))

            door_dist = distance_to_door(last_x, last_y, DOOR_REGION)
            print(f"Track ID {finished_id} vanished at ({last_x},{last_y}), distance to door: {door_dist:.2f} pixels.")

            if is_inside_ellipse(last_x, last_y, door_center, ellipse_axes):
                label = "enter"
                out_dir = ENTER_DIR
            else:
                label = "pass"
                out_dir = PASS_DIR

            new_name = recordings[finished_id]["video_name"].replace("track", label)
            new_path = os.path.join(out_dir, new_name)
            os.rename(recordings[finished_id]["video_path"], new_path)

            print(f"Saved: {new_path} for track ID {finished_id}")

            del recordings[finished_id]
            tracked_ids.remove(finished_id)

    # === Draw visuals ===
    for (vx, vy) in vanish_points:
        cv2.circle(frame, (vx, vy), 5, (0, 0, 255), -1)

    cv2.ellipse(frame, door_center, ellipse_axes, 0, 180, 360, (255, 0, 0), 2)
    
    # --- FPS ---
    current_time = time.time()
    actual_fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 30),
		    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    display_frame = cv2.resize(frame, (1920, 1200))
    cv2.imshow("Live Tracking (Undistorted)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
for rec in recordings.values():
    rec["writer"].release()
