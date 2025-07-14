import cv2
import os
import datetime
from ultralytics import YOLO
import torch
import math
import numpy as np

# Check for CUDA
print("CUDA Available:", torch.cuda.is_available())

def distance_to_door(x, y, door_region):
    door_center_x = (door_region[0] + door_region[2]) // 2
    door_center_y = (door_region[1] + door_region[3]) // 2
    return math.sqrt((x-door_center_x) ** 2 + (y-door_center_y) ** 2)

def is_inside_ellipse(x, y, center, axes):
    dx = x - center[0]
    dy = y - center[1]
    if dy > 0:
        return False
    return (dx*dx)/(axes[0]*axes[0]) + (dy*dy)/(axes[1]*axes[1]) <= 1


# Define output directories
OUTPUT_DIR = "live_output"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
ENTER_DIR = os.path.join(OUTPUT_DIR, "enter")
PASS_DIR = os.path.join(OUTPUT_DIR, "pass")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ENTER_DIR, exist_ok=True)
os.makedirs(PASS_DIR, exist_ok=True)

# Load YOLO model
yolo_model = YOLO("yolo11n.pt")

# Setup the camera
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 30

mtx = np.array([[882.97786113, 0, 969.24568304],
             [0, 863.01861637, 504.95560309],
             [0, 0, 1]])
dist = np.array([-0.24485893, 0.04001458, 0.0040836, 0.00174867, -0.00255292])

# Precompute undistortion map and get ROI for cropping
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (frame_width, frame_height), 1, (frame_width, frame_height))
map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (frame_width, frame_height), cv2.CV_16SC2)
clean_width, clean_height = roi[2], roi[3]  # width, height after crop

# Define door region (bottom center portion of the frame)
x_roi, y_roi, w_roi, h_roi = roi

door_center = (
    int((w_roi * 0.4 + w_roi * 0.7) / 2),
    int(h_roi)
)

ellipse_axes = (70, 70)

DOOR_REGION = (
    int(w_roi * 0.4),                    # x1 (left)
    int(h_roi - ellipse_axes[1] * 2),   # y1 (top, some margin above ellipse center)
    int(w_roi * 0.7),                    # x2 (right)
    int(h_roi)                          # y2 (bottom)
)

vanish_counter = {}
VANISH_THRESHOLD = 60

tracked_ids = set()
recordings = {}
frame_count = 0

results = None
last_results = None
vanish_points = []

print("Starting live tracking. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    
    # Undistort and crop frame
    frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = roi
    frame = frame[y:y+h, x:x+w]
    
    
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

                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (clean_width, clean_height))

                recordings[track_id] = {
                    "writer": writer,
                    "positions": [],
                    "video_path": video_path,
                    "video_name": video_name
                }

            recordings[track_id]["writer"].write(frame)
            recordings[track_id]["positions"].append(center)

    # Handle vanished IDs
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
            cv2.circle(frame, (int(last_x), int(last_y)), 5, (0, 0, 255), -1)  # Red dot
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

    # Draw door region
    # cv2.rectangle(frame, (DOOR_REGION[0], DOOR_REGION[1]), (DOOR_REGION[2], DOOR_REGION[3]), (0, 255, 0), 2)
    for (vx, vy) in vanish_points:
        cv2.circle(frame, (vx, vy), 5, (0, 0, 255), -1)
    
    cv2.ellipse(
        frame,
        door_center,        # center of the ellipse
        ellipse_axes,           # axes lengths (width radius, height radius)
        0,                  # angle of rotation
        180, 360,             # startAngle, endAngle → 0 to 180 = upper half
        (255, 0, 0),        # blue
        2                   # thickness
    )
    
    display_frame = cv2.resize(frame, (640, 360))
    cv2.imshow("Live Tracking", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
for rec in recordings.values():
    rec["writer"].release()