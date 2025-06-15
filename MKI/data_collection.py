import cv2
import os
import datetime
from ultralytics import YOLO

OUTPUT_DIR = "live_output"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
ENTER_DIR = os.path.join(OUTPUT_DIR, "enter")
PASS_DIR = os.path.join(OUTPUT_DIR, "pass")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ENTER_DIR, exist_ok=True)
os.makedirs(PASS_DIR, exist_ok=True)

yolo_model = YOLO("yolov8s.pt")

DOOR_REGION = () # Define the door region (x1, y1, x2, y2)

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

tracked_ids = set()
recordings = {}

print("Starting live tracking. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
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
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    frame_rate,
                    (frame_width, frame_height)
                )
                recordings[track_id] = {
                    "writer": writer,
                    "positions": [],
                    "video_path": video_path,
                    "video_name": video_name
                }

            # Write frame and store position
            recordings[track_id]["writer"].write(frame)
            recordings[track_id]["positions"].append(center)
        
        # Check for completed tracks
        finished_ids = tracked_ids - current_ids
        for finished_id in list(finished_ids):
            if finished_id in recordings:
                writer = recordings[finished_id]["writer"]
                writer.release()

                last_x, last_y = recordings[finished_id]["positions"][-1]
                if (DOOR_REGION[0] <= last_x <= DOOR_REGION[2]) and (DOOR_REGION[1] <= last_y <= DOOR_REGION[3]):
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

        # Draw door
        cv2.rectangle(frame, (DOOR_REGION[0], DOOR_REGION[1]), (DOOR_REGION[2], DOOR_REGION[3]), (0, 255, 0), 2)
        cv2.imshow("Live Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    for rec in recordings.values():
        rec["writer"].release()