import torch
from ultralytics import YOLO
import cv2
from collections import defaultdict
import csv
import os
import json


model = YOLO("yolo11s.pt")

vid = r'C:\Users\hanna\Documents\Thesis\exjobb\MKI\live_output\enter\20250726_183411_enter_1.mp4'


def no_boxes_json():
    cap = cv2.VideoCapture(vid)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writers = {}
    person_data = defaultdict(list)

    results = model.track(
        source=vid,
        stream=True,
        persist=True,
        classes=[0]
    )

    frame_idx = 0
    for result in results:
        frame = result.orig_img.copy()

        if result.boxes is None:
            frame_idx += 1
            continue

        boxes = result.boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None else []

        for i, boxes in enumerate(boxes):
            track_id = int(ids[i]) if i < len(ids) else None
            xyxy = boxes.xyxy[0].cpu().numpy().astype(int).tolist()

            person_data[track_id].append({
                "frame": frame_idx,
                "bbox": xyxy
            })

            if track_id not in writers:
                out_path = f"person_{track_id}.mp4"
                writers[track_id] = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            writers[track_id].write(frame)
        
        frame_idx += 1
    
    for track_id, data in person_data.items():
        json_path = f"person_{track_id}.json"
        with open(json_path, 'w') as jf:
            json.dump(data, jf, indent=2)

    for writer in writers.values():
        writer.release()

def no_boxes():
    cap = cv2.VideoCapture(vid)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writers = {}

    output_file = open('tracking_data.csv', 'w', newline='')
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["frame", "id", "x1", "y1", "x2", "y2"])

    results = model.track(
        source=vid,
        stream=True,
        persist=True,
        classes=[0]
    )

    frame_idx = 0

    for result in results:
        frame = result.orig_img.copy()

        if result.boxes is None:
            frame_idx += 1
            continue

        boxes = result.boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None else []

        for i, box in enumerate(boxes):
            track_id = int(ids[i]) if i < len(ids) else None
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            csv_writer.writerow([frame_idx, track_id, *xyxy])

            if track_id not in writers:
                out_path = f'person_{track_id}.mp4'
                writers[track_id] = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            writers[track_id].write(frame)
        frame_idx += 1
    
    output_file.close()
    for writer in writers.values():
        writer.release()

def boxes():
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    results = model.track(
        source=vid,
        stream=True,
        persist=True,
        classes=[0]
    )

    writers = {}
    id_frames = defaultdict(list)

    frame_idx = 0

    for result in results:
        frame = result.orig_img
        if result.boxes is None:
            frame_idx += 1
            continue

        boxes = result.boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None else []

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            track_id = int(ids[i]) if i < len(ids) else None

            if cls == 0 and track_id is not None:
                focused_frame = frame.copy()

                
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                cv2.putText(frame, f'ID {track_id}', (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if track_id not in writers:
                    out_path = f'person_{track_id}.mp4'
                    writers[track_id] = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                writers[track_id].write(focused_frame)
        frame_idx += 1

    for writer in writers.values():
        writer.release()

no_boxes_json()

# model.track(
#     source=vid,
#     persist=True,
#     show=True,
#     save=True,
#     classes=[0],
#     tracker="botsort.yaml"
# )