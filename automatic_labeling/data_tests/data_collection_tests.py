import cv2
import json
import os
import time
import datetime
import math
import numpy as np
import subprocess
import re
import csv
import traceback
import select

from ultralytics import YOLO
from collections import defaultdict
from jtop import jtop

try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False
    

class LiveTracker:
    def __init__(self,
                 yolo_model="yolo11s.pt",
                 ellipse_axes=(70, 70),
                 vanish_threshold=90,
                 extra_seconds=1,
                 camera_index=0,
                 computer="Jetson",
                 distort=False,
                 fullscreen=False,
                 debug=False,
                 sample_interval=30,
                 cadence=1):
        
        scenario_name = input("Please input a scenario, e.g. 'ms_cad1' (Model Small, Cadence 1): ")
        
        # --- Output dirs ---
        self.OUTPUT_DIR = f"{scenario_name}_live_output"
        self.TEMP_DIR = os.path.join(self.OUTPUT_DIR, "temp")
        self.ENTER_DIR = os.path.join(self.OUTPUT_DIR, "enter")
        self.PASS_DIR = os.path.join(self.OUTPUT_DIR, "pass")
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.ENTER_DIR, exist_ok=True)
        os.makedirs(self.PASS_DIR, exist_ok=True)

        # --- Params ---
        self.ellipse_axes = ellipse_axes
        self.vanish_threshold = int(vanish_threshold / cadence)
        self.extra_seconds = extra_seconds
        self.camera_index = camera_index
        self.computer = computer
        self.distort = distort
        self.fullscreen = fullscreen
        self.debug = debug
        self.sample_interval = sample_interval
        self.cadence = cadence

        # --- Camera calibration ---
        if computer=="ASUS":
            self.mtx = np.array([[882.97786113, 0, 969.24568304],
                                 [0, 863.01861637, 504.95560309],
                                 [0, 0, 1]])
            self.dist = np.array([-0.24485893, 0.04001458, 0.0040836, 0.00174867, -0.00255292])
        elif computer=="Jetson":
            self.mtx = np.array([[307.8047385, 0, 355.19676862],
                                 [0, 302.44366762, 233.22849986],
                                 [0, 0, 1]])
            self.dist = np.array([-0.290496, 0.07539763, -0.00075077, -0.00159761, -0.00811828])
        
        # --- YOLO ---
        self.yolo_model = YOLO(yolo_model)

        # --- Capture ---
        self.cap = cv2.VideoCapture(self.camera_index)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25

        # --- Frame --- 
        if self.distort:
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, 
                (self.frame_width, self.frame_height), 1,
                (self.frame_width, self.frame_height)
            )

            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.newcameramtx, 
                (self.frame_width, self.frame_height), cv2.CV_32FC1
            )

            self.clean_width, self.clean_height = self.roi[2], self.roi[3]
            self.door_center = (self.clean_width // 2, self.clean_height)
        
        else:
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist,
                (self.frame_width, self.frame_height), 1,
                (self.frame_width, self.frame_height)
            )
            self.clean_width, self.clean_height = self.frame_width, self.frame_height
            self.door_center = (self.clean_width // 2, self.clean_height)

        # --- Tracking memory ---
        self.tracked_ids = set()
        self.recordings = {}
        self.vanish_counter = {}
        self.vanish_points = []
        self.frame_counter = 0
        self.last_detected_ids = set()
        
        # --- FPS tracking ---
        self.frame_count_for_fps = 0
        self.fps = 0
        self._prev_fps_time = time.time()
        
        # --- Performance CSV (one file per session) ---
        self.performance_csv = os.path.join(self.OUTPUT_DIR, f"{scenario_name}_performance_log.csv")
        os.makedirs(os.path.dirname(self.performance_csv), exist_ok=True)
        
        self.last_sample_time = time.time()


    def is_inside_ellipse(self, x, y):
        """
        Function to check if point is in ellipse.
        
        Args:
            x (float) - x position
            y (float) - y position
            
        Returns:
            Binary
        """
        dx = x - self.door_center[0]
        dy = y - self.door_center[1]
        if dy > 0:
            return False
        return ((dx*dx)/(self.ellipse_axes[0]**2))+((dy*dy)/(self.ellipse_axes[1]**2)) <= 1

    def sample_system_stats_desktop(self):
        data = {
            "ram_used_mb": None, "ram_total_mb": None,
            "cpu_load_percent": None, "gpu_load_percent": None,
            "emc_freq_percent": None, "gpu_freq_percent": None,
            "gpu_mem_used_mb": None, "gpu_mem_total_mb": None,
            "power_mw": None
        }
        try:
            if psutil:
                vm = psutil.virtual_memory()
                data["ram_used_mb"] = int((vm.total - vm.available) / 1024**2)
                data["ram_total_mb"] = int(vm.total / 1024**2)
                data["cpu_load_percent"] = round(psutil.cpu_percent(interval=0.0), 1)
            # GPU stats via NVML
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    data["gpu_load_percent"] = util.gpu if util else None
                    data["gpu_mem_used_mb"] = meminfo.used // 1024**2
                    data["gpu_mem_total_mb"] = meminfo.total // 1024**2
                    try:
                        data["power_mw"] = pynvml.nvmlDeviceGetPowerUsage(handle)
                    except Exception:
                        pass
                except Exception as e:
                    print("[WARN] NVML read failed in desktop sampler:", e)
        except Exception:
            pass
        return data
    
    def get_performance_sample(self):
        sample = {"timestamp": int(time.time()), "fps": int(self.fps)}
        try:
            with jtop() as jetson:
                stats = jetson.stats
                # Flatten the stats dictionary for simple CSV writing
                for k, v in stats.items():
                    if isinstance(v, dict):
                        # Flatten sub-dictionaries (e.g., GPU memory)
                        for sub_k, sub_v in v.items():
                            sample[f"{k}_{sub_k}"] = sub_v
                    else:
                        sample[k] = v
        except Exception as e:
            print("[WARN] jtop read failed:", e)

        if self.debug:
            print("final sample:", sample)

        return sample


    def write_performance_csv(self, sample):
        file_exists = os.path.exists(self.performance_csv)
        
        with open(self.performance_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample.keys()))
            
            if not file_exists or os.stat(self.performance_csv).st_size == 0:
                writer.writeheader()
            
            writer.writerow(sample)


    def run(self):
        print("Starting live tracking. Press 'q' to quit.")
        prev_time = time.time()

        # --- Window setting ---
        cv2.namedWindow("Live Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Tracking", (1280, 720))
        if self.fullscreen:
            cv2.setWindowProperty(
                "Live Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_counter += 1
            track = (self.frame_counter % self.cadence == 0)

            if self.distort:
                frame = cv2.remap(
                    frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR
                )
                x, y, w, h = self.roi
                frame = frame[y:y+h, x:x+w]
            else:
                frame = cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)

            clean_frame = frame.copy()
            
            current_ids = set()
            
            if track:

                results = self.yolo_model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    tracker='botsort.yaml',
                    verbose=False,
                ) # botsort

                boxes = results[0].boxes
                current_ids = set()

                if boxes.id is not None:
                    for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                        track_id = int(track_id)
                        x1, y1, x2, y2 = box
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        current_ids.add(track_id)

                        if track_id not in self.tracked_ids:
                            self.start_recording(track_id)
                        
                        self.recordings[track_id]["frames"].append({
                            "frame_index": len(self.recordings[track_id]["frames"]),
                            "center": [int(center[0]), int(center[1])],
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
                        
                        self.recordings[track_id]["writer"].write(clean_frame)

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID {track_id}",
                                    (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                self.last_detected_ids = current_ids
            
            else:
                # still write frames for existing tracked IDs
                current_ids = self.last_detected_ids
            
            self.handle_vanished(current_ids, clean_frame, frame)

            # --- Draw ellipse for door ---
            cv2.ellipse(frame, self.door_center, self.ellipse_axes, 0, 180, 360, (255, 0, 0), 2)
            
            if self.debug:
                for vid, (vx, vy) in self.vanish_points:
                    cv2.circle(frame, (vx, vy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID {vid}", (vx + 5, vy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # --- FPS ---
            self.frame_count_for_fps += 1
            now = time.time()
            if now - self._prev_fps_time >= 1.0:
                self.fps = self.frame_count_for_fps
                self.frame_count_for_fps = 0
                self._prev_fps_time = now
                
            cv2.putText(frame, f"FPS: {self.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if now - self.last_sample_time >= self.sample_interval:
                self.last_sample_time = now
                sample = self.get_performance_sample()
                self.write_performance_csv(sample)
                if self.debug:
                    print(f"[PERF] sample @ {sample['timestamp']}: fps={sample.get('fps')} cpu={sample.get('cpu_load_percent')} gpu={sample.get('gpu_load_percent')} power_mw={sample.get('power_mw')}")

            if self.fullscreen:
                display_frame = frame
            else:
                display_frame = cv2.resize(frame, (1280, 720))
            
            
            cv2.imshow("Live Tracking", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for track_id in list(self.tracked_ids):
            print(f"[FINALIZE] ID {track_id} did not vanish but session ended.")
            self.finish_recording(track_id, clean_frame)

        self.cleanup()


    def start_recording(self, track_id):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"{timestamp}_id{track_id}.mp4"
        video_path = os.path.join(self.TEMP_DIR, video_name)
        
        if int(self.fps) > 1:
            fps = self.fps
        else:
            fps = self.frame_rate
        
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (self.clean_width, self.clean_height)
        )

        self.recordings[track_id] = {
            "writer": writer,
            "frames": [],
            "video_path": video_path,
            "video_name": video_name,
            "vanish_frames": 0
        }

        self.tracked_ids.add(track_id)
        print(f"[START]Tracking ID {track_id}, file: {video_name}")


    def handle_vanished(self, current_ids, clean_frame, frame):
        finished_ids = set()
        for track_id in list(self.tracked_ids):
            
            if track_id not in current_ids:
                self.recordings[track_id]["vanish_frames"] += 1
                
                if self.recordings[track_id]["vanish_frames"] == self.vanish_threshold:
                    print(f"[VANISH] ID {track_id} vanished, adding {self.extra_seconds}s buffer...")
                
                if self.recordings[track_id]["vanish_frames"] >= self.vanish_threshold+self.frame_rate*self.extra_seconds:
                    finished_ids.add(track_id)
            else:
                self.recordings[track_id]["vanish_frames"] = 0
        
        for finished_id in finished_ids:
            self.finish_recording(finished_id, clean_frame)


    def finish_recording(self, track_id, clean_frame):
        rec = self.recordings[track_id]
        rec["writer"].release()

        last_center = rec["frames"][-1]["center"]
        label = "enter" if self.is_inside_ellipse(*last_center) else "pass"
        out_dir = self.ENTER_DIR if label == "enter" else self.PASS_DIR
        
        if self.debug:
            self.vanish_points.append((track_id, tuple(last_center)))

        base_name = rec["video_name"].replace(".mp4", f"_{label}.mp4")
        final_video_path = os.path.join(out_dir, base_name)
        os.rename(rec["video_path"], final_video_path)

        json_path = final_video_path.replace(".mp4", ".json")
        json_data = {
            "id": track_id,
            "label": label,
            "frames": rec["frames"]
        }

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        
        print(f"[SAVED] {final_video_path}")
        print(f"[SAVED] {json_path}")

        self.tracked_ids.remove(track_id)
        del self.recordings[track_id]


    def cleanup(self):
        for rec in self.recordings.values():
            rec["writer"].release()
        self.cap.release()
        cv2.destroyAllWindows()
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def main():
    tracker = LiveTracker(
        yolo_model="yolo11s.pt",
        ellipse_axes=(160, 120),
        vanish_threshold=30,
        extra_seconds=2,
        camera_index=0,
        computer="Jetson",
        distort=False,
        fullscreen=False,
        debug=False,
        sample_interval=15,
        cadence=2
    )
    tracker.run()

if __name__ == "__main__":
    main()
