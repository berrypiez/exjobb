"""

In this file you prepare the data for the different scenarios.

"""
import os
import json
import math
from typing import List, Dict, Tuple, Optional, Any

import cv2
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import numpy as np


class DataSetup:

    def data_packager(self, repos):
        if isinstance(repos, str):
            repos = [repos]

        pairs = []

        for repo in repos:
            repo = os.path.normpath(repo)
            for scenario in os.listdir(repo):
                scenario_path = os.path.join(repo, scenario)
                if not os.path.isdir(scenario_path):
                    continue

                for label in ("enter", "pass"):
                    label_path = os.path.join(scenario_path, label)
                    if not os.path.isdir(label_path):
                        continue

                    videos, jsons = {}, {}
                    for f in os.listdir(label_path):
                        full_path = os.path.join(label_path, f)
                        key = os.path.splitext(f)[0]
                        if f.lower().endswith(".mp4"):
                            videos[key] = full_path
                        elif f.lower().endswith(".json"):
                            jsons[key] = full_path

                    for key, vpath in videos.items():
                        if key in jsons:
                            pairs.append({
                                "scenario": scenario,
                                "label": label,
                                "video": vpath,
                                "json": jsons[key]
                            })
        return pairs


class DataProcessing:

    def __init__(self, n_landmarks: int = 33):
        self.n_landmarks = n_landmarks
        self.mp_pose = mp.solutions.pose

    def test(self):
        print("You called")

    def qualification(self, video_json, min_frames=12) -> bool:
        # check that the number of frames in json exceeds a certain number
        with open(video_json, "r") as f:
            data = json.load(f)
        return len(data.get('frames', [])) >= min_frames


    def cutting(self, video_mp4: str, video_json: str, tail: float = 0.1, out_dir: str ="cut_videos/"):
        os.makedirs(out_dir, exist_ok=True)

        with open(video_json, "r") as f:
            data = json.load(f)

        total_frames = len(data['frames'])
        cut_frames = max(1, int(total_frames * tail))
        new_total_frames = total_frames - cut_frames

        new_data = dict(data)
        new_data['frames'] = data['frames'][:new_total_frames]

        cap = cv2.VideoCapture(video_mp4)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        base_name = os.path.splitext(os.path.basename(video_mp4))[0]
        out_video_path = os.path.join(out_dir, f"{base_name}_cut.mp4")
        out_json_path = os.path.join(out_dir, f"{base_name}_cut.json")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        while frame_idx < new_total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_idx += 1
        cap.release()
        writer.release()

        with open(out_json_path, "w") as f:
            json.dump(new_data, f, indent=2)

        return out_video_path, out_json_path


    def run_mediapipe(self, video_path: str, frames: List[dict]) -> List[dict]:
        cap = cv2.VideoCapture(video_path)
        pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        out_frames = []

        for i,f_info in enumerate(frames):
            ret, frame = cap.read()
            if not ret:
                f_info['pose_landmarks'] = None
                out_frames.append(f_info)
                continue
            bbox = f_info.get('bbox')
            if bbox is None:
                f_info['pose_landmarks'] = None
                out_frames.append(f_info)
                continue

            x_min, y_min, x_max, y_max = bbox
            h, w = frame.shape[:2]

            x_min = int(max(0, min(x_min, w-1)))
            y_min = int(max(0, min(y_min, h-1)))
            x_max = int(max(0, min(x_max, w-1)))
            y_max = int(max(0, min(y_max, h-1)))

            if x_max <= x_min or y_max <= y_min:
                f_info['pose_landmarks'] = None
                out_frames.append(f_info)
                continue

            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                f_info['pose_landmarks'] = None
                out_frames.append(f_info)
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                landmarks = []
                for lm in res.pose_landmarks.landmark:
                    x = lm.x * (x_max - x_min) + x_min
                    y = lm.y * (y_max - y_min) + y_min
                    z = lm.z
                    visibility = getattr(lm, 'visibility', 0.0)
                    landmarks.append([float(x), float(y), float(z), float(visibility)])
                f_info['pose_landmarks'] = landmarks
            else:
                f_info['pose_landmarks'] = None
            out_frames.append(f_info)
        cap.release()
        pose.close()
        return out_frames


    def skeleton(self, video: str, video_json: str) -> dict:
        # using bounding boxes as guidance, extract the skeleton in the video
        # load the json
        with open(video_json, "r") as f:
            data = json.load(f)

        frames = data.get('frames', [])
        augmented_frames = self.run_mediapipe(video, frames)
        data['frames'] = augmented_frames
        return data
    
    def skeleton_angles(self, video_json: dict) -> dict:
        data = dict(video_json) 

        for frame in data.get('frames', []):
            lm = frame.get('pose_landmarks')
            if lm is None:
                frame['pose_angles'] = None
                continue

            try:
                arr = np.array(lm)[:, :3] # ignore visibility, take x,y,z
                def safe_get(idx):
                    return arr[idx] if idx < len(arr) else np.array([np.nan, np.nan, np.nan])
                
                # Shoulders
                ls = safe_get(11) # left shoulder
                rs = safe_get(12) # right shoulder
                shoulder_vec = rs - ls
                shoulder_angle = float(math.degrees(math.atan2(shoulder_vec[1], shoulder_vec[0])))

                # Hips
                lh = safe_get(23) # left hip
                rh = safe_get(24) # right hip
                hip_vec = rh - lh
                hip_angle = float(math.degrees(math.atan2(hip_vec[1], hip_vec[0])))

                # Feet (ankles)
                la = safe_get(27) # left ankle
                ra = safe_get(28) # right ankle
                feet_vec = ra - la
                feet_angle = float(math.degrees(math.atan2(feet_vec[1], feet_vec[0])))

                # Head direction
                nose = safe_get(0) # nose
                shoulder_mid = (ls + rs) / 2
                head_vec = shoulder_mid - nose
                head_angle = float(math.degrees(math.atan2(head_vec[1], head_vec[0])))

                frame['pose_angles'] = {
                    'shoulder_angle': shoulder_angle,
                    'hip_angle': hip_angle,
                    'feet_angle': feet_angle,
                    'head_angle': head_angle
                }

            except Exception as e:
                frame['pose_angles'] = None
            
        return data
    
    def master_json(self, video_path: str, json_path: str, out_dir: str ='master_jsons/') -> str:
        os.makedirs(out_dir, exist_ok=True)

        with open(json_path, "r") as f:
            base = json.load(f)

        data_with_skeleton = self.skeleton(video_path, json_path)
        master_json = self.skeleton_angles(data_with_skeleton)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        master_json_path = os.path.join(out_dir, f"{base_name}_master.json")
        with open(master_json_path, "w") as f:
            json.dump(master_json, f, indent=4)

        return master_json_path
    
    def box_features(self, frame: dict) -> List[float]:
        if "bbox" in frame and frame["bbox"] is not None:
            x_min, y_min, x_max, y_max = map(float, frame["bbox"])
            w = max(1.0, x_max - x_min)
            h = max(1.0, y_max - y_min)
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            area = w * h
            return [cx, cy, w, h, area]
        else:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    def feature_extraction(
            self, 
            video_json_data: dict, 
            use_skeleton: bool = True, 
            use_angles: bool = True,
            add_velocity: bool = True):
        
        frames = video_json_data.get('frames', [])
        feats = []
        prev_center = None

        for frame in frames:
            frame_feats = []

            basic = self.box_features(frame)
            frame_feats.extend(basic)

            if add_velocity:
                cx, cy = basic[0], basic[1]
                if prev_center is not None:
                    vx = cx - prev_center[0]
                    vy = cy - prev_center[1]
                else:
                    vx, vy = 0.0, 0.0
                frame_feats.extend([vx, vy])
                prev_center = (cx, cy)

            if use_skeleton:
                lm = frame.get('pose_landmarks')
                if lm is not None:
                    coords = []
                    for i in range(self.n_landmarks):
                        if i < len(lm):
                            coords.extend([float(lm[i][0]), float(lm[i][1]), float(lm[i][2])])
                        else:
                            coords.extend([0.0, 0.0, 0.0])
                    frame_feats.extend(coords)
                else:
                    frame_feats.extend([0.0] * (self.n_landmarks * 3))
            
            if use_angles:
                ang = frame.get('pose_angles')
                if ang:
                    frame_feats.extend([
                        float(ang.get('shoulder_angle', 0.0)),
                        float(ang.get('hip_angle', 0.0)),
                        float(ang.get('feet_angle', 0.0)),
                        float(ang.get('head_angle', 0.0))
                    ])
                else:
                    frame_feats.extend([0.0, 0.0, 0.0, 0.0])
            feats.append(frame_feats)
        return np.array(feats, dtype=np.float32)
    
    def prepare_data(
            self, 
            video_path: str, 
            json_path: str, 
            tail: float = 0.1,
            use_skeleton: bool = True,
            use_angles: bool = True,
            add_velocity: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
        master_json_path = self.master_json(video_path, json_path)

        with open(master_json_path, "r") as f:
            data = json.load(f)

        total_frames = len(data.get('frames', []))
        if total_frames == 0:
            return np.zeros((0,0)), np.zeros((0,0)), {}, {}
        
        cut_frames = max(1, int(total_frames * tail))
        new_total_frames = total_frames - cut_frames

        past_data = {"frames": data["frames"][:new_total_frames]}
        future_data = {"frames": data["frames"][new_total_frames:]}

        X = self.feature_extraction(past_data, use_skeleton, use_angles, add_velocity)
        y = self.feature_extraction(future_data, use_skeleton, use_angles, add_velocity)

        return X, y, past_data, future_data
    
    def build_dataset(
            self,
            pairs: List[Dict[str, str]],
            tail: float = 0.1,
            use_skeleton: bool = True,
            use_angles: bool = True,
            add_velocity: bool = True,
            skip_short: int = 6
    ) -> Dict[str, str]:
        Xs, ys, labels, len_x, len_y = [], [], [], [], []
        for p in pairs:
            try:
                X, y, _, _ = self.prepare_data(
                    p['video'], 
                    p['json'], 
                    tail,
                    use_skeleton,
                    use_angles,
                    add_velocity
                )
                if X.shape[0] < skip_short or y.shape[0] < 1:
                    continue

                Xs.append(X)
                ys.append(y)
                labels.append(p.get('label', 'unknown'))
                len_x.append(X.shape[0])
                len_y.append(y.shape[0])
            except Exception as e:
                print(f"Error processing {p.get('video')}: {e}")
                continue
            
        return {
            "Xs": Xs,
            "ys": ys,
            "labels": labels,
            "len_x": len_x,
            "len_y": len_y
        }
    
    def pad_sequences(
            self,
            sequences: List[np.ndarray],
            padding: str = 'post',
            dtype = "float32"
    ):
        if len(sequences) == 0:
            return np.zeros((0,0,0), dtype=dtype)
        
        max_t = max([s.shape[0] for s in sequences])
        feat = sequences[0].shape[1]
        out = np.zeros((len(sequences), max_t, feat), dtype=dtype)
        lengths = []
        for i, s in enumerate(sequences):
            t = s.shape[0]
            lengths.append(t)
            if padding == 'post':
                out[i, :t, :] = s
            else:
                out[i, -t:, :] = s
        return out, np.array(lengths, dtype=np.int32)
    
    def fit_scaler(self, Xs: List[np.ndarray]) -> StandardScaler:
        all_data = np.vstack(Xs)
        scaler = StandardScaler()
        scaler.fit(all_data)
        return scaler
    
    def transform_with_scaler(self, seqs: List[np.ndarray], scaler: StandardScaler) -> List[np.ndarray]:
        return [scaler.transform(s) for s in seqs]

dataprocessing = DataProcessing()
datasetup = DataSetup()