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

    REDUCED_LANDMARKS = [0, 11, 12, 23, 24, 25, 26, 27, 28] # nose, shoulders, hips, knees, ankles

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
        """
        Run Mediapipe pose on the video, cropping each frame to bbox, and store
        pose landmarks in image coordinates in each frame dict under 'pose_landmarks'.
        Also attach 'frame_size' (width, height) onto the top-level data in skeleton().
        """
        cap = cv2.VideoCapture(video_path)
        pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        out_frames = []

        # We will read frames in sync with frames list — if the video has different
        # number of frames the original code handled that, keep same behavior.
        frame_idx = 0
        frame_w = None
        frame_h = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # store frame size once
            if frame_w is None or frame_h is None:
                frame_h, frame_w = frame.shape[:2]

            # If frames list shorter than video, we still should process until frames length
            if frame_idx >= len(frames):
                break

            f_info = frames[frame_idx].copy()
            bbox = f_info.get('bbox')
            if bbox is None:
                f_info['pose_landmarks'] = None
                out_frames.append(f_info)
                frame_idx += 1
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
                frame_idx += 1
                continue

            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                f_info['pose_landmarks'] = None
                out_frames.append(f_info)
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                landmarks = []
                for lm in res.pose_landmarks.landmark:
                    # convert landmark back to original image coordinates
                    x = lm.x * (x_max - x_min) + x_min
                    y = lm.y * (y_max - y_min) + y_min
                    z = lm.z
                    visibility = getattr(lm, 'visibility', 0.0)
                    landmarks.append([float(x), float(y), float(z), float(visibility)])
                f_info['pose_landmarks'] = landmarks
            else:
                f_info['pose_landmarks'] = None
            out_frames.append(f_info)
            frame_idx += 1

        cap.release()
        pose.close()

        # we will return out_frames; the caller (skeleton) will insert frame_size into the data dict
        return out_frames


    def skeleton(self, video: str, video_json: str) -> dict:
        """
        Load JSON, run mediapipe on it, and attach pose_landmarks.
        Also attaches top-level 'frame_size' = [width, height] to returned data dict.
        """
        with open(video_json, "r") as f:
            data = json.load(f)

        frames = data.get('frames', [])
        augmented_frames = self.run_mediapipe(video, frames)
        data['frames'] = augmented_frames

        # Try to get frame size from the video (useful for door point). If not available,
        # attempt to infer from largest bbox in frames. Default to (0,0) if not found.
        try:
            cap = cv2.VideoCapture(video)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                data['frame_size'] = [int(w), int(h)]
            cap.release()
        except Exception:
            data['frame_size'] = [0, 0]

        # fallback: if frame_size still zero, try to infer from bbox extents
        if data.get('frame_size', [0, 0]) == [0, 0]:
            max_w = 0
            max_h = 0
            for f in frames:
                bb = f.get('bbox')
                if bb:
                    max_w = max(max_w, bb[2])
                    max_h = max(max_h, bb[3])
            if max_w > 0 and max_h > 0:
                data['frame_size'] = [int(max_w), int(max_h)]

        return data
    
    def skeleton_angles(self, video_json: dict, selected_landmarks=[0,11,12,23,24]) -> dict:
        data = dict(video_json)
        for frame in data.get('frames', []):
            lm = frame.get('pose_landmarks')
            if lm is None:
                frame['pose_angles'] = None
                continue
            try:
                arr = np.array(lm)[:, :3]  # x, y, z
                h, w = frame['frame_height'], frame['frame_width']
                door_point = np.array([w/2, h, 0])
                angles = {}
                for idx in selected_landmarks:
                    joint = arr[idx] if idx < len(arr) else np.array([np.nan, np.nan, np.nan])
                    vec = door_point - joint
                    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
                    angles[f"joint{idx}_to_door"] = angle
                frame['pose_angles'] = angles
            except Exception:
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
            use_reduced_skeleton: bool = True
            ):
        
        frames = video_json_data.get('frames', [])
        feats = []
        prev_center = None

        for frame in frames:
            frame_feats = []

            basic = self.box_features(frame)
            frame_feats.extend(basic)

            if use_skeleton:
                lm = frame.get('pose_landmarks')
                if lm is not None:
                    coords = []
                    if use_reduced_skeleton:
                        # use the reduced list of landmark indexes
                        for idx in self.REDUCED_LANDMARKS:
                            if idx < len(lm):
                                coords.extend([float(lm[idx][0]), float(lm[idx][1]), float(lm[idx][2])])
                            else:
                                coords.extend([0.0, 0.0, 0.0])
                    else:
                        # legacy: include up to self.n_landmarks
                        for i in range(self.n_landmarks):
                            if i < len(lm):
                                coords.extend([float(lm[i][0]), float(lm[i][1]), float(lm[i][2])])
                            else:
                                coords.extend([0.0, 0.0, 0.0])
                    frame_feats.extend(coords)
                else:
                    # pad with zeros for reduced skeleton length
                    if use_reduced_skeleton:
                        frame_feats.extend([0.0] * (len(self.REDUCED_LANDMARKS) * 3))
                    else:
                        frame_feats.extend([0.0] * (self.n_landmarks * 3))
            
            if use_angles:
                ang = frame.get('pose_angles')
                if ang:
                    # we store six angles as described earlier
                    frame_feats.extend([
                        float(ang.get('shoulder_left_angle', 0.0)) if ang.get('shoulder_left_angle') is not None else 0.0,
                        float(ang.get('shoulder_right_angle', 0.0)) if ang.get('shoulder_right_angle') is not None else 0.0,
                        float(ang.get('shoulder_door_angle', 0.0)) if ang.get('shoulder_door_angle') is not None else 0.0,
                        float(ang.get('hip_left_angle', 0.0)) if ang.get('hip_left_angle') is not None else 0.0,
                        float(ang.get('hip_right_angle', 0.0)) if ang.get('hip_right_angle') is not None else 0.0,
                        float(ang.get('hip_door_angle', 0.0)) if ang.get('hip_door_angle') is not None else 0.0
                    ])
                else:
                    frame_feats.extend([0.0] * 6)
            feats.append(frame_feats)
        return np.array(feats, dtype=np.float32)
    
    def prepare_data(
            self, 
            video_path: str, 
            json_path: str, 
            tail: float = 0.1,
            use_skeleton: bool = True,
            use_angles: bool = True,
            use_reduced_skeleton: bool = True
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

        X = self.feature_extraction(past_data, use_skeleton=use_skeleton,
                                    use_angles=use_angles,
                                    use_reduced_skeleton=use_reduced_skeleton)
        y = self.feature_extraction(future_data, use_skeleton=use_skeleton,
                                    use_angles=use_angles,
                                    use_reduced_skeleton=use_reduced_skeleton)

        return X, y, past_data, future_data
    
    def build_dataset(
            self,
            pairs: List[Dict[str, str]],
            tail: float = 0.1,
            use_skeleton: bool = True,
            use_angles: bool = True,
            use_reduces_skeleton: bool = True,
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
            return np.zeros((0,0,0), dtype=dtype), np.array([], dtype=np.int32)
        
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
    
    def build_features(self, frames, use_bbox=True, use_skeleton=True, use_angles=True, selected_landmarks=[0,11,12,23,24]):
        """
        Returns concatenated feature array for one video.
        """
        features = []
        for f in frames:
            frame_feats = []
            if use_bbox:
                bbox = f.get('bbox')
                frame_feats.append(np.array(bbox).flatten() if bbox is not None else np.zeros(4))
            if use_skeleton:
                lm = f.get('pose_landmarks')
                lm_sel = np.array(lm)[selected_landmarks,:3].flatten() if lm is not None else np.zeros(3*len(selected_landmarks))
                frame_feats.append(lm_sel)
            if use_angles:
                ang = f.get('pose_angles')
                ang_vals = np.array(list(ang.values())).flatten() if ang is not None else np.zeros(len(selected_landmarks))
                frame_feats.append(ang_vals)
            features.append(np.concatenate(frame_feats))
        return np.array(features)  # shape: (T, F)

dataprocessing = DataProcessing()
datasetup = DataSetup()