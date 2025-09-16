"""

In this file you prepare the data for the different scenarios.

"""
import json
import cv2
import mediapipe as mp
import math
import numpy as np
import os

class DataSetup:

    def data_packager(self, repos):
        if isinstance(repos, str):
            repos = [repos]

        pairs = []

        for repo in repos:
            repo = os.path.normpath(repo)
            scenario = os.path.basename(repo)

            for label in ("enter", "pass"):
                label_path = os.path.join(repo, label)

                if not os.path.isdir(label_path):
                    continue

                videos, jsons = {}, {}

                for f in os.listdir(label_path):
                    full_path = os.path.normpath(os.path.join(label_path, f))
                    if f.endswith(".mp4"):
                        key = os.path.splitext(f)[0]
                        videos[key] = full_path
                    elif f.endswith(".json"):
                        key = os.path.splitext(f)[0]
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

    def test(self):
        print("You called")

    def qualification(self, video_json, min_frames=12):
        # check that the number of frames in json exceeds a certain number
        with open(video_json, "r") as f:
            data = json.load(f)
        
        if len(data['frames']) >= min_frames:
            return True
        else:
            return False

    def cutting(self, video_mp4, video_json, tail=0.1, out_dir="cut_videos/"):
        os.makedirs(out_dir, exist_ok=True)

        with open(video_json, "r") as f:
            print("opened json")
            data = json.load(f)

        total_frames = len(data['frames'])
        cut_frames = max(1, int(total_frames * tail))
        new_total_frames = total_frames - cut_frames

        new_data = data.copy()
        new_data['frames'] = data['frames'][:new_total_frames]

        print("new total frames:", new_total_frames)

        cap = cv2.VideoCapture(video_mp4)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        base_name = os.path.splitext(os.path.basename(video_mp4))[0]
        out_video_path = os.path.join(out_dir, f"{base_name}_cut.mp4")
        out_json_path = os.path.join(out_dir, f"{base_name}_cut.json")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= new_total_frames:
                break
            out_writer.write(frame)
            frame_idx += 1
        cap.release()
        out_writer.release()

        with open(out_json_path, "w") as f:
            json.dump(new_data, f, indent=4)

        return out_video_path, out_json_path

    def skeleton(self, video, video_json):
        # using bounding boxes as guidance, extract the skeleton in the video
        # load the json
        with open(video_json, "r") as f:
            data = json.load(f)

        # open video
        cap = cv2.VideoCapture(video)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True)

        for i, frame_info in enumerate(data['frames']):
            ret, frame = cap.read()
            if not ret:
                frame_info['pose_landmarks'] = None
                continue

            x_min, y_min, x_max, y_max = frame_info['bbox']
            crop_frame = frame[y_min:y_max, x_min:x_max]
            if crop_frame.size == 0:
                frame_info['pose_landmarks'] = None
                continue

            rgb_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    x = lm.x * (x_max - x_min) + x_min
                    y = lm.y * (y_max - y_min) + y_min
                    z = lm.z
                    visibility = lm.visibility
                    landmarks.append([x, y, z, visibility])
                frame_info['pose_landmarks'] = landmarks
            else:
                frame_info['pose_landmarks'] = None
        cap.release()
        pose.close()
        return data
    
    def skeleton_angles(self, video_json):
        updated_json = video_json.copy()

        for frame in updated_json['frames']:
            lm = frame.get('pose_landmarks')
            if lm is None:
                frame['pose_angles'] = None
                continue

            try:
                landmarks = np.array(lm)[:, :3] # ignore visibility, take x,y,z

                # Shoulders
                ls = landmarks[11] # left shoulder
                rs = landmarks[12] # right shoulder
                shoulder_vec = rs - ls
                shoulder_angle = math.degrees(math.atan2(shoulder_vec[1], shoulder_vec[0]))

                # Hips
                lh = landmarks[23] # left hip
                rh = landmarks[24] # right hip
                hip_vec = rh - lh
                hip_angle = math.degrees(math.atan2(hip_vec[1], hip_vec[0]))

                # Feet (ankles)
                la = landmarks[27] # left ankle
                ra = landmarks[28] # right ankle
                feet_vec = ra - la
                feet_angle = math.degrees(math.atan2(feet_vec[1], feet_vec[0]))

                # Head direction
                nose = landmarks[0] # nose
                shoulder_mid = (ls + rs) / 2
                head_vec = shoulder_mid - nose
                head_angle = math.degrees(math.atan2(head_vec[1], head_vec[0]))

                frame['pose_angles'] = {
                    'shoulder_angle': shoulder_angle,
                    'hip_angle': hip_angle,
                    'feet_angle': feet_angle,
                    'head_angle': head_angle
                }

            except Exception as e:
                frame['pose_angles'] = None
            
        return updated_json
    
    def master_json(self, video_path, json_path, out_dir='master_jsons/'):
        os.makedirs(out_dir, exist_ok=True)

        with open(json_path, "r") as f:
            data = json.load(f)

        data_with_skeleton = self.skeleton(video_path, json_path)

        master_json = self.skeleton_angles(data_with_skeleton)

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        master_json_path = os.path.join(out_dir, f"{base_name}_master.json")
        with open(master_json_path, "w") as f:
            json.dump(master_json, f, indent=4)

        return master_json_path

    def feature_extraction(self, video_json, use_skeleton=True, use_angles=True):
        n_landmarks = 33
        skeleton_len = n_landmarks * 3 if use_skeleton else 0
        angles_len = 4 if use_angles else 0
        
        features = []

        for frame in video_json['frames']:
            frame_feats = []

            if 'bbox' in frame:
                x_min, y_min, x_max, y_max = frame['bbox']
                w = x_max - x_min
                h = y_max - y_min
                cx, cy = frame['center']
                frame_feats.extend([cx, cy, w, h])
            else:
                frame_feats.extend([0,0,0,0])

            if use_skeleton and frame.get('pose_landmarks') is not None:
                for lm in frame['pose_landmarks']:
                    frame_feats.extend(lm[:3])
                    # padding for missing landmarks
                    n_missing = n_landmarks - len(frame["pose_landmarks"])
                    frame_feats.extend([0]*n_missing*3)
            else:
                frame_feats.extend([0]*skeleton_len)
                
            if use_angles:
                if frame.get('pose_angles') is not None:
                    angles = frame['pose_angles']
                    frame_feats.extend([
                        angles['shoulder_angle'],
                        angles['hip_angle'],
                        angles['feet_angle'],
                        angles['head_angle']
                    ])
                else:
                    frame_feats.extend([0]*angles_len)

            features.append(frame_feats)
        return np.array(features, dtype=float)
    
    def prepare_data(self, video_path, json_path, tail=0.1):
        master_json_path = self.master_json(video_path, json_path)

        with open(master_json_path, "r") as f:
            data = json.load(f)

        total_frames = len(data['frames'])
        cut_frames = max(1, int(total_frames * tail))
        new_total_frames = total_frames - cut_frames

        past_data = {"frames": data["frames"][:new_total_frames]}
        future_data = {"frames": data["frames"][new_total_frames:]}

        X = self.feature_extraction(past_data)
        y = self.feature_extraction(future_data)

        return X, y, past_data, future_data

dataprocessing = DataProcessing()
datasetup = DataSetup()