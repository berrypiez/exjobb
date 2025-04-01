import cv2
import mediapipe as mp
import numpy as np
from math import atan2
import pyrealsense2 as rs
import requests

class PoseEstimator():
    def __init__(self, server_url):
        self.lower_limit_armpit = 50
        self.upper_limit_armpit = 130
        self.lower_limit_elbow = 100
        self.upper_limit_elbow = 180

        self.server_url = server_url

        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.speed = "slow"  # Start with slow speed by default

    def angle(self, p_1, p_2, p_3):
        a = atan2(p_3[1] - p_2[1], p_3[0] - p_2[0]) - atan2(p_1[1] - p_2[1], p_1[0] - p_2[0])
        a = np.rad2deg(a)
        return abs(a if a <= 180 else 360 - a)

    def is_open(self, handLandmarks):
        tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        mcps = [
            self.mp_hands.HandLandmark.THUMB_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]
        wrist = np.array([
            handLandmarks.landmark[self.mp_hands.HandLandmark.WRIST].x,
            handLandmarks.landmark[self.mp_hands.HandLandmark.WRIST].y,
            handLandmarks.landmark[self.mp_hands.HandLandmark.WRIST].z
        ])
        extended = 0
        for tip, mcp in zip(tips, mcps):
            tip_pos = np.array([handLandmarks.landmark[tip].x, handLandmarks.landmark[tip].y, handLandmarks.landmark[tip].z])
            mcp_pos = np.array([handLandmarks.landmark[mcp].x, handLandmarks.landmark[mcp].y, handLandmarks.landmark[mcp].z])
            if np.linalg.norm(tip_pos - wrist) > np.linalg.norm(mcp_pos - wrist) + 0.05:
                extended += 1
        return True if extended == 5 else False if extended == 0 else None

    #two speeds: slow and fast
    def get_speed(self, hands):
        for hand in hands:
            tips = [
                self.mp_hands.HandLandmark.THUMB_TIP,
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                self.mp_hands.HandLandmark.RING_FINGER_TIP,
                self.mp_hands.HandLandmark.PINKY_TIP
            ]
            mcps = [
                self.mp_hands.HandLandmark.THUMB_MCP,
                self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                self.mp_hands.HandLandmark.RING_FINGER_MCP,
                self.mp_hands.HandLandmark.PINKY_MCP
            ]
            wrist = np.array([
                hand.landmark[self.mp_hands.HandLandmark.WRIST].x,
                hand.landmark[self.mp_hands.HandLandmark.WRIST].y,
                hand.landmark[self.mp_hands.HandLandmark.WRIST].z
            ])
            ext = []
            for tip, mcp in zip(tips, mcps):
                tip_pos = np.array([hand.landmark[tip].x, hand.landmark[tip].y, hand.landmark[tip].z])
                mcp_pos = np.array([hand.landmark[mcp].x, hand.landmark[mcp].y, hand.landmark[mcp].z])
                ext.append(np.linalg.norm(tip_pos - wrist) > np.linalg.norm(mcp_pos - wrist) + 0.05)

            # slow: only thumb up, rest down
            if ext[0] and not any(ext[1:]):
                return "slow"
            # fast: index and middle up, rest down
            if ext[1] and ext[2] and not any([ext[0], ext[3], ext[4]]):
                return "fast"

    def get_gesture(self, laa, raa, lea, rea, x_vals, open_hand):
        pointing_left = (self.lower_limit_armpit < laa < self.upper_limit_armpit and self.lower_limit_elbow < lea < self.upper_limit_elbow and not (self.lower_limit_armpit < raa < self.upper_limit_armpit) and (x_vals[2] > x_vals[0] + 0.2))
        pointing_right = (self.lower_limit_armpit < raa < self.upper_limit_armpit and self.lower_limit_elbow < rea < self.upper_limit_elbow and not (self.lower_limit_armpit < laa < self.upper_limit_armpit) and (x_vals[1] > x_vals[3] + 0.2))
        if pointing_left:
            return "left"
        elif pointing_right:
            return "right"
        elif open_hand is not None:
            return "openhand" if open_hand else "closedhand"
        else:
            return None

    def run(self):
        # ---- Using RealSense camera ----
        # pipe = rs.pipeline()
        # cfg = rs.config()
        # cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # pipe.start()

        # ---- Using regular webcam ----
        cap = cv2.VideoCapture(0)

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
            self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while True:
                ret, frame = cap.read()
                # ---- OR RealSense ----
                # frames = pipe.wait_for_frames()
                # color_frame = frames.get_color_frame()
                # frame = np.asanyarray(color_frame.get_data())

                if not ret:
                    continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                pose_results = pose.process(image)
                hands_results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if not pose_results.pose_landmarks:
                    continue

                lms = pose_results.pose_landmarks.landmark
                self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                def get_coords(idx):
                    return np.array([lms[idx].x, lms[idx].y, lms[idx].z])

                ls, rs = get_coords(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value), get_coords(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                le, re = get_coords(self.mp_pose.PoseLandmark.LEFT_ELBOW.value), get_coords(self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                lw, rw = get_coords(self.mp_pose.PoseLandmark.LEFT_WRIST.value), get_coords(self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
                lh, rh = get_coords(self.mp_pose.PoseLandmark.LEFT_HIP.value), get_coords(self.mp_pose.PoseLandmark.RIGHT_HIP.value)

                laa, raa = self.angle(lh, ls, le), self.angle(rh, rs, re)
                lea, rea = self.angle(ls, le, lw), self.angle(rs, re, rw)

                #checking open or close only when hand position is close to the shoulders vertically and horizontally
                left_active = (
                abs(lw[1] - ls[1]) < 0.5 and  # vertical proximity
                abs(lw[0] - ls[0]) < 0.5      # horizontal proximity
                )

                right_active = (
                abs(rw[1] - rs[1]) < 0.5 and
                abs(rw[0] - rs[0]) < 0.5
                )

                hand_open = None
                detected_hands = []

                if hands_results.multi_hand_landmarks:
                    for i, hl in enumerate(hands_results.multi_hand_landmarks):
                        label = hands_results.multi_handedness[i].classification[0].label
                        active = (label == "Left" and left_active) or (label == "Right" and right_active)
                        if active:
                            self.mp_drawing.draw_landmarks(image, hl, self.mp_hands.HAND_CONNECTIONS)
                            detected_hands.append(hl)
                            if self.is_open(hl):
                                hand_open = True
                            elif self.is_open(hl) is False and hand_open is not True:
                                hand_open = False

                gesture = self.get_gesture(laa, raa, lea, rea, [ls[0], rs[0], lw[0], rw[0], le[0], re[0], lh[0], rh[0]], hand_open)

                if detected_hands:
                    self.speed = self.get_speed(detected_hands)

                if gesture:
                    print(f"Detected gesture: {gesture}")
                    # requests.post(self.server_url, json={'command': gesture})
                else:
                    print("No gesture detected")

                if self.speed:
                    print(f"Detected speed: {self.speed}")
                    # requests.post(self.server_url, json={'speed': speed})

                cv2.imshow('Mediapipe Feed', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        # pipe.stop()  
        cv2.destroyAllWindows()

if __name__ == "__main__":
    server_url = "http://192.168.0.2:5000/command"
    PoseEstimator(server_url).run()
