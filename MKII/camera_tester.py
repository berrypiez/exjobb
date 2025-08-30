# Test camera fisheye/regular

import cv2
import numpy as np
import time

def main():
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = int(cap.get(cv2.CAP_PROP_FPS)) # or 30

    mtx = np.array([[307.8047385, 0, 355.19676862],
                    [0, 302.44366762, 233.22849986],
                    [0, 0, 1]])
    dist = np.array([-0.290496, 0.07539763, -0.00075077, -0.00159761, -0.00811828])
    
    show_undistorted = False
    
    prev_time = time.time()
    actual_fps = 0

    while True:# Read the first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            cap.release()
            exit()
            
        if show_undistorted:
            frame = cv2.undistort(frame, mtx, dist, None, mtx)
        
        current_time = time.time()
        actual_fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(frame, f"Reported FPS: {reported_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Actual FPS: {actual_fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        display_frame = cv2.resize(frame, (640*2, 360*2))
        cv2.imshow("Camera", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('d'):
            show_undistorted = True
        elif key == ord('f'):
            show_undistorted = False
        elif key == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
        
    
if __name__ == "__main__":
    main()
    
