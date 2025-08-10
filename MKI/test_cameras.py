import cv2


# for i in range(5):
#     cap = cv2.VideoCapture(i)  # Adjust the camera index as needed
#     if cap is None or not cap.isOpened():
#         print(f"Camera {i} is not available.")
#         continue

#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     print(f"Camera {i} is available.")
#     cv2.imshow("Live Tracking", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#         break

cap = cv2.VideoCapture(1)  # Adjust the camera index as needed

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:# Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        cap.release()
        exit()

    # Display the frame
    cv2.imshow("Live Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
