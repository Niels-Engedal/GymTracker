import cv2

cap = cv2.VideoCapture(0)  # 0 is usually the default camera index

if not cap.isOpened():
    print("Could not open video device")
else:
    print("Video device is open")

# Set resolution if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Test Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
