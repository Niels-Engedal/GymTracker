import cv2

def test_camera():
    cap = cv2.VideoCapture(1)  # Try default camera index
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return

    print("Press 'q' to quit the test.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

test_camera()
