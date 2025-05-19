import cv2
import numpy as np

# Define color range for object detection (e.g., red object)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

cap = cv2.VideoCapture(0)  # Use drone camera feed or video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and convert to HSV
    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for red object
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    direction = "Searching..."
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

            # Decide direction based on x position
            if x < 200:
                direction = "Move Left"
            elif x > 440:
                direction = "Move Right"
            else:
                direction = "Move Forward"

    cv2.putText(frame, f'Direction: {direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Drone Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() # type: ignore