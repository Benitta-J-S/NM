import cv2
import numpy as np

# Load the image
image = cv2.imread('image.png')  # Ensure 'image.png' exists in your script directory
if image is None:
    print("Image not found.")
    exit()

# Resize for easier processing
image = cv2.resize(image, (600, 400))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise reduction
blur = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection
edges = cv2.Canny(blur, 30, 200)

# Find contours in the edge map
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and keep the largest ones
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

plate = None

# Loop over contours to find a rectangle (license plate-like)
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        plate = approx
        break

# Draw the license plate contour if found
output = image.copy()
if plate is not None:
    cv2.drawContours(output, [plate], -1, (0, 255, 0), 3)
    cv2.putText(output, "License Plate Detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
else:
    cv2.putText(output, "No Plate Found", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show result
cv2.imshow("Input Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()