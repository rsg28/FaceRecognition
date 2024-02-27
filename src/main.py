import cv2
import dlib
import numpy as np
from faceshapeCalculator import calculate_face_shape

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load the shape predictor
predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Capture one frame
ret, frame = cap.read()
    
if ret:
    # Convert frame to grayscale for the face detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    if len(faces) > 0:
        # Get the landmarks/parts for the first face in the frame
        landmarks = predictor(gray, faces[0])
        
        # Draw the landmarks on the face
        points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        # Estimate forehead points
        # We'll use points 18, 19, 24, 25 (based on the 68 point model) which are at the ends and inner edges of the eyebrows
        brow_left = np.array(points[17])
        brow_right = np.array(points[26])
        nose_bridge = np.array(points[27])
        # Estimate the distance between the eyebrows and the top of the nose bridge
        forehead_height = np.linalg.norm(nose_bridge - np.mean([brow_left, brow_right], axis=0)) * 3
        # Estimate forehead points by moving up `forehead_height` from the eyebrow points
        for x, y in [points[17], points[18], points[24], points[25], points[26]]:
            estimated_forehead_point = (x, int(y - forehead_height))
            cv2.circle(frame, estimated_forehead_point, 1, (0, 255, 0), -1)

        # Save the captured image with landmarks to disk
        cv2.imwrite('captured_face_with_forehead.jpg', frame)

        # Display the resulting frame with landmarks
        cv2.imshow('Captured Face', frame)
        cv2.waitKey(0)
else:
    print("Failed to capture image")

# When everything is done, release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()

# Assume that 'points' is a list of tuples containing the detected landmark points
face_shape = calculate_face_shape(points)
print(f"The estimated face shape is: {face_shape}")