import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

def apply_elliptical_blush(center, axes, color, image, alpha=0.5):
    # Create a mask for the blush
    mask = np.zeros_like(image, dtype=np.uint8)

    # Draw an ellipse on the mask
    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=color, thickness=-1)

    # Blur the mask to create a smooth gradient effect
    mask = cv2.GaussianBlur(mask, (21, 21), 30)

    # Blend the mask with the original image
    cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Convert landmarks to a list of tuples
            landmarks = [(int(l.x * image.shape[1]), int(l.y * image.shape[0])) for l in face_landmarks.landmark]

            # Left and right cheek landmarks for center calculation
            left_cheek_center = np.mean([landmarks[117], landmarks[123], landmarks[187], landmarks[205], landmarks[101], landmarks[118]], axis=0).astype(int)
            right_cheek_center = np.mean([landmarks[346], landmarks[347], landmarks[330], landmarks[425], landmarks[411], landmarks[352]], axis=0).astype(int)

            # Define the axes of the ellipse (width and height)
            ellipse_axes = (int(image.shape[1] * 0.02), int(image.shape[0] * 0.01))  # Adjust these values for size

            # Apply elliptical blush to cheeks
            blush_color = (76, 58, 162)  # Pinkish color (BGR format)
            apply_elliptical_blush(tuple(left_cheek_center), ellipse_axes, blush_color, image)
            apply_elliptical_blush(tuple(right_cheek_center), ellipse_axes, blush_color, image)

    # Display the processed video
    cv2.imshow('Elliptical Blush Effect', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()