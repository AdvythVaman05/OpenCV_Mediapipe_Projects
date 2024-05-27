import cv2
import mediapipe as mp
import tensorflow as tf
import logging
import absl.logging
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    # Convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image and detect hands
    results = hands.process(image)

    # Reset hand state
    left_hand_open = False
    right_hand_open = False

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Determine if the hand is open
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = ((thumb_tip.x - index_finger_tip.x)*2 + (thumb_tip.y - index_finger_tip.y)2)*0.5
            hand_side = "Left" if handedness.classification[0].label == "Left" else "Right"

            if distance > 0.1:  # Adjust this threshold based on your needs
                if hand_side == "Left":
                    left_hand_open = True
                elif hand_side == "Right":
                    right_hand_open = True

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract handedness information
            label = handedness.classification[0].label
            score = handedness.classification[0].score
            text = f"{label} ({int(score * 100)}%)"

            # Get the position of the wrist landmark to place the label
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = image.shape
            cx, cy = int(wrist.x * w), int(wrist.y * h)

            # Draw the label on the image
            cv2.putText(image, text, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2, cv2.LINE_AA)

    # Control keyboard based on hand state
    if left_hand_open:
        pyautogui.keyDown('left')
    if right_hand_open:
        pyautogui.keyDown('right')

    if not left_hand_open:
        pyautogui.keyUp('left')
    if not right_hand_open:
        pyautogui.keyUp('right')

    # Display the resulting frame
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()