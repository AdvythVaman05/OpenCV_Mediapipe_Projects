import cv2
import numpy as np
from Modules.FaceModule import HandDetector as htm
import time
import autopy

##########################
cam_width, cam_height = 640, 480
frame_reduction = 100  # Reduce the frame size for better accuracy
smooth_factor = 7  # Smoothing factor for mouse movement
#########################

prev_time = 0
prev_x, prev_y = 0, 0
current_x, current_y = 0, 0

video_capture = cv2.VideoCapture(0)
video_capture.set(3, cam_width)
video_capture.set(4, cam_height)
hand_detector = htm.handDetector(maxHands=1)
screen_width, screen_height = autopy.screen.size()
# print(screen_width, screen_height)

while True:
    # 1. Detect hand landmarks in the frame
    success, frame = video_capture.read()
    frame = hand_detector.findHands(frame)
    landmarks_list, bounding_box = hand_detector.findPosition(frame)
    
    # 2. Get the coordinates of the index and middle finger tips
    if len(landmarks_list) != 0:
        index_x, index_y = landmarks_list[8][1:]
        middle_x, middle_y = landmarks_list[12][1:]
        # print(index_x, index_y, middle_x, middle_y)

    # 3. Determine which fingers are raised
    fingers_status = hand_detector.fingersUp()
    # print(fingers_status)
    cv2.rectangle(frame, (frame_reduction, frame_reduction), (cam_width - frame_reduction, cam_height - frame_reduction),
                  (255, 0, 255), 2)
    
    # 4. If only the index finger is raised: Enable cursor movement
    if fingers_status[1] == 1 and fingers_status[2] == 0:
        # 5. Map the finger coordinates to screen size
        mapped_x = np.interp(index_x, (frame_reduction, cam_width - frame_reduction), (0, screen_width))
        mapped_y = np.interp(index_y, (frame_reduction, cam_height - frame_reduction), (0, screen_height))
        
        # 6. Apply smoothing to the cursor movement
        current_x = prev_x + (mapped_x - prev_x) / smooth_factor
        current_y = prev_y + (mapped_y - prev_y) / smooth_factor

        # 7. Move the mouse pointer based on finger movement
        autopy.mouse.move(screen_width - current_x, current_y)
        cv2.circle(frame, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)
        prev_x, prev_y = current_x, current_y

    # 8. If both index and middle fingers are raised: Enable click action
    if fingers_status[1] == 1 and fingers_status[2] == 1:
        # 9. Measure the distance between the two fingers
        distance, frame, line_info = hand_detector.findDistance(8, 12, frame)
        print(distance)
        
        # 10. Perform a mouse click if fingers are close enough
        if distance < 40:
            cv2.circle(frame, (line_info[4], line_info[5]),
                       15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()

    # 11. Calculate and display the frame rate
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    
    # 12. Show the video feed with the drawn landmarks
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == 27:
        break
