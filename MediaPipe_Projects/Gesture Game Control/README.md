
# Gesture Game Control

This project allows you to play the popular game **Hill Climb Racing** using hand gestures, leveraging **OpenCV**, **Mediapipe**, and **PyAutoGUI**.

## Features
- **Hand Tracking:** Uses Mediapipe to detect and track hands.
- **Gesture Recognition:** Recognizes specific gestures for controlling acceleration and braking.
- **Keyboard Automation:** Simulates key presses for the right and left arrow keys.
- **Real-Time Processing:** Runs smoothly with a webcam for real-time hand gesture detection.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install opencv-python mediapipe pyautogui
```

## How It Works
- The script captures video from the webcam.
- It detects hands and extracts key landmarks using **Mediapipe**.
- The following gestures are recognized:
  - **Right Hand (Thumb & Index Finger Up)** → Accelerate (Presses Right Arrow Key)
  - **Left Hand (Thumb & Index Finger Up)** → Brake (Presses Left Arrow Key)
- Gestures are continuously monitored, and key presses are simulated using **PyAutoGUI**.

## Usage
1. Run the script:
   ```bash
   python your_script.py
   ```
2. Ensure your webcam is enabled.
3. Use your right hand with thumb and index finger up to accelerate.
4. Use your left hand with thumb and index finger up to brake.
5. Press `q` to exit.

## Implementation Details
- **Hand Detection:** Utilizes `mediapipe.solutions.hands` to detect hand landmarks.
- **Landmark Processing:** Extracts x, y coordinates of fingertips to determine gestures.
- **Key Simulation:** Uses `pyautogui` to send keyboard inputs.
- **Real-Time Visualization:** Displays webcam feed with detected hand landmarks.

## Future Enhancements
- Add more gestures for additional controls.
- Improve accuracy for varied hand positions.
- Optimize performance for lower-latency input.

## Credits
- **Mediapipe** for hand tracking.
- **OpenCV** for video processing.
- **PyAutoGUI** for key automation.

Enjoy gesture-controlled gaming! 🚀

