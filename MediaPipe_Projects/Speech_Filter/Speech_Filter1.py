import cv2
import threading
import pyaudio
import json
import mediapipe as mp
from vosk import Model, KaldiRecognizer
import textwrap

# Initialize the Vosk model and recognizer
model_path = r"F:\OpenCV Python\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

# Variable to store the recognized text
caption = ""

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def recognize_speech():
    global caption
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open the microphone stream
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()

    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            caption = result.get("text", "")

# Start the speech recognition thread
thread = threading.Thread(target=recognize_speech)
thread.daemon = True
thread.start()

# Initialize webcam
cap = cv2.VideoCapture(0)

def draw_speech_bubble(frame, mouth_coords, text):
    if not mouth_coords:
        return frame

    x_min = min(mouth_coords, key=lambda x: x[0])[0]
    y_min = min(mouth_coords, key=lambda x: x[1])[1]

    # Wrap the text to fit inside the bubble
    wrapped_text = textwrap.fill(text, width=25)

    # Calculate the size of the speech bubble based on text length
    line_height = 20
    lines = wrapped_text.split('\n')
    bubble_w = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in lines]) + 20
    bubble_h = line_height * len(lines) + 20

    bubble_x = 10
    bubble_y = 10

    # Draw the cartoon speech bubble
    cv2.rectangle(frame, (bubble_x, bubble_y), (bubble_x + bubble_w, bubble_y + bubble_h), (255, 255, 255), -1)
    cv2.rectangle(frame, (bubble_x, bubble_y), (bubble_x + bubble_w, bubble_y + bubble_h), (0, 0, 0), 2)
    cv2.line(frame, (bubble_x + bubble_w // 2, bubble_y + bubble_h), (x_min, y_min), (0, 0, 0), 2)

    # Put the text inside the speech bubble
    y_text = bubble_y + 20
    for line in lines:
        cv2.putText(frame, line, (bubble_x + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y_text += line_height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame color to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    result = face_mesh.process(rgb_frame)

    mouth_coords = []
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for id, lm in enumerate(face_landmarks.landmark):
                ih, iw, _ = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                if id in [13, 14, 78, 308, 82, 312, 324, 87]:  # Indices of the mouth landmarks
                    mouth_coords.append((x, y))

    # Draw the speech bubble and add the caption
    draw_speech_bubble(frame, mouth_coords, caption)

    # Display the resulting frame
    cv2.imshow('Live Stream', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
