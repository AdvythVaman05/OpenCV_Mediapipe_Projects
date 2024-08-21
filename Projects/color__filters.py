import cv2
import numpy as np

def apply_filter(frame, filter_type):
    if filter_type == 'gray':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    elif filter_type == 'sepia':
        # Create sepia filter
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return cv2.transform(frame, sepia_filter)
    
    elif filter_type == 'warm':
        warm_filter = np.array([[1.2, 0.2, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.8]])
        return cv2.transform(frame, warm_filter)
    
    elif filter_type == 'cool':
        cool_filter = np.array([[0.8, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.2, 0.2, 1.2]])
        return cv2.transform(frame, cool_filter)
    
    elif filter_type == 'invert':
        return cv2.bitwise_not(frame)
    
    elif filter_type == 'blur':
        return cv2.GaussianBlur(frame, (15, 15), 0)
    
    elif filter_type == 'edge_detection':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
    elif filter_type == 'cartoon':
        # Apply bilateral filter to smoothen the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    
    else:
        return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Apply the desired filter
    filter_type = 'warm'  # Change this to 'gray', 'sepia', 'invert', 'blue', or others
    filtered_frame = apply_filter(frame, filter_type)
    
    # if filter_type == 'gray':
    #     # Convert to BGR to keep consistent output for imshow
    #     filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)

    cv2.imshow('Filtered Frame', filtered_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
