import cv2
import numpy as np

img = cv2.imread("Resources/my_photo-white.jpeg")
print(img.shape)

imgResized = cv2.resize(img, (200,300))
print(imgResized.shape)

imgCropped = img[200:500,0:200] 

cv2.imshow("Image",img)
cv2.imshow("Resized Image", imgResized)
cv2.imshow("Cropped Image", imgCropped)
cv2.waitKey(0)
