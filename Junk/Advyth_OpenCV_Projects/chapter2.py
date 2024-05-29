import cv2
import numpy as np

img = cv2.imread("Resources/my_photo-white.jpeg")
kernel = np.ones((2,2),np.uint8)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img,(7,7),0)
imgCanny = cv2.Canny(img, 75, 75)
imgDilation = cv2.dilate(imgCanny,kernel,iterations=2)

# cv2.imshow("Gray Image",imgGray)
# cv2.imshow("Blur Image", imgBlur)
# cv2.imshow("Canny Image", imgCanny)
cv2.imshow("Dilated Canny Image", imgDilation)

cv2.waitKey(0)
