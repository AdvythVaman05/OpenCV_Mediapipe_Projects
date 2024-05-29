import cv2
import numpy as np
img = cv2.imread("Resources/lambo.jpg")
print(img.shape)


cv2.imshow("Image",img)

#IMAGE RESIZE, CHANGING D NUMBER OF PIXELS
imgResize = cv2.resize(img,(1000,500))
print(imgResize.shape)
cv2.imshow("Image Resize",imgResize)

#IMAG CROPPING
imgCropped = img[0:200,200:370]#height comes first nd then width
cv2.imshow("Image Cropped",imgCropped)
cv2.waitKey(0)