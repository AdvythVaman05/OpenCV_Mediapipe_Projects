import cv2
import numpy as np

width,height = 250,350
img = cv2.imread("Resources/cards.jpg")
pts1 = np.float32([[402,40],[562,135],[257,288],[417,381]]) 
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)

imgOutput = cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow("Image", img)
cv2.imshow("Output Image", imgOutput)   
cv2.waitKey(0)