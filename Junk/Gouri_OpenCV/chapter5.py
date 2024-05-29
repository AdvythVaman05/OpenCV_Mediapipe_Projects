import cv2
import numpy as np

#WARP PERSPECTIVE ON IMG TO GET ITS BIRD EYE VIEW

img = cv2.imread("Resources/card.png")

width,height = 250,350
# pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
pts1 = np.float32([[232,254],[436,220],[276,544],[504,495]])#found out the pixels through PAINT
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

#transformation matrix
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOutput = cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow("Image",img)
cv2.imshow("Output",imgOutput)
cv2.waitKey(0)