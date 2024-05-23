import cv2
import numpy as np

#importing image using imread()
img = cv2.imread("Resources/girl.jpg")
kernel = np.ones((5,5),np.uint8)#All the values r 1, matrix size 5x5,type of obj defined-unsigned int of 8 bit


#cvtCOLOR()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#cvtColor converts the image defined - img to diff color spaces
cv2.imshow("Grey Image",imgGray)


#GaussianBlur()
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)#define kernel size to blur it(odd no.s)
cv2.imshow("Blur Image",imgBlur)


#Canny()-image with edges
imgCanny = cv2.Canny(img,150,200)#2 threshold values, inc d value for less number of edges
cv2.imshow("Canny Image",imgCanny)

#dilate()-increase thickness of d edges of canny img
imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)#iteration-defines d thickness, itn inc thick inc
cv2.imshow("Dialation Image",imgDialation)

#erode()-dec thickness of edges of dialated img
imgEroded = cv2.erode(imgDialation,kernel,iterations=1)
cv2.imshow("Eroded Image",imgEroded)

cv2.waitKey(0)

