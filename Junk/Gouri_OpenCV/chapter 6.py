import cv2
import numpy as np

#JOINING IMAGES
img = cv2.imread("Resources/girl.jpg")
#stack d img w/ itself

# imgHor = np.hstack((img,img))#horizontal stacking
# imgVer = np.vstack((img,img))

# cv2.imshow("Horizontal",imgHor)
# cv2.imshow("Vertical",imgVer)
#this method will not work if imgs r of diff channels, 1 rgb nd 1 grey lyk tht, and also we cant resize imgs also


cv2.waitKey(0)

