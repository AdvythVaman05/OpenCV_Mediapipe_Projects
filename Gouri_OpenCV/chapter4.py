import cv2
import numpy as np

#draw shapes nd texts on images

#matrix created using numpy
img = np.zeros((512,512,3),np.uint8)#0-black,to add colour , give dimensionality - 3
# print(img)#matrix gets displayed
#color d img
# img[:]=255,0,0#: - d whole matrix gets colored

#CREATE A LINE
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)

#CREATE A RECTANGLE
cv2.rectangle(img,(0,0),(250,350),(0,0,255),cv2.FILLED)#in thickness part write FILLED colors d rectangle

#CREATE A CIRCLE
cv2.circle(img,(400,50),30,(255,255,0),5)#1st 255-blue, 2nd-green, 3rd-red

#PUTTING TEXTS ON IMG
cv2.putText(img," OPENCV ",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),3)
cv2.imshow("Image",img)
cv2.waitKey(0)