import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)
# img[10:20,-20:-10] = 255,0,0
# img[10:20,10:20] = 255,0,0
# img[-20:-10,10:20] = 255,0,0
# img[-20:-10,-20:-10] = 255,0,0

cv2.line(img,(0,0),(512,512),(255,0,0),3)
cv2.line(img,(0,512),(512,0),(255,0,0),3)
img[240:270,240:270] = 0,0,255
cv2.rectangle(img,(0,0),(256,256),(0,255,0),cv2.FILLED)
cv2.rectangle(img,(256,256),(512,512),(0,255,0),cv2.FILLED)
cv2.circle(img,(256,256),180,(255,0,255),3)
cv2.putText(img, "OPENCV",(190,256), cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0),3)

cv2.imshow("Image", img)
cv2.waitKey(0)