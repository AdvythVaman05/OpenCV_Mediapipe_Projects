import cv2
print("Package Imported")


# img = cv2.imread("Resources/girl.jpg")
#
# cv2.imshow("Output",img)
# cv2.waitKey(0)

# cap = cv2.VideoCapture("Resources/test_video.mp4")
#while loop used - becoz video is a sequence of imgs
# while True:
#     success, img = cap.read()
    #success variable(bool) - to show that each imgs of d video is stored successfully in d img
    # cv2.imshow("Video", img)
    # if cv2.waitKey(1) & 0xFF ==ord('q'):
    #     break
    #adds a delay nd wait for q press to break d loop
cap = cv2.VideoCapture(0)
#0 - for one lap connected webcam, more than 1 add d id
cap.set(3,640) #width
cap.set(4,480) #height
cap.set(10,100)#brightness
while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break



