import cv2
import mediapipe as mp
import math
import pyautogui
import os
import random
import screen_brightness_control as sbc
import numpy as np
from Modules.HandModule import HandDetector

class AutoGUI :

    def __init__(self):
        pass
    def screenshot(self):
        snap = pyautogui.screenshot()

        while True:
            file = f'D:\\{random.randint(0, 100000)}.png'

            if not os.path.isfile(file):
                snap.save(file)
                break

    def left(self):
        pyautogui.press('left')
    
    def right(self):
        pyautogui.press('right')
    
    def up(self):
        pyautogui.press('up')
    
    def down(self):
        pyautogui.press('down')
    
    def enter(self):
        pyautogui.press('enter')

    def zoomin(self):
        pyautogui.hotkey("ctrl", "+")

    def zoomout(self):
        pyautogui.hotkey("ctrl", "-")

    def close_tab(self):
        pyautogui.hotkey("ctrl", "w")

    def shift_tab_right(self):
        pyautogui.hotkey("ctrl", "tab")
    
    def shift_tab_left(self):
        pyautogui.hotkey("ctrl", "shift", "tab")
    
    def new_tab(self):
        pyautogui.hotkey("ctrl", "t")

    def new_window(self):
        pyautogui.hotkey("ctrl", "n")

    def set_system_volume(self, volume_level):
        pass

    def scroll_up(self):
        pyautogui.scroll(-100)
    
    def scroll_down(self):
        pyautogui.scroll(100)

class HandActions:

    def __init__(self):
       self.detector = HandDetector(detectionCon=0.8, maxHands=3)
       self.autogui = AutoGUI()

    def swipe(self):

        cap = cv2.VideoCapture(0)

        left_threshold = 400
        right_threshold = 270
        start_time = 0
        swipe_ended = False
        swipe_started = False
        lswipe_ended = False
        lswipe_started = False
         
        while True:
                
                success, img = cap.read()
                img = cv2.flip(img, 1)
                hands, img = self.detector.findHands(img)

                if hands:
                    hand1 = hands[0]
                    lmList1 = hand1["lmList"]
                    bbox1 = hand1["bbox"]
                    centerPoint1 = hand1["center"]
                    handType1 = hand1["type"]

                    fingers1 = self.detector.fingersUp(hand1)

                    if fingers1[1] and fingers1[2] and fingers1[3] and fingers1[4] and not fingers1[0]:
                        indMid = [lmList1[8][0], lmList1[12][0], lmList1[16][0], lmList1[16][0]]

                        if not swipe_started and max(indMid) > left_threshold:
                            swipe_started = True

                        if swipe_started and min(indMid) < right_threshold:
                            swipe_started = False
                            swipe_ended = True

                        if swipe_ended:
                            print("Swipe detected!")
                            self.autogui.right()
                            swipe_ended = False

                    if fingers1[1] and fingers1[2] and fingers1[3] and not fingers1[4]:
                        indMid = [lmList1[8][0], lmList1[12][0], lmList1[16][0], lmList1[16][0]]

                        if not lswipe_started and min(indMid) < right_threshold:
                            lswipe_started = True

                        if lswipe_started and max(indMid) > left_threshold:
                            lswipe_started = False
                            lswipe_ended = True

                        if lswipe_ended:
                            print("Left Swipe detected!")
                            self.autogui.left()
                            lswipe_ended = False

                else:
                    start_time = 0

                cv2.imshow("frame", img)
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        
    def zoom(self):


        cap = cv2.VideoCapture(0)

        zoom_started = False
        zoom_ended = False
        zout_started = False
        zout_ended = False

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands, img = self.detector.findHands(img)

            if hands:
                hand1 = hands[0]
                lmList1 = hand1["lmList"]
                bbox1 = hand1["bbox"]
                centerPoint1 = hand1["center"]
                handType1 = hand1["type"]

                fingers1 = self.detector.fingersUp(hand1)

                if (
                    fingers1[0]
                    and fingers1[1]
                    and not fingers1[2]
                    and not fingers1[4]
                    and not fingers1[3]
                ):
                    length, info, _ = self.detector.findDistance(
                        lmList1[4][0:2], lmList1[8][0:2], img
                    )
                    if not zoom_started and length < 50:
                        zoom_started = True

                    if zoom_started and length > 120:
                        zoom_started = False
                        zoom_ended = True

                    if zoom_ended:
                        zoom_ended = False
                        print("Zoomed")
                        self.autogui.zoomin()

                if (
                    fingers1[0]
                    and fingers1[1]
                    and fingers1[2]
                    and not fingers1[3]
                    and not fingers1[4]
                ):
                    length, info, _ = self.detector.findDistance(
                        lmList1[4][0:2], lmList1[8][0:2], img
                    )
                    if not zout_started and length > 120:
                        zout_started = True

                    if zout_started and length < 50:
                        zout_started = False
                        zout_ended = True

                    if zout_ended:
                        zout_ended = False
                        print("Zoomed Out")
                        self.autogui.zoomout()
            else:
                start_time = 0

            cv2.imshow("frame", img)
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    def screenshot(self):
        cap = cv2.VideoCapture(0)
        detect = HandDetector(detectionCon=0.8, maxHands=2)

        while cap.isOpened():
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands, img = detect.findHands(img)

            if hands:
                joints = detect.findJoints(img, draw=False)

                if len(joints) != 0:
                    x1, y1 = joints[8][1], joints[8][2]
                    x2, y2 = joints[12][1], joints[12][2]

                    cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    line_len = math.hypot(x2 - x1, y2 - y1)

                    if line_len < 20:
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        self.autogui.screenshot()
                        print("Screenshot taken")

            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    def brightness(self):
        cap = cv2.VideoCapture(0)
        detect = HandDetector(detectionCon=0.8, maxHands=2)

        while cap.isOpened():
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands, img = detect.findHands(img)

            if hands:
                hand1 = hands[0]
                lmList1 = hand1["lmList"]


                fingers1 = detect.fingersUp(hand1)

                if (
                    fingers1[0]
                    and fingers1[1]
                    and not fingers1[2]
                    and not fingers1[4]
                    and not fingers1[3]
                ):
                    x1, y1 = lmList1[8][0], lmList1[8][1]  
                    x2, y2 = lmList1[4][0], lmList1[4][1]  

                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    brightness_value = int(np.interp(distance, [50, 200], [0, 100]))

                    sbc.set_brightness(brightness_value)

                    print("Brightness set to:", brightness_value)

            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def volume_control(self):
        cap = cv2.VideoCapture(0)
        detect = HandDetector(detectionCon=0.8, maxHands=2)

        while cap.isOpened():
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands, img = detect.findHands(img)

            if hands:
                hand1 = hands[0]
                lmList1 = hand1["lmList"]

                fingers1 = detect.fingersUp(hand1)

                if (
                    fingers1[0]
                    and fingers1[1]
                    and not fingers1[2]
                    and not fingers1[4]
                    and not fingers1[3]
                ):
                    x1, y1 = lmList1[8][0], lmList1[8][1]  
                    x2, y2 = lmList1[4][0], lmList1[4][1]  

                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    volume_value = int(np.interp(distance, [50, 200], [0, 100]))

                    self.autogui.set_system_volume(volume_value)

                    print("Volume set to:", volume_value)

            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def scroll(self):
        cap = cv2.VideoCapture(0)
        detect = HandDetector(detectionCon=0.8, maxHands=2)

        while cap.isOpened():
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands, img = detect.findHands(img)

            if hands:
                hand1 = hands[0]
                lmList1 = hand1["lmList"]

                fingers1 = detect.fingersUp(hand1)

                if (
                    fingers1[0]
                    and fingers1[1]
                    and not fingers1[2]
                    and not fingers1[4]
                    and not fingers1[3]
                ):
                    h, w, c = img.shape
                    if lmList1[8][0] > 2*h//3 : 
                        self.autogui.scroll_up()
                    if lmList1[8][0] < h//3 :
                        self.autogui.scroll_down()

            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    


obj = HandActions()
obj.swipe()