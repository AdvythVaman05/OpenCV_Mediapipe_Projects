import cv2
import pyautogui
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.minTrackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(
                self.results.multi_handedness, self.results.multi_hand_landmarks
            ):
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        if draw:
            return allHands, img
        else:
            return allHands, img

    def fingersUp(self, myHand):
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info
        
    def findJoints(self, img, hand_no=0, draw=True):
        joints = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]

            for id, joint in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(joint.x * w), int(joint.y * h)
                joints.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return joints

    def set_system_volume(self, volume_level):
        pass

def main():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    acceleration = False
    brake = False

    while True:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img, draw=False)
        
        for hand in hands:
            handType = hand['type']
            fingers = detector.fingersUp(hand)
            
            if handType == 'Right':
                if fingers[0] == 1 and fingers[1] == 1:
                    if not acceleration:
                        pyautogui.keyDown('right')
                        acceleration = True
                else:
                    if acceleration:
                        pyautogui.keyUp('right')
                        acceleration = False

            if handType == 'Left':
                if fingers[0] == 1 and fingers[1] == 1:
                    if not brake:
                        pyautogui.keyDown('left')
                        brake = True
                else:
                    if brake:
                        pyautogui.keyUp('left')
                        brake = False

            thumb_tip = hand['lmList'][4]
            index_tip = hand['lmList'][8]

            cv2.circle(img, (thumb_tip[0], thumb_tip[1]), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (index_tip[0], index_tip[1]), 10, (255, 0, 255), cv2.FILLED)

        cv2.imshow('Image', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()