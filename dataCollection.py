import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# adding capture object.
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
ImgSize = 300
counter = 0
folder = "dataset/C"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        imgWhite = np.ones((ImgSize, ImgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

        aspectratio = h / w

        if aspectratio > 1:
            k = ImgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, ImgSize))
            imgResizeshape = imgResize.shape
            wGap = math.ceil((ImgSize - wCal) / 2)
            imgWhite[:, wGap : wCal + wGap] = imgResize
        else:
            k = ImgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (ImgSize, hCal))
            imgResizeshape = imgResize.shape
            hGap = math.ceil((ImgSize - hCal) / 2)
            imgWhite[hGap : hCal + hGap, :] = imgResize
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Imagewhite", imgWhite)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)  # gives 1 millisecond delay
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)
    elif key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
