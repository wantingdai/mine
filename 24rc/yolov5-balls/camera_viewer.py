# Author: Ethan Lee
# 2024/5/21 下午9:14

import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

cap2 = cv2.VideoCapture(6)
cap2.set(3, 640)
cap2.set(4, 360)

while cap.isOpened():
    _, frame = cap.read()
    _, frame2 = cap2.read()

    cv2.imshow('Monitor', frame)
    cv2.imshow('Monitor2', frame2)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
