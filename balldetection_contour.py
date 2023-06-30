import cv2
import numpy as np
import time
import os

def resize(img):
    return cv2.resize(img, (512,512))

cap = cv2.VideoCapture(0)

lowerBound_blue = np.array([60, 35, 140]) # sets lower and upper bounds for blue
upperBound_blue = np.array([180, 255, 255])

pTime=0
while True:
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound_blue, upperBound_blue)
    
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)

    frame = cv2.drawContours(frame, [cnt],0, (0,255,255), 2)

    frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 0), 3)
    
    cv2.imshow('frame', frame)
    cv2.imshow("mask", mask)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break
cv2.destroyAllWindows()
