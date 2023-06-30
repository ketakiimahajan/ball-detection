import cv2
import numpy as np
import time
lower =np.array([100,60,60])
upper =np.array([104,340,340])
    
cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((7,7),np.uint8)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res=cv2.bitwise_and(frame,frame,mask=mask)


    indices = np.where(mask == 255)  
    if len(indices[0]) > 100:
        x = int(np.mean(indices[1]))
        y = int(np.mean(indices[0]))   
        print(x,y)
        cv2.circle(frame, (x,y), 4, (255, 255, 0), -1)  
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (100, 100), cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 0), 2)
     
    closing=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel) 
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow("idk",closing)
    cv2.imshow("res",res)

    
    if cv2.waitKey(1) & 0xFF == ord('K'):
        break