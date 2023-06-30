import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# fgbg = cv2.createBackgroundSubtractorMOG2()

lowerBound_blue = np.array([60, 35, 140]) 
upperBound_blue = np.array([180, 255, 255]) 

while True:
    ret, frame = cap.read()
    # fgmask = fgbg.apply(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound_blue, upperBound_blue)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    # kernel = np.ones((5,5), np.uint8)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # fg = cv2.bitwise_and(frame, frame, mask = fgmask)
    
    corners = cv2.findNonZero(mask) # all points of non-zero points = white pixels
    
    indices = np.where(mask == 255)  
    if len(indices[0]) > 100:

        # x, y, w, h = cv2.boundingRect(corners)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # cx, cy = x + w // 2, y + h // 2
        # cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
        
        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10'] // M['m00'])
            cy = int(M['m01'] // M['m00'])
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            print(cx, cy)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    # cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == ord("m"):
        break

cap.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)

# lowerBound_blue = np.array([100, 60, 60]) 
# upperBound_blue = np.array([105, 340, 340])

# while True:
#     ret, frame = cap.read()
    
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY) # converting to grayscale
#     ret, thresh = cv2.threshold(gray, 127, 255, 0) # grayscale --> binary
#     mask = cv2.inRange(hsv, lowerBound_blue, upperBound_blue)
    
#     mask_white = (mask > 21)
    
#     width = sum(mask_white.any(axis=0))
#     height = sum(mask_white.any(axis=1))
    
#     ratio_px_mm = 153 / 14
#     width_cm = (width / ratio_px_mm)/10
#     height_cm = (height / ratio_px_mm)/10
    
    
#     if (width <= 8) and (height <= 8):
#         moment = cv2.moments(thresh)
#         if moment["m00"] > 200:
        
#             cX = int(moment["m10"] / moment["m00"])
#             cY = int(moment["m01"] / moment["m00"])
#             print(cX, cY)
            
#             cv2.circle(frame, (cX, cY), 5, (0, 0, 0), -1)

#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     #cv2.imshow('res', res) 

#     if cv2.waitKey(1) & 0xFF == ord('m'): 
#         break

# cv2.destroyAllWindows()

# """ ratio_px_mm = 153 / 14
#             mm = width / ratio_px_mm
#             cm = mm / 10 """