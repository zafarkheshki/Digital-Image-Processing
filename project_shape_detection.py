import numpy as np
from cv2 import *
import argparse
import imutils
from pyimagesearch.shapedetector import ShapeDetector
cap = cv2.VideoCapture(0)

#RETR_CCOMP, RETR_EXTERNAL, RETR_TREE

while(cap.isOpened() ):
       # ret = cap.set(3,320)
        #ret = cap.set(4,240)
    ret, frame = cap.read()
            
    if ret==True:
        resized = imutils.resize(frame, width=300)
        ratio = frame.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]


        x, thresh1 = cv2.threshold(blurred, 120, 255, cv2.THRESH_TOZERO) #120 value
        x2,thresh2 = cv2.threshold(blurred,120,255,cv2.THRESH_TOZERO_INV)
        IM2, cnts, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        IM22, contours2, hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


        # find contours in the thresholded image and initialize the
        # shape detector
        #cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(cnts)
        sd = ShapeDetector()

        # loop over the contours
        for cnt in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(cnt);
            if M["m00"] != 0:
                cX = int((M["m10"] / M["m00"]) * ratio)
            else:
                cX = 0
            if M["m01"] != 0:
                cY = int((M["m01"] / M["m00"]) * ratio)
            else:
                cY = 0
            shape = sd.detect(cnt)

            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            cnt = cnt.astype("float")
            cnt *= ratio
            cnt = cnt.astype("int")
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
            #cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
        
            
            approx = cv2.approxPolyDP(cnt,0.03*cv2.arcLength(cnt,True),True) #if this is more, circles are not detected
            
            # triangle, red
            if len(approx)==3:
                cv2.drawContours(frame,[cnt],0,(0,0,255),3)
            
            # pentagon in white
            elif len(approx) == 5:
                cv2.drawContours(frame,[cnt],0,(255,255,255),3)
            
             # rectanle, blue
            elif len(approx)==4:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame,[box],0,(255,0,0),2)

            # circle, yellow
            elif len(approx)> 10:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(frame,ellipse,(0,255,255),2)
                x = approx.ravel()[0]
                y = approx.ravel()[1]
                #cv2.putText(frame, "Circle", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0))
            
            #bigger rectangle, blue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(255,0,0),2)
            

            
        #cv2.drawContours(frame, contours,-1, (255,255,0), 1) 
        cv2.imshow('Image with contours',frame)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()
