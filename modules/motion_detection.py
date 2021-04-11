# Imports
import os
import time
import datetime
import numpy as np
import cv2
import pyautogui

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

def frame_preprocess(frame):
    '''
    fast_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    hsvim = cv2.cvtColor(fast_frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0.1 * 255, 0.10 * 255, 230], dtype = "uint8")
    upper = np.array([0.7 * 255, 0.6 * 255, 245], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    _,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    '''
    fast_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    gray = cv2.cvtColor(fast_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray
    
    #return thresh

if __name__ == '__main__':
    # init camera
    execution_path = os.getcwd()
    camera = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    # Grab a single frame of video
    ret, frame = camera.read()
    # Initialize the 'background'
    gray = frame_preprocess(frame)
    background = gray
    
    prevCnt = (0,0,0,0)
    movement_direction = None

    while True:
        # Set background to previous frame - remove to detect any motion compared to background
        background = gray

        # Init and FPS process
        start_time = time.time()

        # Grab a single frame of video
        ret, frame = camera.read()

        # calculate FPS >> FPS = 1 / time to process loop
        fpsInfo = "FPS: " + str(1.0 / (time.time() - start_time)) 
        #print(fpsInfo)
        cv2.putText(frame, fpsInfo, (10, 10), font, 0.4, (255, 255, 255), 1)

        text = "No Movement"

        gray = frame_preprocess(frame)

        # compute the absolute difference between the current frame and background
        frameDelta = cv2.absdiff(background, gray)
        # speed threshold - 25 pixels
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        # loop over the contours
        if cnts: print('-'*30)
        biggestCnt = None
        biggestCntArea = 0
        for c in cnts:
            area = cv2.contourArea(c)
            # if the contour is too small, ignore it
            if area < 20:
                continue
            
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            #print((x, y, w, h))
            #cv2.rectangle(frame, (x*4, y*4), (x*4 + w*4, y*4 + h*4), (0, 255, 0), 2)
            text = "Detecting"

            if area > biggestCntArea:
                biggestCnt = (x, y, w, h)

        if prevCnt and biggestCnt:
            (x, y, w, h) = biggestCnt
            cv2.rectangle(frame, (x*4, y*4), (x*4 + w*4, y*4 + h*4), (0, 255, 0), 2)
            
            speed = 50
            xOffset, yOffset = 0, 0
            #xOffset = 10*(prevCnt[0] - biggestCnt[0])
            #yOffset = 10*(biggestCnt[1] - prevCnt[1])

            print('Movement:', end='\t')
            if biggestCnt[0] > prevCnt[0]:
                print('Left', end='\t')
            elif biggestCnt[0] < prevCnt[0]:
                print('Right', end='\t')
            else:
                print(end='\t')

            if biggestCnt[1] > prevCnt[1]:
                print('Down', end='\t')
            elif biggestCnt[1] < prevCnt[1]:
                print('Up', end='\t')
            else:
                print(end='\t')

            offset = [prevCnt[0] - biggestCnt[0], biggestCnt[1] - prevCnt[1]]
            if abs(offset[0]) > abs(offset[1]):
                '''
                if offset[0] > 0:
                    xOffset = speed
                elif offset[0] < 0:
                    xOffset = speed*(-1)
                '''
                xOffset = speed*offset[0]//2
            elif abs(offset[1]) > abs(offset[0]):
                '''
                if offset[1] > 0:
                    yOffset = speed
                elif offset[1] < 0:
                    yOffset = speed*(-1)
                '''
                yOffset = speed*offset[1]//2
            pyautogui.moveRel(xOffset, yOffset)
            print()

        # draw the text and timestamp on the frame
        cv2.putText(frame, "Status: {}".format(text), (10, 20), font, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), font, 0.35, (0, 0, 255), 1)

        prevCnt = biggestCnt
        
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    camera.release()
    cv2.destroyAllWindows()