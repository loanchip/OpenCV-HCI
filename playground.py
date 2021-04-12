# Imports
import os
import time
import datetime
import numpy as np
import cv2
import pyautogui
pyautogui.FAILSAFE = False
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def image_preprocess(image):
    #fast_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    return image

if __name__ == '__main__':
    # init camera
    execution_path = os.getcwd()
    camera = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.9)

    index = 0
    refresh_rate = 10
    speed = 1
    threshold = 50
    prev_data = [0,0,0,0,0,0]
    finger_change = [0,0,0,0]

    while camera.isOpened():
        # Init and FPS process
        start_time = time.time()

        if index == refresh_rate:
            print(finger_change)
            # Thumb Movement - Horizontal
            if abs(finger_change[2]) > abs(finger_change[3]):
                # Right swipe
                if finger_change[2] > 0:
                    pyautogui.click()
                # Left swipe
                else:
                    pyautogui.doubleClick()
            # Horizontal Gesture
            if abs(finger_change[0]) > abs(finger_change[1]):
                pyautogui.hscroll(speed*(finger_change[0]/abs(finger_change[0]))*min(abs(finger_change[0] * pyautogui.size()[1]),30))
            # Vertical Gesture
            elif abs(finger_change[1]) > abs(finger_change[0]):
                pyautogui.scroll(speed*(-finger_change[1]/abs(finger_change[1]))*min(abs(finger_change[1] * pyautogui.size()[0]),30))
            index = 0
            prev_data = [0,0,0,0,0,0]
            finger_change = [0,0,0,0]

        # Grab a single frame
        success, image = camera.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        # Preprocess Image
        image = image_preprocess(image)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        # Hand Detection
        results = hands.process(image)
        
        # Draw the hand annotations on an empty image.
        #image.flags.writeable = True
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        empty_frame = np.zeros(shape=image.shape, dtype=np.uint8)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    empty_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                pyautogui.moveTo(
                    (
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x + 
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x + 
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x + 
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x #+ 
                        #hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                    ) * pyautogui.size()[0] / 4,
                    (
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y + 
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y + 
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y + 
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y #+ 
                        #hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                    ) * pyautogui.size()[1] / 4
                )

                threshold = abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                    - hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x) / 3

                if index == 0:
                    prev_data = [
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                    ]
                else:
                    new_hand_change = [
                        prev_data[2] - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                        prev_data[3] - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                    ]
                    new_finger_change = [
                        prev_data[0] - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        prev_data[1] - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                        prev_data[4] - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                        prev_data[5] - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                    ]

                    if abs(new_hand_change[0]) <= threshold:
                        if abs(new_finger_change[0]) >= threshold:
                            finger_change[0] += new_finger_change[0]
                        if abs(new_finger_change[2]) >= threshold*2:
                            finger_change[2] += new_finger_change[2]
                    if abs(new_hand_change[1]) <= threshold:
                        if abs(new_finger_change[1]) >= threshold:
                            finger_change[1] += new_finger_change[1] 
                        if abs(new_finger_change[3]) >= threshold*2:
                            finger_change[3] += new_finger_change[3] 

                    prev_data = [
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                    ]
        
        # Display Result Image + Info
        # Calculate FPS >> FPS = 1 / time to process loop
        fpsInfo = "FPS: " + str(1.0 / (time.time() - start_time)) 
        #print(fpsInfo)
        cv2.putText(empty_frame, fpsInfo, (10, 10), font, 0.4, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', empty_frame)
        
        # Hit 'q' on the keyboard to quit!    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        index += 1

    # Release handle to the webcam
    camera.release()
    cv2.destroyAllWindows()