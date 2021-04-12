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
    ''' Image Preprocessing - Resize and Flip
    '''
    # Reduce image size for faster processing
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    return image

def start_tracking(max_num_hands=1, min_detection_confidence=0.5, 
            min_tracking_confidence=0.9, display_skeleton=True):
    ''' OpenCV and Mediapipe Hands based module to detect and track
    hand movements. Mouse control using hand gestures based on rule 
    based movements.

    Gestures:
        Index Finger - 
            Up, Down, Left, Right -> Scrolling
        Thumb - 
            Right -> Double Click
            Left - Single Click
    '''
    # Mediapipe Hands Tracking Config
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands, 
        min_detection_confidence=min_detection_confidence, 
        min_tracking_confidence=min_tracking_confidence
    )

    # Tracking Parameters
    refresh_rate = 10

    # Usage Parameters
    scrolling_speed = 1

    # Local Variables
    frame_index = 0
    prev_hand_landmarks = [0,0,0,0,0,0]
    finger_change = [0,0,0,0]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    # Start Video Camera Capture
    execution_path = os.getcwd()
    camera = cv2.VideoCapture(0)
    print('Tracking..')

    while camera.isOpened():
        # Init and FPS process
        start_time = time.time()

        if frame_index == refresh_rate:
            print(finger_change)
            # Thumb Movement - Horizontal
            if abs(finger_change[2]) > abs(finger_change[3]):
                if finger_change[2] > 0: # Right swipe
                    pyautogui.click()
                else: # Left swipe
                    pyautogui.doubleClick()
            # Horizontal Gesture
            if abs(finger_change[0]) > abs(finger_change[1]):
                pyautogui.hscroll(
                    scrolling_speed * 
                    (finger_change[0]/abs(finger_change[0])) * # direction
                    min(abs(finger_change[0] * pyautogui.size()[1]),30) # velocity
                )
            # Vertical Gesture
            elif abs(finger_change[1]) > abs(finger_change[0]):
                pyautogui.scroll(
                    scrolling_speed * 
                    (-finger_change[1]/abs(finger_change[1])) * # direction
                    min(abs(finger_change[1] * pyautogui.size()[0]),30) # velocity
                )

            # Reset variables
            frame_index = 0
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
        
        #image.flags.writeable = True
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Draw the hand annotations on an empty image.
        if display_skeleton:
            skeleton_image = np.zeros(shape=image.shape, dtype=np.uint8)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if display_skeleton:
                    mp_drawing.draw_landmarks(
                        skeleton_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Threshold based on hand size
                threshold = abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                    - hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x) / 3
                
                # Average Hand Movement
                x_movement = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x + 
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x + 
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x + 
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x #+ 
                    #hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                )
                x_movement /= 4
                y_movement = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y + 
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y + 
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y + 
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y #+ 
                    #hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                )
                y_movement /= 4

                new_hand_change = [
                    prev_hand_landmarks[4] - x_movement,
                    prev_hand_landmarks[5] - y_movement
                ]

                # Move Mouse Position
                if abs(new_hand_change[0]) >= threshold/15 or abs(new_hand_change[1]) >= threshold/15:
                    pyautogui.moveTo(x_movement * pyautogui.size()[0],y_movement * pyautogui.size()[1])

                # Finger Movement
                if frame_index != 0:
                    new_finger_change = [
                        prev_hand_landmarks[0] - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        prev_hand_landmarks[1] - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                        prev_hand_landmarks[2] - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                        prev_hand_landmarks[3] - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                    ]

                    '''
                    # Calculate finger movement if no significant hand movement
                    if abs(new_hand_change[0]) <= threshold:
                        if abs(new_finger_change[0]) >= threshold: # index finger
                            finger_change[0] += new_finger_change[0]
                        if abs(new_finger_change[2]) >= threshold: # thumb
                            finger_change[2] += new_finger_change[2]
                    if abs(new_hand_change[1]) <= threshold:
                        if abs(new_finger_change[1]) >= threshold: # index finger
                            finger_change[1] += new_finger_change[1] 
                        if abs(new_finger_change[3]) >= threshold: # thumb
                            finger_change[3] += new_finger_change[3]
                    '''
                    # Calculate finger movement relative to hand movement
                    if abs(new_hand_change[0] - new_finger_change[0]) >= threshold: # index finger
                        finger_change[0] += new_finger_change[0]
                    if abs(new_hand_change[0] - new_finger_change[2]) >= threshold: # thumb
                        finger_change[2] += new_finger_change[2]
                    if abs(new_hand_change[1] - new_finger_change[1]) >= threshold: # index finger
                        finger_change[1] += new_finger_change[1]
                    if abs(new_hand_change[1] - new_finger_change[3]) >= threshold: # thumb
                        finger_change[3] += new_finger_change[3]

                # Storing current step hand informaiton
                prev_hand_landmarks = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                    x_movement,
                    y_movement
                ]
        
        if display_skeleton:
            # Display Result Image + Info
            # Calculate FPS >> FPS = 1 / time to process loop
            fpsInfo = "FPS: " + str(1.0 / (time.time() - start_time)) 
            #print(fpsInfo)
            cv2.putText(skeleton_image, fpsInfo, (10, 10), font, 0.4, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', skeleton_image)
        
            # Hit 'q' on the keyboard to quit!    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_index += 1

    # Release handle to the webcam
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_tracking()