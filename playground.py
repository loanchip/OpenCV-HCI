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

from modules.hand_tracking import start_tracking

if __name__ == '__main__':
    start_tracking(display_skeleton=True)