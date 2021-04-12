# OpenCV-HCI
OpenCV based Human - Computer Interaction
  
Docs: https://docs.opencv.org/master/  
Python Tutorial Docs: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html  
Mediapipe Hands: https://google.github.io/mediapipe/solutions/hands  

## Implemented Modules:  
- faces.py: Face Recognition  
- motion_detection.py: Motion Detection  
- motion_scroll.py: Scrolling based on motion detection  
- hand_tracking.py: Mouse Control based on Hand Movements and Gestures  
  
### Current Progress:  
- Mouse Control using Hand Movements and Gestures  
  - Implemented hand movement based mouse movements  
  - Implemented hand gesture based scrolling and clicking  

### To Do:  
Use Mask-based Hand Detection to identify hand movements and gestures  
  
Dual camera is used for all rgb-d images for depth view - Microsoft/Azure Kinect  

Single camera based approach:  
 - Hand detection / segmentation - Done  
 - Fingertip detection - Done  
 - Between frame movements - Done  
 - Gesture classification - Done (Rule Based)  
 - Gesture classification using ML  