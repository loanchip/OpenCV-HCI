# OpenCV-HCI
OpenCV based Human - Computer Interaction
  
Docs: https://docs.opencv.org/master/  
Python Tutorial Docs: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html  

## Implemented Modules:  
- faces.py: Face Recognition  
- motion_detection.py: Motion Detection  
- motion_scroll.py: Scrolling based on motion detection  
  
### Current Progress:  
- Mouse Control using Movement Detection  
  - Implemented simple motion detection based mouse movements  
  - Implemented simple motion detection based scrolling  

### To Do:  
Use Mask-based Hand Detection to identify hand movements and gestures  
  
Dual camera is used for all rgb-d images for depth view - Microsoft/Azure Kinect  

Single camera based approach:  
 - Hand detection / segmentation  
 - Fingertip detection  
 - Between frame movements  
 - Gesture classification  