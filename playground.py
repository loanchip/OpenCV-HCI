# Imports
import os
import time
import numpy as np
import cv2
import face_recognition

def LoadFaces(path):
    files_list = [files_list for _,_,files_list in os.walk(top=path)][0]
    print(files_list)
    
    known_face_encodings = []
    known_face_names = []

    for fname in files_list:
        face_image = face_recognition.load_image_file(path+fname)
        face_encoding = face_recognition.face_encodings(face_image)[0]

        known_face_names.append(fname[:-4])
        known_face_encodings.append(face_encoding)

    return known_face_encodings, known_face_names

if __name__ == '__main__':
    
    # Load Known Faces
    known_face_encodings, known_face_names = LoadFaces('data/faces/')

    # init camera
    execution_path = os.getcwd()
    camera = cv2.VideoCapture(0)

    faces_refresh_rate = 30
    ticker = faces_refresh_rate-1

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    faces_left, faces_right, faces_top, faces_bottom, faces_names = [], [], [], [], []

    while True:
        ticker += 1
        # Init and FPS process
        start_time = time.time()

        # Grab a single frame of video
        ret, frame = camera.read()

        # calculate FPS >> FPS = 1 / time to process loop
        fpsInfo = "FPS: " + str(1.0 / (time.time() - start_time)) 
        #print(fpsInfo)

        cv2.putText(frame, fpsInfo, (10, 10), font, 0.4, (255, 255, 255), 1)

        if ticker == faces_refresh_rate:
            fast_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = fast_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            faces_left, faces_right, faces_top, faces_bottom, faces_names = [], [], [], [], []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                
                faces_left.append(left*4)
                faces_right.append(right*4)
                faces_top.append(top*4)
                faces_bottom.append(bottom*4)
                faces_names.append(name)
            print('In Frame: ', faces_names)
            ticker = 0

        for index,name in enumerate(faces_names):
            cv2.rectangle(frame, (faces_left[index], faces_top[index]), (faces_right[index], faces_bottom[index]), (0, 0, 255), 2)
            cv2.rectangle(frame, (faces_left[index], faces_bottom[index] - 25), (faces_right[index], faces_bottom[index]), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (faces_left[index] + 6, faces_bottom[index] - 6), font, 0.7, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    camera.release()
    cv2.destroyAllWindows()