# Capture Video from Camera
import cv2 as cv
import numpy as np
import pickle

cam = cv.VideoCapture('test_file.avi')
face_cascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_alt2.xml')

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")

labels = {"person_name":1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
while(cam.isOpened()):
    ret, frame = cam.read() # Read frame by frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # converting to greyscale
    
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for(x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] 
        #roi_color = frame[y:y+h, x:x+w]
        
        # recognizer
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            #print(id_)
            #print(labels[id_])
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)
            
        
        
        img_item = "pic.png"
        cv.imwrite(img_item, roi_gray)
        
        color = (0, 255, 0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv.rectangle(frame, (x,y),( end_cord_x, end_cord_y), color, stroke) 
        
        
        
        

    cv.imshow('frame', frame) # Displaying with color
    # cv.imshow('frame', gray) # You can uncomment this to get output in gray scale 
    if cv.waitKey(1) & 0xFF ==('q'):
        break

cam.release() # Releasing the camera
cv.destroyAllWindows()






