import os
from PIL import Image
import numpy as np
import cv2 as cv
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
# print(base_dir)
image_dir = os.path.join(base_dir, "Images")
# print (image_dir)

face_cascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
x_train = []
y_labels = []


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").replace("_", "-").lower()
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]   
            #print(label_ids) 
            #print(id_)
            
            pil_image = Image.open(path).convert("L") # Converting to Grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, 1.5, 5)
            
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
# print(x_train)
#print(y_labels) 

with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids, f)
    
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trained_model.yml")