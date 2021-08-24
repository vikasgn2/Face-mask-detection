# Importing necessary packages
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Loading a model generated after training
model=load_model("model2-007.model")

# Result to be displayed 
results={0:'without mask',1:'mask'}
GR_dict={0:(0,0,255),1:(0,255,0)}

#Window size and command to start real-time Detection using camera
rect_size = 5
cap = cv2.VideoCapture(0) 

#Object detection algorithm for real-time video pointing to it's directory
haarcascade = cv2.CascadeClassifier(r'C:\Users\Vikas\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

while True:
    (rval, im) = cap.read()
    im=cv2.flip(im,1,1) 
    
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = im[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(150,150))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    
    if key == 27: 
        break

cap.release()

#Closes all windows after hitting ESC key
cv2.destroyAllWindows()