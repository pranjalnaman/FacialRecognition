import numpy as np
import cv2
from Dataset import idname

face_cascade = cv2.CascadeClassifier('C:\\Users\\hp\\Desktop\\Facial Recognition\\haarcascade_frontalface_alt.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("C:\\Users\\hp\\Desktop\\Facial Recognition\\New Folder\\trainer\\trainingdata.yml")

choice = int(input("1. TO USE THE WEBCAM FOR DETECTION\n2. TO DETECT FACES IN AN IMAGE"))

if choice == 1:

    cap = cv2.VideoCapture(0)
    
    id = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            id,conf = rec.predict(gray[y:y+h,x:x+w])
            cv2.putText(img,str(IDnam[id]),(x,y+h),font,1,(0,0,255))
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
           
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

elif choice == 2:

    dir1 = input("ENTER THE PATH TO THE IMAGE")
    directory = "r\""+dir1+"\""

    img = cv2.imread(dir1)
    faces = face_cascade.detectMultiScale(img)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

     cv2.imshow('img',img)


