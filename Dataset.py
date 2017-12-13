import numpy as np
import cv2
import glob
import os

face_cascade = cv2.CascadeClassifier('C:\\Users\\hp\\Desktop\\Facial Recognition\\haarcascade_frontalface_alt.xml')

IDname = [""]

def idname():
    return IDname


id = input("Enter the ID:\n")

idname = input("\nEnter the name associated with the ID: ")
IDname.append(idname)
choice = input("1. Use WebCam to setup database\n2. Use images to setup database\nEnter your choice: ")
choice = int(choice)

if choice == 1:
    cap = cv2.VideoCapture(0)

    sampleno = 0;
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            sampleno+=1
            cv2.imwrite("dataset/User."+str(id)+"."+str(sampleno)+".jpg",gray[y:y+h, x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        
        cv2.imshow('face',img)
        cv2.waitKey(10)
            

        if sampleno>=50:
            break
    cap.release()
    cv2.destroyAllWindows()

elif choice == 2:

    sampleno=0       
    
    imgPaths = []
    
    imgPaths = glob.glob("C:\\Users\\hp\\Desktop\\Facial Recognition\\New Folder\\dataset\\2\\*.jpg")
    for imgPath in imgPaths:
        print(imgPath)
        img = cv2.imread(imgPath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            sampleno+=1
            cv2.imwrite("dataset/User."+str(id)+"."+str(sampleno)+".jpg",gray[y:y+h, x:x+w])

        cv2.imshow('face', img)
        cv2.waitKey(50)       


