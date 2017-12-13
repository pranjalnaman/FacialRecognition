import cv2
import numpy as np
import glob
import os
from PIL import Image
import csv

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

idlist = ["","trump","trudeau"]
def getlist(str):
	if(str!=""):
		idlist.append(str)
		return idlist
	else:
		return idlist

#DATASET CREATOR

global IDname
IDname = []
def dataset_creator():
	global IDname

	id = input("Enter the ID:\n")

	idname = input("\nEnter the name associated with the ID: ")
	IDname = getlist(idname)
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
		
		imgPaths = glob.glob("Data/"+idname+"/*.jpg")
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

	c = input("\nDO YOU WANT TO ADD MORE DATA SETS?(Y/N): ")
	
	if ((c == 'y') or (c == 'Y')):
		dataset_creator()
	else:
		trainer()  


#TRAIN THE DETECTOR

def getImagesWithID(path):
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	#listdir() provides the list of all the paths of all the files in the directory 'path'
	#list of faces
	faceSamples = []
	#list of IDs
	IDs = []

	#Looping through all the image paths and loading the IDs and faces

	for imagePath in imagePaths:
		#loading the image and converting it to grayscale
		pilImage = Image.open(imagePath).convert('L')

		#Converting the PIL image to a numpy array because openCV works with numpy arrays
		imageNP = np.array(pilImage,'uint8')

		#Getting the ID
		ID = int(os.path.split(imagePath)[-1].split(".")[1])


		faceSamples.append(imageNP)
		IDs.append(ID)
		cv2.imshow('training',imageNP)
		cv2.waitKey(10)


	return faceSamples, np.array(IDs)

def trainer():

	recognizer = cv2.face.LBPHFaceRecognizer_create()
	path = 'dataset'

	faces,IDs = getImagesWithID(path)
	recognizer.train(faces,np.array(IDs))
	recognizer.write('trainer/trainingdata.yml')
	cv2.destroyAllWindows()


#DETECTOR or RECOGNIZER

def detector():
	rec = cv2.face.LBPHFaceRecognizer_create()
	rec.read("trainer/trainingdata.yml")
	IDname = getlist("")
	print(IDname)
	choice = int(input("1. TO USE THE WEBCAM FOR DETECTION\n2. TO DETECT FACES IN AN IMAGE\nCHOICE: "))
	font = cv2.FONT_HERSHEY_SIMPLEX
	if choice == 1:
		cap = cv2.VideoCapture(0)
		id=0
		font = cv2.FONT_HERSHEY_SIMPLEX
		while 1:
			ret, img = cap.read()
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray)

			for (x,y,w,h) in faces:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
				id,conf = rec.predict(gray[y:y+h,x:x+w])
				cv2.putText(img,str(IDname[id]),(x,y+h),font,1,(0,0,255))
				roi_gray = gray[y:y+h, x:x+w]
				roi_color = img[y:y+h, x:x+w]
			   
			cv2.imshow('img',img)
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break

		cap.release()
		cv2.destroyAllWindows()

	elif choice == 2:
		dir1 = input("ENTER THE PATH TO THE IMAGE: ")
		img = cv2.imread(dir1)
		cv2.imshow('img',img)
		cv2.waitKey(100)
		faces = face_cascade.detectMultiScale(img)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
			id,conf = rec.predict(gray[y:y+h,x:x+w])
			cv2.putText(img,str(IDname[id]),(x,y+h),font,1,(0,0,255),4)
			roi_color = img[y:y+h, x:x+w]

		cv2.imshow('FinalImg',img)
		cv2.waitKey(100000)


#main() to call the abovementioned functions

def main():
	print("----------FACE RECOGNITION USING CV2----------\n\n")
	print(IDname)
	choice = input("TO RUN THE DETECTOR, PRESS 1\nTO CREATE A NEW DATASET AND TRAIN, PRESS ANY OTHER KEY\n\nCHOICE: ")

	if(int(choice) == 1):
		detector()

	else:
		dataset_creator()
		ch = int(input("TO RUN THE DETECTOR, PRESS 1: "))
		if ch == 1:
			detector()
		else:
			print("WRONG SELECTION. STARTING AGAIN...")
			cv2.waitKey(1000)
			main()


main()
