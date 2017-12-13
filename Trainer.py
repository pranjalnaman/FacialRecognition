import os   #To read the directory
import cv2  #The computer vision library
import numpy as np
from PIL import Image


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('C:\\Users\\hp\\Desktop\\Facial Recognition\\haarcascade_frontalface_alt.xml')


path = 'dataset'

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

faces,IDs = getImagesWithID(path)
recognizer.train(faces,np.array(IDs))
recognizer.write('trainer/trainingdata.yml')
cv2.destroyAllWindows()