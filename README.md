# FacialRecognition
Facial Recognition in Python using OpenCV

# OpenCV

OpenCV is the most popular library for computer vision. Originally written in C/C++, it now provides bindings for Python.

OpenCV uses machine learning algorithms to search for faces within a picture. For something as complicated as a face, there isn’t one simple test that will tell you if it found a face or not. Instead, there are thousands of small patterns/features that must be matched. The algorithms break the task of identifying the face into thousands of smaller, bite-sized tasks, each of which is easy to solve. These tasks are also called classifiers.

# Dependencies
1. OpenCV
2. NumPy
3. Glob
4. Pillow
5. OS

# Execution of the Code
The code is mainly divided into three parts:
          i)    Dataset Creator
          ii)   Training 
          iii)  Detecting
    
 The above three parts are implemented using three functions: i)dataset_creator() ii)trainer()  iii)detector()
 
1. Execute the FacialRecgn.py
2. The user can choose whether to create the dataset or run the detector. Create a dataset if you don't have a dataset yet. 
3. The dataset can be created using a batch of images or with a webcam. Choose the option you prefer. 
          i)      Every time while adding a dataset for a new face, append the person's name in the list idlist.
          ii)     The batch of photos to create a dataset should be placed in a folder with the person's name(in all lower case) as the                   folder name and this folder is in turn placed in the Data folder.
          iii)    Increment the user ID by one eveytime you add a new face's dataset
4. After the dataset is created, the trainer is run and the trainingdata.yml file is created in the folder trainer.
5. Run the detector. The detection can also be done on webcams or on images. The folder TestImages has some test images to run detection on.
 
   
