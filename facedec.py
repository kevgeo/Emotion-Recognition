#Face detection and corner points on eyes		
import cv2
import sys
import argparse
#import imutils 
import numpy as np
from imutils import paths

count = 0
# We are building an argument parser which will take command line arguments 
#which will give paths to both facedataset and cascade file
arg = argparse.ArgumentParser()
arg.add_argument("-i", "--images", required=True , help="Path to image dataset")
#arg.add_argument("-f", "--cascades", required=True , help="Path to cascade file")
args = vars(arg.parse_args())

#Loading the required XML classifiers
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
#faceCascade = cv2.CascadeClassifier(args["cascades"])

# Making a list of all the images in the face dataset folder
images = list(paths.list_images(args["images"]))

#Going through each image in the image list
for img in images:
	image = cv2.imread(img)
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#converting the image into grayscale, only have to deal with 255 values in each pixel   
	faces = faceCascade.detectMultiScale(
		gray_img, 
		scaleFactor = 1.5,
		minNeighbors = 5,
		)
	#The above function will detect faces of different sizes and detected objects are returned as a list of rectangles

	#print "Found {0} faces!".format(len(faces)) #fancier way of outputting, makes it easy when we want to display many variables

	#Draw rectangle around the faces
	for (x,y,w,h) in faces:
		print x,y,w,h
		#x,y are coordinates of top-left corner of rectangle
		#w,h are width and height respectively		
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2) #third param is basically the BGR values, so we want to draw green rectangle
		roi_gray = gray_img[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		#print roi_gray,roi_color
		eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.2, minSize=(70, 40), minNeighbors=3,flags = 1)
		#print "Found {0} eyes!".format(len(eyes))
		#if ( len(eyes) != 2):
			#count += 1
		
		#Detecting corner points on eyes
		for (ex,ey,ew,eh) in eyes:
			#blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
			#wide = cv2.Canny(blurred, 100, 200)
			#cv2.imshow("Edgess Found", wide)
			#cv2.waitKey(0)
			#gray = np.float32(wide)
			#corners = cv2.goodFeaturesToTrack(gray, 10, 0.40, 10)
			#corners = np.int0(corners)
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			#for corner in corners:
				#A,B = corner.ravel()
				#print A,B
				#cv2.circle(roi_color, (A,B), 3, 255, -1)
				#cv2.circle(roi_color, (95,165), 3, 255, -1)
				#cv2.circle(roi_color, (170,165), 3, 255, -1)
				#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	cv2.imshow("Faces Found", image)
	cv2.waitKey(0)

#print count 
