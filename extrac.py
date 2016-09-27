#feature extraction using harris corner detection
import cv2
#import cv
import sys	
import argparse
import numpy as np
import imutils
from imutils import paths

arg = argparse.ArgumentParser()
arg.add_argument("-i", "--images", required=True , help="Path to image dataset")
args = vars(arg.parse_args())

# Making a list of all the images in the face dataset folder
images = list(paths.list_images(args["images"]))


#Going through each image in the image list
for img in images:
	image = cv2.imread(img)		#reading the image
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   #converting the image into grayscale, only have to deal with 255 values in each pixel
	corners = cv2.goodFeaturesToTrack( gray_img, 25, 0.01, 10)
	corners = np.int0(corners)
	for i in corners:
		x,y = i.ravel()
		cv2.circle( image, (x,y), 3, 255, -1)

	cv2.imshow('face', image)
	cv2.waitKey(0)

