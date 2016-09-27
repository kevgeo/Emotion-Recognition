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
	gray_img = cv2.cvtColor(image, cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)   #converting the image into grayscale, only have to deal with 255 values in each pixel
	cornerMap = cv2.cv.CreateMat( gray_img.height, gray_img.width, cv.CV_32FC1)
	# OpenCV corner detection
	cv2.CornerHarris( gray_img, cornerMap, 3)
	for y in range(0, image.height):
		for x in range(0, image.width):
			harris = cv.Get2D(cornerMap, y, x) # get the x,y value
			if harris[0] > 10e-06: # check the corner detector response
				cv.Circle(imcolor,(x,y),2,cv.RGB(155, 0, 25)) # draw a small circle on the original image

	cv2.imshow('dst', image)
	cv2.waitKey(0)

	#gray_img = np.float32(gray_img)

	#dst = cv2.cornerHarris(gray_img,2,3,0.04)
	
	#readingsult is dilated for marking the corners, not important
	#dst = cv2.dilate(dst,None)
	
	# Threshold for an optimal value, it may vary depending on the image.
	#image[dst>0.01*dst.max()]=[0,0,255]

	#cv2.imshow('dst',image)
	#if cv2.waitKey(0) & 0xff == 27:
	#	cv2.destroyAllWindows()