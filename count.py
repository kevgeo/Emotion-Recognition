# Counting number of images in dataset
import cv2
import sys
import argparse
import imutils 
from imutils import paths

count = 0
# We are building an argument parser which will take command line arguments 
#which will give paths to both facedataset and cascade file
arg = argparse.ArgumentParser()
arg.add_argument("-i", "--images", required=True , help="Path to image dataset")
args = vars(arg.parse_args())

# Making a list of all the images in the face dataset folder
images = list(paths.list_images(args["images"]))

print len(images)