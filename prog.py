#To execute program on command line, 
#type python prog.py --images <Name of folder containing faces> --cascade <haarcascade_frontalface_default.xml>



import cv2
import sys
import argparse
import imutils #download package from pyimagesearch
from imutils import paths

#imagePath = sys.argv[1]
#cascPath = sys.argv[2]
arg = argparse.ArgumentParser()
arg.add_argument("-i", "--images", required=True, help="path to images folder")
arg.add_argument("-f", "--cascade", required=True, help="path to cascade file")
args = vars(arg.parse_args())


faceCascade = cv2.CascadeClassifier(args["cascade"])
images = list(paths.list_images(args["images"]))
for imagep in images:
	image = cv2.imread(imagep)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
	    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	print "Found {0} faces!".format(len(faces))


	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
	    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	cv2.imshow("Faces found" ,image)
	cv2.waitKey(0)






