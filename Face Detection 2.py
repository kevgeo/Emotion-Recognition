import cv2
import numpy as np
#from skimage.feature import greycomatrix

img = cv2.imread("F:\Projects\Emotion Detection\Database\AM10SAS.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('F:\Program Files\OpenCV\sources\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('F:\Program Files\OpenCV\sources\data\haarcascades\haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('F:\Program Files\OpenCV\sources\data\haarcascades\haarcascade_mcs_mouth.xml')

faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.2, minNeighbors = 3, flags = 1, minSize = (70, 40))
    mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor = 2.8, minNeighbors = 11, flags = 1, minSize = (150, 80))
    count = 0
    for (ex,ey,ew,eh) in eyes:
        ey = int(ey + 0.3*eh)
        eh = int(eh - 0.4*eh)
        cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

##        kernel = np.ones((3,3), np.uint8)
##        eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
##        erosion = cv2.erode(eye_roi, kernel, iterations = 1)
##        dilation = cv2.dilate(erosion, kernel, iterations = 1)
##        eye_edges = cv2.Canny(dilation, threshold1 = 0, threshold2 = 100, apertureSize = 3, L2gradient = True)
##
##
##        if count == 0:
##            strname = "Left"
##        else:
##            strname = "Right"
##        cv2.namedWindow(strname, cv2.WINDOW_NORMAL)
##        cv2.imshow(strname, eye_edges)
##        x_edge = np.where(eye_edges == 255)[0]
##        y_edge = np.where(eye_edges == 255)[1]
##        x_cord = max(x_edge)
##        y_cord = y_edge[np.where(x_edge == x_cord)[0][0]]
##        cv2.circle(roi_color[ey:ey+eh, ex:ex+ew], (x_cord, y_cord), 2, (0, 255, 0), 3)
##        count = count + 1

##        eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
##        circles = cv2.HoughCircles(eye_roi, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, param1=50, param2=30, minRadius = 0, maxRadius=0)
##        circles = np.uint16(np.around(circles))
##        for i in circles[0, :]:
##            cv2.circle(roi_color[ey:ey+eh, ex:ex+ew],(i[0],i[1]),i[2],(0,255,0),2)
##            cv2.circle(roi_color[ey:ey+eh, ex:ex+ew],(i[0],i[1]),2,(0,0,255),3)


    for (mx, my, mw, mh) in mouth:
        mxorg = mx
        myorg = my
        my = int(my + 0.25*mh)
        mh = int(mh - 0.5*mh)
        cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)

        mouth_roi = roi_gray[my:my+mh, mx:mx+mw]
        mouth_roi_color = roi_color[my:my+mh, mx:mx+mw]
        kernel = np.ones((3,3), np.uint8)
        imgl = mouth_roi[:, 0:mouth_roi.shape[1]/2]
        imgr = mouth_roi[:, mouth_roi.shape[1]/2:mouth_roi.shape[1]]

        corners = cv2.cornerHarris(np.float32(imgl), 2, 5, 0.04)
        dst = cv2.dilate(corners, kernel, iterations = 2)
        mx1_cord = int(np.where(dst == dst.max())[0].mean())
        my1_cord = int(np.where(dst == dst.max())[1].mean())
        cv2.circle(mouth_roi_color, (my1_cord, mx1_cord), 5, (0, 0, 255), -1)

        corners = cv2.cornerHarris(np.float32(imgr), 2, 5, 0.04)
        dst = cv2.dilate(corners, kernel, iterations = 2)
        mx2_cord = int(np.where(dst == dst.max())[0].mean())
        my2_cord = int(np.where(dst == dst.max())[1].mean())
        cv2.circle(mouth_roi_color, (mouth_roi.shape[1]/2+my2_cord, mx2_cord), 5, (0, 0, 255), -1)

        #cv2.imshow('mouth', mouth_roi_color)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()