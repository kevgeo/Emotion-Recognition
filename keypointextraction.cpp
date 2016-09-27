#include <numeric>
#include <string.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <math.h>
#include <vector>

using namespace cv; 
using namespace std; 



int main()
{
	int minHessian = 400;
	vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;

    Mat image_1 = imread("face.JPG");
	SurfFeatureDetector detector( minHessian );
    detector.detect( image_1, keypoints_1 );
    //detector.detect( image_2, keypoints_2 );
    Mat image_2;
    drawKeypoints( image_1, keypoints_1, image_2,(255,0,0) );

    imshow("Detected Features",image_2);
    waitKey(0);

	return 0;
}