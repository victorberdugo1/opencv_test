//g++ -L./lib main.cpp -o my_program -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs
//export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	int	iLastX = -1;
	int	iLastY = -1;

	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}
	namedWindow("Control", WINDOW_AUTOSIZE);
	while (true)
	{
		Mat imgOriginal, imgGray, imgThresholded;
		bool bSuccess = cap.read(imgOriginal);
		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		cvtColor(imgOriginal, imgGray, COLOR_BGR2GRAY);
		threshold(imgGray, imgThresholded, 100, 255, THRESH_BINARY);
		vector<vector<Point>> contours;
		findContours(imgThresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		if (!contours.empty())
		{
			size_t largestContourIndex = 0;
			double largestArea = 0;
			for (size_t i = 0; i < contours.size(); i++)
			{
				double area = contourArea(contours[i]);
				if (area > largestArea)
				{
					largestArea = area;
					largestContourIndex = i;
				}
			}
			Rect rect = boundingRect(contours[largestContourIndex]);
			rectangle(imgOriginal, rect.tl(), rect.br(), Scalar(0, 255, 0), 2);
			Point center = (rect.tl() + rect.br()) / 2;
			if (iLastX >= 0 && iLastY >= 0)
			{
				line(imgOriginal, Point(iLastX, iLastY), center, Scalar(0, 0, 255), 2);
			}
			iLastX = center.x;
			iLastY = center.y;
		}
		imshow("Thresholded Image", imgThresholded);
		imshow("Original", imgOriginal);
		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	return 0;
}
