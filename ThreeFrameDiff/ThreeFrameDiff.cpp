
#include "core/core.hpp"

#include "highgui/highgui.hpp"

#include "imgproc/imgproc.hpp"

#include <iostream>

#include <stdio.h>


using namespace cv;
using namespace std;

vector<vector<cv::Point>> g_vContours;
vector<Vec4i> g_vHierarchy;



int main(int argc, char* argv[])

{

	VideoCapture videoCap("EZVZ0048.mp4");

	if (!videoCap.isOpened())

	{

		return -1;

	}

	double videoFPS = videoCap.get(CV_CAP_PROP_FPS);  //获取帧率

	double videoPause = 1000 / videoFPS;

	Mat framePrePre; //上上一帧

	Mat framePre; //上一帧

	Mat frameNow; //当前帧

	Mat frameDet; //运动物体

	videoCap >> framePrePre;

	videoCap >> framePre;

	cvtColor(framePrePre, framePrePre, CV_RGB2GRAY);

	GaussianBlur(framePrePre, framePrePre, Size(7, 7), 0);

	cvtColor(framePre, framePre, CV_RGB2GRAY);

	//Canny(framePrePre, framePrePre, 100, 250);

	GaussianBlur(framePre, framePre, Size(7, 7), 0);

	//Canny(framePre, framePre, 100, 250);

	int save = 0;

	while (true)

	{

		videoCap >> frameNow;

		if (frameNow.empty() || waitKey(videoPause) == 27)

		{

			break;

		}

		cvtColor(frameNow, frameNow, CV_RGB2GRAY);

		GaussianBlur(frameNow, frameNow, Size(7, 7), 0);

		//Canny(frameNow, frameNow, 100, 250);

		Mat Det1;

		Mat Det2;

		absdiff(framePrePre, framePre, Det1);  //帧差1

		absdiff(framePre, frameNow, Det2);     //帧差2

		threshold(Det1, Det1, 0, 255, CV_THRESH_OTSU);  //自适应阈值化

		threshold(Det2, Det2, 0, 255, CV_THRESH_OTSU);

		Mat element = getStructuringElement(0, Size(30, 7));  //膨胀核

		dilate(Det1, Det1, element);    //膨胀

		dilate(Det2, Det2, element);

		bitwise_and(Det1, Det2, frameDet);

		framePrePre = framePre;

		framePre = frameNow;

		namedWindow("Video", WINDOW_NORMAL);

		imshow("Video", frameNow);

		namedWindow("Detection", WINDOW_NORMAL);

		findContours(frameDet, g_vContours , g_vHierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		cout << g_vContours[0];

		imshow("Detection", frameDet);

	}

	return 0;

}
