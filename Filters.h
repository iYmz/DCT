#pragma once
#ifndef FILTERS_H
#define FILTERS_H

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "ExampleCaptureCamera.h"
#include "Tools.h"
using namespace cv;
using namespace std;
class Filters {
public:
	Filters();
	int Average_8DGrey_3x3(const String& imgPath);
	int H_3x3[3][3] = {
				{1,1,1},
				{1,1,1},
				{1,1,1}
	};
	int H_Sobel[3][3] = {
		{-1,0,1},
		{-1,0,2},
		{-1,0,1}
	};
	/** @breif Create a 3x3 filter to process 8bits grey image

	*/
	~Filters();

private:
};
Filters::Filters() {}
Filters::~Filters() {}
int Filters::Average_8DGrey_3x3(const String& imgPath) {

	int H_3x3[3][3] = {
					{1,1,1},
					{1,1,1},
					{1,1,1}
	};

	Mat image = LoadImageAndShow("Window", imgPath, 0);
	Mat copy = Mat(image.rows, image.cols, CV_8UC1);
	int rows = image.rows;
	int cols = image.cols;
	int step = image.step;
	cout << "Image informations:" << endl << "rows:" << rows << endl << "cols:" << cols << endl << "steps:" << step << endl;

	int** pixels = Create2DArray(0, rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int value = (int)image.at<uchar>(i, j);
			pixels[i][j] = value;
			//copy.data[i * cols + j] = pixels[i][j] = (int)image.ptr<uchar>(i)[j];
		}
	}
	for (int u = 1; u < rows - 1; u++) {
		for (int v = 1; v < cols - 1; v++) {
			int sum = 0;
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					int p = pixels[u + i][v + j] * H_3x3[1 + i][1 + j];
					sum += p;
				}
			}
			copy.data[u * cols + v] = sum / 6;
		}

	}

	imshow("AverageBlur", copy);
	waitKey();
	return 0;
}

#endif // !FILTERS_H

