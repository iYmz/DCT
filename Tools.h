#pragma once
#ifndef TOOLS_H
#define TOOLS_H
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

template<typename T>
T **Create2DArray(T init,int rows,int cols){
	T ** pixels = new T* [rows];
	for (int i = 0; i < rows; i++)
		{
			pixels[i] = new T[cols];
			memset(pixels[i], init, cols * sizeof(T));
	}
	return pixels;
}

/**
@param windowName Name of created window
@param filePath Path of image
@param readFlags OpenCV enum type of how to load an image
*/
Mat LoadImageAndShow(String const &windowName,String const &filePath,int readFlags = 1) {
/** @brief Read an image from path of disk, show it in the window and return a Mat Object
Use the function to create a Mat object
8 bits grey image could be loaded correctly as the readFlag's value is 0;
*/
	Mat mat = imread(filePath, readFlags);
	if (mat.empty()) {
		return mat = NULL;
	};
	imshow(windowName, mat);
	return mat;
/* @example
int main(){
	Filters filter;
	filter.Average_8DGrey_3x3("C://Users//qq941//Pictures//imagingbook-images-en1//imgs-by-chap//ch02//airfield-05small-auto.tif");
	}
	*/

}


/*
@param mat OpenCV Mat object
@param u x
@param v y
*/
int* GetRGB(Mat mat, int u, int v) {
	/* @brief Return int value of  R, G, B
	red:array[0]
	green:array[1]
	blue:array[2]
	*/
	static int rgb[3];
	
	rgb[2] = mat.at<Vec3b>(u,v)[2];// Color of Red
	rgb[1] = mat.at<Vec3b>(u,v)[1];// Color of Green
	rgb[0] = mat.at<Vec3b>(u,v)[0];// Color of Blue
	return rgb;

	/* @example
	int main(){
		Mat rgbMat = LoadImageAndShow("RGB IMAGE", imagePath);
		int *rgb = GetRGB(rgbMat, 10, 10);
		cout <<endl<< "R:" << rgb[2] << " G:" << rgb[1] << " B:" << rgb[0];
		}

	*/
}

/*
@param mat OpenCV Mat object
@param u x
@param v y
*/
int* GetRGB(int color) {
	/* @brief Return int value of  R, G, B
	blue:rgb[0]
	green:rgb[1]
	red:rgb[2]
	*/
	static int rgb[3];
	rgb[0] = (color & 0xff0000) >> 16;// Color of Blue
	rgb[1] = (color & 0x00ff00) >> 8;// Color of Green
	rgb[2] = color & 0x0000ff;// Color of Red
	return rgb;

	/* @example
	int main(){
		int color = ComposeRGB(1,1,1);
		int *rgb = GetRGB();
		cout <<endl<< "R:" << rgb[0] << " G:" << rgb[1] << " B:" << rgb[2];
		}

	*/
}

int ComposeRGB(int r, int g, int b) {


	int color = ((r & 0xff) << 16 | (g & 0xff) << 8 | (b & 0xff));
	return color;

	/*
	Mat rgbMat = LoadImageAndShow("RGB IMAGE", path);
	int* rgb = GetRGB(rgbMat, x, y);
	cout << endl << "R:" << rgb[2] << " G:" << rgb[1] << " B:" << rgb[0];
	int color = ComposeRGB(rgb[0], rgb[1], rgb[2]);
	cout << endl << "color:	" << color;
	rgb = GetRGB(color);
	cout << endl << "R:" << rgb[2] << " G:" << rgb[1] << " B:" << rgb[0];
	*/

}



#endif // !TOOLS_H

