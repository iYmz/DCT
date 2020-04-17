
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2\opencv.hpp>  
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2\core\mat.hpp>
#include <thread>
#include<functional>
#include "Tools.h"
#include "Fourier.h"

using namespace cv;
using namespace std;

const int QY[8][8] =
{
{16,11,10,16,24,40,51,61},
{12,12,14,19,26,58,60,55},
{14,13,16,24,40,57,69,56},
{14,17,22,29,51,87,80,62},
{18,22,37,56,68,109,103,77},
{24,35,55,64,81,104,113,92},
{49,64,78,87,103,121,120,101},
{72,92,95,98,112,100,103,99}
};
const int QC[8][8] = {
	{17,18,24,47,99,99,99,99},
	{18,21,26,66,99,99,99,99},
	{24,26,56,99,99,99,99,99},
	{47,66,99,99,99,99,99,99},
	{99,99,99,99,99,99,99,99},
	{99,99,99,99,99,99,99,99},
	{99,99,99,99,99,99,99,99},
	{99,99,99,99,99,99,99,99}
};

Mat getChannel(Mat img, Mat dst, int channel) {

	for (int u = 0; u < img.rows; u++) {

		for (int v = 0; v < img.cols; v++) {

			dst.at<Vec3b>(u, v)[channel] = img.at<Vec3b>(u, v)[channel];
		}
	}
	return dst;
}
vector<vector<double>>  dDCT(vector<vector<double>> g) {

	double cm = 1.0;
	double cn = 1.0;
	if (g.empty() == 1)// 如果二维向量为空则返回；
		return g;
	int M = g.size(); // M rows
	int N = g[0].size();// N columns
	double const COMMON_FACTOR = 2.0 / sqrt(M * N);
	vector<vector<double>> G(M);

	//double MN_Sqrt = sqrt(M * N);

	for (int m = 0; m < M; m++) {
		double cm = 1.0;
		if (m == 0)cm = 1.0 / sqrt(2);
		for (int n = 0; n < N; n++) {
			double cn = 1.0;
			if (n == 0)cn = 1.0 / sqrt(2);

			double sum = 0;
			//double sum1 = 0;
			for (int u = 0; u < M; u++) {
				for (int v = 0; v < N; v++) {
					//double wux = PI * (2.0 * u + 1.0) * m / (2.0 * M);
					//double wvx = PI * (2.0 * v + 1.0) * n / (2.0 * N);
					//double cosux = cos(wux);
					//double cosvx = cos(wvx);
					//sum += COMMON_FACTOR * g[u][v] * cm * cosux * cn * cosvx;
					sum += COMMON_FACTOR * g[u][v] * cm * cos(PI * (2 * u + 1) * m / (2 * M)) * cn * cos(PI * (2 * v + 1) * n / (2 * N));
				}
			}

			G[m].push_back(sum);
		}
	}
	return G;
}

void dDCT(Mat& src, Mat& dst)
{
	double pi = 3.141592657;
	Mat C_Mat(src.rows, src.cols, CV_64FC1);
	Mat CT_Mat(src.rows, src.cols, CV_64FC1);

	for (int j = 0; j < C_Mat.rows; j++)
		C_Mat.at<double>(0, j) = sqrt(2.0 / (C_Mat.rows)) * sqrt(1.0 / 2);

	for (int i = 1; i < C_Mat.rows; i++)
		for (int j = 0; j < C_Mat.cols; j++)
			C_Mat.at<double>(i, j) = sqrt(2.0 / (C_Mat.rows)) * cos(pi * (i - 1) * (2 * j - 1) / 2 / (C_Mat.rows));

	CT_Mat = C_Mat.t();

	dst = C_Mat * src * CT_Mat;
}

Mat splitedDCT(Mat rgbMat) {
	if (rgbMat.empty())return rgbMat;
	Mat dst;
	double pi = 3.141592657;
	rgbMat.convertTo(rgbMat, CV_64FC1);
	Mat C_Mat(rgbMat.rows, rgbMat.cols, CV_64FC1);
	Mat CT_Mat(rgbMat.rows, rgbMat.cols, CV_64FC1);

	for (int j = 0; j < C_Mat.rows; j++)
		C_Mat.at<double>(0, j) = sqrt(2.0 / (C_Mat.rows)) * sqrt(1.0 / 2);

	for (int i = 1; i < C_Mat.rows; i++)
		for (int j = 0; j < C_Mat.cols; j++)
			C_Mat.at<double>(i, j) = sqrt(2.0 / (C_Mat.rows)) * cos(pi * (i - 1) * (2 * j - 1) / 2 / (C_Mat.rows));

	CT_Mat = C_Mat.t();

	return dst = C_Mat * rgbMat * CT_Mat;
}

vector<Mat> splitedIDCT(Mat& rgbMat) {
	if (rgbMat.empty())return rgbMat;
	vector<Mat> dst;
}

vector<Mat> splitMatTo8s(Mat mat) {

	if (mat.empty())return mat;
	int row = mat.rows / 8;
	int col = mat.cols / 8;
	vector<Mat> pieces(row * col);
	
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			pieces[i * col + j].create(8, 8, CV_64FC1);
			for (int u = 0; u < 8; ++u) {
				for (int v = 0; v < 8; ++v) {
					pieces[i * col + j].at<double>(u, v) = mat.at<double>(u + i * 8, v + j * 8);
				}
			}

		}
	}
	return pieces;
}
/***********************************************华丽的分隔符*****************************************************************/
void bgr2Yuv(Mat src,Mat &dst) {
	cvtColor(src, dst, COLOR_BGR2YUV);
}

vector<Mat> splitYuvChannels(Mat src) {
	vector<Mat> dst;
	
	split(src, dst);
	return dst;
}

void quantify(Mat& src, const int quantifyArray[8][8]) {
	int rows = src.rows;
	int cols = src.cols;
	for (int i = 0; i < rows/8; ++i) {
		for (int j = 0; j < cols/8; ++j) {
			for (int u = 0; u < 8; ++u) {
				for (int v = 0; v < 8; ++v) {
					int x = u + i * 8;
					int y = v + j * 8;
					
						src.at<double>(x,y) = round(src.at<double>(x, y) / quantifyArray[u][v]);
				}
			}
		}
	}

}

Mat mergeYUVChannels(vector<Mat> idctYuv) {
	Mat mergedIDCTYuv(idctYuv[0].size(), CV_8UC3);
	if (idctYuv.empty()) return mergedIDCTYuv;
	for (int i = 0; i < idctYuv[0].rows; ++i) {
		for (int j = 0; j < idctYuv[0].cols; ++j) {
			mergedIDCTYuv.at<Vec3b>(i, j)[0] = (uchar)idctYuv[0].at<double>(i, j);
			mergedIDCTYuv.at<Vec3b>(i, j)[1] = (uchar)idctYuv[1].at<double>(i, j);
			mergedIDCTYuv.at<Vec3b>(i, j)[2] = (uchar)idctYuv[2].at<double>(i, j);
		}
	}
	return mergedIDCTYuv;
}

void antiQuantify(Mat& src,int const quantifyArray[8][8]) {
	int rows = src.rows;
	int cols = src.cols;
	for (int i = 0; i < rows / 8; ++i) {
		for (int j = 0; j < cols / 8; ++j) {
			for (int u = 0; u < 8; ++u) {
				for (int v = 0; v < 8; ++v) {
					int x = u + i * 8;
					int y = v + j * 8;

					src.at<double>(x, y) = round(src.at<double>(x, y) * quantifyArray[u][v]);
				}
			}
		}
	}

}

void copyRGBMat(Mat const src, Mat& ouput) {
	int w = src.rows;
	int h = src.cols;
	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			ouput.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
			ouput.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
			ouput.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
		}
	}
}

Mat  extendImg(Mat& src) {
	int height = src.rows;
	int width = src.cols;
	Mat dst;
	int h =8- height % 8;
	int w =8- width % 8;
	dst.create(src.rows + h, src.cols + w, CV_8UC3);
	copyRGBMat(src, dst);
	if (h != 0 && w != 0)
	{
		for (int i = 0; i < h; ++i) {
			int x = height + i - 1;
			for (int j = 0; j < dst.cols; ++j) {
				dst.at<Vec3b	>(x, j)[0] = 255;
				dst.at<Vec3b	>(x, j)[1] = 255;
				dst.at<Vec3b	>(x, j)[2] = 255;
			}
		}
		for (int j =0; j < w; ++j) {
			int y = width + j - 1;
			for (int i = 0; i < dst.rows; ++i) {
				dst.at<Vec3b>(i,y)[0] = 255;
				dst.at<Vec3b>(i,y)[1] = 255;
				dst.at<Vec3b>(i,y)[2] = 255;
			}
		}

	}
	else if (h != 0) {

		dst.create(src.rows + 8 - h, src.cols , CV_8UC3);
		for (int i = h ; i < 8; ++i) {
			for (int j = 0; j < width; ++j) {

				int x = height + i - 1;
				dst.at<Vec3b	>(x, j)[0] = 255;
				dst.at<Vec3b	>(x, j)[1] = 255;
				dst.at<Vec3b	>(x, j)[2] = 255;
			}
		}

		for (int j = width % 8; j < 8; ++j) {
			for (int i = 0; i < height; ++i) {
				src.at<Vec3b>(i, width + j - 1)[0] = 255;
				src.at<Vec3b>(i, width + j - 1)[1] = 255;
				src.at<Vec3b>(i, width + j - 1)[2] = 255;
			}
		}
	}
	else	if (w  != 0) {
		dst.create(src.rows, src.cols + 8 - w, CV_8UC3);
		for (int j = width % 8; j < 8; ++j) {
			for (int i = 0; i < height; ++i) {
				src.at<Vec3b>(i, width + j - 1)[0] = 255;
				src.at<Vec3b>(i, width + j - 1)[1] = 255;
				src.at<Vec3b>(i, width + j - 1)[2] = 255;
			}
		}
	}
	return dst;
}
int main() {
	String img1 = "C://Users//qq941//Pictures//imagingbook-images-en1//imgs-by-chap//ch12//alps-01.jpg";
	String img2 = "D://OneDrive - hdu.edu.cn//image//pic//GirlWithHat1.jpg";
	String img3 = "C://Users//qq941//Pictures//image//comic.jpg";
	//Konachan.com - 305031 sample
	Mat rgb = imread(img1 );
	Mat yuv;
	rgb = extendImg(rgb);
	//convert to yuv ,done.
	bgr2Yuv(rgb, yuv);

	//split yuv channels done.
	vector<Mat> yuvChannels = splitYuvChannels(yuv);


	//merge YUV channels
	Mat yuvMerged(rgb.size(), CV_8UC3);
	merge(yuvChannels, yuvMerged);


	//DCT begin
	Mat dctY, dctU,dctV;
	dct(Mat_<double>(yuvChannels[0]), dctY);
	dct(Mat_<double>(yuvChannels[1]), dctU);
	dct(Mat_<double>(yuvChannels[2]), dctV);

	imshow("DCTY", dctY);
	waitKey();


	//quantify begin
	quantify(dctY, QY);
	quantify(dctU, QC);
	quantify(dctV, QC);
	imshow("Quantified dctY", dctY);
	imshow("Quantified ,dctU", dctU);
	imshow("Quantified dctV", dctV);
	waitKey();





	antiQuantify(dctY, QY);
	antiQuantify(dctU, QC);
	antiQuantify(dctV, QC);


	//IDCT Test that result is true
	Mat idctY, idctU, idctV;
	idct(dctY, idctY);
	idct(dctU, idctU);
	idct(dctV, idctV);
	vector<Mat> idctYuv;
	idctYuv.push_back(idctY);
	idctYuv.push_back(idctU);
	idctYuv.push_back(idctV);
	Mat mergedIDCTYuv;//
	mergedIDCTYuv=mergeYUVChannels(idctYuv);
	imshow("mergedIDCTYuv", mergedIDCTYuv);
	waitKey();

	Mat processedRGB;
	cvtColor(mergedIDCTYuv, processedRGB,COLOR_YUV2BGR);

	imshow("processedRGB", processedRGB);
	waitKey();
	return 0;
}


