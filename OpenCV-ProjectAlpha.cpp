
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2\opencv.hpp>  
#include <opencv2\core\mat.hpp>
#include <thread>
#include<functional>
#include "Tools.h"
#include "Fourier.h"

using namespace cv;
using namespace std;

Mat getChannel(Mat img,Mat dst, int channel) {

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
int main()
{

	
	String geryimgPath = "C://Users//qq941//Pictures//imagingbook-images-en1//imgs-by-chap//ch02//airfield-05small-auto.tif";
	String rgbimgPath = "C://Users//qq941//Pictures//imagingbook-images-en1//imgs-by-chap//ch12//alps-01.jpg";
	String rgbimgPath1 = "C://Users//qq941//Desktop//22596167.jpg";
	String rgbimgPath2 = "C://Users//qq941//Pictures//image//GirlWithHat1.jpg";

	Mat  rgbMat = LoadImageAndShow("rgbimg", rgbimgPath, 1);
	constexpr int M = 8;
	constexpr int N = 8;
	int rows = rgbMat.rows;
	int cols = rgbMat.cols;

	Mat img;
	img = imread(rgbimgPath);
	Mat b(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));
	Mat g(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));
	Mat r(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));

	getChannel(img, r, RED_CHANNEL);
	getChannel(img, g, GREEN_CHANNEL);
	getChannel(img, b, BLUE_CHANNEL);

	//for (int u = 0; u < img.rows;u++) {
	//
	//	for (int v = 0; v < img.cols; v++) {

	//		r.at<Vec3b>(u, v)[2] = img.at<Vec3b>(u, v)[2];// Color of Red
	//		g.at<Vec3b>(u, v)[1] = img.at<Vec3b>(u, v)[1];// Color of Green
	//		b.at<Vec3b>(u, v)[0] = img.at<Vec3b>(u, v)[0];// Color of Blue
	//	}
	//}
	
	imshow("r", r);
	imshow("g", g);
	imshow("b", b);

	waitKey();


	
	vector<vector<double>> rgfull(rows, vector<double>(cols));
	vector<vector<double>> ggfull(rows, vector<double>(cols));
	vector<vector<double>> bgfull(rows, vector<double>(cols));
	for(int x=0;x<rows/8;x++){
		for(int y=0; y<cols/8; y++){
			vector<vector<double>> gr(M, vector<double>(N));
			vector<vector<double>> gg(M, vector<double>(N));
			vector<vector<double>> gb(M, vector<double>(N));
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			gr[m][n] = (double)r.at<Vec3b>(m + x * M, n + y * N)[2];
			gg[m][n] = (double)g.at<Vec3b>(m + x * M, n + y * N)[1];
			gb[m][n] = (double)b.at<Vec3b>(m + x * M, n + y * N)[0];
		}
	}

	vector<vector<double>>Gr(M, vector<double>(N));
	
	thread tr(DCT, ref(gr),ref(Gr));
	
	vector<vector<double>> Gg(M, vector<double>(N));
	thread tg(DCT, ref(gg), ref(Gg));
	//Gg = DCT(gg);

	vector<vector<double>> Gb(M, vector<double>(N));
	//Gb = DCT(gb);
	thread tb(DCT, ref(gb),ref(Gb));

	tr.join();
	tg.join();
	tb.join();

	vector<vector<double>> igr(M, vector<double>(N));
	thread tir(iDCT, Gr, ref(igr));
	//igr = iDCT(Gr);

	vector<vector<double>> igg(M, vector<double>(N));
	//igg = iDCT(Gg);
	thread tig(iDCT, ref(Gg), ref(igg));

	vector<vector<double>> igb(M, vector<double>(N));
	//igb= iDCT(Gb);
	thread tib(iDCT, ref(Gb), ref(igb));
	tir.join();
	tig.join();
	tib.join();


	//copy.data[i * cols + j] = pixels[i][j] = (int)image.ptr<uchar>(i)[j];

	
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			rgfull[i + x * 8][j + y * 8] = igr[i][j];
			ggfull[i + x * 8][j + y * 8] = igg[i][j];
			bgfull[i + x * 8][j + y * 8] = igb[i][j];
			
		}
	}
	
	}
	}

	Mat copy(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));



	
	for (int x = 0; x < rows ; x++) {
		for (int y = 0; y < cols ; y++) {
			//gr[m][n] = (double)r.at<Vec3b>(m + x * M, n + y * N)[2];
			copy.at<Vec3b>(x * cols + y)[2]=rgfull[x][y];
			copy.at<Vec3b>(x * cols + y)[1] = ggfull[x][y];
			copy.at<Vec3b>(x * cols + y)[0] = bgfull[x][y];
		}
	}
	
	imshow("IDCT original img", copy);
	waitKey();

		


		return 0;

		
}