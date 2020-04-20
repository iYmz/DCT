
#pragma comment(lib,"zlibwapi.lib")
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
#include "bundle.h"
#include <fstream>
#include <cassert>
#include"zlib.h"
using namespace cv;
using namespace std;
const int QY[8][8] ={
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

vector<Mat_<int>> splitMatTo8s(Mat_<int> mat) {

	if (mat.empty())return mat;
	int row = mat.rows / 8;
	int col = mat.cols / 8;
	vector<Mat_<int>> pieces(row * col);

	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			int block = i * col + j;
			pieces[block].create(8, 8);
			for (int u = 0; u < 8; ++u) {
				for (int v = 0; v < 8; ++v) {
					int x = u + i * 8;
					int y = v + j * 8;
					pieces[block][u][v] = mat(x, y);

				}
			}

		}
	}
	return pieces;
}
/***********************************************华丽的分隔符*****************************************************************/
void bgr2Yuv(Mat src,Mat &dst) {
	cvtColor(src, dst,COLOR_BGR2YUV);
}
void bgr2Yuv(vector<Mat_<int>>src, vector<Mat_<int>>& dst) {
	
	for (int i=0; i < src[0].rows; ++i) {
		for (int j = 0; j < src[0].cols; ++j) {
			dst[0][i][j] = round(0.299 * src[2][i][j] + 0.587 * src[1][i][j] + 0.114 * src[0][i][j]);//Y
			dst[2][i][j] = round(0.500 * src[2][i][j] - 0.419 * src[1][i][j] + -0.081 * src[0][i][j])+128;
			dst[1][i][j] = round(-0.169 * src[2][i][j] - 0.331 * src[1][i][j] + 0.500 * src[0][i][j])+128;
		}
	}
	
}

vector<Mat> splitYuvChannels(Mat src) {
	vector<Mat> dst;
	
	split(src, dst);
	return dst;
}

void quantify(Mat_<int>& src, const int quantifyArray[8][8]) {
	int rows = src.rows;
	int cols = src.cols;
	for (int i = 0; i < rows/8; ++i) {
		for (int j = 0; j < cols/8; ++j) {
			for (int u = 0; u < 8; ++u) {
				for (int v = 0; v < 8; ++v) {
					int x = u + i * 8;
					int y = v + j * 8;
					
						src[x][y] = round(src[x][y] / quantifyArray[u][v]);
				}
			}
		}
	}

}

Mat mergeYUVChannels(vector<Mat_<int>> idctYuv) {
	Mat mergedIDCTYuv(idctYuv[0].size(), CV_8UC3);
	if (idctYuv.empty()) return mergedIDCTYuv;
	for (int i = 0; i < idctYuv[0].rows; ++i) {
		for (int j = 0; j < idctYuv[0].cols; ++j) {
			mergedIDCTYuv.at<Vec3b>(i, j)[0] = (uchar)idctYuv[0][i][j];
			mergedIDCTYuv.at<Vec3b>(i, j)[1] = (uchar)idctYuv[1][i][j];
			mergedIDCTYuv.at<Vec3b>(i, j)[2] = (uchar)idctYuv[2][i][j];
		}
	}
	return mergedIDCTYuv;
}
Mat mergeYUVChannelsT(vector<Mat> idctYuv) {
	Mat mergedIDCTYuv(idctYuv[0].size(), CV_8UC3);
	if (idctYuv.empty()) return mergedIDCTYuv;
	for (int i = 0; i < idctYuv[0].rows; ++i) {
		for (int j = 0; j < idctYuv[0].cols; ++j) {
			mergedIDCTYuv.at<Vec3b>(i, j)[0] = idctYuv[0].at<uchar>(i, j);
			mergedIDCTYuv.at<Vec3b>(i, j)[1] = idctYuv[1].at<uchar>(i, j);
			mergedIDCTYuv.at<Vec3b>(i, j)[2] = idctYuv[2].at<uchar>(i, j);
		}
	}
	return mergedIDCTYuv;
}
void antiQuantify(Mat_<int>& src,int const quantifyArray[8][8]) {
	int rows = src.rows;
	int cols = src.cols;
	for (int i = 0; i < rows / 8; ++i) {
		for (int j = 0; j < cols / 8; ++j) {
			for (int u = 0; u < 8; ++u) {
				for (int v = 0; v < 8; ++v) {
					int x = u + i * 8;
					int y = v + j * 8;
					//int val =round(src.at<double>(x, y) * quantifyArray[u][v]);
					//if (val > 255)val = 255;
					src[x][y] = round(src[x][y] * quantifyArray[u][v]);
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

 void piecesDct(vector<Mat_<int>> &pieces) {
	vector<Mat_<double>>  temp(pieces.size());
	for (int i = 0; i < pieces.size(); ++i) {
		dct(Mat_<double>(pieces[i]), temp[i]);
		pieces[i] = Mat_<int>(temp[i]);
	}
}
 void piecesQuantity(vector<Mat_<int>>& pieces, int flag = 0) {
	 if (flag == 0) {
		 for (int i = 0; i < pieces.size(); ++i) {
			 quantify(pieces[i], QY);
		 }
	 }
	 else {
		 for (int i = 0; i < pieces.size(); ++i) {
			 quantify((pieces[i]), QC);
		 }

	 }
 }
void piecesAntiquantity(vector<Mat_<int>>& pieces, int flag = 0) {

	if (flag == 0)
		for (int i = 0; i < pieces.size(); ++i) {
			antiQuantify((pieces[i]), QY);
		}
	else {
		for (int i = 0; i < pieces.size(); ++i) {
			antiQuantify((pieces[i]), QC);
		}
	}

}

void  piecesIDct(vector<Mat_<int>>& pieces) {
	vector<Mat_<double>>  temp(pieces.size());
	for (int i = 0; i < pieces.size(); ++i) {
		idct(Mat_<double>(pieces[i]), temp[i]);
		pieces[i] = temp[i];
	}

}
Mat  extendImg(Mat& src) {
	int height = src.rows;
	int width = src.cols;
	Mat dst;
	int h = 0;
	int w = 0;
	if(height % 8!=0)
		h =8- height % 8;
	if (width % 8 != 0)
	w =8- width % 8;
	dst.create(src.rows + h, src.cols + w, CV_8UC3);
	copyRGBMat(src, dst);
	//imshow("原图", src);
	//imshow("扩展", dst);
	//waitKey();
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
	else {
		return src;
	}

	//imshow("扩展赋值", dst);
	//waitKey();
	return dst;
}
int balanceRangeCheck(int val) {
	if (val<127 && val>-128)return val;
	if (val > 127)return 127;
	else if (val < -128)return -128;
}
void balanceMatRangeWithRangeCheck(Mat src, Mat& dst, int balance) {

	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {

			dst.at<Vec3b>(i, j)[0] = balanceRangeCheck(src.at<Vec3b>(i, j)[0] - balance);
			dst.at<Vec3b>(i, j)[1] = balanceRangeCheck(src.at<Vec3b>(i, j)[1] - balance);
			dst.at<Vec3b>(i, j)[2] = balanceRangeCheck(src.at<Vec3b>(i, j)[2] - balance);
		}
	}
	//return dst;
}
Mat  balanceMatRange(Mat_<int> src, int balance) {
	Mat dst(src.size(), CV_8SC3);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			//int srcVal = src.at<Vec3b>(i, j)[0];
			//int temp = src.at<Vec3b>(i, j)[0] - balance;
			//cout << "src val:" << srcVal << "  " << "dst val:" << (int)dst.at < Vec3b>(i, j)[0] << "Temp:" << temp << "  " << "char:" << src.at<Vec3b>(i, j)[0] - balance;
			dst.at<Vec3b>(i,j)[0] = short(src.at<Vec3b>(i, j)[0] - 128);
			//std::cout << endl << "dst:" << short(dst.at<Vec3b>(i, j)[0])<<endl;
			dst.at<Vec3b>(i, j)[1] = short(src.at<Vec3b>(i, j)[1] - 128);
			dst.at<Vec3b>(i, j)[2] = short(src.at<Vec3b>(i, j)[2] - 128);
			//dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1] - balance;
			//dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2] - balance;
			
		}
	}
	return dst;
}
vector<Mat_<int>>  balanceMat_Range(vector<Mat_<int>> src, int balance) {
	//Mat dst(src.size(), CV_8SC3);
	Mat_<int> mat_[3];
	vector<Mat_<int>> channels = src;
	mat_[0] = channels[0];
	mat_[1] = channels[1];
	mat_[2] = channels[2];

	for (int i = 0; i < src[0].rows; ++i) {
		for (int j = 0; j < src[0].cols; ++j) {
			int srcVal = src[0][i][j];
			int temp = src[0][i][j] - balance;
			mat_[0][i][j] = src[0][i][j] - balance;
			mat_[1][i][j] = src[1][i][j] - balance;
			mat_[2][i][j] = src[2][i][j] - balance;
		}
	}
	vector<Mat_<int>> temp;
	temp.push_back(mat_[0]);
	temp.push_back(mat_[1]);
	temp.push_back(mat_[2]);
	return temp;
}
vector<Mat_<int>>  balanceMat_Range(Mat src, int balance) {
	//Mat dst(src.size(), CV_8SC3);
	Mat_<int> mat_[3];
	vector<Mat> channels = splitYuvChannels(src);
	mat_[0] = channels[0];
	mat_[1] = channels[1];
	mat_[2] = channels[2];

	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			int srcVal = src.at<Vec3b>(i, j)[0];
			int temp = src.at<Vec3b>(i, j)[0] - balance;
			mat_[0][i][j] = src.at<Vec3b>(i, j)[0] - balance;
			//cout << mat_[0][i][j];
			mat_[1][i][j] = src.at<Vec3b>(i, j)[1] - balance;
			mat_[2][i][j] = src.at<Vec3b>(i, j)[2] - balance;
		}
	}
	vector<Mat_<int>> temp;
	temp.push_back(mat_[0]);
	temp.push_back(mat_[1]);
	temp.push_back(mat_[2]);
	return temp;
}
Mat_<int> pieces2Full(vector<Mat_<int>> pieces,int vBlocks,int hBlocks) {
	int width = hBlocks * 8;
	int height = vBlocks * 8;
	Mat_<int> fullMat(height, width);
	for (int v = 0; v < vBlocks;++v ) {
		for(int u=0;u<hBlocks;++u){

			int blocks = v * hBlocks + u;
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j){
					int x = i + v * 8;
					int y = j + u * 8;

					fullMat[x][y] = pieces[blocks][i][j];

			}
		}
	}
}
	return fullMat;
}
Mat pieces2FullT(vector<Mat> pieces, int vBlocks, int hBlocks) {
	int width = hBlocks * 8;
	int height = vBlocks * 8;
	Mat fullMat(height, width, CV_8UC1);
	for (int v = 0; v < vBlocks; ++v) {
		for (int u = 0; u < hBlocks; ++u) {

			int blocks = v * hBlocks + u;
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					int x = i + v * 8;
					int y = j + u * 8;

					fullMat.at<uchar>(x, y) = pieces[blocks].at<uchar>(i, j);

				}
			}
		}
	}
	return fullMat;
}
void piecesFindAndSet(vector<Mat>& pieces,  int greaterThanVal, int setVal) {
	int rows = pieces[0].rows;
	int cols = pieces[0].cols;
	for (int x = 0; x < rows; ++x) {
		for (int y = 0; y < cols; ++y)
		{
			if (pieces[0].at<double>(x, y) > 127 && pieces[1].at<double>(x, y) > 127 && pieces[2].at<double>(x, y) > 127) {
				pieces[0].at<double>(x, y) = pieces[1].at<double>(x, y) = pieces[2].at<double>(x, y) = 127;
			}
		}
	}
}
String verifyPieces(vector<Mat> pieces, Mat mat, int vBlocks, int hBlocks) {
	Mat pieces2FullMat = pieces2FullT(pieces, vBlocks, hBlocks);
	if (pieces2FullMat.rows != mat.rows || pieces2FullMat.cols != mat.cols)
		return "rows\\cols";
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			if (pieces2FullMat.at<uchar>(i, j) != mat.at<uchar>(i, j))
				return i+" ,"+j;
		}
	}
	return "相等";
}


void writeMat_ToFile(vector<Mat_<int>> pieces, const char* filename)
{
	std::ofstream fout(filename);

	if (!fout)
	{
		std::cout << "File Not Opened" << std::endl;
		return;
	}
	int rows = pieces[0].rows;
	int cols = pieces[0].cols;
	int blocks = pieces.size();

	for (int block = 0; block < blocks; ++block)
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				fout << (char)pieces[block][i][j];
				//fout << pieces[block][i][j] << " ";
			}
		}

	//fout << std::endl;
	fout.close();
}
void writeSting_ToFile(string string, const char* filename)
{
	std::ofstream fout(filename,ios_base::binary);

	if (!fout)
	{
		std::cout << "File Not Opened" << std::endl;
		return;
	}
	fout << string;
	fout.close();
}
string matPieces2String(vector<Mat_<int>> pieces, int vBlocks, int hBlocks) {
	string s;
	int width = hBlocks * 8;
	int height = vBlocks * 8;
	for (int v = 0; v < vBlocks; ++v) {
		for (int u = 0; u < hBlocks; ++u) {
			int blocks = v * hBlocks + u;
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
//					int pos =blocks*64+ i * hBlocks + j;
					
					s += (char)pieces[blocks][i][j];
				}
			}
		}
	}
	return s;
}

vector<Mat_<int>> getStringDataToPieces(String data,int vBlocks, int hBlocks){

	vector<Mat_<int>> pieces(vBlocks * hBlocks);
	for (int i = 0; i < vBlocks; ++i) {
		for (int j = 0; j < hBlocks; ++j) {
			int blocks = i * hBlocks + j;
			pieces[blocks].create(8, 8);
			for (int u = 0; u < 8; ++u) {
				for (int v = 0; v < 8; ++v) {
					int pos = blocks * 64 + u* 8 + v;
					pieces[blocks][u][v] = (int)data[pos];
					//cout << pieces[blocks][u][v] << endl;
				}
			}

		}
	}
	return pieces;
}
using namespace bundle;
int main() {
	String img1 = "C://Users//qq941//Pictures//imagingbook-images-en1//imgs-by-chap//ch12//alps-01.jpg";
	String img2 = "D://OneDrive - hdu.edu.cn//image//pic//GirlWithHat1.jpg";
	String img3 = "C://Users//qq941//Pictures//image//comic.jpg";
	String img4 = "C://Users//qq941//Pictures//imagingbook-images-en1//imgs-by-chap//ch14//fachwerk1-orig.jpg";
	String img5 = "C://Users//qq941//Pictures//image//comic1.jpg";
	String img6 = "C://Users//qq941//Pictures//image//p1.png";
	String img7 = "C://Users//qq941//Pictures//image//white.png";
	String img8 = "C://Users//qq941//Pictures//image//black.png";
	String img9 = "C://Users//qq941//Pictures//image//woman-in-black-hijab-taking-selfie-1.jpg";
	String img10 = "C://Users//qq941//Pictures//image//8bit.jpg";
	String img11 = "C://Users//qq941//Pictures//image//t.png";
	String img12 = "C://Users//qq941//Pictures//image//pexels-photo-3324591.jpeg";
	String img13 = "C://Users//qq941//Pictures//image//fjord.jpg";



	/******************************************************************************************************************************
																				Encode  Stage
	******************************************************************************************************************************/

	
	Mat rgb = imread(img2);
	int vBlocks = (int)ceil(rgb.rows/8);
	int hBlocks =(int) ceil(rgb.cols / 8);
	//if img's size % 8 is not 0
	//extend 
	rgb = extendImg(rgb);
	cout << "载入图片成功！" << endl;
	//Mat balancedYUVMat(rgb.size(), CV_8SC3);
	vBlocks = (int)ceil(rgb.rows / 8);
	hBlocks = (int)ceil(rgb.cols / 8);
	//convert to yuv ,done.

	cout << "平衡像素区间" << endl;
	Mat yuv;
	bgr2Yuv(rgb, yuv);

	cout << "转换RGB到YUV " << endl;
	vector<Mat_<int>> balancedYUVMat(3);
	balancedYUVMat = balanceMat_Range(yuv, 128);

	cout << "图片切割为若干8*8小块" << endl;
	//spilt balanced yuv channels
	vector<Mat_<int>> yBalancedYUVPieces = splitMatTo8s(balancedYUVMat[0]);
	vector<Mat_<int>> uBalancedYUVPieces = splitMatTo8s(balancedYUVMat[1]);
	vector<Mat_<int>> vBalancedYUVPieces = splitMatTo8s(balancedYUVMat[2]);
	
	
	cout << "DCT Stage" << endl;
	//balanced pieces DCT
	piecesDct(yBalancedYUVPieces);
	piecesDct(uBalancedYUVPieces);
	piecesDct(vBalancedYUVPieces);
	vector<Mat_<int>> yuvfull(3);
	yuvfull[0] = pieces2Full(yBalancedYUVPieces, vBlocks, hBlocks);
	yuvfull[1] = pieces2Full(uBalancedYUVPieces, vBlocks, hBlocks);
	yuvfull[2] = pieces2Full(vBalancedYUVPieces, vBlocks, hBlocks);
	Mat temp = mergeYUVChannels(yuvfull);
	imshow("temp", temp);
	waitKey();
	//0-255
	//-128-127

	cout << "DCT系数量化" << endl;
	piecesQuantity(yBalancedYUVPieces, 0);
	piecesQuantity(uBalancedYUVPieces, 1);
	piecesQuantity(vBalancedYUVPieces, 1);


	//writeMat_ToFile(yBalancedYUVPieces, "yBalancedYUVPieces.txt");
	//writeMat_ToFile(uBalancedYUVPieces, "uBalancedYUVPieces.txt");
	//writeMat_ToFile(vBalancedYUVPieces, "vBalancedYUVPieces.txt");
	//string data = matPieces2String(yBalancedYUVPieces, vBlocks, hBlocks);

	string fullStringDataY = matPieces2String(yBalancedYUVPieces, vBlocks, hBlocks);
	string fullStringDataU = matPieces2String(uBalancedYUVPieces, vBlocks, hBlocks);
	string fullStringDataV = matPieces2String(vBalancedYUVPieces, vBlocks, hBlocks);
	//writeSting_ToFile(fullStringDataY, "fullStringDataY.txt");
	//writeSting_ToFile(fullStringDataU, "fullStringDataU.txt");
	//writeSting_ToFile(fullStringDataV, "fullStringDataV.txt");

	//// 以读模式打开文件
	//std::ifstream in1("fullStringDataY.txt", ios_base::in | ios_base::binary);
	//std::istreambuf_iterator<char> begin(in1);
	//std::istreambuf_iterator<char> end;
	//std::string some_str(begin, end);
	//in1.close();

	//
	//bool isEqualful = some_str == fullStringDataY;
	//cout << "some str 与 full 关系:" << isEqualful << endl;

	//compress data

	cout << "写入文件信息" << endl;
	string full = fullStringDataY + fullStringDataU + fullStringDataV;
	string full_with_file_info = to_string(vBlocks) + " " + to_string(hBlocks) +"\n"+full;
	//preinfo += " ";
	//full = preinfo + full;
	string packedfull = pack(bundle::LZMA20, full_with_file_info);

	cout << "写入压缩数据" << endl;
	ofstream output("Full Compressed packedYUV data.dz7", ios_base::binary);
	output << packedfull;
	output.close();


	/******************************************************************************************************************************
																					Decode  Stage
	******************************************************************************************************************************/

	 
	//read compressed data to uncompress
	ifstream input("Full Compressed packedYUV data.dz7", ios_base::in | ios_base::binary);
	istreambuf_iterator<char> begindz7(input);
	istreambuf_iterator<char> enddz7;
	string whole_data(begindz7, enddz7);
	input.close();
	cout << "Check  compressed data if is equal:" << (whole_data == packedfull) << endl;

	//unpack data
	string uncompress_str = unpack(whole_data);
	cout << "Check origin full string if is equals to  umcompress full string :" << (full == uncompress_str) << endl;
	int preinfo_end = uncompress_str.find_first_of('\n');
	int yuvSize = (uncompress_str.size() - preinfo_end)/ 3;
	int  space_pos = uncompress_str.find_first_of(" ");
	string vb = uncompress_str.substr(0, space_pos);
	string hb = uncompress_str.substr(space_pos + 1, preinfo_end);
	int vblock_num = stoi(vb);
	int hblock_num = stoi(hb);
	string Y = uncompress_str.substr(preinfo_end+1, yuvSize);
	string U = uncompress_str.substr(yuvSize+preinfo_end+1, yuvSize);
	string V = uncompress_str.substr(((double)yuvSize)*2+preinfo_end+1, yuvSize);
	cout << "Check uncompressed Y if is euqal to origin:" << (Y == fullStringDataY) << endl
			<< "Check uncompressed U if is euqal to origin:" << (U == fullStringDataU) << endl 
			<< "Check uncompressed V if is euqal to origin:" << (V == fullStringDataV) << endl;





	//____________________
	yBalancedYUVPieces = getStringDataToPieces(Y, vblock_num, hblock_num);
	uBalancedYUVPieces = getStringDataToPieces(U, vblock_num, hblock_num);
	vBalancedYUVPieces = getStringDataToPieces(V, vblock_num, hblock_num);
	//------------------------------
	piecesAntiquantity(yBalancedYUVPieces, 0);
	piecesAntiquantity(uBalancedYUVPieces, 1);
	piecesAntiquantity(vBalancedYUVPieces, 1);
	
	piecesIDct(yBalancedYUVPieces);
	piecesIDct(uBalancedYUVPieces);
	piecesIDct(vBalancedYUVPieces);




	vector<Mat_<int>> balancedFullIDct(3);
	balancedFullIDct[0] = pieces2Full(yBalancedYUVPieces, vblock_num, hblock_num);
	balancedFullIDct[1] = pieces2Full(uBalancedYUVPieces, vblock_num, hblock_num);
	balancedFullIDct[2] = pieces2Full(vBalancedYUVPieces, vblock_num, hblock_num);
//	piecesFindAndSet(balancedFullIDct, 127, 127);

	Mat orginYUVMerge(rgb.size(), CV_8UC3);
	vector<Mat_<int>>unrollMat_Range= balanceMat_Range(balancedFullIDct, -128);


	Mat unrollMatYUVMerge = mergeYUVChannels(unrollMat_Range);
	//orginYUVMerge = balanceMatRange(balancedYUVMerge, -128);
	imshow("Origin YUV ", unrollMatYUVMerge);
	waitKey();
	Mat idctRGB;
	cvtColor(unrollMatYUVMerge, idctRGB, COLOR_YUV2BGR);
	imshow("compression img", idctRGB);
	waitKey();

	//imwrite("testbak.jpg", idctRGB);

	return 0;
}


