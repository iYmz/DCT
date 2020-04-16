
#include <iostream>

using namespace std;


int example()
{

	//Filters filter;
	//filter.Average_8DGrey_3x3("C://Users//qq941//Pictures//imagingbook-images-en1//imgs-by-chap//ch02//airfield-05small-auto.tif");
	/*
	Mat rgbMat = LoadImageAndShow("RGB IMAGE", "C://Users//qq941//Pictures//color//azuref0ffff.png");
	int *rgb = GetRGB(rgbMat, 2, 2);
	cout << endl << "R:" << rgb[2] << " G:" << rgb[1] << " B:" << rgb[0];
	int color = ComposeRGB(rgb[0], rgb[1], rgb[2]);
	cout << endl<<"color:	" << color;
	rgb = GetRGB(color);
	cout << endl << "R:" << rgb[2] << " G:" << rgb[1] << " B:" << rgb[0];

	
	String imgPath = "C://Users//qq941//Pictures//imagingbook-images-en1//imgs-by-chap//ch02//airfield-05small-auto.tif";
	Mat  greyImage = LoadImageAndShow("GreyImage", imgPath, 0);
	int M = 8;
	int N = 8;
	int rows = greyImage.rows;
	int cols = greyImage.cols;

	//resize(src, src, Size(512, 512));
	greyImage.convertTo(greyImage, CV_32F);

	vector<vector<double>> gfull(rows, vector<double>(cols));


	for (int x = 0; x < rows / 8; x++) {
		for (int y = 0; y < cols / 8; y++) {
			vector<vector<double>> g(M, vector<double>(N));
			for (int m = 0; m < M; m++) {
				for (int n = 0; n < N; n++) {
					g[m][n] = (double)greyImage.at<float>(m + x * M, n + y * N);
				}
			}


			vector<vector<double>> G(M, vector<double>(N));
			G = DCT(g);


			vector<vector<double>> g1(M, vector<double>(N));
			g1 = iDCT(G);


			//copy.data[i * cols + j] = pixels[i][j] = (int)image.ptr<uchar>(i)[j];
			for (int i = 0; i < M; i++) {
				for (int j = 0; j < N; j++) {
					gfull[i + x * 8][j + y * 8] = g1[i][j];

				}
			}

		}
	}

	Mat copy = Mat(rows, cols, CV_8UC1);
	for (int x = 0; x < rows; x++) {
		for (int y = 0; y < cols; y++) {
			copy.data[x * cols + y] = gfull[x][y];
		}
	}
	imshow("IDCT img", copy);
	waitKey();




	vector<vector<double>> v(4, vector<double>(4));
	vector<vector<double>> v1(4, vector<double>(4));
	v = { {42,66,68,66},{92,4,76,17},{79,85,74,71},{96,93,39,3} };
	v1 = { { 61,19,50,20 }, { 82,26,61,45 }, { 89,90,82,43 }, { 93,59,53,97 } };
	cout << "Origin";

	cout << endl << endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << v1[i][j] << "  ";
			if (j == 3)cout << endl;
		}
	}
	cout << endl << endl;
	cout << "DCT";

	cout << endl;
	vector<vector<double>> vDct(4, vector<double>(4));
	vDct = DCT(v1);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << vDct[i][j] << "  ";
			if (j == 3)cout << endl;
		}
	}
	cout << endl; cout << endl;
	cout << "IDCT" << endl;
	v1 = iDCT(vDct);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << v1[i][j] << "  ";
			if (j == 3)cout << endl;
		}
	}
	*/




	return 0;

}