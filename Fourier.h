#pragma once
#ifndef FOURIER_H
#define FOURIER_H
#include <iostream>
#include <vector>
#include <math.h>
#include<future>

constexpr auto RED_CHANNEL = 2;
constexpr auto GREEN_CHANNEL = 1;
constexpr auto BLUE_CHANNEL = 0;
#define PI acos(-1)
using namespace std;


vector<vector<double>>  DCT(vector<vector<double>> g,vector<vector<double>>&G) {
	
	double cm = 1.0;
	double cn = 1.0;
	if (g.empty()==1)// 如果二维向量为空则返回；
		return g;
	int M = g.size(); // M rows
	int N = g[0].size();// N columns
	double const COMMON_FACTOR = 2.0 / sqrt(M * N);
	//vector<vector<double>> G(M);

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
					sum +=   g[u][v] * cm * cos(PI * (2 * u + 1) * m / (2 * M)) * cn * cos(PI * (2 * v + 1) * n / (2 * N));
				}
			}
			G[m][n] = COMMON_FACTOR * sum;
		}
	}	
	return G;
}

vector<vector<double>>  iDCT(vector<vector<double>> G, vector<vector<double>> &g) {
	if (G.empty() == true) return G;
	int M = G.size();
	int N = G[0].size();
	//vector<vector<double>> g(M);
	double const COMMON_FACTOR = 2 / sqrt(M * N);
	for (int u = 0; u < M; u++) {
		for (int v = 0; v < N; v++) {
			double sum = 0;
			for (int m = 0; m < M; m++) {
				double cm = 1;
				if (m == 0) cm = 1 / sqrt(2);
				for (int n = 0; n < N; n++) {
					double cn = 1;
					if (n == 0) cn = 1 / sqrt(2);
					sum +=G[m][n] * cm * cos(PI * (2 * u + 1) * m / (2 * M)) * cn * cos(PI * (2 * v + 1) * n / (2 * N));
				}
			}
			g[u][v] = COMMON_FACTOR * sum;
		}
	}
	return g;
}
#endif // !FOURIER_H


