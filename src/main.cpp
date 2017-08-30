#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

// #include "FGS.h"
void FGS(cv::Mat disp, int T = 3, double lambda = 900, double sigma = 0.03){

	sigma *= 255;
	int H = disp.rows;
	int W = disp.cols;

	cv::Mat u = disp;
	cv::Mat Ah = cv::Mat::zeros(W, W, CV_64FC1);
	cv::Mat Av = cv::Mat::zeros(H, H, CV_64FC1);
	for(int t = 1; t <= T; t++){
		double lambda_t = 1.5 * pow(4.0, T - t) / (pow(4.0, T) - 1) * lambda;
		
		for(int i = 0; i < H; i++){
			//a
			for(int j = 1; j < W; j++)
				Ah.at<double>(j, j-1) = -lambda_t * exp( -abs(disp.at<double>(i,j) - disp.at<double>(i,j-1)) / sigma);
			// c
			for(int j = 0; j < W-1; j++)
				Ah.at<double>(j, j+1) = -lambda_t * exp( -abs(disp.at<double>(i,j) - disp.at<double>(i,j+1)) / sigma);
			// b
			Ah.at<double>(0, 0) = 1 - Ah.at<double>(0, 1);
			for(int j = 1; j < W-1; j++)
				Ah.at<double>(j, j) = 1 - Ah.at<double>(j, j-1) - Ah.at<double>(j, j+1);
			Ah.at<double>(W-1, W-1) = 1 - Ah.at<double>(W-1, W-2);

			// calculate
			u.row(i) = (Ah.inv()*u.row(i).t()).t();
		}

		for(int j = 0; j < W; j++){
			for(int i = 1; i < H; i++)
				Av.at<double>(i, i-1) = -lambda_t * exp( -abs(disp.at<double>(i,j) - disp.at<double>(i-1,j)) / sigma);

			for(int i = 0; i < H-1; i++)
				Av.at<double>(i, i+1) = -lambda_t * exp( -abs(disp.at<double>(i,j) - disp.at<double>(i+1,j)) / sigma);

			Av.at<double>(0, 0) = 1 - Av.at<double>(0, 1);
			for(int i = 1; i < H-1; i++)
				Av.at<double>(i, i) = 1 - Av.at<double>(i, i-1) - Av.at<double>(i, i+1);

			Av.at<double>(H-1, H-1) = 1 - Av.at<double>(H-1, H-2);

			u.col(j) = Av.inv()*u.col(j);
		}
	}
}

int main(){
	cv::Mat tmp(3, 3, CV_64FC1);

	tmp.at<double>(0,0) = 1;
	tmp.at<double>(0,1) = 0;
	tmp.at<double>(0,2) = 2;
	tmp.at<double>(1,0) = -1;
	tmp.at<double>(1,1) = 5;
	tmp.at<double>(1,2) = 0;
	tmp.at<double>(2,0) = 0;
	tmp.at<double>(2,1) = 3;
	tmp.at<double>(2,2) = -9;
	cv::Mat disp = tmp.clone();

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// cv::Mat disp = cv::imread("teddyDL.png", -1); // CV_16U
	// disp.convertTo(disp, CV_64FC1);
	FGS(disp);
	std::cout << disp << std::endl;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



	// cv::imshow("disp", u);
	// cv::waitKey(0);
	return 0;
}