#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>

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

	// cv::Mat tmprow  = cv::Mat::zeros(1,3,CV_64FC1);
	// tmprow.copyTo(tmp.row(1));

// 	std::cout << tmp.reshape(0, 1) << std::endl;
// 	cv::Mat tmp_transpose;
// 	cv::transpose(tmp.reshape(0, 1), tmp_transpose);
// 	std::cout << tmp_transpose << std::endl;

	// cv::Mat t;
	// tmp.row(1).copyTo(t);
	// std::cout << tmp << std::endl;
	// std::cout << t << std::endl;
	// t.at<double>(0,0) = 100000;
	// std::cout << t << std::endl;
	// std::cout << tmp << std::endl;
	// std::cout << tmp.reshape(0, 9) << std::endl;
	// std::cout << tmp.reshape(0, 3) << std::endl;
	// cv::Mat tmp_inv;
	// tmp.copyTo(tmp_inv);
	// tmp_inv.at<double>(0,0) = 0;

	// std::cout << tmp << std::endl;
	// std::cout << tmp_inv << std::endl;

	// std::cout << cv::invert(tmp, tmp_inv) << std::endl;
	// std::cout << tmp_inv << std::endl;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat disp = tmp.clone();
	// disp.at<double>(0,0) = 100;
	// std::cout << tmp << std::endl;
	// cv::Mat disp = cv::imread("teddyDL.png", -1);
	// std::cout << disp.type() << std::endl; // CV_16U
	disp.convertTo(disp, CV_64FC1);

	// cv::Mat guide;
	// disp.copyTo(guide);

	// disp.convertTo(disp, CV_16UC1);

	int T = 3;
	int H = disp.rows;
	int W = disp.cols;
	int S = H*W;
	double lambda = 900;
	double sigma = 1 * 255;

	cv::Mat u = disp;
	cv::Mat Ah = cv::Mat::zeros(W, W, CV_64FC1);
	cv::Mat Av = cv::Mat::zeros(H, H, CV_64FC1);
	for(int t = 1; t <= T; t++){
		double lambda_t = 1.5 * pow(4.0, T - t) / (pow(4.0, T) - 1) * lambda;
		std::cout << "iteration " << t-1 << std::endl;
		
		for(int i = 0; i < H; i++){
			
			// cv::Mat fh = u.row(i).clone().t();

			//a
			for(int j = 1; j < W; j++){
				Ah.at<double>(j, j-1) = -lambda_t * exp( -abs(disp.at<double>(i,j) - disp.at<double>(i,j-1)) / sigma);
				// std::cout << "a: " << Ah.at<double>(j, j-1) << std::endl;
			}
			// c
			for(int j = 0; j < W-1; j++){
				Ah.at<double>(j, j+1) = -lambda_t * exp( -abs(disp.at<double>(i,j) - disp.at<double>(i,j+1)) / sigma);
				// std::cout << "c: " << Ah.at<double>(j, j+1) << std::endl;
			}

			// b
			Ah.at<double>(0, 0) = 1 - Ah.at<double>(0, 1);
			for(int j = 1; j < W-1; j++){
				Ah.at<double>(j, j) = 1 - Ah.at<double>(j, j-1) - Ah.at<double>(j, j+1);
				// std::cout << "a: " << Ah.at<double>(j, j-1) << std::endl;
				// std::cout << "c: " << Ah.at<double>(j, j+1) << std::endl;
				// std::cout << "b: " << Ah.at<double>(j, j) << std::endl;
			}
			Ah.at<double>(W-1, W-1) = 1 - Ah.at<double>(W-1, W-2);

			// cv::Mat r = (Ah.inv()*fh).t();
			// r.copyTo(u.row(i));
			u.row(i) = (Ah.inv()*u.row(i).t()).t();
			

		}

		for(int j = 0; j < W; j++){
			// cv::Mat fv = u.col(j).clone();
			for(int i = 1; i < H; i++){
				Av.at<double>(i, i-1) = -lambda_t * exp( -abs(disp.at<double>(i,j) - disp.at<double>(i-1,j)) / sigma);
			}
			for(int i = 0; i < H-1; i++){
				Av.at<double>(i, i+1) = -lambda_t * exp( -abs(disp.at<double>(i,j) - disp.at<double>(i+1,j)) / sigma);
			}
			Av.at<double>(0, 0) = 1 - Av.at<double>(0, 1);
			for(int i = 1; i < H-1; i++){
				Av.at<double>(i, i) = 1 - Av.at<double>(i, i-1) - Av.at<double>(i, i+1);
			}
			Av.at<double>(H-1, H-1) = 1 - Av.at<double>(H-1, H-2);

			// cv::Mat c = (Av.inv()*fv);
			// c.copyTo(u.col(j));
			u.col(j) = Av.inv()*u.col(j);
		}
		
	}
	// u.convertTo(u, CV_8UC1);
	std::cout << u << std::endl;
	// cv::Mat disp_vis;
	// int vis_mult = 1;
 //    cv::ximgproc::getDisparityVis(u, disp_vis, vis_mult);
 //    cv::namedWindow("raw disparity", cv::WINDOW_AUTOSIZE);
 //    cv::imshow("raw disparity", disp_vis);
 //    cv::waitKey(0);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



	// cv::imshow("disp", u);
	// cv::waitKey(0);
	return 0;
}