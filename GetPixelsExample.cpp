#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


void Example(){
cv::Mat img = cv::imread("lenna.png");
for (int j = 0; j < img.rows; j++)
{
    for (int i = 0; i < img.cols; i++)
    {
        std::cout << "Matrix of image loaded is: " << img.at<uchar>(i, j);
    }
}
}