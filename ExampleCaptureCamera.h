#pragma once
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
class ExampleCaptureCamera
{
public:
    ExampleCaptureCamera();
    ~ExampleCaptureCamera();
    int Example1(int, char**);
    int CaptureCamera();
private:

};

ExampleCaptureCamera::ExampleCaptureCamera()
{
}

ExampleCaptureCamera::~ExampleCaptureCamera()
{
}

//opening camrea and capture scenes 
//convert camera scenes to 2 depth grey scenes
int ExampleCaptureCamera::  Example1(int, char**)
{

    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    Mat frame, edges;
    namedWindow("edges", 1);
    for (;;)
    {
        cap >> frame;
        cvtColor(frame, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
        Canny(edges, edges, 0, 30, 3);
        imshow("edges", edges);
        if (waitKey(30) >= 0) break;
    }
    return 0;
}

//Simple example of opening camrea and capture scenes
int ExampleCaptureCamera::CaptureCamera() {
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    Mat frame;
    namedWindow("Capture Camera", 1);
    for (;;) {
        cap >> frame;
        imshow("Capture Camera", frame);
        if (waitKey(30) >= 0)break;
    }
    return 0;
}