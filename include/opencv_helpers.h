#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

namespace ktrack {

using std::cout;
using std::endl;

struct Options {
  int vid;
};

unsigned char*  mat2raw( const cv::Mat&  frame );

void raw2mat( const unsigned char*  VideoData, cv::Mat& frame);

void display_trackpoint( const cv::Mat& frame, const cv::Point2f& pt_xy );

void frame2frame_feature_based(const cv::Mat &im1, const cv::Mat &im2,
                               cv::Mat& warped, cv::Mat& H, bool displayMatches);

int blobtrack_frame( const cv::Mat& frame, cv::Point2f& track_point);

void setup_blob_tracker();


}
