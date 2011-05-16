#include "contour.h"
#include <algorithm>
#include <vector>
#include <omp.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::vector;
using std::cout;
using std::endl;
using namespace cv;

namespace {
  void test_OpenMP()
  {
    int nthreads, tid;
    omp_set_num_threads(8);
    nthreads = omp_get_num_threads();
  #pragma omp parallel shared(nthreads, tid)
    { // fork some threads, each one does one expensive operation
      tid = omp_get_thread_num();
      if( tid == 0 )      { }
      else if( tid == 1 ) { }
      else if( tid == 2 ) { }
    }
  }

}

namespace  ktrack
{

void waterMark(const std::string& text, Mat & img)
{
  int baseline = 0;
  Size textSize = getTextSize(text, CV_FONT_HERSHEY_PLAIN, 3, 2, &baseline);
  Point textOrg((img.cols - textSize.width) / 2, (img.rows + textSize.height) / 2);
  putText(img, text, textOrg, CV_FONT_HERSHEY_PLAIN, 3, Scalar::all(1), 2, 8);

}



//Mat f2f = cv::estimateRigidTransform(Iprev,Icurr,1);
//f2f.at<float>(0,1) = 0.0;      f2f.at<float>(0,0) = 1.0;
//f2f.at<float>(1,0) = 0.0;      f2f.at<float>(1,1) = 1.0;
//f2f.copyTo(affineFrame2Frame);
//pixel_velocity_x = f2f.at<double>(0,2);
//pixel_velocity_y = f2f.at<double>(1,2);

//bool useWarpedInDifference = true;
//if( useWarpedInDifference ) { // can be unstable at edges
//  warpAffine(Iprev_smooth,Iprev_smooth_warped,f2f,Iprev.size());



} // end namespace


