#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <fstream>
#include <iomanip>

#include "opencv_helpers.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include <ctype.h>
#include <vector>
#include <iostream>

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

namespace {
Mat temp_rgb2gray;
}

namespace ktrack  {



unsigned char*  mat2raw( const Mat&  frame )
{
  //*ProcessFrameY8( short Width, short Height, unsigned char *VideoData )
  bool    isRGB   = (frame.channels() == 3);
  long int nBytes = frame.cols * frame.rows * (1+2*isRGB);
  unsigned char*  VideoData = (unsigned char*) malloc( nBytes );

  if( isRGB ) { // !!! Check logic here...
    cv::cvtColor(frame,temp_rgb2gray,CV_RGB2GRAY);
  }  else {
    temp_rgb2gray = frame;
  }

  memcpy( VideoData, temp_rgb2gray.ptr<unsigned char>(0), nBytes );
  return VideoData;
}

void raw2mat( unsigned char*  VideoData, Mat& frame)
{
  int     nchans  = frame.channels();
  bool    isRGB   = (nchans == 3);
  long int nBytes = frame.cols * frame.rows * (1+2*isRGB);

  assert( nBytes > 16 );
  memcpy( frame.ptr<unsigned char>(0), VideoData, nBytes );

}


}
