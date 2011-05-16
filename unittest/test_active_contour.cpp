#include "opencv_helpers.h"
#include "ktrack/ktrack.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "contour.h"

using namespace std;
using namespace cv;
using namespace ktrack;

#define Assert(x) ( (x) ? int(0) :  throw std::logic_error("broke") )

namespace {
   string window_name = "ActiveContourTracker";


}

int main( int argc, char* argv [] )
{
  boost::shared_ptr<ActiveContourSegmentor>  segmentor;
  int idx = 0;
  cv::Mat frame,segmap;
  cv::VideoCapture capture;

  for (;;)   // Main Loop
  {
    capture >> frame; // grab frame data from webcam
    if (frame.empty())
      continue;

    if( 0 == idx ) {
      frame.copyTo(segmap);
      cv::cvtColor(segmap.clone(),segmap,CV_RGB2GRAY);
      cv::circle(frame,Point( frame.cols/2,frame.rows/2),20,
                 Scalar(255,255,255),5,CV_FILLED);
      segmentor = boost::shared_ptr<ActiveContourSegmentor>(
                                  new ActiveContourSegmentor(frame,segmap));
    }

    segmentor->initializeData();
    segmentor->update();

    imshow(window_name,frame);

    char key = waitKey(10);
    if( 'q' == key ) // quick if we hit q key
      break;


  }
  return 0;
}
