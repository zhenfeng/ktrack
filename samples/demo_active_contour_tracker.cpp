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
  VideoCapture capture( 0 ); // assume its at /dev/video0
  capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);
  if (!capture.isOpened()) {
    cerr << "unable to open video device /dev/video0 " << endl;
    return 1;
  }

  for (;;)   // Main Loop
  {
    capture >> frame; // grab frame data from webcam
    if (frame.empty()) {
      cout << "dropped a frame ... " << endl;
      continue;
    }

    if( 0 == idx ) {
      cv::cvtColor(0*frame,segmap,CV_RGB2GRAY);
      cv::circle(segmap,Point( frame.cols/2,frame.rows/2),20,
                 Scalar(255,255,255),50,CV_FILLED);
      segmentor = boost::shared_ptr<ActiveContourSegmentor>(
                                  new ActiveContourSegmentor(frame,segmap));
      idx += 1;
    }

    segmentor->setImage(frame);

    segmentor->update();
    segmentor->getDisplayImage(frame);
    segmentor->getPreviousPhi(segmap);
    imshow(window_name,frame);

    char key = waitKey(15);
    if( 'q' == key ) // quick if we hit q key
      break;


  }
  return 0;
}
