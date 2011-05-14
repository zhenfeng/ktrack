#include "opencv_helpers.h"
#include "ktrack/ktrack.hpp"

using namespace std;
using namespace cv;
using namespace ktrack;

class SimpleMotionTracker : public TrackingProblem
{
  virtual void evaluateLikelihood(const std::vector<cv::Mat> &particles,
                          std::vector<double> &likelihood) {

  }

  virtual void propagateDynamics(const std::vector<cv::Mat> &particles_in,
                         const std::vector<cv::Mat> &particles_out) {

  }

  virtual size_t N() const {
    return 4; // x,y,vx,vy
  }

  virtual void getLowDimOutput(const std::vector<cv::Mat> &particles,
                               const std::vector<double> &weights,
                               std::vector<double> &output) {

  }


};

int main( int argc, char* argv [] )
{
   boost::shared_ptr<SimpleMotionTracker> prob( new SimpleMotionTracker );
   PFTracker  tracker( prob );

   cout << "successfully created a PFTracker object. " << endl;

   Options opts;
  // if (options(argc, argv, opts))
  //     return 1;
    opts.vid = 0; // assume its at /dev/video0
    VideoCapture capture(opts.vid);
    if (!capture.isOpened()) {
           cerr << "unable to open video device " << opts.vid;
           return 1;
    }
    pftracker  pft;
    Mat frame;
    vector<Mat>  image_container(3);
    void* meta_data = NULL;

    for (;;)
    {
       capture >> frame; // grab frame data from webcam
       if (frame.empty())
           continue;

       image_container[0] = frame;
       tracker.processNewData(image_container, meta_data );

       cv::Point2f pt_xy;
       cout << "tracking at:  x = " << pt_xy.x << ", y = " << pt_xy.y << endl;

       char key = cv::waitKey(10);
       if( 'q' == key ) // quick if we hit q key
          break;


     }
    return 0;
}
