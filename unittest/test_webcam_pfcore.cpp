#include "opencv_helpers.h"
#include "ktrack/ktrack.hpp"

using namespace std;
using namespace cv;
using namespace ktrack;

namespace {
string window_name = "SimpleMotionTracker";
}

class SimpleMotionTracker : public TrackingProblem
{
  virtual void evaluateLikelihood(const vector<cv::Mat> &particles,
                                  vector<double> &likelihood)
  {
    // p(z_k | \hat{x}_k )
    // good particles => measurement is solid
  }

  virtual void propagateDynamics(const vector<cv::Mat> &particles_in,
                                 vector<cv::Mat> &particles_out)
  {

  }

  virtual void registerCurrentData(const std::vector<cv::Mat> &images,
                                   void *extra_data)
  {
    image_data = images;

    image_data[0].copyTo(Icurr);
    cv::GaussianBlur(Icurr,Icurr_smooth,Size(5,5),2.0,2.0);

    if( Iprev.empty() ) {
      dIdt = 0 * Icurr;
    } else {
      cv::absdiff(Icurr_smooth,Iprev_smooth,dIdt);
    }
  }

  virtual size_t N() const {
    return 4; // x,y,vx,vy
  }

  virtual void outputCallback(const vector<cv::Mat> &particles,
                              const vector<double> &weights)
  {
    // Assign the "low dimensional outputs"

    track_box.x      = Icurr.cols / 2;
    track_box.y      = Icurr.rows / 2;
    track_box.width  = 20;
    track_box.height = 20;

    Icurr.copyTo(Iprev);
    Icurr_smooth.copyTo(Iprev_smooth);
  }

  // **** Functions specific to this incarnation of TrackingProblem
public:
  /** draw the track_box on top of an image, for display */
  void drawTrackBox( cv::Mat& outimg ) {
    dIdt.copyTo(outimg);

    cv::rectangle(outimg, track_box, Scalar(255,255,255),3,CV_AA);
  }

  // **** Data specific to this incarnation of TrackingProblem

private:
  Mat dIdt;
  Mat Icurr;
  Mat Iprev;
  Mat Iprev_smooth;
  Mat Icurr_smooth;

  cv::Rect  track_box;

};

int main( int argc, char* argv [] )
{
  boost::shared_ptr<SimpleMotionTracker> prob( new SimpleMotionTracker );
  PFTracker  pft( prob );

  cv::namedWindow(window_name);
  cout << "successfully created a PFTracker object. " << endl;

  Options opts;
  // if (options(argc, argv, opts))
  //     return 1;
  opts.vid = 0; // assume its at /dev/video0
  VideoCapture capture(opts.vid);
  capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);
  if (!capture.isOpened()) {
    cerr << "unable to open video device " << opts.vid;
    return 1;
  }

  Mat frame;
  vector<Mat>  image_container(3);
  void* meta_data = NULL;

  for (;;)
  {
    capture >> frame; // grab frame data from webcam
    if (frame.empty())
      continue;

    image_container[0] = frame;
    image_container[1] = frame;
    pft.processNewData(image_container, meta_data );

    prob->drawTrackBox(image_container[1]);
    cv::imshow(window_name,image_container[1]);

    //cv::Point2f pt_xy;
    //cout << "tracking at:  x = " << pt_xy.x << ", y = " << pt_xy.y << endl;

    char key = cv::waitKey(10);
    if( 'q' == key ) // quick if we hit q key
      break;


  }
  return 0;
}
