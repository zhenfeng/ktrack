#include "opencv_helpers.h"
#include "ktrack/ktrack.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace ktrack;

#define Assert(x) ( (x) ? int(0) :  throw std::logic_error("broke") )

namespace {
   string window_name = "SimpleMotionTracker";


}

class SimpleMotionTracker : public TrackingProblem
{
  virtual void evaluateLikelihood(const vector<Mat> &particles,
                                  vector<double> &likelihood)
  {
    // p(z_k | \hat{x}_k )
    // good particles => measurement is solid
    int N = Nparticles();
    for( int k = 0; k < N; k++ ) {
      int x = (int) particles[k].at<double>(0);
      int y = (int) particles[k].at<double>(1);
      likelihood[k] = 0.05 + dIdt_grey.at<unsigned char>(y,x) * 1.0;
    }
  }

  virtual void propagateDynamics(const vector<Mat> &particles_in,
                                 vector<Mat> &particles_out)
  {
    // \hat{x}_k = F( x_{k-1}, z_k )
    int N = Nparticles();
    double DiffusionRange = 1.0;
    if( weights.empty() ) { // initialize to center
      for( int k = 0; k < N; k++ ) {
        particles_out[k].at<double>(0) = Icurr.cols/2.0;
        particles_out[k].at<double>(1) = Icurr.rows/2.0;
      }
    } else {

      for( int k = 0; k < N; k++ ) {
        diffusion.at<double>(0) = DiffusionRange * (-1.0 + 2.0 * (rand()%107)/107.0 );
        diffusion.at<double>(1) = DiffusionRange * (-1.0 + 2.0 * (rand()%107)/107.0 );
        double newx = particles_in[k].at<double>(0);
        double newy = particles_in[k].at<double>(1);
        newx       += pixel_velocity_x + diffusion.at<double>(0);
        newy       += pixel_velocity_y + diffusion.at<double>(1);
        newx       += 5e-3 * (Icurr.cols/2  - newx );
        newy       += 5e-3 * (Icurr.rows/2  - newy );
        newx        = std::max(0.0,newx);
        newy        = std::max(0.0,newy);
        newy        = std::min(newy,Icurr.rows-1.0);
        newx        = std::min(newx,Icurr.cols-1.0);
        particles_out[k].at<double>(0) = newx;
        particles_out[k].at<double>(1) = newy;
      }
    }

  }

  virtual void registerCurrentData(const vector<Mat> &images,
                                   void *extra_data)
  {
    images[0].copyTo(Icurr);
    GaussianBlur(Icurr,Icurr_smooth,Size(5,5),2.0,2.0);

    if( Iprev.empty() ) {
      Icurr.copyTo(Iprev);
      dIdt = 0 * Icurr;
    } else {
      absdiff(Icurr_smooth,Iprev_smooth,dIdt);
    }
    cv::cvtColor( dIdt, dIdt_grey, CV_RGB2GRAY );
    Mat f2f = cv::estimateRigidTransform(Iprev,Icurr,1);
    f2f.copyTo(affineFrame2Frame);
    pixel_velocity_x = f2f.at<double>(0,2);
    pixel_velocity_y = f2f.at<double>(1,2);

    bool printF2F = false;
    if( printF2F ) {
      cout << "rigidFrame2Frame = " << f2f << endl;
    }
  }

  virtual size_t Nstates() const {
    return 2; // x,y,vx,vy
  }

  virtual size_t Nparticles() const {
    return 10; // x,y,vx,vy
  }

  virtual void outputCallback(const vector<Mat> &particles,
                              const vector<double> &weights)
  {
    // Assign the "low dimensional outputs"

    track_box.x      = Icurr.cols / 2;
    track_box.y      = Icurr.rows / 2;
    track_box.width  = 20;
    track_box.height = 20;

    Icurr.copyTo(Iprev);
    Icurr_smooth.copyTo(Iprev_smooth);

    this->particles = particles;
    this->weights   = weights;
  }

  // **** Functions specific to this incarnation of TrackingProblem  ***
  // *******************************************************************

public:

  /** draw the track_box on top of an image, for display */
  void drawTrackBox( Mat& outimg ) {
    Icurr.copyTo(outimg);
    rectangle(outimg, track_box, Scalar(255,255,255),3,CV_AA);
    int N = Nparticles();
    double dmin, dmax;
    cv::minMaxLoc( Mat(weights), &dmin, &dmax);
    for( int k = 0; k < N; k++ ) {
      int x    = (int) particles[k].at<double>(0);
      int y    = (int) particles[k].at<double>(1);
      int rval = (int) (sqrt(weights[k] / dmax) * 255.0);
      circle(outimg, Point2f(x,y),3,Scalar(rval,0,0),1,CV_AA);
    }
  }

  void setupTrackingProblem( ) {
    diffusion = Mat::zeros(Nstates(),1,CV_64F);


    Assert( 2 == Nstates() );
  }

  // **** Data specific to this incarnation of TrackingProblem

private:
  Mat dIdt;
  Mat dIdt_grey;
  Mat Icurr;
  Mat Iprev;
  Mat Iprev_smooth;
  Mat Icurr_smooth;

  Mat affineFrame2Frame;
  Mat diffusion;
  vector<Mat> particles;
  vector<double> weights;

  double pixel_velocity_x;
  double pixel_velocity_y;

  Rect  track_box;

};

int main( int argc, char* argv [] )
{
  boost::shared_ptr<SimpleMotionTracker> prob( new SimpleMotionTracker );
  prob->setupTrackingProblem();
  PFTracker  pft( prob );
  cout << "successfully created a PFTracker object. " << endl;

  namedWindow(window_name);
  VideoCapture capture( 0 ); // assume its at /dev/video0
  capture.set(CV_CAP_PROP_FRAME_WIDTH,320);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
  if (!capture.isOpened()) {
    cerr << "unable to open video device /dev/video0 " << endl;
    return 1;
  }

  Mat frame;
  vector<Mat>  image_container(3);
  void* meta_data = NULL;



  for (;;)   // Main Loop
  {
    capture >> frame; // grab frame data from webcam
    if (frame.empty())
      continue;

    image_container[0] = frame;
    image_container[1] = frame;
    pft.processNewData(image_container, meta_data );

    prob->drawTrackBox(image_container[1]);
    imshow(window_name,image_container[1]);

    //Point2f pt_xy;
    //cout << "tracking at:  x = " << pt_xy.x << ", y = " << pt_xy.y << endl;

    char key = waitKey(10);
    if( 'q' == key ) // quick if we hit q key
      break;


  }
  return 0;
}
