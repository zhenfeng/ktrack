
#include "opencv_helpers.h"
#include "ktrack/ktrack.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "file_io.h"

using namespace std;
using namespace cv;
using namespace ktrack;

#define Assert(x) ( (x) ? int(0) :  throw std::logic_error("broke") )

namespace {
string window_name = "SimpleMotionTracker";


}

/**
  * FOR TRACKING SMALL BLOBS IN PRESENCE OF CAMERA MOTION
  *
  */
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
      likelihood[k] = 1e-3 + (dIdt_grey.at<unsigned char>(y,x) * 1.0 ) /
                             Icurr_grey.at<unsigned char>(y,x);
    }
  }

  virtual void propagateDynamics(const vector<Mat> &particles_in,
                                 vector<Mat> &particles_out)
  {
    // \hat{x}_k = F( x_{k-1}, z_k )
    int N = Nparticles();
    double DiffusionRange = 1.0;
    double dmin,dmax;
    Point pmin,pmax;
    cv::minMaxLoc(dIdt_grey,&dmin,&dmax,&pmin,&pmax);

    if( weights.empty() ) { // initialize to center
      weights = std::vector<double>(N,1.0/N);
      rand_x  = std::vector<double>(N,0);
      rand_y  = std::vector<double>(N,0);
      for( int k = 0; k < N; k++ ) {
        particles_out[k].at<double>(0) = Icurr.cols/2.0;
        particles_out[k].at<double>(1) = Icurr.rows/2.0;
      }
    } else {
      gen_rand::random_vector(rand_x,DiffusionRange);
      gen_rand::random_vector(rand_y,DiffusionRange);
      for( int k = 0; k < N; k++ ) {

        double newx = particles_in[k].at<double>(0);
        double newy = particles_in[k].at<double>(1);
        newx       += pixel_velocity_x + 2*rand_x[k] - DiffusionRange;
        newy       += pixel_velocity_y + 2*rand_y[k] - DiffusionRange;

        // using data inside the prior
        newx       += 1e-3 * (Icurr.cols/2  - newx );
        newy       += 1e-3 * (Icurr.rows/2  - newy );
        if( rand()%7 == 1  &&  weights[k] < (1.0/N) ) {
          newx       += (pmax.x  - newx );
          newy       += (pmax.y  - newy );
        }
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
    GaussianBlur(Icurr,Icurr_smooth,Size(7,7),2.0,2.0);

    if( Iprev.empty() ) {
      Icurr.copyTo(Iprev);
      Icurr_smooth.copyTo(Iprev_smooth);
      dIdt = 0 * Icurr;
    }
    cv::cvtColor( Icurr_smooth, Icurr_grey, CV_RGB2GRAY );
    Mat f2f = cv::estimateRigidTransform(Iprev,Icurr,1);
    f2f.at<float>(0,1) = 0.0;      f2f.at<float>(0,0) = 1.0;
    f2f.at<float>(1,0) = 0.0;      f2f.at<float>(1,1) = 1.0;
    f2f.copyTo(affineFrame2Frame);
    pixel_velocity_x = f2f.at<double>(0,2);
    pixel_velocity_y = f2f.at<double>(1,2);

    bool useWarpedInDifference = true;
    if( useWarpedInDifference ) { // can be unstable at edges
      warpAffine(Iprev_smooth,Iprev_smooth_warped,f2f,Iprev.size());
      absdiff(Icurr_smooth,Iprev_smooth_warped,dIdt);
    } else {
      absdiff(Icurr_smooth,Iprev_smooth,dIdt);
    }
    cv::cvtColor( dIdt, dIdt_grey, CV_RGB2GRAY );
    applyGaussianWeight(dIdt_grey);

    bool printF2F = true;
    if( printF2F ) {
      cout << "rigidFrame2Frame vx,vy= " << pixel_velocity_x << ", "
           << pixel_velocity_y << endl;
    }
  }

  virtual size_t Nstates() const {
    return 2; // x,y,vx,vy
  }

  virtual size_t Nparticles() const {
    return 100; // x,y,vx,vy
  }

  virtual void outputCallback(const vector<Mat> &particles,
                              const vector<double> &weights)
  {
    // Assign the "low dimensional outputs"
    track_box.width  = 20;
    track_box.height = 20;
    track_box.x      = Icurr.cols / 2 - track_box.width/2;
    track_box.y      = Icurr.rows / 2 - track_box.height/2;

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
    outimg = Icurr;
    //rectangle(outimg, track_box, Scalar(255,255,255),3,CV_AA);
    int N = Nparticles();
    double dmin, dmax;
    cv::minMaxLoc( Mat(weights), &dmin, &dmax);
    for( int k = 0; k < N; k++ ) {
      int x    = (int) particles[k].at<double>(0);
      int y    = (int) particles[k].at<double>(1);
      Point2f pix_point;
      pix_point.x = x;
      pix_point.y = y;
      circle(outimg, pix_point,2,Scalar(55,100,255),1,CV_AA);
    }
  }

  void getDiffImg( Mat& outimg ) {
    dIdt_grey.copyTo(outimg);
  }

  void applyGaussianWeight( Mat& img_grey ) {

    for( int i = 0; i < img_grey.rows; i++ ) {
      for( int j = 0; j < img_grey.cols; j++ ) {
        double val = exp(-(pow( ( i * 1.2 )/img_grey.rows - 0.6, 6.0 ) +
                          pow( ( j * 1.2 )/img_grey.cols - 0.6, 6.0 ))/0.05 );
        img_grey.at<unsigned char>(i,j) *= val;
      }
    }

  }

  void setupTrackingProblem( ) {

    // put any initializations here

  }

  // **** Data specific to this incarnation of TrackingProblem

private:
  Mat dIdt;
  Mat dIdt_grey;
  Mat dIdt_kernel;
  Mat Icurr;
  Mat Icurr_grey;
  Mat Iprev;
  Mat Iprev_smooth;
  Mat Iprev_smooth_warped;
  Mat Icurr_smooth;

  Mat affineFrame2Frame;
  vector<double> rand_x;
  vector<double> rand_y;
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
  cout << "attempting to read jpeg images from  argv[1] " << endl;
  cout << argv[1] << endl;

  string file_ext = ".jpg";
  if( argc > 2 ) {
    file_ext = argv[2];
  }
  cout << "looking for files of extension " << file_ext << endl;
  vector<string> image_files;
  getFilePrefixes(argv[1],file_ext,image_files);
  bool bWriteImages = false; // get it from arg!

  namedWindow(window_name);

  Mat frame;
  vector<Mat>  image_container(3);
  void* meta_data = NULL;

  for (int k = 0; k < (int) image_files.size(); k++ )   // Main Loop
  {
    string filename = argv[1] + string("/") + image_files[k] + file_ext;
    //cout << "loading image: " << filename << endl;
    frame = imread( filename );
    if (frame.empty())
      continue;

    image_container[0] = frame;

    pft.processNewData(image_container, meta_data );
    prob->drawTrackBox(image_container[0]);
    prob->getDiffImg(image_container[1]);
    imshow(window_name,image_container[0]);
    imshow(window_name + "diff",image_container[1]);

    if( bWriteImages ) {
      imwrite(filename + ".out.png", image_container[0] );
    }

    char key = waitKey(15);
    if( 'q' == key ) // quick if we hit q key
      break;


  }
  return 0;
}
