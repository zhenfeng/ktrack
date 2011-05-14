#include "ktrack/ktrack.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <algorithm>

using std::vector;
using namespace cv;

namespace ktrack
{

#define Assert(x) ( (x) ? int(0) :  throw std::logic_error("bad") )

struct PFTracker::PFCore
{
  /** convenience definitions */
  typedef boost::shared_ptr<PFCore> Ptr;
  typedef boost::shared_ptr<const PFCore> ConstPtr;

  PFCore(TrackingProblem::Ptr trackprob)
  {
    tracking_problem = trackprob;
  }

  std::vector<cv::Mat>  particles;

  std::vector<double>   weights;

  std::vector<double>   likelihood;

  /** keep a handle on the user-defined problem */
  TrackingProblem::Ptr tracking_problem;


  void processNewData(const vector<Mat>& image_data, void* extra_data )
  {
    // register the data with TrackingProblem
    tracking_problem->registerCurrentData( image_data, extra_data);

    tracking_problem->evaluateLikelihood( particles, likelihood);

  }
};


PFTracker::PFTracker(TrackingProblem::Ptr trackprob) :
                                 pf_core( new PFCore(trackprob) )
{
}


void PFTracker::processNewData( const vector<Mat>& image_data, void* extra_data )
{

  pf_core->processNewData( image_data, extra_data );

}



} // end namespace


#if 0    //  Kolyma

// use this to grab a pointer to the first element in a vector
// #define Vp(x) (&(x[0]))
// reinterpret void* to an NLOptCore_impl
// #define Thiz(x) ( static_cast<NLOptCore_impl*> (x))


#endif

