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

#define Assert(x) ( (x) ? int(0) :  throw std::logic_error("broke") )


/** Implement the 'opaque pointer'
  */
struct PFTracker::PFCore
{
  /** convenience definitions */
  typedef boost::shared_ptr<PFCore> Ptr;
  typedef boost::shared_ptr<const PFCore> ConstPtr;

  PFCore(TrackingProblem::Ptr trackprob)
  {
    tracking_problem = trackprob;
  }

  std::vector<cv::Mat>  particles_estm; // estimate
  std::vector<cv::Mat>  particles_pred; // prediction

  std::vector<double>   weights;
  std::vector<double>   likelihood;

  TrackingProblem::Ptr tracking_problem;

  void updateWeights( )
  {

  }

  void resampleParticles( )
  {

  }

  void processNewData(const vector<Mat>& image_data, void* extra_data )
  {
    // register the data with TrackingProblem
    tracking_problem->registerCurrentData( image_data, extra_data);

    // compute \hat{x}_k | x_{k-1}
    tracking_problem->propagateDynamics( particles_estm, particles_pred );

    // compute p(z_k | \hat{x}_k )
    tracking_problem->evaluateLikelihood( particles_pred, likelihood);

    // compute w_k \prop w_{k-1} * p(z_k | \hat{x}_k )
    updateWeights();     // modifies weights

    // re-sample to get rid of negligible weight particles
    resampleParticles(); // modifies particles_estm

    // display callback on the TrackingProblem with current x_k
    tracking_problem->outputCallback( particles_estm, weights );
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

