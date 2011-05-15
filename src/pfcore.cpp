#include "ktrack/ktrack.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <algorithm>
#include <list>

using std::vector;
using std::cout;
using std::endl;
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

  PFCore(TrackingProblem::Ptr trackprob) : tracking_problem(trackprob)
  {
  }

  void initPFCore()
  {
    cout << "initializing PFCore ... " ;
    Assert(tracking_problem);

    int Nstates    = tracking_problem->Nstates();
    int Nparticles = tracking_problem->Nparticles();

    particles_estm = vector<Mat>( Nparticles );
    particles_pred = vector<Mat>( Nparticles );
    weights        = vector<double>(Nparticles,0.0);
    likelihood     = vector<double>(Nparticles,0.0);
    cdf            = vector<double>(Nparticles,0.0);

    for( int k = 0; k < Nparticles; k++ ) {
      particles_estm[k] = Mat::zeros(Nstates,1,CV_64F);
      particles_pred[k] = Mat::zeros(Nstates,1,CV_64F);
    }
    cout << "Nstates="<<Nstates<<", Nparticles="<<Nparticles<<endl;
  }

  std::vector<cv::Mat>  particles_estm; // estimate
  std::vector<cv::Mat>  particles_pred; // prediction
  std::vector<double>   weights;
  std::vector<double>   cdf;
  std::vector<double>   likelihood;

  TrackingProblem::Ptr tracking_problem;

  void updateWeights( )
  {
    int N = tracking_problem->Nparticles();
    double wsum = 1e-9;
    for( int k = 0; k < N; k++ ) {
      weights[k] = weights[k] * likelihood[k];
      wsum      += weights[k];
    }
    for( int k = 0; k < N; k++ ) {
      weights[k] /= wsum;
      int kprev = (k > 0 ) ? (k-1) : 0;
      cdf[k]    = cdf[kprev] + weights[k];
    }

  }

  void resampleParticles( )
  {
    int N = tracking_problem->Nparticles();
    double wmin = 1.0;
    double wmax = 0.0;
    int idx_max = 0;
    int idx_min = 0;




    for( int k = 0; k < N; k++ )
    { // TODO: resample properly
      particles_pred[k].copyTo(particles_estm[k]);
      if( weights[k] > wmax ) {
        wmax = weights[k];
        idx_max = k;
      } else if ( weights[k] < wmin ) {
        wmin = weights[k];
        idx_min = k;
      }
    }
    particles_estm[idx_max].copyTo(particles_estm[idx_min]);
  }

  void processNewData(const vector<Mat>& image_data, void* extra_data )
  {
    if( particles_estm.empty() ) {
      initPFCore();
    }

    // register the data with TrackingProblem
    tracking_problem->registerCurrentData( image_data, extra_data);

    // compute \hat{x}_k | x_{k-1}
    tracking_problem->propagateDynamics( particles_estm, particles_pred );

    // compute p(z_k | \hat{x}_k )
    tracking_problem->evaluateLikelihood( particles_pred, likelihood);

    // compute w_k \prop w_{k-1} * p(z_k | \hat{x}_k )
    updateWeights();     // modifies weights

    // re-sample to get rid of negligible weight particles
    resampleParticles(); // particles_pred -> particles_estm

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

