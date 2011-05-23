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

double gen_rand::range = 1.0; // initialize static variable


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
    weights        = vector<double>(Nparticles,1.0/Nparticles);
    likelihood     = vector<double>(Nparticles,0.0);
    cdf            = vector<double>(Nparticles,0.0);
    sampler        = vector<double>(Nparticles,0.0);
    resample_index = vector<int>(Nparticles,0);

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
  std::vector<double>   sampler;
  std::vector<double>   likelihood;
  std::vector<int>      resample_index;

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

  static int closestIndex( const vector<double>& svec, double val)
  { // TODO: do this more smartly, e.g. binary search
    double distMin = 1e6;
    double dist    = 0.0;
    double sval    = 0.0;
    int    idxMin  = -1;
    for( int k = 0; k < (int) svec.size(); k++ ) {
      sval  = svec[k];
      dist  = std::fabs(sval - val);
      if( dist < distMin ) {
        distMin = dist;
        idxMin  = k;
      }
    }
    return idxMin;
  }

  void resampleParticles( )
  {
    int N = tracking_problem->Nparticles();
    gen_rand::random_vector(sampler,1.0);
    double wmin = 1.0/N;
    for( int k = 0; k < N; k++ )
    {
      if( weights[k] < wmin ) {
        double sampval = sampler[k];
        int    sampidx = closestIndex( cdf, sampval ); // EVIL SLOW ALGORITHM
        particles_pred[sampidx].copyTo(particles_estm[k]);
        resample_index[k] = sampidx; // where k-th was resampled from
        weights[k]        = weights[sampidx]/2.0;
        weights[sampidx]  = weights[sampidx]/2.0;
      } else {
        resample_index[k] = k;
      }
    }
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

