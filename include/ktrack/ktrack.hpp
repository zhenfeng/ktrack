/**
- define the interfaces
- "clients" must implement them
- it just happens that a few particular clients are
  must-haves so we'll make them in the library

*/

#ifndef KTRACK_HPP
#define KTRACK_HPP

#include <vector>
#include <stdexcept>
#include <new>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

namespace ktrack
{


/** \abstract, \interface defines the problem-specific stuff
   clients must implement evaluateLikelihood, propagateDynamics, N,
   outputCallback, and must perform any GUI / display / output. */
class TrackingProblem
{
public:
  /** virtual destructor for inheritance. */
  virtual ~TrackingProblem()
  {
  }

  //*************************************
  // required interfaces
  //*************************************

  /** evaluate likelihood p(z_k | \hat{x}_k ), 'score'
      assigned to the newly sampled particles */
  virtual void evaluateLikelihood( const std::vector<cv::Mat>& particles,
                                   std::vector<double>& likelihood ) = 0;


  /** propagate dynamics, sample from the proposal density.
      the client implementing this should utilize the registered data
      if proposal density does not depend solely on the previous state.*/
  virtual void sampleNewParticles( const std::vector<cv::Mat>& particles_in,
                                        std::vector<cv::Mat>& particles_out ) = 0;

  /** Number of states in particles.
   * This is called at registration time and must not change! */
  virtual size_t Nstates() const = 0;

  /** Number of particles in filter.
   * This is called at registration time and must not change! */
  virtual size_t Nparticles() const = 0;

  /** application-specific processing of the distributed particles.
      e.g. taking the mean or k-means of coordinates
      */
  virtual void outputCallback( const std::vector<cv::Mat>& particles,
                               const std::vector<double>& weights ) = 0;


  //*************************************
  // optional interfaces
  //*************************************

  virtual void registerCurrentData( const std::vector<cv::Mat>& images,
                                    void* extra_data = 0 ) {
    image_data = images;
  }


  typedef boost::shared_ptr<TrackingProblem> Ptr; //!< Convenience boost pointer type
  typedef boost::shared_ptr<const TrackingProblem> ConstPtr; //!< Convenience const boost pointer type

protected:
  std::vector<cv::Mat> image_data;
};





/** concrete, base-class member shared pointer
 *  to TrackingProblem and PFCore
 *  handles the IO, "on new data received"
 */
class PFTracker
{

public:
  /** Setup: pass a pointer to derived class */
  PFTracker( TrackingProblem::Ptr trackprob );

  void processNewData( const std::vector<cv::Mat>& image_data, void* extra_data );

private:
  struct PFCore; // opaque pointer, defined elsewhere
  boost::shared_ptr<PFCore> pf_core;

private:
  PFTracker()
  { //no default constructor!
  }
  PFTracker(const PFTracker&)
  { //no copying!
  }
  void operator=(const PFTracker&)
  { //no copying!
  }

};



struct gen_rand {
    static double range;
public:
    gen_rand(){}
    double operator()() {
        return (rand()/(double)RAND_MAX) * range;
    }
    static void random_vector( std::vector<double>& x, double maxval = 1.0) {
      range = maxval;
      std::generate_n(x.begin(), x.size(), gen_rand());
    }

    /** \example
           std::vector<double> x(num_items);
           std::generate_n(x.begin(), num_items, gen_rand());
           // now "x" contains random numbers
      */
};


}

#endif
