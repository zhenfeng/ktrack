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
   getLowDimOutput, and must perform any GUI / display / output. */
class TrackingProblem
{
public:
  /** virtual destructor for inheritance. */
  virtual ~TrackingProblem()
  {
  }

  /** evaluate likelihood */
  virtual void evaluateLikelihood( const std::vector<cv::Mat>& particles,
                                   std::vector<double>& likelihood ) = 0;

  /** propagate dynamics */
  virtual void propagateDynamics( const std::vector<cv::Mat>& particles_in,
                                 const std::vector<cv::Mat>& particles_out ) = 0;

  /** Number of states in particles.
   * This is called at registration time and must not change! */
  virtual size_t N() const = 0;

  /** evaluate a low dimensional output, such as from "contour" to
     "centroid" for example. */
  virtual void getLowDimOutput( const std::vector<cv::Mat>& particles,
                                const std::vector<double>& weights,
                                std::vector<double>& output ) = 0;


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




}

#endif
