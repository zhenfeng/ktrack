#ifndef CONTOUR_INTERFACE_H
#define CONTOUR_INTERFACE_H


#include <opencv2/core/core.hpp>

struct LL; // used in sfm, forward declare here

namespace ktrack
{


/** write string on top of image data in-place*/
void waterMark(const std::string& text, cv::Mat & img);

class ActiveContourSegmentor
{


public:
  ActiveContourSegmentor( ); // prevent invalid initialization
  ActiveContourSegmentor( cv::Mat& image, cv::Mat& segmap );
  virtual ~ActiveContourSegmentor( );
  void initializeData();
  void update();
  void setRadius( int radNew );
private:

  void intializeLevelSet();


  cv::Mat        image,  imageF64;  //
  cv::Mat        phimap, phimapF64; //
  cv::Mat        maskinitF64;
  cv::Mat        labelF64;



  // Warning: everything below is delicate and might leak or do something crazy
  unsigned char *ptrCurrImage;
  unsigned char *ptrCurrLabel;

  std::vector<int> mdims; //dimensions of "image" we are segmenting (ex.512x512x212)
  double *imgRange;       //[minImageVal, maxImageVal]
  double* labelRange;     //[minLabVal, maxLabVal]

  double *ptr_img;
  double *ptr_phi;
  double *ptr_lbl;
  double *ptr_msk;

  int iters_sfls;      //number of iterations to execute
  double lambda_curv;  //curvature penalty
  double rad;          //radius of ball used in local-global energies
  double dthresh;      // similarity ??

  short *iList;        //row indices of points on zero level set from last run
  short *jList;        //column indices of points on zero level set from last run
  long lengthZLS;      //number of point on the zero level set from last run


  //Level Set Variables Stay persistent
  double *B, *C;
  double *F;
  double usum, vsum;
  int    countdown;
  long    dims[5];
  long dimz,dimy,dimx;
  LL *Lz, *Ln1, *Ln2, *Lp1, *Lp2;
  LL *Sz, *Sn1, *Sn2, *Sp1, *Sp2;
  LL *Lin2out, *Lout2in;

};


}



#endif
