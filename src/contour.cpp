
#include <string>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <algorithm>
#include <list>
#include "contour.h"
#include "llist.h"
#include "sfm_local_chanvese_mex.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using namespace cv;

namespace ktrack
{
#define Assert(x) ( (x) ? int(0) :  throw std::logic_error("broke") )

//these global variables are no good, need to fix later
extern double ain, aout, auser; // means
extern double *pdfin, *pdfout, *pdfuser;
extern long numdims;
extern double engEval;
extern bool UseInitContour;
extern double *Ain, *Aout, *Sin, *Sout; //local means

void ActiveContourSegmentor::setRadius( int radNew ) {
  rad = radNew;
}

void ActiveContourSegmentor::setImage( const cv::Mat& img )
{

  // TODO: use RGB
  cv::cvtColor(img,image,CV_RGB2GRAY);
  image.copyTo(image_show);
  image.convertTo(imageF64,CV_64F,0.1);
  ptr_img = imageF64.ptr<double>(0);

  if( frame_buffer.empty() ) {
    frame_buffer.push_front(imageF64);
  }
  imageF64 = 0.9 * imageF64 + 0.1 * frame_buffer.front();
  frame_buffer.push_front(imageF64);
  if( (int) frame_buffer.size() == buff_len ) {
    frame_buffer.pop_back();
  }
}

void ActiveContourSegmentor::setLabel(const cv::Mat &segmap)
{
  segmap.copyTo(phimap);
  segmap.convertTo(phimapF64,CV_64F);
  segmap.convertTo(maskinitF64,CV_64F);
  segmap.convertTo(labelF64,CV_64F);

  this->ptr_phi    = phimapF64.ptr<double>(0);
  this->ptr_msk    = maskinitF64.ptr<double>(0);
  this->ptr_lbl    = labelF64.ptr<double>(0);

  Lz=NULL;
  Ln1= NULL;
  Ln2= NULL;
  Lp1=NULL ;
  Lp2=NULL ;
  Sz=NULL ;
  Sn1=NULL ;
  Sn2=NULL ;
  Sp1=NULL ;
  Sp2=NULL ;
  Lin2out=NULL ;
  Lout2in=NULL ;
  this->iList=NULL;
  this->jList=NULL;

  dims[2] = 1;
  dimx    = (int)mdims[1];
  dims[1] = dimx;
  dimy    = (int)mdims[0];
  dims[0] = dimy;
  dims[3] = dims[0]*dims[1];
  dims[4] = dims[0]*dims[1]*dims[2];

  intializeLevelSet();


}


void ActiveContourSegmentor::getDisplayImage( cv::Mat& img_out)
{
  Assert( img_out.size() == image.size() );
  static Mat rc,gc,bc;
  static vector<Mat> chans(3);
  image_show.copyTo(rc);
  image_show.copyTo(gc);
  image_show.copyTo(bc);
  unsigned char oppval = 255;
  for( int k = 0; k < lengthZLS; k++ ) {
    rc.at<unsigned char>( iList[k], jList[k] )     = 0;
    bc.at<unsigned char>( iList[k]-1, jList[k] )   = oppval;
    bc.at<unsigned char>( iList[k]+1, jList[k] )   = oppval;
    bc.at<unsigned char>( iList[k], jList[k]-1)    = oppval;
    bc.at<unsigned char>( iList[k], jList[k]+1 )   = oppval;
  }
  chans[0] = rc;
  chans[1] = gc;
  chans[2] = bc;
  merge(chans,img_out);
}

void ActiveContourSegmentor::getPreviousPhi(cv::Mat &phi_out)
{
  Assert( phi_out.size() == phimapF64.size() );
  phimapF64.copyTo(phi_out);
}

ActiveContourSegmentor::ActiveContourSegmentor(Mat &img_in, Mat &segmap)
{
  Assert( img_in.cols == segmap.cols && img_in.rows == segmap.rows );
  setImage(img_in);


  this->mdims     = std::vector<int>(2);
  mdims[0]        = img_in.cols;
  mdims[1]        = img_in.rows;

  this->rad           = 20;
  this->iters_sfls    = 10;
  this->lambda_curv   = .5;
  this->buff_len      = 5.0;


  this->imgRange   = std::vector<double>(2);
  this->labelRange = std::vector<double>(2);

  setLabel(segmap);

}

void ActiveContourSegmentor::intializeLevelSet(){

  if (Lz!=NULL){
    //destroy linked lists
    ll_destroy(Lz);
    ll_destroy(Ln1);
    ll_destroy(Ln2);
    ll_destroy(Lp1);
    ll_destroy(Lp2);
    ll_destroy(Lin2out);
    ll_destroy(Lout2in);
  }

  //create linked lists
  Lz  = ll_create();
  Ln1 = ll_create();
  Ln2 = ll_create();
  Lp1 = ll_create();
  Lp2 = ll_create();
  Lin2out = ll_create();
  Lout2in = ll_create();

  //initialize lists, phi, and labels
  ls_mask2phi3c(ptr_msk,ptr_phi,ptr_lbl,dims,Lz,Ln1,Ln2,Lp1,Lp2);

}

void ActiveContourSegmentor::update(int iters_in)
{

  iters_sfls = iters_in;

  int mode = 0;
  switch( mode ) {
  case 0:
    chanvese(ptr_img,ptr_phi,ptr_lbl,dims,
             Lz,Ln1,Lp1,Ln2,Lp2,Lin2out,Lout2in,
             iters_sfls,lambda_curv,NULL,0);
    break;
  case 1:
    lrbac_chanvese(ptr_img,ptr_phi,ptr_lbl,dims,
                   Lz,Ln1,Lp1,Ln2,Lp2,Lin2out,Lout2in,
                   iters_sfls,rad,lambda_curv,NULL,0);
    break;
  case 2:
    bhattacharyya(ptr_img,ptr_phi,ptr_lbl,dims,
                  Lz,Ln1,Lp1,Ln2,Lp2,Lin2out,Lout2in,
                  iters_sfls,lambda_curv,NULL,0);
  default:
    break;
  }



  // delete the old (i,j) levelset indices
  if(iList!=NULL){
    delete[] iList;
  }
  if(jList!=NULL){
    delete[] jList;
  }

  //get number and coordinates of point (row, col) on the zero level set
  prep_C_output(Lz,dims,ptr_phi, &iList, &jList, lengthZLS);
}

ActiveContourSegmentor::~ActiveContourSegmentor(){

  delete [] this->iList;
  delete [] this->jList;

  ll_destroy(Lz);
  ll_destroy(Ln1);
  ll_destroy(Ln2);
  ll_destroy(Lp1);
  ll_destroy(Lp2);
  ll_destroy(Lin2out);
  ll_destroy(Lout2in);
}


}

