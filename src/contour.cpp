
#include <string>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
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

ActiveContourSegmentor::ActiveContourSegmentor(Mat &img_in, Mat &segmap)
{
  // Assert( img_in.)
  this->image     = img_in; // = by reference!
  this->phimap    = segmap; // = by reference!
  image.convertTo(imageF64,CV_64FC3);
  segmap.convertTo(phimapF64,CV_64F);
  segmap.convertTo(maskinitF64,CV_64F);
  segmap.convertTo(labelF64,CV_64F);

  this->mdims     = std::vector<int>(2);
  mdims[0]        = img_in.rows;
  mdims[1]        = img_in.cols;

  this->rad           = 10;
  this->dthresh       = 500; // what is this!?
  this->iters_sfls    = 200;
  this->lambda_curv   = .2;

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

  this->ptr_img    = imageF64.ptr<double>(0);
  this->ptr_phi    = phimapF64.ptr<double>(0);
  this->ptr_msk    = maskinitF64.ptr<double>(0);
  this->ptr_lbl    = labelF64.ptr<double>(0);
  this->imgRange   = new double[2];
  this->labelRange = new double[2];

  this->iList=NULL;
  this->jList=NULL;

  dims[2] = 1;
  dims[1] = 1;
  numdims = 2; // fixed for 2D image stream.
  switch(numdims){
  case 3:
    dimz = (int)mdims[2];
    dims[2] = dimz;
  case 2:
    dimx = (int)mdims[1];
    dims[1] = dimx;
  case 1:
    dimy = (int)mdims[0];
    dims[0] = dimy;
  }
  dims[3] = dims[0]*dims[1];
  dims[4] = dims[0]*dims[1]*dims[2];

  initializeData();
  intializeLevelSet();
}

void ActiveContourSegmentor::initializeData()
{
  this->ptrCurrImage = image.ptr<unsigned char>(0);
  this->ptrCurrLabel = phimap.ptr<unsigned char>(0);

  cv::minMaxLoc(imageF64,&(imgRange[0]),&(imgRange[1]) );
  cv::minMaxLoc(phimapF64,&(labelRange[0]),&(labelRange[1]) );

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

void ActiveContourSegmentor::update()
{
  lrbac_chanvese(ptr_img,ptr_phi,ptr_lbl,dims,
           Lz,Ln1,Lp1,Ln2,Lp2,Lin2out,Lout2in,
           iters_sfls,rad,lambda_curv,NULL,0);

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

  delete [] this->imgRange;
  delete [] this->labelRange;
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

