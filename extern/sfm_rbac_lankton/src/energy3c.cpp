/*********************************************************************
 * energy3c.cpp
 *
 * This file holds functions that initialize image means, compute the
 * curve flow for various energies
 *
 * written by: Shawn Lankton (4/17/2009) - www.shawnlankton.com
 *
 ********************************************************************/



#include "energy3c.h"
#include "math.h"
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

static double uin, uout, uuser, sumin, sumout,sumuser;
double ain, aout, auser; // means
static double sum2in, sum2out, varin, varout;      // variances
static double du_orig;       //used in yezzi speed
double *gball, *Ain, *Aout, *Sin, *Sout; //local means
double *pdfin, *pdfout, *pdfuser;
static int nbins=256;
static double imgMin, imgMax,   binWidth;
long numdims;
double engEval;
bool UseInitContour=1;

double *en_lrbac_vessel_yz_compute(LL *Lz,double *phi, double *img, long *dims, double *scale, double lam, double rad, double dthresh){
  int x,y,z,idx,n;
  double *F, *kappa;
  double a,Fmax,u,v,I;

  // allocate space for F
  F = (double*)malloc(Lz->length*sizeof(double));    if(F==NULL) return NULL;
  kappa = (double*)malloc(Lz->length*sizeof(double));if(kappa==NULL) return NULL;

  ll_init(Lz); n=0; Fmax = 1.0e-6; //begining of list;
  while(Lz->curr != NULL){          //loop through list
    x = Lz->curr->x; y = Lz->curr->y; z = Lz->curr->z; idx = Lz->curr->idx;
    I = img[idx];

    if(I>=dthresh){
      if(Ain[idx] <0){
        en_lrbac_vessel_cv_init_point(img,phi,idx,x,y,z,dims,rad,dthresh);
      }
      if(Ain[idx] >0) u = Sin[idx] /Ain[idx];
      if(Aout[idx]>0) v = Sout[idx]/Aout[idx];
      else v = u;

      a = -(u-v)*((I-u)/Ain[idx]+(I-v)/Aout[idx]);
      //a = ((I-u)*(I-u)/Ain[idx]-(I-v)*(I-v)/Aout[idx]);
      kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
      if(fabs(a)>Fmax) Fmax = fabs(a);
      F[n] = a;
    }
    else F[n] = 0;
    ll_step(Lz); n++;       //next point
  }

  if(scale[0]==0) scale[0]=Fmax;

  for(int j=0;j<Lz->length;j++){
    F[j] = (1-lam)*F[j]/scale[0]+lam*kappa[j];
  }
  free(kappa);
  return F;
}

void en_lrbac_vessel_yz_init_point(double* img, double* phi, int idx, int x, int y, int z, long *dims, double rad, double dthresh){
  double usum,vsum,au,av;
  int i,j,k,irad,idia,ridx,bidx;

  if(img[idx]<dthresh) return;

  usum=vsum=au=av=0;
  irad = (int)(floor(rad));
  idia = irad*2+1;

  for(i=-irad;i<=irad;i++){
    if((x+i)<0 | (x+i)>=DIMX) continue;
    for(j=-irad;j<=irad;j++){
      if((y+j)<0 | (y+j)>=DIMY) continue;
      for(k=-irad;k<=irad;k++){
        if((z+k)<0 | (z+k)>=DIMZ) continue;
        ridx = idx+(i*OFFX)+(j*OFFY)+(k*OFFZ);
        bidx = (j+irad)+((i+irad)*idia)+((k+irad)*idia*idia);

        if(img[ridx]>dthresh){
          if(phi[ridx]<=0){
            usum += img[ridx]*gball[bidx];
            au   += gball[bidx];
          }
          else{
            vsum += img[ridx]*gball[bidx];
            av   += gball[bidx];
          }
        }
      }
    }
  }
  Ain[idx] = au;   Aout[idx] = av;
  Sin[idx] = usum; Sout[idx] = vsum;
}

void en_lrbac_vessel_yz_update(double* img, long *dims, LL *Lin2out, LL *Lout2in, double rad, double dthresh){
  int x,y,z;
  int idx;
  int i,j,k,irad,idia,ridx,bidx;

  irad = (int)(floor(rad));
  idia = irad*2+1;

  ll_init(Lin2out);
  while(Lin2out->curr != NULL){
    x = Lin2out->curr->x; y = Lin2out->curr->y; z = Lin2out->curr->z; idx = Lin2out->curr->idx;
    if(img[idx]>=dthresh){
      for(i=-irad;i<=irad;i++){
        if((x+i)<0 | (x+i)>=DIMX) continue;
        for(j=-irad;j<=irad;j++){
          if((y+j)<0 | (y+j)>=DIMY) continue;
          for(k=-irad;k<=irad;k++){
            if((z+k)<0 | (z+k)>=DIMZ) continue;
            ridx = idx+(i*OFFX)+(j*OFFY)+(k*OFFZ);
            bidx = (j+irad)+((i+irad)*idia)+((k+irad)*idia*idia);

            if(Ain[ridx]>=0)
            {
              Sin[ridx]  -= img[idx]*gball[bidx];
              Ain[ridx]  -= gball[bidx];
              Sout[ridx] += img[idx]*gball[bidx];
              Aout[ridx] += gball[bidx];
            }
          }
        }
      }
    }
    ll_remcurr_free(Lin2out);
  }
  ll_init(Lout2in);
  while(Lout2in->curr != NULL){
    x = Lout2in->curr->x; y = Lout2in->curr->y; z = Lout2in->curr->z; idx = Lout2in->curr->idx;

    if(img[idx]>=dthresh){
      for(i=-irad;i<=irad;i++){
        if((x+i)<0 | (x+i)>=DIMX) continue;
        for(j=-irad;j<=irad;j++){
          if((y+j)<0 | (y+j)>=DIMY) continue;
          for(k=-irad;k<=irad;k++){
            if((z+k)<0 | (z+k)>=DIMZ) continue;
            ridx = idx+(i*OFFX)+(j*OFFY)+(k*OFFZ);
            bidx = (j+irad)+((i+irad)*idia)+((k+irad)*idia*idia);

            if(Ain[ridx]>=0)
            {
              Sin[ridx]  += img[idx]*gball[bidx];
              Ain[ridx]  += gball[bidx];
              Sout[ridx] -= img[idx]*gball[bidx];
              Aout[ridx] -= gball[bidx];
            }
          }
        }
      }
    }
    ll_remcurr_free(Lout2in);
  }
  if(uin>0)  uin  = sumin/ain;
  if(uout>0) uout = sumout/aout;
}

double *en_lrbac_vessel_cv_compute(LL *Lz,double *phi, double *img, long *dims, double *scale, double lam, double rad, double dthresh){
  int x,y,z,idx,n;
  double *F, *kappa;
  double a,Fmax,u,v,I;

  // allocate space for F
  F = (double*)malloc(Lz->length*sizeof(double));    if(F==NULL) return NULL;
  kappa = (double*)malloc(Lz->length*sizeof(double));if(kappa==NULL) return NULL;

  ll_init(Lz); n=0; Fmax = 1.0e-6; //begining of list;
  while(Lz->curr != NULL){          //loop through list
    x = Lz->curr->x; y = Lz->curr->y; z = Lz->curr->z; idx = Lz->curr->idx;
    I = img[idx];

    if(I>=dthresh){
      if(Ain[idx] <0){
        en_lrbac_vessel_cv_init_point(img,phi,idx,x,y,z,dims,rad,dthresh);
      }
      if(Ain[idx] >0) u = Sin[idx] /Ain[idx];
      if(Aout[idx]>0) v = Sout[idx]/Aout[idx];
      else v = u;
      a = (I-u)*(I-u)-(I-v)*(I-v);
      kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
      if(fabs(a)>Fmax) Fmax = fabs(a);
      F[n] = a;
    }
    else F[n] = 0;
    ll_step(Lz); n++;       //next point
  }

  if(scale[0]==0) scale[0]=Fmax;

  for(int j=0;j<Lz->length;j++){
    F[j] = (1-lam)*F[j]/scale[0]+lam*kappa[j];
  }
  free(kappa);
  return F;
}

void en_lrbac_vessel_cv_init_point(double* img, double* phi, int idx, int x, int y, int z, long *dims, double rad, double dthresh){
  double usum,vsum,au,av;
  int i,j,k,irad,idia,ridx,bidx;

  if(img[idx]<dthresh) return;

  usum=vsum=au=av=0;
  irad = (int)(floor(rad));
  idia = irad*2+1;

  for(i=-irad;i<=irad;i++){
    if((x+i)<0 | (x+i)>=DIMX) continue;
    for(j=-irad;j<=irad;j++){
      if((y+j)<0 | (y+j)>=DIMY) continue;
      for(k=-irad;k<=irad;k++){
        if((z+k)<0 | (z+k)>=DIMZ) continue;
        ridx = idx+(i*OFFX)+(j*OFFY)+(k*OFFZ);
        bidx = (j+irad)+((i+irad)*idia)+((k+irad)*idia*idia);

        if(img[ridx]>dthresh){
          if(phi[ridx]<=0){
            usum += img[ridx]*gball[bidx];
            au   += gball[bidx];
          }
          else{
            vsum += img[ridx]*gball[bidx];
            av   += gball[bidx];
          }
        }
      }
    }
  }
  Ain[idx] = au;   Aout[idx] = av;
  Sin[idx] = usum; Sout[idx] = vsum;
}

void en_lrbac_vessel_cv_update(double* img, long *dims, LL *Lin2out, LL *Lout2in, double rad, double dthresh){
  int x,y,z;
  int idx;
  int i,j,k,irad,idia,ridx,bidx;

  irad = (int)(floor(rad));
  idia = irad*2+1;

  ll_init(Lin2out);
  while(Lin2out->curr != NULL){
    x = Lin2out->curr->x; y = Lin2out->curr->y; z = Lin2out->curr->z; idx = Lin2out->curr->idx;
    if(img[idx]>=dthresh){
      for(i=-irad;i<=irad;i++){
        if((x+i)<0 | (x+i)>=DIMX) continue;
        for(j=-irad;j<=irad;j++){
          if((y+j)<0 | (y+j)>=DIMY) continue;
          for(k=-irad;k<=irad;k++){
            if((z+k)<0 | (z+k)>=DIMZ) continue;
            ridx = idx+(i*OFFX)+(j*OFFY)+(k*OFFZ);
            bidx = (j+irad)+((i+irad)*idia)+((k+irad)*idia*idia);

            if(Ain[ridx]>=0)
            {
              Sin[ridx]  -= img[idx]*gball[bidx];
              Ain[ridx]  -= gball[bidx];
              Sout[ridx] += img[idx]*gball[bidx];
              Aout[ridx] += gball[bidx];
            }
          }
        }
      }
    }
    ll_remcurr_free(Lin2out);
  }
  ll_init(Lout2in);
  while(Lout2in->curr != NULL){
    x = Lout2in->curr->x; y = Lout2in->curr->y; z = Lout2in->curr->z; idx = Lout2in->curr->idx;

    if(img[idx]>=dthresh){
      for(i=-irad;i<=irad;i++){
        if((x+i)<0 | (x+i)>=DIMX) continue;
        for(j=-irad;j<=irad;j++){
          if((y+j)<0 | (y+j)>=DIMY) continue;
          for(k=-irad;k<=irad;k++){
            if((z+k)<0 | (z+k)>=DIMZ) continue;
            ridx = idx+(i*OFFX)+(j*OFFY)+(k*OFFZ);
            bidx = (j+irad)+((i+irad)*idia)+((k+irad)*idia*idia);

            if(Ain[ridx]>=0)
            {
              Sin[ridx]  += img[idx]*gball[bidx];
              Ain[ridx]  += gball[bidx];
              Sout[ridx] -= img[idx]*gball[bidx];
              Aout[ridx] -= gball[bidx];
            }
          }
        }
      }
    }
    ll_remcurr_free(Lout2in);
  }
  if(uin>0)  uin  = sumin/ain;
  if(uout>0) uout = sumout/aout;
}

void en_lrbac_init(LL *Lz,double *img,double *phi, long *dims, double rad){
  int i,j,k,n,x,y,z,idx,ridx,bidx;

  //create ball
  gball = en_lrbac_gball(rad);

  //allocate memory for lookups
  Ain  = (double*)malloc(NUMEL*sizeof(double)); if(Ain==NULL) return;
  Sin  = (double*)malloc(NUMEL*sizeof(double)); if(Sin==NULL) return;
  Aout = (double*)malloc(NUMEL*sizeof(double)); if(Aout==NULL) return;
  Sout = (double*)malloc(NUMEL*sizeof(double)); if(Sout==NULL) return;

  //poision "uninitialized" points
  for(i=0;i<NUMEL;i++){
    Ain[i] = -1; Aout[i] = -1;
  }
}

void en_lrbac_init_point(double* img, double* phi, int idx, int x, int y, int z, long *dims, double rad){
  double usum,vsum,au,av;
  int i,j,k,irad,idia,ridx,bidx;

  usum=vsum=au=av=0;
  irad = (int)(floor(rad));
  idia = irad*2+1;

  for(i=-irad;i<=irad;i++){
    if((x+i)<0 || (x+i)>=DIMX) continue;
    for(j=-irad;j<=irad;j++){
      if((y+j)<0 || (y+j)>=DIMY) continue;
      for(k=-irad;k<=irad;k++){
        if((z+k)<0 || (z+k)>=DIMZ) continue;
        ridx = idx+(i*OFFX)+(j*OFFY)+(k*OFFZ);
        bidx = (j+irad)+((i+irad)*idia)+((k+irad)*idia*idia);

        if(phi[ridx]<=0){
          usum += img[ridx]*gball[bidx];
          au   += gball[bidx];
        }
        else{
          vsum += img[ridx]*gball[bidx];
          av   += gball[bidx];
        }
      }
    }
  }
  Ain[idx] = au;   Aout[idx] = av;
  Sin[idx] = usum; Sout[idx] = vsum;
}

void en_lrbac_update(double* img, long *dims, LL *Lin2out, LL *Lout2in, double rad){
  int x,y,z;
  int idx;
  int i,j,k,irad,idia,ridx,bidx;

  irad = (int)(floor(rad));
  idia = irad*2+1;


  ll_init(Lin2out);
  while(Lin2out->curr != NULL){
    x = Lin2out->curr->x; y = Lin2out->curr->y; z = Lin2out->curr->z; idx = Lin2out->curr->idx;

    for(i=-irad;i<=irad;i++){
      if((x+i)<0 | (x+i)>=DIMX) continue;
      for(j=-irad;j<=irad;j++){
        if((y+j)<0 | (y+j)>=DIMY) continue;
        for(k=-irad;k<=irad;k++){
          if((z+k)<0 | (z+k)>=DIMZ) continue;
          ridx = idx+(i*OFFX)+(j*OFFY)+(k*OFFZ);
          bidx = (j+irad)+((i+irad)*idia)+((k+irad)*idia*idia);

          if(Ain[ridx]>=0)
          {
            Sin[ridx]  -= img[idx]*gball[bidx];
            Ain[ridx]  -= gball[bidx];
            Sout[ridx] += img[idx]*gball[bidx];
            Aout[ridx] += gball[bidx];
          }
        }
      }
    }
    ll_remcurr_free(Lin2out);
  }
  ll_init(Lout2in);
  while(Lout2in->curr != NULL){
    x = Lout2in->curr->x; y = Lout2in->curr->y; z = Lout2in->curr->z; idx = Lout2in->curr->idx;

    for(i=-irad;i<=irad;i++){
      if((x+i)<0 | (x+i)>=DIMX) continue;
      for(j=-irad;j<=irad;j++){
        if((y+j)<0 | (y+j)>=DIMY) continue;
        for(k=-irad;k<=irad;k++){
          if((z+k)<0 | (z+k)>=DIMZ) continue;
          ridx = idx+(i*OFFX)+(j*OFFY)+(k*OFFZ);
          bidx = (j+irad)+((i+irad)*idia)+((k+irad)*idia*idia);

          if(Ain[ridx]>=0)
          {
            Sin[ridx]  += img[idx]*gball[bidx];
            Ain[ridx]  += gball[bidx];
            Sout[ridx] -= img[idx]*gball[bidx];
            Aout[ridx] -= gball[bidx];
          }
        }
      }
    }
    ll_remcurr_free(Lout2in);
  }
  if(uin>0)  uin  = sumin/ain;
  if(uout>0) uout = sumout/aout;
}


void en_lrbac_destroy(){
  if(gball!=NULL)
    free(gball);
  if(Ain!=NULL)
    free(Ain);
  if(Aout!=NULL)
    free(Aout);
  if(Sin!=NULL)
    free(Sin);
  if(Sout!=NULL)
    free(Sout);
}

double *en_lrbac_compute(LL *Lz,double *phi, double *img, long *dims, double *scale, double lam, double rad ){
  int x,y,z, idx;
  double *F, *kappa;
  double a,u,v,I;
  // allocate space for F
  F = (double*)malloc(Lz->length*sizeof(double));    if(F==NULL) throw "Failed Allocating F!" ;
  kappa = (double*)malloc(Lz->length*sizeof(double)); if(kappa==NULL) throw "Failed Allocating kappa!" ;

  ll_init(Lz);
  int        n= 0;
  double Fmax = 1.0e-6; //begining of list;
  while(Lz->curr != NULL){          //loop through list
    x = Lz->curr->x; y = Lz->curr->y; z = Lz->curr->z; idx = Lz->curr->idx;
    I = img[idx];

    if(Ain[idx] <0){
      en_lrbac_init_point(img,phi,idx,x,y,z,dims,rad);
    }
    if(Ain[idx] >0) u = Sin[idx] /Ain[idx];
    if(Aout[idx]>0) v = Sout[idx]/Aout[idx];
    a = (I-u)*(I-u)-(I-v)*(I-v);
    if(fabs(a)> Fmax)   Fmax = fabs(a);
    F[n] = a;
    kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
    ll_step(Lz); n++;       //next point
  }
  if(scale[0]==0)
    scale[0] = Fmax;
  for(int j=0;j<Lz->length;j++){
    F[j] = F[j]/scale[0]+lam*kappa[j];
  }
  free(kappa);
  return F;
}

// allocates and populates memory for a 3D Gaussian,
// size (floor(rad)*2+1)^3 centered in the middle with sigma = rad/2.
double *en_lrbac_gball(double rad){
  double *gball;
  int dia,dia2,i,j,k,idx;
  double cen,x2,y2,z2,sig2;
  double gsum;
  dia = (int)(floor(rad)*2+1);
  dia2 = dia*dia;
  cen = (int)(floor(rad));
  sig2 = (rad/2)*(rad/2);

  gball = (double*)malloc(sizeof(double)*dia*dia*dia);
  if(gball == NULL) return NULL;

  gsum = 0;
  for(i=0;i<dia;i++){
    for(j=0;j<dia;j++){
      for(k=0;k<dia;k++){
        idx = i+j*dia+k*dia2;
        x2 = ((double)i-cen)*((double)i-cen);
        y2 = ((double)j-cen)*((double)j-cen);
        z2 = ((double)k-cen)*((double)k-cen);
        gball[idx] = exp(-(x2+y2+z2)/(2*sig2));
        gsum += gball[idx];
      }
    }
  }
  for(i=0;i<(dia2*dia);i++){
    gball[i] = gball[i]/gsum;
  }
  return gball;
}

double *en_yezzi_compute(LL *Lz,double *phi, double *img, long *dims, double *scale, double lam){
  //int x,y,z;
  int idx,n,j;
  double *F, *kappa;
  double a,Fmax,u,v,I;
  double Gamuu, Gamuv, Gamvv, gamu, gamv, du, dv;
  double sumuu, sumvv, sumuv, Ibar, I2bar;
  bool ubad, vbad;
  // allocate space for F & kappa
  F = (double*)malloc(Lz->length*sizeof(double));
  if(F == NULL) return NULL;
  kappa = (double*)malloc(Lz->length*sizeof(double));
  if(kappa == NULL) return NULL;

  u = uin; v = uout;
  Ibar = I2bar = 0;
  ll_init(Lz);               //begining of list;
  while(Lz->curr != NULL){   //loop through list
    idx = Lz->curr->idx;
    I = img[idx];
    Ibar  += I;
    I2bar += I*I;
    ll_step(Lz);       //next point
  }
  Ibar  = Ibar/Lz->length;
  I2bar = I2bar/Lz->length;
  sumuu = I2bar - 2*Ibar*u + u*u;
  sumvv = I2bar - 2*Ibar*v + v*v;
  sumuv = I2bar - Ibar*(u+v) + u*v;
  Gamuu =  1/( ain*ain )*sumuu;
  Gamuv = -1/( ain*aout)*sumuv;
  Gamvv =  1/(aout*aout)*sumvv;
  gamu = sumuv/sumuu;
  gamv = sumuv/sumvv;
  du = (u-v)*(Gamuu-Gamuv);
  dv = (u-v)*(Gamuv-Gamvv);

  ubad = vbad = 0;
  if((Gamuv*(Gamuu+Gamvv))>(Gamuu*Gamvv+Gamuv*Gamuv)){
    if(du*du_orig<0){ubad = 1;}// mexPrintf("cu ");}
    else            {vbad = 1;}// mexPrintf("cv ");}
  }

  ll_init(Lz); n = 0; Fmax = 0.0001; //begining of list;
  while(Lz->curr != NULL){           //loop through list
    idx = Lz->curr->idx;

    if(ubad){
      //mexPrintf("ubad\n");
      a = -(u-v)/aout *((img[idx]-v)-gamu*(img[idx]-u)); //preserve u
    }
    else if(vbad){
      //mexPrintf("vbad\n");
      a = -(u-v)/ain*((img[idx]-u)-gamv*(img[idx]-v));   //preserve v
    }
    else{
      a = -(u-v)*((img[idx]-u)/ain+(img[idx]-v)/aout);   //change both
    }

    if(fabs(a)>Fmax) Fmax = fabs(a);
    F[n] = a;
    kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
    ll_step(Lz); n++;       //next point
  }

  if(scale[0]==0) scale[0] = Fmax*4;
  for(j=0;j<Lz->length;j++){
    F[j] = F[j]/scale[0]+lam*kappa[j];
  }
  //mexPrintf("%f,%f\n",u,v);
  free(kappa);
  return F;
}

void en_yezzi_init(LL* Lz, double *img, double *phi, long *dims){
  /*int x,y,z */ int idx;
  double Gamuu, Gamuv, sumuu, sumuv, Ibar, I2bar;
  sumin = 0; sumout = 0; ain = 0; aout = 0;
  uin = 0; uout = 0; du_orig = 0;

  for(int i=0; i<NUMEL; i++){
    if(phi[i]<=0){
      sumin  = sumin  + img[i];
      ain++;
    }
    else{
      sumout = sumout + img[i];
      aout++;
    }
  }
  if(ain>0)  uin = sumin/ain;
  if(aout>0) uout = sumout/aout;

  Ibar = I2bar = 0;
  ll_init(Lz);               //begining of list;
  while(Lz->curr != NULL){   //loop through list
    idx = Lz->curr->idx;
    Ibar  += img[idx];
    I2bar += img[idx]*img[idx];
    ll_step(Lz);       //next point
  }
  Ibar  = Ibar /Lz->length;
  I2bar = I2bar/Lz->length;
  sumuu = I2bar-2*Ibar*uin+uin*uin;
  sumuv = I2bar-Ibar*(uin+uout)+uin*uout;
  Gamuu =  1/( ain*ain )*sumuu;
  Gamuv = -1/(ain*aout )*sumuv;
  du_orig = (uin-uout)*(Gamuu-Gamuv);
}

void en_yezzi_update(double* img, long *dims, LL *Lin2out, LL *Lout2in){
  /*int x,y,z */ int idx;
  ll_init(Lin2out);
  while(Lin2out->curr != NULL){
    idx = Lin2out->curr->idx;

    sumin  -= img[idx]; ain--;
    sumout += img[idx]; aout++;
    ll_remcurr_free(Lin2out);
  }
  ll_init(Lout2in);
  while(Lout2in->curr != NULL){
    idx = Lout2in->curr->idx;

    sumout -= img[idx]; aout--;
    sumin  += img[idx]; ain++;
    ll_remcurr_free(Lout2in);
  }
  if(uin>0)  uin  = sumin/ain;
  if(uout>0) uout = sumout/aout;
}

double *en_grow_compute(LL *Lz, double *img, double* phi, long *dims, double lam, double dthresh){
  double *F,*kappa;
  int n = 0;
  F = (double*)malloc(Lz->length*sizeof(double));
  if(F == NULL) return NULL;
  kappa = (double*)malloc(Lz->length*sizeof(double));
  if(kappa == NULL) return NULL;
  ll_init(Lz); n=0;         //begining of list;
  while(Lz->curr != NULL){  //loop through list
    F[n] = -1;
    kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
    if(img[Lz->curr->idx]<dthresh){
      F[n] = 0;
      kappa[n] = 0;
    }
    ll_step(Lz); n++;    //  next point
  }
  for(int j=0;j<Lz->length;j++){
    F[j] = (1-lam)*F[j]+lam*kappa[j];
  }
  free(kappa);
  return F;
}

double *en_shrink_compute(LL *Lz,double *img, double* phi,long *dims, double rad, double lam, double *scale){
  double *F, *kappa;
  double dx,dy,dz,kmax,fmax,pmin;
  int dxi,dyi,dzi;
  int x,y,z,idx,n,idxN;
  double MIN_INTERIOR;

  F = (double*)malloc(Lz->length*sizeof(double));
  if(F == NULL) return NULL;
  kappa = (double*)malloc(Lz->length*sizeof(double));
  if(kappa == NULL) return NULL;

  if(DIMZ == 1)
    MIN_INTERIOR = .01;
  else
    MIN_INTERIOR = 0.25;

  kmax = 0; fmax=0.0001;
  ll_init(Lz); n=0;         //begining of list;
  while(Lz->curr != NULL){  //loop through list
    x = Lz->curr->x; y = Lz->curr->y; z = Lz->curr->z; idx = Lz->curr->idx;
    kappa[n] = en_kappa_norm_pt(Lz->curr, phi, dims, &dx, &dy, &dz); //compute kappa

    if(Ain[idx]<0){
      en_lrbac_init_point(img,phi,idx,x,y,z,dims,rad);
    }

    idxN = idx;
    F[n] = Ain[idx]-MIN_INTERIOR;
    if(fabs(F[n])>fmax) fmax = fabs(F[n]);
    ll_step(Lz); n++;    //  next point
  }

  if(scale[0]==0) scale[0] = fmax;
  for(int j=0;j<Lz->length;j++){
    //F[j] = F[j]/scale[0]+lam*kappa[j];
    F[j] = F[j]/scale[0]+lam*kappa[j];
  }

  free(kappa);
  return F;
}

double *en_chanvese_compute(LL *Lz, double *phi, double *img, long *dims, double *scale, double lam)
{
  int x,y,z,idx,n;
  double *F, *kappa;
  double a,I,Fmax;
  // allocate space for F
  F = (double*)malloc(Lz->length*sizeof(double));
  if(F == NULL) return NULL;

  kappa = (double*)malloc(Lz->length*sizeof(double));
  if(kappa == NULL) return NULL;

  ll_init(Lz);n=0;Fmax=0.0001; //begining of list;
  while(Lz->curr != NULL){     //loop through list
    idx = Lz->curr->idx;
    I = img[idx];
    a = (I-uin)*(I-uin)-(I-uout)*(I-uout);
    if(fabs(a)>Fmax) Fmax = fabs(a);
    F[n] = a;
    kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
    ll_step(Lz); n++;       //next point
  }
  if(scale[0] == 0) scale[0] = Fmax;

  for(int j=0;j<Lz->length;j++){
    F[j] = F[j]/scale[0]+lam*kappa[j];
  }
  free(kappa);
  return F;
}


void en_lrbac_user_init(LL *Lz,double *img,double *phi, long *dims, double rad, double* seed){
  int i,j,k,n,x,y,z,idx,ridx,bidx;
  double I;
  auser=0;
  uuser = 0;
  sumuser=0;



  //create ball
  gball = en_lrbac_gball(rad);

  //allocate memory for lookups
  Ain  = (double*)malloc(NUMEL*sizeof(double)); if(Ain==NULL) return;
  Sin  = (double*)malloc(NUMEL*sizeof(double)); if(Sin==NULL) return;
  Aout = (double*)malloc(NUMEL*sizeof(double)); if(Aout==NULL) return;
  Sout = (double*)malloc(NUMEL*sizeof(double)); if(Sout==NULL) return;

  //poision "uninitialized" points and compute mean seeds
  for(i=0;i<NUMEL;i++){
    Ain[i] = -1; Aout[i] = -1;

    I = img[i];
    if(UseInitContour==0){
        if(seed[i]!=0){
            sumuser =sumuser+I; //we compute mean from the defined seeds
            auser++;
        }
    }else{
        if(phi[i]<=0){
            sumuser =sumuser+I; //we compute mean from initialization
            auser++;
        }

    }
  }

  if(auser>0) uuser = sumuser/auser;
  std::cout<<"The mean user is: "<<uuser<<std::endl;
  std::cout<<"The number of elements is : "<<NUMEL<<std::endl;
  std::cout<<"The sumuser is : "<<sumuser<<std::endl;
  std::cout<<"The auser is : "<<auser<<std::endl;
}

double *en_lrbac_user_compute(LL *Lz,double *phi, double *img,double penaltyAlpha, long *dims, double *scale, double lam, double rad){
  int x,y,z,idx,n;
  double *F, *kappa;
  double a,Fmax,u,v,I;
  int binNum;
  int numPointsOnLevSet=0;

  // allocate space for F
  F = (double*)malloc(Lz->length*sizeof(double));    if(F==NULL) return NULL;
  kappa = (double*)malloc(Lz->length*sizeof(double));if(kappa==NULL) return NULL;

  ll_init(Lz); n=0; Fmax = 1.0e-6; //begining of list;
  while(Lz->curr != NULL){          //loop through list
    x = Lz->curr->x;
    y = Lz->curr->y;
    z = Lz->curr->z;
    idx = Lz->curr->idx;
    I = img[idx];

   numPointsOnLevSet++;


    if(Ain[idx] <0){
      en_lrbac_init_point(img,phi,idx,x,y,z,dims,rad);
    }

    if(Ain[idx] >0){
        u = Sin[idx] /Ain[idx];
    }

    if(Aout[idx]>0){
        v = Sout[idx]/Aout[idx];
    }

    a =  (I-u)*(I-u) - (I-v)*(I-v) + penaltyAlpha*(I-uuser)*(I-uuser);

    if(fabs(a)>Fmax){
        Fmax = fabs(a);
    }

    F[n] = a;
    kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
    ll_step(Lz); //next point
    n++;
  }

//  std::cout<<"We still have "<<numPointsOnLevSet<<" points."<<std::endl;
  if(scale[0]==0) scale[0] = Fmax;
  for(int j=0;j<Lz->length;j++){
    F[j] = F[j]/scale[0]+lam*kappa[j];
  }
  free(kappa);
  return F;
}

void en_chanvese_init(double* img, double* phi, long *dims){
  double I;
  sumin = 0; sumout = 0; ain = 0; aout = 0;
  uin = 0; uout = 0; du_orig = 0;

  for(int i=0; i<NUMEL; i++){
    I = img[i];
    if(phi[i]<=0){
      sumin  = sumin  + I;
      ain++;
    }
    else{
      sumout = sumout + I;
      aout++;
    }
  }
  if(ain>0)  uin = sumin/ain;
  if(aout>0) uout = sumout/aout;
  //mexPrintf("uin=%f uout=%f\n",uin,uout);
}

double *en_user_chanvese_compute(LL *Lz, double *phi, double *img,double penaltyAlpha, long *dims, double *scale, double lam)
{
  int x,y,z,idx,n;
  double *F, *kappa;
  double a,I,Fmax;
  // allocate space for F
  F = (double*)malloc(Lz->length*sizeof(double));
  if(F == NULL) return NULL;

  kappa = (double*)malloc(Lz->length*sizeof(double));
  if(kappa == NULL) return NULL;

  ll_init(Lz);n=0;Fmax=0.0001; //begining of list;
  while(Lz->curr != NULL){     //loop through list
    idx = Lz->curr->idx;
    I = img[idx];
    a = -(I-uin)*(I-uin)+(I-uout)*(I-uout) - penaltyAlpha*(uin-uuser)*(uin-uuser);
    if(fabs(a)>Fmax) Fmax = fabs(a);
    F[n] = a;
    kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
    ll_step(Lz); n++;       //next point
  }
  if(scale[0] == 0) scale[0] = Fmax;

  for(int j=0;j<Lz->length;j++){
    F[j] = F[j]/scale[0]+lam*kappa[j];
  }
  free(kappa);
  return F;
}



void en_user_chanvese_init(double* img, double* phi, long *dims, double* seed){
  double I;
  sumin = 0; sumout = 0; sumuser=0;ain = 0; aout = 0; auser=0;
  uin = 0; uout = 0; du_orig = 0;



  for(int i=0; i<NUMEL; i++){
    I = img[i];
    if(phi[i]<=0){
      sumin  = sumin  + I;
      ain++;
    }
    else{
      sumout = sumout + I;
      aout++;
    }

    if(UseInitContour==0){
        if(seed[i]!=0){
            sumuser =sumuser+I; //we compute mean from the defined seeds
            auser++;
        }
    }else{
        if(phi[i]<=0){
            sumuser =sumuser+I; //we compute mean from initialization
            auser++;
        }

    }


  }
  if(ain>0)  uin = sumin/ain;
  if(aout>0) uout = sumout/aout;
  if(auser>0) uuser = sumuser/auser;
}

void en_chanvese_update(double* img, long *dims, LL *Lin2out, LL *Lout2in){
  /*int x,y,z */ int idx;
  ll_init(Lin2out);
  while(Lin2out->curr != NULL){
    idx = Lin2out->curr->idx;

    sumin  -= img[idx]; ain--;
    sumout += img[idx]; aout++;
    ll_remcurr_free(Lin2out);
  }
  ll_init(Lout2in);
  while(Lout2in->curr != NULL){
    idx = Lout2in->curr->idx;

    sumout -= img[idx]; aout--;
    sumin  += img[idx]; ain++;
    ll_remcurr_free(Lout2in);
  }
  if(uin>0)  uin  = sumin/ain;
  if(uout>0) uout = sumout/aout;
  //mexPrintf("uin=%f uout=%f\n",uin,uout);
}

double *en_meanvar_compute(LL *Lz, double *phi, double *img, long *dims, double *scale, double lam)
{
  int x,y,z,idx,n;
  double *F, *kappa;
  double a,I,Fmax;
  // allocate space for F
  F = (double*)malloc(Lz->length*sizeof(double));
  if(F == NULL) return NULL;

  kappa = (double*)malloc(Lz->length*sizeof(double));
  if(kappa == NULL) return NULL;

  ll_init(Lz);n=0;Fmax=0.0001; //begining of list;
  while(Lz->curr != NULL){     //loop through list
    idx = Lz->curr->idx;
    I = img[idx];
    a = -(log(varout/varin)-(I-uin)*(I-uin)/varin+(I-uout)*(I-uout)/varout);
    if(fabs(a)>Fmax) Fmax = fabs(a);
    F[n] = a;
    kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
    ll_step(Lz); n++;       //next point
  }
  if(scale[0] == 0) scale[0] = Fmax;

  for(int j=0;j<Lz->length;j++){
    F[j] = F[j]/scale[0]+lam*kappa[j];
  }
  free(kappa);
  return F;
}

void en_meanvar_init(double* img, double* phi, long *dims){
  double I;
  sumin = 0; sumout = 0; ain = 0; aout = 0;
  sum2in=0; sum2out=0; varin = 0; varout = 0;
  uin = 0; uout = 0; du_orig = 0;

  for(int i=0; i<NUMEL; i++){
    I = img[i];
    if(phi[i]<=0){
      sumin  += I;
      sum2in += I*I;
      ain++;
    }
    else{
      sumout  += I;
      sum2out += I*I;
      aout++;
    }
  }
  if(ain>0){
    uin = sumin/ain;
    varin = sum2in/ain-uin*uin;
  }
  if(aout>0){
    uout = sumout/aout;
    varout = sum2out/aout-uout*uout;
  }
  //mexPrintf("varin=%f varout=%f\n",varin,varout);
}

void en_meanvar_update(double* img, long *dims, LL *Lin2out, LL *Lout2in){
  /*int x,y,z */ int idx;
  double I,I2;
  ll_init(Lin2out);
  while(Lin2out->curr != NULL){
    idx = Lin2out->curr->idx;
    I = img[idx]; I2 = I*I;
    sumin  -= I; sum2in  -= I2;  ain--;
    sumout += I; sum2out += I2; aout++;
    ll_remcurr_free(Lin2out);
  }
  ll_init(Lout2in);
  while(Lout2in->curr != NULL){
    idx = Lout2in->curr->idx;
    I = img[idx]; I2 = I*I;
    sumout -= I; sum2out -= I2; aout--;
    sumin  += I; sum2in  += I2;  ain++;
    ll_remcurr_free(Lout2in);
  }
  if(ain>0){
    uin = sumin/ain;
    varin = sum2in/ain-uin*uin;
  }
  if(aout>0){
    uout = sumout/aout;
    varout = sum2out/aout-uout*uout;
  }

  //mexPrintf("uin=%f uout=%f\n",uin,uout);
}

double *en_bhattacharyya_compute(LL *Lz, double *phi, double *img, long *dims, double *scale, double lam)
{
  int i,x,y,z,idx,n;
  double *F, *kappa, *lookup;
  double a,Fmax,KappaMax,p,q,B,integral;
  int I;
  // allocate space for F
  F = (double*)malloc(Lz->length*sizeof(double));
  if(F == NULL) return NULL;

  kappa = (double*)malloc(Lz->length*sizeof(double));
  if(kappa == NULL) return NULL;

  lookup = (double*)malloc(nbins*sizeof(double));
  if(lookup==NULL) return NULL;

  double *pdfin_smooth= (double *)malloc(nbins*sizeof(double));
  double *pdfout_smooth= (double *)malloc(nbins*sizeof(double));

  smoothHist(pdfin, pdfin_smooth );
  smoothHist(pdfout, pdfout_smooth );

  B = 0;
  for(i=0;i<nbins;i++){
    B += sqrt(pdfin_smooth[i]/ain*pdfout_smooth[i]/aout);
  }

  for(i=0;i<nbins;i++){
    p = pdfin_smooth[i]/ain   + 0.0000001;
    q = pdfout_smooth[i]/aout + 0.0000001;
    integral = sqrt(p/q)/aout-sqrt(q/p)/ain;
    lookup[i]=B/2*(1/ain-1/aout)+integral/2;
  }

  ll_init(Lz);
  n=0;
  Fmax=0.000000001; //begining of list;
  KappaMax=0.000000001;
  while(Lz->curr != NULL){     //loop through list
    idx = Lz->curr->idx;
    I = (int) floor((img[idx]-imgMin)/binWidth);
    a = -lookup[I];
    if(fabs(a)>Fmax) Fmax = fabs(a);
    F[n] = a;
    kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
    if(fabs(kappa[n])>KappaMax) KappaMax = fabs(kappa[n]);
    ll_step(Lz); n++;       //next point
  }
  if(scale[0] == 0) scale[0] = Fmax;

  for(int j=0;j<Lz->length;j++){
    F[j] =F[j]/scale[0]+lam*kappa[j]/KappaMax;
  }
  free(lookup);
  free(kappa);

  free(pdfin_smooth);
  free(pdfout_smooth);

  engEval=B;
  return F;
}

void en_bhattacharyya_init(double* img, double* phi, long *dims){
  int I;
  nbins = 256;


  pdfin  = (double*)malloc(nbins*sizeof(double)); if(pdfin ==NULL) return;
  pdfout = (double*)malloc(nbins*sizeof(double)); if(pdfout==NULL) return;
  ain = 0; aout = 0;
  for(int i=0;i<nbins;i++){
    pdfin[i]=0; pdfout[i]=0;
  }

  findMinMax(img,dims);

  binWidth=(imgMax-imgMin+1)/nbins;

  for(int i=0; i<NUMEL; i++){

    I = (int) floor((img[i]-imgMin)/binWidth);

    if(I<0 || I>nbins){
      //mexPrintf("WARNING: Image should c++be in the range [0-%d]\n",nbins);
      return;
    }

    if(phi[i]<=0){
      pdfin[I] += 1;
      ain++;
    }
    else{
      pdfout[I] += 1;
      aout++;
    }
  }

}

void findMinMax(double *img,long *dims){
    imgMin=0;
    imgMax=0;
    long elemNum=0;

    for (int k=0; k<dims[2]; k++) {
        for (int j=0; j<dims[1]; j++)  {
            for (int i=0; i<dims[0]; i++) {
                if(img[elemNum]>imgMax) imgMax = img[elemNum];
                if(img[elemNum]<imgMin) imgMin = img[elemNum];
                elemNum++;
            }
        }
    }

}

double *en_user_bhattacharyya_compute(LL *Lz, double *phi, double *img, double penaltyAlpha, long *dims, double *scale, double lam)
{
  int i,x,y,z,idx,n;
  double *F, *kappa, *lookup, *lookupUser;
  double a,Fmax,KappaMax,p,q,u,B,Buser,integral, integralUser;
  int I;
  // allocate space for F
  F = (double*)malloc(Lz->length*sizeof(double));
  if(F == NULL) return NULL;

  kappa = (double*)malloc(Lz->length*sizeof(double));
  if(kappa == NULL) return NULL;

  lookup = (double*)malloc(nbins*sizeof(double));
  if(lookup==NULL) return NULL;

  lookupUser = (double*)malloc(nbins*sizeof(double));
  if(lookupUser==NULL) return NULL;

  double *pdfin_smooth= (double *)malloc(nbins*sizeof(double));
  double *pdfout_smooth= (double *)malloc(nbins*sizeof(double));
  double *pdfuser_smooth= (double *)malloc(nbins*sizeof(double));

  smoothHist(pdfin, pdfin_smooth );
  smoothHist(pdfout, pdfout_smooth );
  smoothHist(pdfuser, pdfuser_smooth );


  B = 0;
  Buser = 0;
  for(i=0;i<nbins;i++){
    B += sqrt(pdfin_smooth[i]/ain*pdfout_smooth[i]/aout);
    Buser += sqrt((pdfuser_smooth[i]/auser)*(pdfin_smooth[i]/ain));
  }

  for(i=0;i<nbins;i++){
    p = pdfin_smooth[i]/ain   + 0.0000001;
    q = pdfout_smooth[i]/aout + 0.0000001;
    u = pdfuser_smooth[i]/auser + 0.0000001;
    integral = sqrt(p/q)/aout-sqrt(q/p)/ain;
    lookup[i]=B/2*(1/ain-1/aout)+integral/2;
    integralUser=sqrt(u/p);
    lookupUser[i]= .5*(1/ain)*(Buser - integralUser);
  }

//  std::ofstream myfile_user("lookupUser");
//  for(int i=0; i<nbins;i++){
//      //myfile <<((double *) (smoothPDF.data))[i]<<std::endl;
//      myfile_user <<lookupUser[i]<<std::endl;
//  }
//  myfile_user.close();


//  std::ofstream myfile_bhatt("lookup");
//  for(int i=0; i<nbins;i++){
//      //myfile <<((double *) (smoothPDF.data))[i]<<std::endl;
//      myfile_bhatt <<lookup[i]<<std::endl;
//  }
//  myfile_bhatt.close();



  ll_init(Lz);
  n=0;
  Fmax=0.000000001; //begining of list;
  KappaMax=0.000000001;


  while(Lz->curr != NULL){     //loop through list
    idx = Lz->curr->idx;
    I =  (int) floor((img[idx]-imgMin)/binWidth);
    a = -(lookup[I] + penaltyAlpha*lookupUser[I]);
    if(fabs(a)>Fmax){
        Fmax = fabs(a);
    }
    F[n] = a;
    kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa
    if(fabs(kappa[n])>KappaMax){
        KappaMax = fabs(kappa[n]);
    }
    ll_step(Lz);
    n++;       //next point
  }
  if(scale[0] == 0){
      scale[0] = Fmax;
  }

  for(int j=0;j<Lz->length;j++){
    F[j] = F[j]/scale[0]+lam*kappa[j]/KappaMax;
  }

  free(lookup);
  free(lookupUser);
  free(kappa);


  free(pdfin_smooth);
  free(pdfout_smooth);
  free(pdfuser_smooth);

  engEval=B + penaltyAlpha*Buser;

  //std::cout<<"Buser: "<<Buser<<" B: "<<B<<" penaltyAlpha: "<<penaltyAlpha<<" engEval: "<<engEval<<std::endl;
  return F;
}

void en_user_bhattacharyya_init(double* img, double* phi, long *dims, double* seed){
  int I;

  pdfin  = (double*)malloc(nbins*sizeof(double)); if(pdfin ==NULL) return;
  pdfout = (double*)malloc(nbins*sizeof(double)); if(pdfout==NULL) return;
  pdfuser= (double*)malloc(nbins*sizeof(double)); if(pdfuser==NULL) return;

  ain = 0; aout = 0; auser=0;

  for(int i=0;i<nbins;i++){
    pdfin[i]=0; pdfout[i]=0; pdfuser[i]=0;
  }

  findMinMax(img,dims);

  binWidth=(imgMax-imgMin+1)/nbins;



  for(int i=0; i<NUMEL; i++){
    I = (int) floor((img[i]-imgMin)/binWidth);

//    if(I<0 || I>nbins){
//      mexPrintf("WARNING: Image should be in the range [0-%d]\n",nbins);
//      return;
//    }

    if(phi[i]<=0){
      pdfin[I] += 1;
      ain++;
    }
    else{
      pdfout[I] += 1;
      aout++;
    }

    if(UseInitContour==0){
        if(seed[i]!=0){
            pdfuser[I] +=1; //we compute constant pdf from the defined seeds
            auser++;
        }
    }else{
        if(phi[i]<=0){
            pdfuser[I] +=1; //we simply remember this original pdf
            auser++;
        }

    }
  }

std::ofstream myfile_pdfuser("pdfuser.txt");
myfile_pdfuser<<"The image min is: "<<imgMin<<std::endl;
myfile_pdfuser<<"The image max is: "<<imgMax<<std::endl;
myfile_pdfuser<<"The image binWidth is: "<<binWidth<<std::endl;

for(int i=0; i<nbins;i++){
    myfile_pdfuser <<pdfuser[i]<<std::endl;
}
myfile_pdfuser<<"The total number of points is: " <<auser<<std::endl;
myfile_pdfuser.close();

}


void en_bhattacharyya_destroy(){
  if(pdfin!=NULL)  free(pdfin);
  if(pdfout!=NULL) free(pdfout);
}
void en_user_bhattacharyya_destroy(){
  if(pdfin!=NULL)  free(pdfin);
  if(pdfout!=NULL) free(pdfout);
  if(pdfuser!=NULL) free(pdfuser);
}

void en_bhattacharyya_update(double* img, long *dims, LL *Lin2out, LL *Lout2in){
  /*int x,y,z */ int idx;
  int I;
  ll_init(Lin2out);
  while(Lin2out->curr != NULL){
    idx = Lin2out->curr->idx;
    I = (int) floor((img[idx]-imgMin)/binWidth);
    pdfin[I]--;   ain--;
    pdfout[I]++;  aout++;
    ll_remcurr_free(Lin2out);
  }
  ll_init(Lout2in);
  while(Lout2in->curr != NULL){
    idx = Lout2in->curr->idx;
    I = (int) floor((img[idx]-imgMin)/binWidth);
    pdfout[I]--;  aout--;
    pdfin[I]++;   ain++;
    ll_remcurr_free(Lout2in);
  }
  //mexPrintf("uin=%f uout=%f\n",uin,uout);
}

double en_kappa_pt(PT* p, double *phi, long *dims){
  double dx,dy,dz;
  return en_kappa_norm_pt(p, phi, dims, &dx, &dy, &dz);
}

double *en_kappa_compute(LL *Lz, double *phi, long *dims)
{
  double *kappa;
  int n = 0;
  // allocate space for F
  kappa = (double*)malloc(Lz->length*sizeof(double));
  if(kappa == NULL) return NULL;

  ll_init(Lz);              //begining of list;
  while(Lz->curr != NULL){  //loop through list
    kappa[n] = en_kappa_pt(Lz->curr, phi, dims); //compute kappa[n]
    ll_step(Lz); n++;       //next point
  }
  return kappa;
}

double en_kappa_norm_pt(PT* p, double *phi, long *dims, double *pdx, double *pdy, double *pdz){
  double kappa;
  double dx,dy,dz,dxx,dyy,dzz,dxy,dxz,dyz,dx2,dy2,dz2;
  int idx,x,y,z,n;
  int xok,yok,zok;

  x = p->x; y = p->y; z = p->z; idx = p->idx;

  dx=dy=dz=dxx=dyy=dzz=dxy=dxz=dyz=dx2=dy2=dz2=0;
  xok = yok = zok = 0;

  if((x+1)<DIMX && (x-1)>=0) xok = 1;
  if((y+1)<DIMY && (y-1)>=0) yok = 1;
  if((z+1)<DIMZ && (z-1)>=0) zok = 1;

  if(xok){
    dx  = (phi[idx-OFFX]-phi[idx+OFFX])/2;// (l-r)/2
    dxx = (phi[idx-OFFX]-2*phi[idx]+phi[idx+OFFX]); // l-2c+r
    dx2 = dx*dx;
  }
  if(yok){
    dy  = (phi[idx-OFFY]-phi[idx+OFFY])/2;// (u-d)/2
    dyy = (phi[idx-OFFY]-2*phi[idx]+phi[idx+OFFY]);// u-2c+d
    dy2 = dy*dy;
  }
  if(zok){
    dz  = (phi[idx-OFFZ]-phi[idx+OFFZ])/2;// (b-f)/2
    dzz = (phi[idx-OFFZ]-2*phi[idx]+phi[idx+OFFZ]);// b-2c+f
    dz2 = dz*dz;
  }
  if(xok && yok){// (ul+dr-ur-dl)/4
    dxy = (phi[idx-OFFY-OFFX]+phi[idx+OFFY+OFFX]-phi[idx-OFFY+OFFX]-phi[idx+OFFY-OFFX])/4;
  }
  if(xok && zok){// (lf+rb-rf-lb)/4
    dxz = (phi[idx+OFFZ-OFFX]+phi[idx-OFFZ+OFFX]-phi[idx+OFFZ+OFFX]-phi[idx+OFFZ+OFFX])/4;
  }
  if(zok && yok){// (uf+db-df-ub)/4
    dyz = (phi[idx-OFFY+OFFZ]+phi[idx+OFFY-OFFZ]-phi[idx+OFFY+OFFZ]-phi[idx-OFFY-OFFZ])/4;
  }

  kappa = (dxx*(dy2+dz2)+dyy*(dx2+dz2)+dzz*(dx2+dy2)-
           2*dx*dy*dxy-2*dx*dz*dxz-2*dy*dz*dyz)/
           (dx2+dy2+dz2+.00000001);

  pdx[0] = dx;
  pdy[0] = dy;
  pdz[0] = dz;
  return kappa;
}

void en_null_update(double* img, long *dims, LL *Lin2out, LL *Lout2in){
  ll_init(Lin2out);
  while(Lin2out->curr != NULL){
    ll_remcurr_free(Lin2out);
  }

  ll_init(Lout2in);
  while(Lout2in->curr != NULL){
    ll_remcurr_free(Lout2in);
  }
}

void draw_histogram(double *pdf,char* pdfInFileName, double a, int display ){


    //create an image to hold the histogram
    cv::Mat histImage =  255 * cv::Mat::ones(400,600,CV_8U); // cvCreateImage(cvSize(320,200), 8, 1);
//    cv::Mat pdfMat=cv::Mat(1, nbins,CV_64F);
//    pdfMat.data=pdf;
//    cv::boxFilter(pdfMat,pdfMat,-1,cv::Size(4,1), cv::Point(-1,1), true, cv::BORDER_DEFAULT);

    //create a factor for scaling along the width
    int bin_w = round((double)histImage.cols/nbins);

    std::ofstream myfile(pdfInFileName);
    for(int i = 0; i < nbins-1; i++ ) {
        //draw the histogram data onto the histogram image
        int x1=i*bin_w;
        int y1=histImage.rows-round((pdf[i]/a)*histImage.rows);
        int x2=(i+1)*bin_w;
        int y2=histImage.rows-round((pdf[i+1]/a)*histImage.rows);
        cv::line(histImage, cv::Point(x1,y1), cv::Point(x2,y2), CV_RGB(255, 0, 0), 1, 8, 0);
        if (myfile.is_open())
        {
            myfile << pdf[i]<<"     "<<x1<<"     "<<y1<<"     "<<x2<<"     "<<y2<<"\n";
        }
    }
    myfile.close();

    if(display==1){
        cv::namedWindow(pdfInFileName,CV_WINDOW_AUTOSIZE);
        cv::imshow(pdfInFileName,histImage );

        //hold the images until a key is pressed
        cv::waitKey(30);
    }

}

void smoothHist(double *pdfRough, double *pdfSmooth ){

//    //create matrix with our data
//    cv::Mat roughPDF =  cv::Mat(nbins,1, CV_32FC1,&pdfRough);
//    cv::Mat smoothPDF=  cv::Mat(nbins,1, CV_64FC1,&pdfSmooth);
//    //cv::GaussianBlur(roughPDF, smoothPDF, cv::Size(1,0), 1, 1, cv::BORDER_DEFAULT);
//    cv::blur(roughPDF, smoothPDF, cv::Size(4,1), cv::Point(-1, -1), cv::BORDER_DEFAULT);



    int kerSize=7;
    int kerSide=3;
    int currInd;

    double testSumBefore=0;
    double testSumAfter=0;
    for(int l=0;l<nbins;l++){
        testSumBefore+=pdfRough[l];
    }

    double kernel[7]={0.0044,0.0540, 0.2420, 0.3989, 0.2420, 0.0540,0.0044};
    for(int i=0;i<nbins; i++){
        pdfSmooth[i]=0;
        for(int j=0;j<kerSize;j++){
            if (i+j-kerSide<0){
                currInd=0;
            }else if ((i+j-kerSide)>=nbins){
                currInd=nbins-1;
            }else{
                currInd=i+j-kerSide;
            }
            pdfSmooth[i]=pdfSmooth[i] + kernel[j]*pdfRough[currInd];
        }
    }


//        std::ofstream myfile("testSmooth");
//        for(int i=0; i<nbins;i++){
//            //myfile <<((double *) (smoothPDF.data))[i]<<std::endl;
//            myfile <<pdfSmooth[i]<<std::endl;
//        }

//        myfile.close();

//    for(int m=0;m<nbins;m++){
//        testSumAfter+=pdfSmooth[m];
//    }
    //std::cout<<testSumAfter<<std::endl;
}

void display_slice(double *img,int *dims, int sliceNum, char *name, double *imgRange ){

    //create matrix with our data
    int offsetInt=dims[0]*dims[1]*sliceNum;

    cv::Mat histImage =  cv::Mat(dims[0],dims[1], CV_64FC1,&img[offsetInt]);
    histImage  = (histImage - imgRange[0])* (1.0 / (imgRange[1]-imgRange[0]) );

    cv::namedWindow(name,CV_WINDOW_AUTOSIZE);
    cv::imshow(name,histImage );

    //hold the images until a key is pressed
    cv::waitKey(30);



}
