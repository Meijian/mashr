#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <R.h>
#include <Rinternals.h>
#include "proj_gauss_main.h"
#include "proj_gauss_mixtures.h"

void bovy_randvec(gsl_vector * eps, int d, double length){
  length /= sqrt((double)d);
  int dd;
  for (dd = 0; dd != d; ++dd)
    gsl_vector_set(eps,dd,(2.*gsl_rng_uniform(randgen)-1.)*length);

  return;
}

void calc_splitnmerge(struct datapoint * data,int N,
		      struct gaussian * gaussians, int K, 
		      gsl_matrix * qij, int * snmhierarchy){
  gsl_matrix * tempqij = gsl_matrix_alloc(N,K);
  gsl_matrix_memcpy(tempqij,qij);
  gsl_matrix * Jmerge = gsl_matrix_alloc(K,K);
  gsl_matrix_set_all(Jmerge,-1.);
  int kk1, kk2, kk, ii,maxsnm= K*(K-1)*(K-2)/2;
  int d = (gaussians->VV)->size1;//dim of mm
  double temp1,temp2,temp;
  for (kk1 = 0; kk1 != K; ++kk1)
    for (kk2 = kk1+1; kk2 != K; ++kk2){
      temp = 0.;
      for (ii=0; ii != N; ++ii){
	//make them all exps
	temp1 = exp(gsl_matrix_get(qij,ii,kk1));
	temp2 = exp(gsl_matrix_get(qij,ii,kk2));
	temp += temp1*temp2;
      }
      gsl_matrix_set(Jmerge,kk1,kk2,temp);
    }

  //Then calculate Jsplit
  gsl_vector * Jsplit = gsl_vector_alloc(K);
  gsl_vector * Jsplit_temp = gsl_vector_alloc(K);
  gsl_vector_set_all(Jsplit,-1.);
  
  //if there is missing data, fill in the missing data
  struct missingdatapoint{
    gsl_vector *ww;
  };
  struct missingdatapoint * missingdata;
  missingdata = (struct missingdatapoint *) malloc(N * sizeof (struct missingdatapoint) );
  for (ii=0; ii != N; ++ii){
    missingdata->ww = gsl_vector_alloc(d);
    ++missingdata;
  }
  missingdata -= N;

  gsl_matrix * tempRR,* tempVV;
  gsl_vector * tempSS,* tempwork;
  gsl_vector * expectedww = gsl_vector_alloc(d);
  //gsl_vector_view tempUcol;
  double lambda;
  int di,signum;
  for (ii=0; ii != N; ++ii){
    //First check whether there is any missing data
    if ((data->ww)->size == d){
      gsl_vector_memcpy(missingdata->ww,data->ww);
      ++missingdata;
      ++data;
      continue;
    }

    /*AS IT STANDS THE MISSING DATA PART IS *NOT* IMPLEMENTED
      CORRECTLY: A CORRECT IMPLEMENTATION NEEDS THE NULL SPACE OF THE
      PROJECTION MATRIX WHICH CAN BE FOUND FROM THE FULL SINGULAR
      VALUE DECOMPOSITIIN, UNFORTUNATELY GSL DOES NOT COMPUTE THE FULL
      SVD, BUT ONLY THE THIN SVD.  LAPACK MIGHT DO, BUT MIGHT NOT BE
      INSTALLED (?) AND THIS MIGHT BE HARD TO IMPLEMENT.

      INDICATED BELOW ARE THE SECTION THAT WOULD HAVE TO BE FIXED TO
      MAKE THIS WORK
     */

    //calculate expectation, for this we need to calculate the bbijs (EXACTLY THE SAME AS IN PROJ_EM, SHOULD WRITE GENERAL FUNCTION TO DO THIS)
    gsl_vector_set_zero(expectedww);
    for (kk = 0; kk != K; ++kk){
      //prepare...
      di = (data->SS)->size1;
      p = gsl_permutation_alloc (di);
      wminusRm = gsl_vector_alloc (di);
      gsl_vector_memcpy(wminusRm,data->ww);
      TinvwminusRm = gsl_vector_alloc (di);
      Tij = gsl_matrix_alloc(di,di);
      gsl_matrix_memcpy(Tij,data->SS);
      Tij_inv = gsl_matrix_alloc(di,di);
      VRT = gsl_matrix_alloc(d,di);
      VRTTinv = gsl_matrix_alloc(d,di);
      Rtrans = gsl_matrix_alloc(d,di);
      //Calculate Tij
      gsl_matrix_transpose_memcpy(Rtrans,data->RR);
      gsl_blas_dsymm(CblasLeft,CblasUpper,1.0,gaussians->VV,Rtrans,0.0,VRT);//Only the upper right part of VV is calculated
      gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,data->RR,VRT,1.0,Tij);//This is Tij
      //Calculate LU decomp of Tij and Tij inverse
      gsl_linalg_LU_decomp(Tij,p,&signum);
      gsl_linalg_LU_invert(Tij,p,Tij_inv);
      //Calculate Tijinv*(w-Rm)
      gsl_blas_dgemv(CblasNoTrans,-1.0,data->RR,gaussians->mm,1.0,wminusRm);
      gsl_blas_dsymv(CblasUpper,1.0,Tij_inv,wminusRm,0.0,TinvwminusRm);
      //Now calculate bij and Bij
      gsl_vector_memcpy(bs->bbij,gaussians->mm);
      gsl_blas_dgemv(CblasNoTrans,1.0,VRT,TinvwminusRm,1.0,bs->bbij);
      //..and add the result to expectedww
      gsl_vector_scale(bs->bbij,exp(gsl_matrix_get(qij,ii,kk)));
      gsl_vector_add(expectedww,bs->bbij);
      //Clean up
      gsl_permutation_free (p);
      gsl_vector_free(wminusRm);
      gsl_vector_free(TinvwminusRm);
      gsl_matrix_free(Tij);
      gsl_matrix_free(Tij_inv);
      gsl_matrix_free(VRT);
      gsl_matrix_free(VRTTinv);
      gsl_matrix_free(Rtrans);
      ++gaussians;
    }
    gaussians -= K;
    //if missing, fill in the missing data
    tempRR = gsl_matrix_alloc((data->RR)->size2,(data->RR)->size1);//will hold the transpose of RR
    //tempVV = gsl_matrix_alloc((data->RR)->size1,(data->RR)->size1);
    //tempSS = gsl_vector_alloc((data->RR)->size1);
    //tempwork = gsl_vector_alloc((data->RR)->size1);
    gsl_matrix_transpose_memcpy(tempRR,data->RR);
    gsl_blas_dgemv(CblasNoTrans,1.,tempRR,data->ww,0.,missingdata->ww);
    //gsl_linalg_SV_decomp(tempRR,tempVV,tempSS,tempwork);
    //compute the missing data THIS PART IS NOT IMPLEMENTED CORRECTLY
    //for (kk = 0; kk != d-(data->ww)->size; ++kk){
      //tempUcol = gsl_matrix_column(tempRR,d-1-kk);
    //gsl_blas_ddot(&(tempUcol.vector),expectedww,&lambda);
    //gsl_vector_scale(&(tempUcol.vector),lambda);
    //gsl_vector_add(missingdata->ww,&(tempUcol.vector));
    //}
    ++missingdata;
    ++data;
    //free
    gsl_matrix_free(tempRR);
    //gsl_matrix_free(tempVV);
    //gsl_vector_free(tempSS);
    //gsl_vector_free(tempwork);
  }
  data -= N;
  missingdata -= N;

  //then for every gaussian, calculate the KL divergence between the local data density and the l-th gaussian
  double tempsplit;
  p = gsl_permutation_alloc (d);
  tempVV= gsl_matrix_alloc(d,d);
  tempRR= gsl_matrix_alloc(d,d);
  tempSS= gsl_vector_alloc(d);
  tempwork= gsl_vector_alloc(d);
  for (kk = 0; kk != K; ++kk){
    //calculate qil/ql factors
    normalize_row(tempqij,kk,false,true,0.);
    //calculate inverse of V and det(V)
    gsl_matrix_memcpy(tempVV,gaussians->VV);
    gsl_linalg_LU_decomp(tempVV,p,&signum);
    gsl_linalg_LU_invert(tempVV,p,tempRR);//tempRR now has the inverse of VV
    tempsplit = d * halflogtwopi + 0.5 * gsl_linalg_LU_lndet(tempVV);
    for (ii=0; ii != N; ++ii){
      if (exp(gsl_matrix_get(tempqij,ii,kk)) == 0.){
	++missingdata;
	continue;
      }
      tempsplit += gsl_matrix_get(tempqij,ii,kk) * exp(gsl_matrix_get(tempqij,ii,kk));
      gsl_vector_memcpy(tempSS,gaussians->mm);
      gsl_vector_scale(tempSS,-1.);
      gsl_vector_add(tempSS,missingdata->ww);
      gsl_blas_dgemv(CblasNoTrans,1.0,tempRR,tempSS,0.,tempwork);
      gsl_blas_ddot(tempSS,tempwork,&lambda);
      tempsplit += 0.5 * exp(gsl_matrix_get(tempqij,ii,kk)) * lambda;
      ++missingdata;
    }
    gsl_vector_set(Jsplit,kk,tempsplit);
    //printf("Jsplit for gaussian %i = %f\n",kk,tempsplit);
    missingdata -= N;
    ++gaussians;
  }
  gaussians -= K;

  //free
  gsl_permutation_free(p);
  gsl_matrix_free(tempRR);
  gsl_matrix_free(tempVV);
  gsl_vector_free(tempSS);
  gsl_vector_free(tempwork);
  

  //and put everything in the hierarchy
  size_t maxj, maxk, maxl;
  for (kk1 = 0; kk1 != maxsnm; kk1 += (K-2)){
    gsl_matrix_max_index(Jmerge,&maxj,&maxk);
    gsl_vector_memcpy(Jsplit_temp,Jsplit);
    gsl_vector_set(Jsplit_temp,maxj,-1.);
    gsl_vector_set(Jsplit_temp,maxk,-1.);
    for (kk2=0; kk2 != K-2; ++kk2){
      maxl = gsl_vector_max_index(Jsplit_temp);
      gsl_vector_set(Jsplit_temp,maxl,-1.);
      *(snmhierarchy++)= maxj;
      *(snmhierarchy++)= maxk;
      *(snmhierarchy++)= maxl;
      //printf("j = %i, k = %i, l = %i\n",(int)maxj,(int)maxk,(int)maxl);
    }
    //then set it to zero and find the next
    gsl_matrix_set(Jmerge,maxj,maxk,-1.);
  }
  snmhierarchy -= 3*maxsnm;
    


  //clean up
  gsl_matrix_free(Jmerge);
  gsl_vector_free(Jsplit);
  gsl_vector_free(Jsplit_temp);

  return ;
}

double logsum(gsl_matrix * q, int row, bool isrow){
  double logxmin = log(DBL_MIN);
  double logxmax = log(DBL_MAX);
  int l = (isrow) ? q->size2 : q->size1;

  /* First find the maximum and mininum */
  double max, min;
  minmax(q,row,isrow,&min,&max);//,allfixed);


  min *= -1.;
  min += logxmin;
  max *= -1.;
  max += logxmax - log(l);
  (min >  max) ? (max=max) : (max=min);
  double loglike=0.0;
  int dd;
  if (isrow)
    for (dd = 0; dd != q->size2; ++dd)
	loglike += exp(gsl_matrix_get(q,row,dd)+max);
  else
    for (dd = 0; dd != q->size1; ++dd)
	loglike += exp(gsl_matrix_get(q,dd,row)+max);
  

  return log(loglike)-max;
}

void minmax(gsl_matrix * q, int row, bool isrow, double * min, 
	    double * max){
  *max = -DBL_MAX;
  *min = DBL_MAX;
  int dd;
  double temp;
  if (isrow) {
    for (dd = 0; dd != q->size2; ++dd){
	temp  = gsl_matrix_get(q,row,dd);
	if (temp > *max && bovy_isfin(temp))
	  *max = temp;
	if (temp < *min && bovy_isfin(temp))
	  *min = temp;
    }
  }
  else {
    for (dd = 0; dd != q->size1; ++dd){
	temp  = gsl_matrix_get(q,dd,row);
	if (temp > *max && bovy_isfin(temp))
	  *max = temp;
	if (temp < *min && bovy_isfin(temp))
	  *min = temp;
    }
  }



  return ;
}

double normalize_row(gsl_matrix * q, int row, bool isrow,
		     bool noweight, double weight){
  double loglike;
  if (isrow)
    loglike = logsum(q,row,true);
  else
    loglike = logsum(q,row,false);

  int dd;
  if (isrow)
    for (dd = 0; dd != q->size2; ++dd) {
      if ( noweight ) 
	gsl_matrix_set(q,row,dd,gsl_matrix_get(q,row,dd)-loglike);
      else
	gsl_matrix_set(q,row,dd,gsl_matrix_get(q,row,dd)-loglike+weight);
    }
  else
    for (dd = 0; dd != q->size1; ++dd) {
      if ( noweight ) 
	gsl_matrix_set(q,dd,row,gsl_matrix_get(q,dd,row)-loglike);
      else
	gsl_matrix_set(q,dd,row,gsl_matrix_get(q,dd,row)-loglike+weight);
    }
  if ( ! noweight ) loglike*= exp(weight);

  return loglike;
}

bool parse_option(char line[]){
  //Define the options
  char K_opt[]="K";
  char fixamp_opt[]="fixamp";
  char fixmean_opt[]="fixmean";
  char fixcovar_opt[]="fixcovar";
  char maxiter_opt[]="maxiter";
  char tol_opt[]="tol";
  char likeonly_opt[]="likeonly";
  char w_opt[]="w";
  char splitnmerge_opt[]="splitnmerge";

  
  //Split the option in the name and value
  int ii=0,valuestart;
  char option[12]="", value[1000]="";
  while (line[ii] != '='){
    option[ii]= line[ii];
    ++ii;
  }
  valuestart=++ii;
  while (line[ii] != '\n'){
    value[ii-valuestart]= line[ii];
    ++ii;
  }
  
  //Set parameters
  if (strcmp(option,K_opt) == 0){
    K=atoi(value);
    //Based on this information, allocated some matrices/vectors
    fixampP = (bool *) malloc(K * sizeof (bool));
    fixmeanP = (bool *) malloc(K * sizeof (bool));
    fixcovarP = (bool *) malloc(K * sizeof (bool));
    gaussians = (struct gaussian *) malloc(K * sizeof (struct gaussian) );
    if (fixampP == NULL || fixmeanP == NULL || fixcovarP == NULL || gaussians == NULL){
      printf("Allocation of arrays failed, not enough free memory?\n");
      return -1;
    }
  }
  else if (strcmp(option,maxiter_opt) == 0) maxiter=atoll(value);
  else if (strcmp(option,tol_opt) == 0) tol=atof(value);
  else if (strcmp(option,w_opt) == 0) w=atof(value);
  else if (strcmp(option,likeonly_opt) == 0) likeonly=atoi(value);
  else if (strcmp(option,splitnmerge_opt) == 0) splitnmerge=atoi(value);
  else if (strcmp(option,fixamp_opt) == 0){
    for (ii=0; ii != K; ++ii){
      if (value[ii] == '0') *fixampP=false;
      else *fixampP=true;
      ++fixampP;
    }
    fixampP -= K;
  }
  else if (strcmp(option,fixmean_opt) == 0){
    for (ii=0; ii != K; ++ii){
      if (value[ii] == '0') *fixmeanP=false;
      else *fixmeanP=true;
      ++fixmeanP;
    }
    fixmeanP -= K;
  }
  else if (strcmp(option,fixcovar_opt) == 0){
    for (ii=0; ii != K; ++ii){
      if (value[ii] == '0') *fixcovarP=false;
      else *fixcovarP=true;
      ++fixcovarP;
    }
    fixcovarP -= K;
  }
  else return false;
  
  return true;
}

void proj_EM(struct datapoint * data, int N, struct gaussian * gaussians, 
	     int K,bool * fixamp, bool * fixmean, bool * fixcovar, 
	     double * avgloglikedata, double tol,long long int maxiter, 
	     bool likeonly, double w, bool keeplog, FILE *logfile,
	     FILE *tmplogfile, bool noproj, bool diagerrs, bool noweight){
  double diff = 2. * tol, oldavgloglikedata;
  int niter = 0;
  int d = (gaussians->mm)->size;
  halflogtwopi  = 0.5 * log(8. * atan(1.0));
  while ( diff > tol && niter < maxiter){
    proj_EM_step(data,N,gaussians,K,fixamp,fixmean,fixcovar,avgloglikedata,
		 likeonly,w,noproj,diagerrs,noweight);
    if (keeplog){
      fprintf(logfile,"%f\n",*avgloglikedata);
      fprintf(tmplogfile,"%f\n",*avgloglikedata);
      fflush(logfile);
      fflush(tmplogfile);
      //printf("%f\n",*avgloglikedata);
    }
    if (niter > 0){
      diff = *avgloglikedata - oldavgloglikedata;
      if (diff < 0){
	printf("Warning: log likelihood decreased by %g\n",diff);
	//fprintf(logfile,"oldavgloglike was %g\navgloglike is %g\n",oldavgloglikedata,*avgloglikedata);
      }
    }
    oldavgloglikedata = *avgloglikedata;
    if (likeonly) break;
    ++niter;
    //write_model("result.dat");
  }
  
 //post-processing: only the upper right of VV was computed, copy this to the lower left of VV
  int dd1,dd2,kk;
  for (kk = 0; kk != K; ++kk){
    for (dd1 = 0; dd1 != d; ++dd1)
      for (dd2 = dd1+1; dd2 != d ; ++dd2)
	gsl_matrix_set(gaussians->VV,dd2,dd1,
		       gsl_matrix_get(gaussians->VV,dd1,dd2));
    ++gaussians;
  }
  gaussians -= K;



  return;
}

#define CHUNKSIZE 1

void proj_EM_step(struct datapoint * data, int N, 
		  struct gaussian * gaussians, int K,bool * fixamp, 
		  bool * fixmean, bool * fixcovar, double * avgloglikedata, 
		  bool likeonly, double w, bool noproj, bool diagerrs,
		  bool noweight){
  *avgloglikedata = 0.0;
  //struct timeval start,time1, time2, time3, time4, time5,end;
  struct datapoint * thisdata;
  struct gaussian * thisgaussian;
  struct gaussian * thisnewgaussian;
  int signum,di;
  double exponent;
  double currqij;
  struct modelbs * thisbs;
  int d = (gaussians->VV)->size1;//dim of mm

  //gettimeofday(&start,NULL);
  //Initialize new parameters
  int kk;
  for (kk=0; kk != K*nthreads; ++kk){
    newgaussians->alpha = 0.0;
    gsl_vector_set_zero(newgaussians->mm);
    gsl_matrix_set_zero(newgaussians->VV);
    ++newgaussians;
  }
  newgaussians= startnewgaussians;
    
  //gettimeofday(&time1,NULL);
  //check whether for some Gaussians none of the parameters get updated
  double sumfixedamps= 0;
  bool * allfixed = (bool *) calloc(K, sizeof (bool) );
  double ampnorm;
  for (kk=0; kk != K; ++kk){
    if (*fixamp == true){
      sumfixedamps += gaussians->alpha;
    }
    ++gaussians;
    if (*fixamp == true && *fixmean == true && *fixcovar == true)
      *allfixed= true;
    ++allfixed;
    ++fixamp;
    ++fixmean;
    ++fixcovar;
  }
  gaussians -= K;
  allfixed -= K;
  fixamp -= K;
  fixmean -= K;
  fixcovar -= K;

  //gettimeofday(&time2,NULL);

  //now loop over data and gaussians to update the model parameters
  int ii, jj, ll;
  double sumSV;
  int chunk;
  chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) \
  private(tid,di,signum,exponent,ii,jj,ll,kk,Tij,Tij_inv,wminusRm,p,VRTTinv,sumSV,VRT,TinvwminusRm,Rtrans,thisgaussian,thisdata,thisbs,thisnewgaussian,currqij) \
  shared(newgaussians,gaussians,bs,allfixed,K,d,data,avgloglikedata)
  for (ii = 0 ; ii < N; ++ii){
    thisdata= data+ii;
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    di = (thisdata->SS)->size1;
    //printf("Datapoint has dimension %i\n",di);
    p = gsl_permutation_alloc (di);
    wminusRm = gsl_vector_alloc (di);
    TinvwminusRm = gsl_vector_alloc (di);
    Tij = gsl_matrix_alloc(di,di);
    Tij_inv = gsl_matrix_alloc(di,di);
    if ( ! noproj ) VRT = gsl_matrix_alloc(d,di);
    VRTTinv = gsl_matrix_alloc(d,di);
    if ( ! noproj ) Rtrans = gsl_matrix_alloc(d,di);
    for (jj = 0; jj != K; ++jj){
      //printf("%i,%i\n",(thisdata->ww)->size,wminusRm->size);
      gsl_vector_memcpy(wminusRm,thisdata->ww);
      //fprintf(stdout,"Where is the seg fault?\n");
      thisgaussian= gaussians+jj;
      //prepare...
      if ( ! noproj ) {
	if ( diagerrs ) {
	  gsl_matrix_set_zero(Tij);
	  for (ll = 0; ll != di; ++ll)
	    gsl_matrix_set(Tij,ll,ll,gsl_matrix_get(thisdata->SS,ll,0));}
	else
	  gsl_matrix_memcpy(Tij,thisdata->SS);
      }
      //Calculate Tij
      if ( ! noproj ) {
	gsl_matrix_transpose_memcpy(Rtrans,thisdata->RR);
	gsl_blas_dsymm(CblasLeft,CblasUpper,1.0,thisgaussian->VV,Rtrans,0.0,VRT);//Only the upper right part of VV is calculated --> use only that part
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,thisdata->RR,VRT,1.0,Tij);}//This is Tij
      else {
	if ( diagerrs ) {
	  for (kk = 0; kk != d; ++kk){
	    gsl_matrix_set(Tij,kk,kk,gsl_matrix_get(thisdata->SS,kk,0)+gsl_matrix_get(thisgaussian->VV,kk,kk));
	    for (ll = kk+1; ll != d; ++ll){
	      sumSV= gsl_matrix_get(thisgaussian->VV,kk,ll);
	      gsl_matrix_set(Tij,kk,ll,sumSV);
	      gsl_matrix_set(Tij,ll,kk,sumSV);}}}
	else {
	  for (kk = 0; kk != d; ++kk){
	    gsl_matrix_set(Tij,kk,kk,gsl_matrix_get(thisdata->SS,kk,kk)+gsl_matrix_get(thisgaussian->VV,kk,kk));
	    for (ll = kk+1; ll != d; ++ll){
	      sumSV= gsl_matrix_get(thisdata->SS,kk,ll)+gsl_matrix_get(thisgaussian->VV,kk,ll);
	      gsl_matrix_set(Tij,kk,ll,sumSV);
	      gsl_matrix_set(Tij,ll,kk,sumSV);}}}}
      //gsl_matrix_add(Tij,thisgaussian->VV);}
      //Calculate LU decomp of Tij and Tij inverse
      gsl_linalg_LU_decomp(Tij,p,&signum);
      gsl_linalg_LU_invert(Tij,p,Tij_inv);
      //Calculate Tijinv*(w-Rm)
      if ( ! noproj ) gsl_blas_dgemv(CblasNoTrans,-1.0,thisdata->RR,thisgaussian->mm,1.0,wminusRm);
      else gsl_vector_sub(wminusRm,thisgaussian->mm);
      //printf("wminusRm = %f\t%f\n",gsl_vector_get(wminusRm,0),gsl_vector_get(wminusRm,1));
      gsl_blas_dsymv(CblasUpper,1.0,Tij_inv,wminusRm,0.0,TinvwminusRm);
      //printf("TinvwminusRm = %f\t%f\n",gsl_vector_get(TinvwminusRm,0),gsl_vector_get(TinvwminusRm,1));
      gsl_blas_ddot(wminusRm,TinvwminusRm,&exponent);
      //printf("Exponent = %f\nDet = %f\n",exponent,gsl_linalg_LU_det(Tij,signum));
      gsl_matrix_set(qij,ii,jj,log(thisgaussian->alpha) - di * halflogtwopi - 0.5 * gsl_linalg_LU_lndet(Tij) -0.5 * exponent);//This is actually the log of qij
      //printf("Here we have = %f\n",gsl_matrix_get(qij,ii,jj));
      //Now calculate bij and Bij
      thisbs= bs+tid*K+jj;
      gsl_vector_memcpy(thisbs->bbij,thisgaussian->mm);
      //printf("%i,%i,%i\n",tid,ii,jj);
      if ( ! noproj ) gsl_blas_dgemv(CblasNoTrans,1.0,VRT,TinvwminusRm,1.0,thisbs->bbij);
      else gsl_blas_dsymv(CblasUpper,1.0,thisgaussian->VV,TinvwminusRm,1.0,thisbs->bbij);
      //printf("bij = %f\t%f\n",gsl_vector_get(bs->bbij,0),gsl_vector_get(bs->bbij,1));
      gsl_matrix_memcpy(thisbs->BBij,thisgaussian->VV);
      if ( ! noproj ) {
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,VRT,Tij_inv,0.0,VRTTinv);
	gsl_blas_dgemm(CblasNoTrans,CblasTrans,-1.0,VRTTinv,VRT,1.0,thisbs->BBij);}
      else {
	gsl_blas_dsymm(CblasLeft,CblasUpper,1.0,thisgaussian->VV,Tij_inv,0.0,VRTTinv);
	gsl_blas_dsymm(CblasRight,CblasUpper,-1.0,thisgaussian->VV,VRTTinv,1.0,thisbs->BBij);}
      gsl_blas_dsyr(CblasUpper,1.0,thisbs->bbij,thisbs->BBij);//This is bijbijT + Bij, which is the relevant quantity
      }
    gsl_permutation_free (p);
    gsl_vector_free(wminusRm);
    gsl_vector_free(TinvwminusRm);
    gsl_matrix_free(Tij);
    gsl_matrix_free(Tij_inv);
    if ( ! noproj ) gsl_matrix_free(VRT);
    gsl_matrix_free(VRTTinv);
    if ( ! noproj ) gsl_matrix_free(Rtrans);
    //Again loop over the gaussians to update the model(can this be more efficient? in any case this is not so bad since generally K << N)
#pragma omp critical
    {
      //Normalize qij properly
      *avgloglikedata += normalize_row(qij,ii,true,noweight,thisdata->logweight);
    }
      //printf("qij = %f\t%f\n",gsl_matrix_get(qij,ii,0),gsl_matrix_get(qij,ii,1));
      //printf("avgloglgge = %f\n",*avgloglikedata);
      for (jj = 0; jj != K; ++jj){
	currqij = exp(gsl_matrix_get(qij,ii,jj));
	//printf("Current qij = %f\n",currqij);
	thisbs= bs+tid*K+jj;
	thisnewgaussian= newgaussians+tid*K+jj;
	gsl_vector_scale(thisbs->bbij,currqij);
	gsl_vector_add(thisnewgaussian->mm,thisbs->bbij);
	gsl_matrix_scale(thisbs->BBij,currqij);
	gsl_matrix_add(thisnewgaussian->VV,thisbs->BBij);
	//printf("bij = %f\t%f\n",gsl_vector_get(bs->bbij,0),gsl_vector_get(bs->bbij,1));
	//printf("Bij = %f\t%f\t%f\n",gsl_matrix_get(bs->BBij,0,0),gsl_matrix_get(bs->BBij,1,1),gsl_matrix_get(bs->BBij,0,1));
      }
  }
  *avgloglikedata /= N;
  if (likeonly) {
    free(allfixed);
    return;
  }
  //gettimeofday(&time3,NULL);

  //gather newgaussians
  if ( nthreads != 1 ) 
#pragma omp parallel for schedule(static,chunk) \
  private(ll,jj)
    for (jj = 0; jj < K; ++jj) 
      for (ll = 1; ll != nthreads; ++ll) {
	gsl_vector_add((newgaussians+jj)->mm,(newgaussians+ll*K+jj)->mm);
	gsl_matrix_add((newgaussians+jj)->VV,(newgaussians+ll*K+jj)->VV);
      }
  
  //gettimeofday(&time4,NULL);

  //Now update the parameters
  //Thus, loop over gaussians again!
  double qj;
#pragma omp parallel for schedule(dynamic,chunk) \
  private(jj,qj)
  for (jj = 0; jj < K; ++jj){
    if (*(allfixed+jj)){
      continue;
    }
    else {
      qj = exp(logsum(qij,jj,false));
      (qj < DBL_MIN) ? qj = 0: 0;
      //printf("qj = %f\n",qj);
      if (*(fixamp+jj) != true) {
	(gaussians+jj)->alpha = qj;
	if (qj == 0) {//rethink this
	  *(fixamp+jj)=1;
	  *(fixmean+jj)=1;
	  *(fixcovar+jj)=1;
	  continue;
	}
      }
      gsl_vector_scale((newgaussians+jj)->mm,1.0/qj);
      if (*(fixmean+jj) != true){
	gsl_vector_memcpy((gaussians+jj)->mm,(newgaussians+jj)->mm);
    }
      if (*(fixcovar+jj) != true){
	//	if (*(fixmean+jj) != true)
	//  gsl_blas_dsyr(CblasUpper,-qj,(gaussians+jj)->mm,(newgaussians+jj)->VV);
	//else {
	gsl_blas_dsyr(CblasUpper,qj,(gaussians+jj)->mm,(newgaussians+jj)->VV);
	gsl_blas_dsyr2(CblasUpper,-qj,(gaussians+jj)->mm,(newgaussians+jj)->mm,(newgaussians+jj)->VV);
	//}
      if (w > 0.){
	gsl_matrix_add((newgaussians+jj)->VV,I);
	gsl_matrix_scale((newgaussians+jj)->VV,1.0/(qj+1.0));
      }
      else gsl_matrix_scale((newgaussians+jj)->VV,1.0/qj);
      gsl_matrix_memcpy((gaussians+jj)->VV,(newgaussians+jj)->VV);
      }
    }
  }
  //gettimeofday(&time5,NULL);

  //normalize the amplitudes
  if ( sumfixedamps == 0. && noweight ){
    for (kk=0; kk != K; ++kk){
      if ( noweight ) (gaussians++)->alpha /= (double) N;
    }
  }
  else {
    ampnorm= 0;
    for (kk=0; kk != K; ++kk){
      if (*(fixamp++) == false) ampnorm += gaussians->alpha;
      ++gaussians;
    }
    fixamp -= K;
    gaussians -= K;
    for (kk=0; kk != K; ++kk){
      if (*(fixamp++) == false){
	gaussians->alpha /= ampnorm;
	gaussians->alpha *= (1. - sumfixedamps);
      }
      ++gaussians;
    }
    fixamp -= K;
    gaussians -= K;
  }

  free(allfixed);

  return;
}

void proj_gauss_mixtures(struct datapoint * data, int N, 
			 struct gaussian * gaussians, int K,
			 bool * fixamp, bool * fixmean, bool * fixcovar, 
			 double * avgloglikedata, double tol,
			 long long int maxiter, bool likeonly, double w, 
			 int splitnmerge, bool keeplog, FILE *logfile, 
			 FILE *convlogfile, bool noproj, bool diagerrs,
			 bool noweight){
  //Allocate some memory
  struct gaussian * startgaussians;
  startgaussians = gaussians;
  int d = (gaussians->VV)->size1;//dim of mm
  //Only give copies of the fix* vectors to the EM algorithm
  bool *fixamp_tmp, *fixmean_tmp,*fixcovar_tmp;
  fixamp_tmp = (bool *) malloc(K * sizeof (bool) );
  fixmean_tmp = (bool *) malloc(K * sizeof (bool) );
  fixcovar_tmp = (bool *) malloc(K * sizeof (bool) );
  int kk;
  for (kk = 0; kk != K; ++kk){
    *(fixamp_tmp++) = *(fixamp++);
    *(fixmean_tmp++) = *(fixmean++);
    *(fixcovar_tmp++) = *(fixcovar++);
  }
  fixamp -= K;
  fixamp_tmp -= K;
  fixmean -= K;
  fixmean_tmp -= K;
  fixcovar -= K;
  fixcovar_tmp -= K;
  //allocate the newalpha, newmm and newVV matrices
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  newgaussians = (struct gaussian *) malloc(K * nthreads * sizeof (struct gaussian) );
  startnewgaussians = newgaussians;
  int ll;
  for (kk=0; kk != K*nthreads; ++kk){
    newgaussians->alpha = 0.0;
    newgaussians->mm = gsl_vector_calloc (d);
    newgaussians->VV = gsl_matrix_calloc (d,d);
    ++newgaussians;
  }
  newgaussians= startnewgaussians;
  double oldavgloglikedata;
  //allocate the q_ij matrix
  qij = gsl_matrix_alloc(N,K);
  double * qstarij = (double *) malloc(N * sizeof (double) );
  I = gsl_matrix_alloc(d,d);
  gsl_matrix_set_identity(I);//Unit matrix
  gsl_matrix_scale(I,w);//scaled to w
  //Also take care of the bbij's and the BBij's
  bs = (struct modelbs *) malloc(nthreads * K * sizeof (struct modelbs) );
  for (kk = 0; kk != nthreads*K; ++kk){
    bs->bbij = gsl_vector_alloc (d);
    bs->BBij = gsl_matrix_alloc (d,d);
    ++bs;
  }
  bs -= nthreads*K;
  //splitnmerge
  int maxsnm = K*(K-1)*(K-2)/2;
  int * snmhierarchy = (int *) malloc(maxsnm*3* sizeof (int) );
  int j,k,l;
  struct gaussian * oldgaussians = (struct gaussian *) malloc(K * sizeof (struct gaussian) );
  gsl_matrix * oldqij = gsl_matrix_alloc(N,K);
  for (kk=0; kk != K; ++kk){
    oldgaussians->mm = gsl_vector_calloc (d);
    oldgaussians->VV = gsl_matrix_calloc (d,d);
    ++oldgaussians;
  }
  oldgaussians -= K;

  //create temporary file to hold convergence info
  FILE *tmpconvfile;
  char* tmpfilename= tmpnam(NULL);
  if (keeplog)
    tmpconvfile= fopen(tmpfilename,"w+");

  //proj_EM
  if (keeplog)
    fprintf(logfile,"#Initial proj_EM\n");
  //printf("Where's the segmentation fault?\n");
  proj_EM(data,N,gaussians,K,fixamp_tmp,fixmean_tmp,fixcovar_tmp,
	  avgloglikedata,tol,maxiter,likeonly,w,
	  keeplog,logfile,convlogfile,noproj,diagerrs,noweight);
  if (keeplog){
    fprintf(logfile,"\n");
    fprintf(convlogfile,"\n");
  }
  //reset fix* vectors
  for (kk = 0; kk != K; ++kk){
    *(fixamp_tmp++) = *(fixamp++);
    *(fixmean_tmp++) = *(fixmean++);
    *(fixcovar_tmp++) = *(fixcovar++);
  }
  fixamp -= K;
  fixamp_tmp -= K;
  fixmean -= K;
  fixmean_tmp -= K;
  fixcovar -= K;
  fixcovar_tmp -= K;

  //Run splitnmerge
  randgen = gsl_rng_alloc(gsl_rng_mt19937);
  bool weretrying = true;
  if (likeonly || splitnmerge == 0 || K < 3)
    ;
  else {
    while (weretrying){
      weretrying = false; /* this is set back to true if an improvement is found */
      //store avgloglike from normal EM and model parameters
      oldavgloglikedata = *avgloglikedata;
      gsl_matrix_memcpy(oldqij,qij);
      for (kk=0; kk != K; ++kk){
	oldgaussians->alpha = gaussians->alpha;
	gsl_vector_memcpy(oldgaussians->mm,gaussians->mm);
	gsl_matrix_memcpy(oldgaussians->VV,gaussians->VV);
	++oldgaussians;
	++gaussians;
      }
      gaussians -= K;
      oldgaussians -= K;
      //Then calculate the splitnmerge hierarchy
      calc_splitnmerge(data,N,gaussians,K,qij,snmhierarchy);
      //Then go through this hierarchy
      kk=0;
      while (kk != splitnmerge && kk != maxsnm){
	//splitnmerge
	j = *(snmhierarchy++);
	k = *(snmhierarchy++);
	l = *(snmhierarchy++);
	splitnmergegauss(gaussians,K,oldqij,j,k,l);
	//partial EM
	//Prepare fixed vectors for partial EM
	for (ll = 0; ll != K; ++ll){
	  *(fixamp_tmp++) = (ll != j && ll != k && ll != l);
	  *(fixmean_tmp++) = (ll != j && ll != k && ll != l);
	  *(fixcovar_tmp++) = (ll != j && ll != k && ll != l);
	}
	fixamp_tmp -= K;
	fixmean_tmp -= K;
	fixcovar_tmp -= K;
	if (keeplog)
	  fprintf(logfile,"#Merging %i and %i, splitting %i\n",j,k,l);
	proj_EM(data,N,gaussians,K,fixamp_tmp,fixmean_tmp,fixcovar_tmp,
		avgloglikedata,tol,maxiter,likeonly,w,keeplog,logfile,
		tmpconvfile,noproj,diagerrs,noweight);
	//reset fix* vectors
	for (ll = 0; ll != K; ++ll){
	    *(fixamp_tmp++) = *(fixamp++);
	    *(fixmean_tmp++) = *(fixmean++);
	    *(fixcovar_tmp++) = *(fixcovar++);
	}
	fixamp -= K;
	fixamp_tmp -= K;
	fixmean -= K;
	fixmean_tmp -= K;
	fixcovar -= K;
	fixcovar_tmp -= K;
	//Full EM
	if (keeplog){
	  fprintf(logfile,"#full EM:\n");
	  fprintf(tmpconvfile,"\n");
	}
	proj_EM(data,N,gaussians,K,fixamp_tmp,fixmean_tmp,fixcovar_tmp,
		avgloglikedata,tol,maxiter,likeonly,w,keeplog,logfile,
		tmpconvfile,noproj,diagerrs,noweight);
	if (keeplog){
	  fprintf(logfile,"\n");
	  fprintf(tmpconvfile,"\n");
	}
	//reset fix* vectors
	for (ll = 0; ll != K; ++ll){
	    *(fixamp_tmp++) = *(fixamp++);
	    *(fixmean_tmp++) = *(fixmean++);
	    *(fixcovar_tmp++) = *(fixcovar++);
	  }
	  fixamp -= K;
	  fixamp_tmp -= K;
	  fixmean -= K;
	  fixmean_tmp -= K;
	  fixcovar -= K;
	  fixcovar_tmp -= K;
	//Better?
	if (*avgloglikedata > oldavgloglikedata){
	  if (keeplog){
	    fprintf(logfile,"#accepted\n");
	    //Copy tmpfile into convfile
	    fseek(tmpconvfile,0,SEEK_SET);
	    while (feof(tmpconvfile) == 0)
	      fputc(fgetc(tmpconvfile),convlogfile);
	    fseek(convlogfile,-1,SEEK_CUR);
	    fclose(tmpconvfile);
	    tmpconvfile= fopen(tmpfilename,"w+");
	  }
	  weretrying = true;
	  ++kk;
	  break;
	}
	else {
	  if (keeplog){
	    fprintf(logfile,"#didn't improve likelihood\n");
	    fclose(tmpconvfile);
	    tmpconvfile= fopen(tmpfilename,"w+");
	  }
	  //revert back to the older solution
	  *avgloglikedata = oldavgloglikedata;
	  /*  gsl_matrix_memcpy(qij,oldqij);*/
	  for (ll=0; ll != K; ++ll){
	    gaussians->alpha = oldgaussians->alpha;
	    gsl_vector_memcpy(gaussians->mm,oldgaussians->mm);
	    gsl_matrix_memcpy(gaussians->VV,oldgaussians->VV);
	    ++oldgaussians;
	    ++gaussians;
	  }
	  gaussians -= K;
	  oldgaussians -= K;
	}
	++kk;
      }
      snmhierarchy -= 3*kk;
    }
  }

  if (keeplog){
    fclose(tmpconvfile);
    fprintf(convlogfile,"\n");
  }


  //Compute some criteria to set the number of Gaussians and print these to the logfile
  int ii;
  int npc,np;
  double pc,aic,mdl;
  if (keeplog){
    //Partition coefficient
    pc=0.;
    for (ii=0; ii != N; ++ii)
      for (kk=0; kk != K; ++kk)
	pc += pow(exp(gsl_matrix_get(qij,ii,kk)),2);
    pc /= N;
    fprintf(logfile,"Partition coefficient \t=\t%f\n",pc);
    //Akaike's information criterion
    npc = 1 + d + d * (d - 1) / 2;
    np = K * npc;
    aic= -2 * ((double) N - 1. - npc - 100)* *avgloglikedata + 3 * np;
    fprintf(logfile,"AIC \t\t\t=\t%f\n",aic);
    //MDL
    mdl = - *avgloglikedata * N + 0.5 * np * log(N);
    fprintf(logfile,"MDL \t\t\t=\t%f\n",mdl);
  }



  
  //Free memory
  gsl_matrix_free(I);
  gsl_matrix_free(qij);
  for (kk = 0; kk != nthreads*K; ++kk){
    gsl_vector_free(bs->bbij);
    gsl_matrix_free(bs->BBij);
    ++bs;
  }
  bs -= nthreads*K;
  free(bs);
  for (kk=0; kk != K*nthreads; ++kk){
    gsl_vector_free(newgaussians->mm);
    gsl_matrix_free(newgaussians->VV);
    ++newgaussians;
  }
  newgaussians= startnewgaussians;
  free(newgaussians);
  gsl_rng_free(randgen);

  gsl_matrix_free(oldqij);
  for (kk=0; kk != K; ++kk){
    gsl_vector_free(oldgaussians->mm);
    gsl_matrix_free(oldgaussians->VV);
    ++oldgaussians;
  }
  oldgaussians -= K;
  free(oldgaussians);
  free(qstarij);
  free(snmhierarchy);
  free(fixamp_tmp);
  free(fixmean_tmp);
  free(fixcovar_tmp);

  return;
}

bool * int2bool(int * a, int K)
{
  bool * x = (bool *) malloc(K * sizeof (bool));
  int kk;
  for (kk = 0; kk != K; ++kk) *(x++) = (bool) *(a++);
  x -=K;
  return x;
}

SEXP proj_gauss_mixtures_IDL(double * ydata, double * ycovar, 
			    double * projection, double * logweights,
			    int *p_N, int *p_dy,
			    double * amp, double * xmean,
			    double * xcovar, int *p_d, int *p_K,
			    int * fixamp_int, int * fixmean_int, int * fixcovar_int,
			    double * avgloglikedata, double *p_tol,
			    int *p_maxiter, int *p_likeonly, double *p_w,
			    int * logfilename, int *p_slen, int *p_splitnmerge,
			    int * convlogfilename, int *p_convloglen,
			    int *p_noprojection, int *p_diagerrors,
			    int *p_noweights) {
  
  // convert variables from R interface
  int N = *p_N, dy = *p_dy, d = *p_d, K = *p_K,
    maxiter = *p_maxiter, slen = *p_slen,
    splitnmerge = *p_splitnmerge, convloglen = *p_convloglen,
    likeonly = *p_likeonly, noprojection = *p_noprojection,
    diagerrors = *p_diagerrors, noweights = *p_noweights;
  double tol = *p_tol, w = *p_w;
  bool * fixamp = int2bool(fixamp_int, K);
  bool * fixmean = int2bool(fixmean_int, K);
  bool * fixcovar = int2bool(fixcovar_int, K);
  //Set up logfiles  
  bool keeplog = true;
  char logname[slen+1];
  char convlogname[convloglen+1];
  int ss;
  if (*logfilename == 0 || likeonly != 0 || slen == 0)
    keeplog = false;
  else {
    for (ss = 0; ss != slen; ++ss)
      logname[ss] = (char) *(logfilename++);
    for (ss = 0; ss != convloglen; ++ss)
      convlogname[ss] = (char) *(convlogfilename++);
    logfilename -= slen;
    convlogfilename -= convloglen;
    logname[slen] = '\0';
    convlogname[convloglen] = '\0';
  }

  if (keeplog) {
    logfile = fopen(logname,"a");
    if (logfile == NULL) return R_NilValue;
    convlogfile = fopen(convlogname,"w");
    if (convlogfile == NULL) return R_NilValue;
  }

  if (keeplog){
    time_t now;
    time(&now);
    fprintf(logfile,"#----------------------------------\n");
    fprintf(logfile,"#\n#%s\n",asctime(localtime(&now)));
    fprintf(logfile,"#----------------------------------\n");
    fflush(logfile);
  }
  
  //Copy everything into the right formats
  struct datapoint * data = (struct datapoint *) malloc( N * sizeof (struct datapoint) );
  struct gaussian * gaussians = (struct gaussian *) malloc (K * sizeof (struct gaussian) );

  bool noproj= (bool) noprojection;
  bool noweight= (bool) noweights;
  bool diagerrs= (bool) diagerrors;
  int ii, jj,dd1,dd2;
  for (ii = 0; ii != N; ++ii){
    data->ww = gsl_vector_alloc(dy);
    if ( ! noweight ) data->logweight = *(logweights++);
    if ( diagerrs ) data->SS = gsl_matrix_alloc(dy,1);
    else data->SS = gsl_matrix_alloc(dy,dy);
    if ( ! noproj ) data->RR = gsl_matrix_alloc(dy,d);
    for (dd1 = 0; dd1 != dy;++dd1)
      gsl_vector_set(data->ww,dd1,*(ydata++));
    if ( diagerrs)
      for (dd1 = 0; dd1 != dy; ++dd1)
	  gsl_matrix_set(data->SS,dd1,0,*(ycovar++));
    else
      for (dd1 = 0; dd1 != dy; ++dd1)
	for (dd2 = 0; dd2 != dy; ++dd2)
	  gsl_matrix_set(data->SS,dd1,dd2,*(ycovar++));
    if ( ! noproj )
      for (dd1 = 0; dd1 != dy; ++dd1)
	for (dd2 = 0; dd2 != d; ++dd2)
	  gsl_matrix_set(data->RR,dd1,dd2,*(projection++));
    else data->RR= NULL;
    ++data;
  }
  data -= N;
  ydata -= N*dy;
  if ( diagerrs ) ycovar -= N*dy;
  else ycovar -= N*dy*dy;
  if ( ! noproj ) projection -= N*dy*d;

  for (jj = 0; jj != K; ++jj){
    gaussians->mm = gsl_vector_alloc(d);
    gaussians->VV = gsl_matrix_alloc(d,d);
    gaussians->alpha = *(amp++);
    for (dd1 = 0; dd1 != d; ++dd1)
      gsl_vector_set(gaussians->mm,dd1,*(xmean++));
    for (dd1 = 0; dd1 != d; ++dd1)
      for (dd2 = 0; dd2 != d; ++dd2)
	gsl_matrix_set(gaussians->VV,dd1,dd2,*(xcovar++));
    ++gaussians;
  }
  gaussians -= K;
  amp -= K;
  xmean -= K*d;
  xcovar -= K*d*d;


  //Print the initial model parameters to the logfile
  int kk;
  if (keeplog){
    fprintf(logfile,"#\n#Using %i Gaussians and w = %f\n\n",K,w);
    fprintf(logfile,"#\n#Initial model parameters used:\n\n");
    for (kk=0; kk != K; ++kk){
      fprintf(logfile,"#Gaussian ");
      fprintf(logfile,"%i",kk);
      fprintf(logfile,"\n");
      fprintf(logfile,"#amp\t=\t");
      fprintf(logfile,"%f",(*gaussians).alpha);
      fprintf(logfile,"\n");
      fprintf(logfile,"#mean\t=\t");
      for (dd1=0; dd1 != d; ++dd1){
	fprintf(logfile,"%f",gsl_vector_get(gaussians->mm,dd1));
	if (dd1 < d-1) fprintf(logfile,"\t");
      }
      fprintf(logfile,"\n");
      fprintf(logfile,"#covar\t=\t");
      for (dd1=0; dd1 != d; ++dd1)
	fprintf(logfile,"%f\t",gsl_matrix_get(gaussians->VV,dd1,dd1));
      for (dd1=0; dd1 != d-1; ++dd1)
	for (dd2=dd1+1; dd2 != d; ++dd2){
	  fprintf(logfile,"%f\t",gsl_matrix_get(gaussians->VV,dd1,dd2));
	}
      ++gaussians;
      fprintf(logfile,"\n#\n");
    }
    gaussians -= K;
    fflush(logfile);
  }
  
  //Then run projected_gauss_mixtures
  proj_gauss_mixtures(data,N,gaussians,K,(bool *) fixamp,
		      (bool *) fixmean, (bool *) fixcovar,avgloglikedata,
		      tol,(long long int) maxiter, (bool) likeonly, w,
		      splitnmerge,keeplog,logfile,convlogfile,noproj,diagerrs,
		      noweight);


  //Print the final model parameters to the logfile
  if (keeplog){
    fprintf(logfile,"\n#Final model parameters obtained:\n\n");
    for (kk=0; kk != K; ++kk){
      fprintf(logfile,"#Gaussian ");
      fprintf(logfile,"%i",kk);
      fprintf(logfile,"\n");
      fprintf(logfile,"#amp\t=\t");
      fprintf(logfile,"%f",(*gaussians).alpha);
      fprintf(logfile,"\n");
      fprintf(logfile,"#mean\t=\t");
      for (dd1=0; dd1 != d; ++dd1){
	fprintf(logfile,"%f",gsl_vector_get(gaussians->mm,dd1));
	if (dd1 < d-1) fprintf(logfile,"\t");
      }
      fprintf(logfile,"\n");
      fprintf(logfile,"#covar\t=\t");
      for (dd1=0; dd1 != d; ++dd1)
	fprintf(logfile,"%f\t",gsl_matrix_get(gaussians->VV,dd1,dd1));
      for (dd1=0; dd1 != d-1; ++dd1)
	for (dd2=dd1+1; dd2 != d; ++dd2){
	  fprintf(logfile,"%f\t",gsl_matrix_get(gaussians->VV,dd1,dd2));
	}
      ++gaussians;
      fprintf(logfile,"\n#\n");
    }
    gaussians -= K;
    fflush(logfile);
  }
  
  //Then update the arrays given to us by IDL
  for (jj = 0; jj != K; ++jj){
    *(amp++) = gaussians->alpha;
    for (dd1 = 0; dd1 != d; ++dd1)
      *(xmean++) = gsl_vector_get(gaussians->mm,dd1);
    for (dd1 = 0; dd1 != d; ++dd1)
      for (dd2 = 0; dd2 != d; ++dd2)
	*(xcovar++) = gsl_matrix_get(gaussians->VV,dd1,dd2);
    ++gaussians;
  }
  gaussians -= K;
  amp -= K;
  xmean -= K*d;
  xcovar -= K*d*d;
  
  //And free any memory we allocated
  for (ii = 0; ii != N; ++ii){
    gsl_vector_free(data->ww);
    gsl_matrix_free(data->SS);
    if ( ! noproj )  gsl_matrix_free(data->RR);
    ++data;
  }
  data -= N;
  free(data);
  
  for (jj = 0; jj != K; ++jj){
    gsl_vector_free(gaussians->mm);
    gsl_matrix_free(gaussians->VV);
    ++gaussians;
  }
  gaussians -= K;
  free(gaussians);

  if (keeplog){
    fclose(logfile);
    fclose(convlogfile);
  }

  return R_NilValue;
}

bool read_IC(char ICfilename[]){
  
  FILE *ICfile;
  if ( (ICfile= fopen(ICfilename,"r")) == NULL){
    printf ("Opening the initial conditions file failed...\n");
    return false;
  }
  char line[100];
  printf("Reading the options section of the initial conditions file...\n");
  while (fgets(line,100,ICfile) != NULL){
    if (line[0] != '#'){
      if (line[0] == '\n') break;
      if (parse_option(line) == false){
	printf("One of the lines in the options section of the initial conditions file is corrupted\n");
	printf("Please check the initial conditions file and try again\n");
	return false;
      }
    }
  }
  
  printf("Successfully read the options\n");

  printf("Reading the initial model parameters...\n");
  

  //Read first block, establish the dimension d of the modeled quantities
  int countd=0;
  double *mmtemp = (double *) malloc (1000000 * sizeof (double));
  while (fgets(line,100,ICfile) != NULL){
    if (line[0] != '#'){
      if (line[0] == '\n') break;
      *(mmtemp++) = atof(line);
      ++countd;
    }
  }
  //now determine d
  d = (int) (-3 + sqrt(9 + 8 * (countd-1)))/2 ; 
  dV = (int) (d*(d+1)/2);

  //allocate the alpha, mm and VV matrices
  int kk;
  for (kk=0; kk != K; ++kk){
    gaussians->mm = gsl_vector_alloc (d);
    gaussians->VV = gsl_matrix_alloc (d,d);;
    ++gaussians;
  }
  gaussians -= K;
  
  //first map the mmtemp values on the right alpha, mm, VV
  mmtemp -= countd;
  (*gaussians).alpha = *(mmtemp++);
  int dd;
  for (dd=0; dd != d; ++dd)
    gsl_vector_set(gaussians->mm,dd,*(mmtemp++));
  int dd1,dd2;
  for (dd1=0; dd1 != d; ++dd1)
    gsl_matrix_set(gaussians->VV,dd1,dd1,*(mmtemp++));
  for (dd1=0; dd1 != d-1; ++dd1)
    for (dd2=dd1+1; dd2 != d; ++dd2){
      gsl_matrix_set(gaussians->VV,dd1,dd2,*mmtemp);
      gsl_matrix_set(gaussians->VV,dd2,dd1,*mmtemp);
      mmtemp++;
    }

  ++gaussians;
  
  //reallocate mmtemp
  mmtemp -= countd;
  mmtemp = (double *) realloc (mmtemp,countd * sizeof (double) );
  if (mmtemp == NULL){
    printf("Error reallocating memory\n");
    printf("Returning\n");
    return false;
  }

  //Then read the rest of the Gaussians.
  for (kk=1; kk != K; ++kk){
    while (fgets(line,100,ICfile) != NULL){
      if (line[0] != '#'){
	if (line[0] == '\n') break;
	*(mmtemp++) = atof(line);
      }
    }
    mmtemp -=countd;
    (*gaussians).alpha = *(mmtemp++);
    for (dd=0; dd != d; ++dd)
      gsl_vector_set(gaussians->mm,dd,*(mmtemp++));
    for (dd1=0; dd1 != d; ++dd1)
      gsl_matrix_set(gaussians->VV,dd1,dd1,*(mmtemp++));
    for (dd1=0; dd1 != d-1; ++dd1)
      for (dd2=dd1+1; dd2 != d; ++dd2){
	gsl_matrix_set(gaussians->VV,dd1,dd2,*mmtemp);
	gsl_matrix_set(gaussians->VV,dd2,dd1,*mmtemp);
	mmtemp++;
      }
    ++gaussians;
    mmtemp -= countd;
  }
  gaussians -= K;
    
  free(mmtemp);

  fclose(ICfile);

  printf("Successfully read initial model parameters from the initial conditions file\n");


  //Print options
  printf("\nThe options are set to:\n");
  printf("K\t\t=\t");
  printf("%i",K);
  printf("\n");
  printf("maxiter\t\t=\t");
  printf("%lli",maxiter);
  printf("\n");
  printf("tol\t\t=\t");
  printf("%f",tol);
  printf("\n");
  printf("splitnmerge\t=\t");
  printf("%i",splitnmerge);
  printf("\n");
  printf("likeonly\t=\t");
  printf("%i",likeonly);
  printf("\n");
  printf("w\t\t=\t");
  printf("%f",w);
  printf("\n");
  printf("fixamp\t\t=\t");
  int ii;
  for (ii=0; ii != K; ++ii){
    printf("%i",*fixampP);
    if (ii < K-1) printf("\t");
    fixampP++;
  }
  fixampP -= K;
  printf("\n");
  printf("fixmean\t\t=\t");
  for (ii=0; ii != K; ++ii){
    printf("%i",*fixmeanP);
    if (ii < K-1) printf("\t");
    fixmeanP++;
  }
  fixmeanP -= K;
  printf("\n");
  printf("fixcovar\t=\t");
  for (ii=0; ii != K; ++ii){
    printf("%i",*fixcovarP);
    if (ii < K-1) printf("\t");
    fixcovarP++;
  }
  fixcovarP -= K;
  printf("\n");


  //Print the initial model parameters
  printf("\nInitial model parameters used:\n\n");
  for (kk=0; kk != K; ++kk){
    printf("Gaussian ");
    printf("%i",kk);
    printf("\n");
    printf("amp\t=\t");
    printf("%f",(*gaussians).alpha);
    printf("\n");
    printf("mean\t=\t");
    for (dd=0; dd != d; ++dd){
      printf("%f",gsl_vector_get(gaussians->mm,dd));
      if (dd < d-1) printf("\t");
    }
    printf("\n");
    printf("covar\t=\t");
    for (dd1=0; dd1 != d; ++dd1)
      printf("%f\t",gsl_matrix_get(gaussians->VV,dd1,dd1));
    for (dd1=0; dd1 != d-1; ++dd1)
      for (dd2=dd1+1; dd2 != d; ++dd2){
	printf("%f\t",gsl_matrix_get(gaussians->VV,dd1,dd2));
      }
    ++gaussians;
    printf("\n\n");
  }
  gaussians -= K;

  return true;
  
}

bool read_data(char inputfilename[]){
  
  FILE *inputfile;
  if ( (inputfile= fopen(inputfilename,"r")) == NULL){
    printf ("Opening the data file failed...\n");
    return false;
  }
  
  //First count the number of datapoints
  N=0;
  while (feof(inputfile) ==0) if (getc(inputfile) == '\n') N++;
  
  printf("%i datapoints found in ",N);
  printf("%s", inputfilename);
  printf("\n");
  
  //Then read the actual data
  fseek(inputfile,0,SEEK_SET);

  //allocate data
  data = (struct datapoint *) malloc(N*sizeof (struct datapoint));
  if (data == NULL){
    printf("Allocation of arrays failed, not enough free memory?\n");
    exit(-1);
  } 
  startdata = data;
    
  //current line holds a maximum of d+dV+d*d elements
  //First read the whole line
  double *curr_line = (double *) calloc(d+dV+d*d, sizeof (double) );
  bool end_line;
  int ii=0,dd=0,di;//di is the dimension of the individual datapoints
  while (feof(inputfile) == 0){
    char curr_value[20]="";
    end_line = read_till_sep(curr_value,inputfile,'|');
    *(curr_line++) = atof(curr_value);
    ++dd;
    if (end_line){
      ++ii;
      curr_line -= dd;
      //Determine the dimension of the datapoint, di, from the equation di+di*(di+1)/2+d*di=dd, or, 
      di = (int) ((-(3 + 2 * d) + sqrt((3 + 2 * d)*(3 + 2 * d)+8 * dd))/2);
      //then write data values to memory
      //first allocate the space for the data
      data->ww = gsl_vector_alloc(di);
      data->SS = gsl_matrix_alloc(di,di);
      data->RR = gsl_matrix_alloc(di,d);
      int dd1,dd2;
      for (dd1=0; dd1 != di; ++dd1)
	gsl_vector_set(data->ww,dd1,*(curr_line++));
      for (dd1=0; dd1 != di; ++dd1)
	gsl_matrix_set(data->SS,dd1,dd1,*(curr_line++));
      for (dd1=0; dd1 != di-1; ++dd1)
	for (dd2=dd1+1; dd2 != di; ++dd2){
	  gsl_matrix_set(data->SS,dd1,dd2,*curr_line);
	  gsl_matrix_set(data->SS,dd2,dd1,*curr_line);
	  curr_line++;
	}
      for (dd1=0; dd1 != di; ++dd1)
	for (dd2=0; dd2 != d; ++dd2)
	  gsl_matrix_set(data->RR,dd1,dd2,*(curr_line++));
      curr_line -= dd;
      dd = 0;
      data++;
      if (ii == N) break;
    }
  }
  
  data = startdata;

  fclose(inputfile);
  free(curr_line);

  return true;
}

bool read_till_sep(char curr_value[],FILE *file,char sep){
  int vv=0;
  bool not_found_sep = true;
  char curr_char;
  while (not_found_sep){
    curr_char = (char) getc(file);
    if (curr_char == sep) break;
    if (curr_char == '\n') return true;
    curr_value[vv++] = curr_char;
  }

  return false;
}

void splitnmergegauss(struct gaussian * gaussians,int K, 
		      gsl_matrix * qij, int j, int k, int l){
  //get the gaussians to be split 'n' merged
  int d = (gaussians->VV)->size1;//dim of mm
  //int partial_indx[]= {-1,-1,-1};/* dummy argument for logsum */
  //bool * dummy_allfixed = (bool *) calloc(K,sizeof(bool));
  //j,k,l gaussians
  struct gaussian gaussianj, gaussiank, gaussianl;
  gaussianj.mm = gsl_vector_alloc(d);
  gaussianj.VV = gsl_matrix_alloc(d,d);
  gaussiank.mm = gsl_vector_alloc(d);
  gaussiank.VV = gsl_matrix_alloc(d,d);
  gaussianl.mm = gsl_vector_alloc(d);
  gaussianl.VV = gsl_matrix_alloc(d,d);
  
  gsl_matrix * unitm = gsl_matrix_alloc(d,d);
  gsl_matrix_set_identity(unitm);
  gsl_vector * eps = gsl_vector_alloc(d);
  double qjj,qjk,detVVjl;
  int kk;
  for (kk = 0; kk != K; ++kk){
    if (kk == j){
      gaussianj.alpha = gaussians->alpha;
      gsl_vector_memcpy(gaussianj.mm,gaussians->mm);
      gsl_matrix_memcpy(gaussianj.VV,gaussians->VV);
      qjj = exp(logsum(qij,j,false));//,dummy_allfixed));
    }
    if (kk == k){
      gaussiank.alpha = gaussians->alpha;
      gsl_vector_memcpy(gaussiank.mm,gaussians->mm);
      gsl_matrix_memcpy(gaussiank.VV,gaussians->VV);
      qjk = exp(logsum(qij,k,false));//,dummy_allfixed));
    }
    if (kk == l){
      gaussianl.alpha = gaussians->alpha;
      gsl_vector_memcpy(gaussianl.mm,gaussians->mm);
      gsl_matrix_memcpy(gaussianl.VV,gaussians->VV);
    }
    ++gaussians;
  }
  gaussians -= K;

  //merge j & k
  gaussianj.alpha += gaussiank.alpha;
  if (qjk == 0. && qjj == 0){
    gsl_vector_add(gaussianj.mm,gaussiank.mm);
    gsl_vector_scale(gaussianj.mm,0.5);
    gsl_matrix_add(gaussianj.VV,gaussiank.VV);
    gsl_matrix_scale(gaussianj.VV,0.5);
  }
  else{
    gsl_vector_scale(gaussianj.mm,qjj/(qjj+qjk));
    gsl_vector_scale(gaussiank.mm,qjk/(qjj+qjk));
    gsl_vector_add(gaussianj.mm,gaussiank.mm);
    gsl_matrix_scale(gaussianj.VV,qjj/(qjj+qjk));
    gsl_matrix_scale(gaussiank.VV,qjk/(qjj+qjk));
    gsl_matrix_add(gaussianj.VV,gaussiank.VV);
  }

  //split l
  gaussianl.alpha /= 2.;
  gaussiank.alpha = gaussianl.alpha;
  detVVjl = bovy_det(gaussianl.VV);
  detVVjl= pow(detVVjl,1./d);
  gsl_matrix_scale(unitm,detVVjl);
  gsl_matrix_memcpy(gaussiank.VV,unitm);
  gsl_matrix_memcpy(gaussianl.VV,unitm);
  gsl_vector_memcpy(gaussiank.mm,gaussianl.mm);
  bovy_randvec(eps,d,sqrt(detVVjl));
  gsl_vector_add(gaussiank.mm,eps);
  bovy_randvec(eps,d,sqrt(detVVjl));
  gsl_vector_add(gaussianl.mm,eps);
  
  //copy everything back into the right gaussians
  for (kk = 0; kk != K; ++kk){
    if (kk == j){
      gaussians->alpha = gaussianj.alpha;
      gsl_vector_memcpy(gaussians->mm,gaussianj.mm);
      gsl_matrix_memcpy(gaussians->VV,gaussianj.VV);
    }
    if (kk == k){
      gaussians->alpha = gaussiank.alpha;
      gsl_vector_memcpy(gaussians->mm,gaussiank.mm);
      gsl_matrix_memcpy(gaussians->VV,gaussiank.VV);
    }
    if (kk == l){
      gaussians->alpha = gaussianl.alpha;
      gsl_vector_memcpy(gaussians->mm,gaussianl.mm);
      gsl_matrix_memcpy(gaussians->VV,gaussianl.VV);
    }
    ++gaussians;
  }
  gaussians -= K;

  //cleanup
  gsl_matrix_free(unitm);
  gsl_vector_free(eps);
  //free(dummy_allfixed);

  return ;
}

bool write_model(char outputfilename[]){

  FILE *outputfile;
  if ( (outputfile= fopen(outputfilename,"w+")) == NULL){
    printf ("Opening the data file failed...\n");
    return false;
  }  
  
  //loop over the gaussians and write the results to a file
  int kk,dd,dd1,dd2;
  for (kk=0; kk != K; ++kk){
    printf("#K=%i\n",kk+1);
    printf("%g\n",gaussians->alpha);
    for (dd=0; dd != d; ++dd)
      printf("%g\n",gsl_vector_get(gaussians->mm,dd));
    for (dd1=0; dd1 != d; ++dd1)
      printf("%g\n",gsl_matrix_get(gaussians->VV,dd1,dd1));
    for (dd1=0; dd1 != d-1; ++dd1)
      for (dd2=dd1+1; dd2 != d; ++dd2){
	printf("%g\n",gsl_matrix_get(gaussians->VV,dd1,dd2));
      }
    printf("\n");
    ++gaussians;
  }
  
  gaussians -= K;

  return true;
}
