/*
Based on MM.c provided by Dr. Srinivasan on COP5522, Oct-Dec 2019
This version is the optimized one, traverseing the arrays in row order, causing  efficient cache access.
 
This version was done after assignment submittal, and incorporates vector intrinsics

Modification to the original program:
 -removed command line arguments since the assignment requirement was to have none
 -N is now set using MATRIX_SIZE definition
    - mat[MATRIX_SIZE * MATRIX_SIZE]
    - vec[MATRIX_SIZE]
    - res[MATRIX_SIZE]
 - removed k loop--not needed since vector and result are one dimensional
 - segfault locally and on cs-ssh w/ N>45000
 - updated Gflops/s calc based on N*N work done instead of N*N*N
*/

#include <stdio.h>
#include "microtime.h"
#include <string.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <mmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>

#define MATRIX_SIZE 10000

int main()
{
    int N = MATRIX_SIZE;
    int i, j;
    float *mat, *vec, *res, result=0.0;
    double time1, time2;
    //float temp[4] __attribute__((aligned(16))); //not used, saving as reference
    __m128 X, Y, Z;
    
    //these don't seem to be needed w/ posix_memalign
    //mat = (void *) malloc(N*N*sizeof(mat[0]));
    //vec = (void *) malloc(N*sizeof(vec[0]));
    //res = (void *) malloc(N*sizeof(res[0]));
                                
    posix_memalign((void **) &mat, 16, N*N*sizeof(mat[0]));
    posix_memalign((void **) &vec, 16, N*sizeof(vec[0]));
    posix_memalign((void **) &res, 16, N*sizeof(res[0]));
    
    if(mat==0 || vec==0 || res==0)
    {
      fprintf(stderr, "Memory allocation failed in file %s, line %d\n", __FILE__, __LINE__);
      exit(1);
    }

    memset(res, 0, N*sizeof(res[0])); //sets results array to zeroes
    
    for(i=0; i<N*N; i++) //sets mat and vec arrays to 1.0
        mat[i] = 1.0;
    for(i=0; i<N; i++)
        vec[i] = 1.0;
    
    time1 = microtime(); //starts time for loop
    for(i=0; i<N; i+=4) //
    {
        Z = _mm_load_ps(&res[i]);  //load current res[i]+3 into Z
        Y = _mm_load_ps(&vec[i]);  //load vec[i]+3 into Y
        for(j=0; j<N; j+=4)
        {
            X = _mm_load_ps(&mat[i*N+j]);  //load mat index+3 into X
            Z = _mm_add_ps(Z, _mm_mul_ps(X, Y)); //sum X*Y
        }
        Z = _mm_hadd_ps(Z, Z); //stackoverflow answer
        Z = _mm_hadd_ps(Z, Z); //you end up w/ z0+z1+z2+z3 in all 4 z floats
        _mm_store_ps(&res[i], Z); //store one of those floats in res[i]
    }
    time2 = microtime();
  
    printf("Time = %g us\tTimer Resolution = %g us\tPerformance = %g Gflop/s\n", time2-time1, get_microtime_resolution(), 2*N*N*1e-3/(time2-time1));
  
    for(i=0; i<N; i++) //Check correctness
        result += res[i];
  
    printf("Sum of matrix elements = %g\tExpected value = %g\n", result, (double) N*N);
  
    return 0;
}
