/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

#include <stdlib.h>
#include <stdio.h>
//#include <string.h>
//#include <math.h>
//#include "fstream.h"
//#include <time.h>

// includes, project
//#include <cutil_inline.h>
//#include <cuda.h>


/*long int testSize = 5;
long int trainSize = 1024;
const int blockSize = 380;
const int window_size = 1024;*/

extern __shared__ float array[];

_global__ void calc_dtw (unsigned SRC_LEN,unsigned TRG_LEN,unsigned TRG_COT,
    float *S, float *TT, float *Result, unsigned int *window, unsigned window_size)
{
    // blockDim.x = TRG_LEN*TRG_COT
    float* path_h1 = (float*)array;
    float* path_h2 = (float*)&path_h1[blockDim.x];
    float* dist    = (float*)&path_h2[blockDim.x];

    //int blockId = (blockIdx.y*gridDim.x + blockIdx.x);
    //int G_idx   = blockId *blockDim.x +threadIdx.x;
    float *T =     TT + (blockIdx.y*gridDim.x +blockIdx.x) *blockDim.x;
    float *R = Result + (blockIdx.y*gridDim.x +blockIdx.x) *TRG_COT;
    
    float *ex;
    int i,j;
    
    int sub_x = threadIdx.x % TRG_LEN;
    int x_cot = threadIdx.x / TRG_LEN;

    // first line speical, do first
    // 1. paralle, first line's every element's dist
    dist[threadIdx.x] = (S[0] -T[threadIdx.x])
                       *(S[0] -T[threadIdx.x]);
    __syncthreads();
    // 2. serie, first line's every element's serie's dist
    if (sub_x == 0){
        path_h1[threadIdx.x] = dist[threadIdx.x];
        for (i=1; i <TRG_LEN; i++) {
            path_h1[i +threadIdx.x] = path_h1[i-1 +threadIdx.x]
                                    +    dist[i   +threadIdx.x];
        }
    }
    __syncthreads();
    
    for (i=1; i <SRC_LEN; i++){ // do circle
        // 1. paralle, calc itself's DISTANCE, for speed follow progress
        //    use memeory to rise speed
        dist[threadIdx.x] = (S[i] -T[threadIdx.x])
                           *(S[i] -T[threadIdx.x]);
        __syncthreads();
        // 2. paralle, get from upper line's "up","left-up"- the min dist
        if (sub_x == 0) // FIRST element speical, add DISTANCE here
            path_h2[threadIdx.x] =  path_h1[threadIdx.x] + dist[threadIdx.x];
        else
            path_h2[threadIdx.x] = min (path_h1[threadIdx.x],
                                        path_h1[threadIdx.x-1]);
        __syncthreads();
        // 2. serie, compare to left(front) element with myself, get the less value
        if (sub_x == 0) // the first element had plused dist, so can use here
            for (j=1; j<TRG_LEN; j++) 
                path_h2[j +threadIdx.x] = min(path_h2[j   +threadIdx.x], 
                                              path_h2[j-1 +threadIdx.x])
                                        + dist[j +threadIdx.x];
        __syncthreads();
        ex      = path_h2;
        path_h2 = path_h1;
        path_h1 = ex;
        __syncthreads();
    }
    // when all done, can return the result
    if (sub_x == 0){
        R[x_cot] = sqrt(ex[threadIdx.x +TRG_LEN -1]);
    }
}

__global__ void dtw_kernel(unsigned len_a, unsigned len_b, float *a, float *b, 
float *cost_matrix, int *path_matrix, int *window, unsigned window_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int i = idx / len_b;
        int j = idx % len_b;

        if (i >= len_a || j >= len_b)
            return;
        int wi;
        int wj; //= (window_size/2)+1;
        int len_w = (window_size/2);
        for (wi=0; wi<len_w; wi++){
            for(wj=len_w+1; wj<window_size; wj++){
                if (i == window[wi] && j == window[wj]){
                    
                }
            }
        }
        if (abs(i - j) <= window_size) {
            float cost = fabs(a[i] - b[j]);  // Example cost function, modify as needed
            cost_matrix[i * len_b + j] = cost;

            // Initialize the path matrix
            if (i == 0 && j == 0) {
                path_matrix[i * len_b + j] = -1;
            } else {
                path_matrix[i * len_b + j] = -1;  // Initialize to an invalid value
            }
        }
    }