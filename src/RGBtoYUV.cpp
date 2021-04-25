#ifndef RGBTOYUV
#define RGBTOYUV

#include <cuda_fp16.h>

#define CHANNELS 3 // we have 3 channels corresponding to RGB
__global__ void YUVandCItoGray(float * grayImage, float * yuvImage, float * rgbImage, float * redData, float * greenData, float * blueData, int width, int height) {//this function combineds yuv, grayscale, and color invarient image processing and also assigns R G B arrays for coalesced reads in proc 5 and reduces register trafic
	
}



__global__ void YUVandCItoGrayV2(float * grayImage, float * yuvImage, float * rgbImage, float * redData, float * greenData, float * blueData, int width, int height) {//this function combineds yuv, grayscale, and color invarient image processing
	
	
}

__global__ void YUVandCItoGrayV1(float * grayImage, float * yuvImage, float * rgbImage, int width, int height) {//this function combineds yuv, grayscale, and color invarient image processing
	
}

__global__ void RGBtoYUVSharedCoal(float * yuvImage, float * rgbImage, int width, int height) {//this is tries to use shared memory to increase speed, however is it slower
  
}




__global__ void RGBtoYUVOriginal(float * yuvImage, float * rgbImage, int width, int height) {//this is the originial rgb to ruv kernel
	
}
#endif
