#ifndef RGBTOYUV
#define RGBTOYUV

#include <cuda_fp16.h>

#define CHANNELS 3 // we have 3 channels corresponding to RGB
__global__ void YUVandCItoGray(float * grayImage, float * yuvImage, float * rgbImage, float * redData, float * greenData, float * blueData, int width, int height) {//this function combineds yuv, grayscale, and color invarient image processing and also assigns R G B arrays for coalesced reads in proc 5 and reduces register trafic

/*

//simple copy ??
int x = threadIdx.x + blockIdx.x * blockDim.x; //col
int y = threadIdx.y + blockIdx.y * blockDim.y; //row

if(x < width && y < height){
	int rgbOffset = (y * width + x);
	
	float r = rgbImage[rgbOffset];
	float g = rgbImage[rgbOffset + 1];
	float b = rgbImage[rgbOffset + 2];

	yuvImage[rgbOffset] = r;
	yuvImage[rgbOffset + 1] = g;
	yuvImage[rgbOffset + 2] = b;
	

}
*/

/*
//printf("width = %d and height = %d\n\n ",width,height);
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if(x < width && y < height){ //thread divergence here
	//int grayOffset = y * width + x;
 	int rgbOffset = (y * width + x)*3 ;
	
	float r = rgbImage[rgbOffset];
	float g = rgbImage[rgbOffset + 1];
	float b = rgbImage[rgbOffset + 2];

	float ci1 = 0;
	float ci2 = 0;
	float ci3 = 0; 	



	if(atan(r * max(g,b)) < 0){
		ci1 = 0; 	
	}else {
		ci1 = atan(r * max(g,b));
	}


	if(atan(g * max(r,b)) < 0){
		ci2 = 0; 	
	}else {
		ci2 = atan(g * max(r,b));
	}


	if(atan(b * max(r,g)) < 0){
		ci3 = 0; 	
	}else {
		ci3 = atan(b * max(r,g));
	}


	yuvImage[rgbOffset] = ci1;
	yuvImage[rgbOffset + 1] = ci2;
	yuvImage[rgbOffset + 2] = ci3;  */



//greyscale

int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if(x < width && y < height){ //thread divergence here // greyscale 
	int grayOffset = y * width + x;
 	int rgbOffset = (y * width + x)*3 ;
	
	float r = rgbImage[rgbOffset];
	float g = rgbImage[rgbOffset + 1];
	float b = rgbImage[rgbOffset + 2];

	//grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b; //greyscale output image

	
	float ci1 = atan(r * max(g,b));
	float ci2 = atan(g * max(r,b));
	float ci3 = atan(b * max(r,g));

	yuvImage[rgbOffset] = ci1;
	yuvImage[rgbOffset + 1] = ci2;
	yuvImage[rgbOffset + 2] = ci3;

	__syncthreads();
	
	grayImage[grayOffset] = 0.21f*ci1 + 0.71f*ci2 + 0.07f*ci3; //greyscale output image

	__syncthreads();
}

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
