	#ifndef RGBTOYUV
#define RGBTOYUV

#include <cuda_fp16.h>

#define CHANNELS 3 // we have 3 channels corresponding to RGB
__global__ void YUVandCItoGray(float * grayImage, float * yuvImage, float * rgbImage, float * redData, float * greenData, float * blueData, int width, int height) {//this function combineds yuv, grayscale, and color invarient image processing and also assigns R G B arrays for coalesced reads in proc 5 and reduces register trafic


int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if(x < width && y < height){ 
	int grayOffset = y * width + x;
 	int rgbOffset = (y * width + x)*3 ;
	
	float r = rgbImage[rgbOffset];
	float g = rgbImage[rgbOffset + 1];
	float b = rgbImage[rgbOffset + 2];
	

	//color invariance image calculations
	float ci1 = atan(r / max(g,b));
	float ci2 = atan(g / max(r,b));
	float ci3 = atan(b / max(r,g));
	

	//yuv image claculations
	/*float y = (16 + (.299*r) + (.587*g) + (.114*b))/255;
	//float u = 128 -.168736*r -.331364*g + .5*b;	
	float u = (128 + (112*b) - (37.397*r) - (74.203*g))/255;//prof suggestion
	float v = (128 + (.5*r) - (.418688*g) - (.081312*b))/255;*/

	
	//group 2 solution
	float y = (16 + (65.481*r) + (128.553*g) + (24.966*b))/255;
	//float u = 128 -.168736*r -.331364*g + .5*b;	
	float u = (128 + (-37.797*r) + (-74.203 * g) + (112*b))/255;//prof suggestion
	float v = (128 + (112*r) + (-93.786*g) + (-18.214*b))/255;
	

	//boundary checks
	if(ci1<0){ci1=0;}
	if(ci2<0){ci2=0;}
	if(ci3<0){ci3=0;}

	//yuv image writes
	yuvImage[rgbOffset] = y;
	yuvImage[rgbOffset + 1] = u;
	yuvImage[rgbOffset + 2] = v;
	//yuvImage[grayOffset] = (128 + (112*b) - (37.397*r) - (74.203*g))/255;//as per prof

	__syncthreads();
	//greyscale image global wirtes
	grayImage[grayOffset] = 0.21f*ci1 + 0.71f*ci2 + 0.07f*ci3; //greyscale output image
	__syncthreads();
	
}

/*
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if(x < width && y < height){
	int rgbOffset = (y * width + x)*3 ;
	
	float r = rgbImage[rgbOffset];
	float g = rgbImage[rgbOffset + 1];
	float b = rgbImage[rgbOffset + 2];
	
	//CI1 = arctan(R * max(G,B)); CI2 = arctan(G * max(R,B)); CI3 = (B * max(R,G));
	float ci1 = atan(r / max(g,b));
	float ci2 = atan(g / max(r,b));
	float ci3 = atan(b / max(r,g));

	if(ci1<0){ci1=0;}
	if(ci2<0){ci2=0;}
	if(ci3<0){ci3=0;}

	yuvImage[rgbOffset] = ci1;
	yuvImage[rgbOffset + 1] = ci2;
	yuvImage[rgbOffset + 2] = ci3;
}*/

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
