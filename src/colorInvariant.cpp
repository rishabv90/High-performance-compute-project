#ifndef COLOR_INVARIANT
#define COLOR_INVARIANT

#include <cuda_fp16.h>//CUDA MATH, faster!


//CI1 = arctan(R * max(G,B)); CI2 = arctan(G * max(R,B)); CI3 = (B * max(R,G));
#define CHANNELS 3 // we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__ void colorInvariant(float * ciImage, float * rgbImage, int width, int height) {
	//calculates color invarient of image


 
}

#endif

