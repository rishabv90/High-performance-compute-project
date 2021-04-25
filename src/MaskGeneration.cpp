#ifndef MASK_GENERATION_CPP
#define MASK_GENERATION_CPP

#include <math.h>

#define NUM_BINS 256 			// Every image we deal with should be 8-bit unsigned
#define MAX_LOOK_BACK 3			// The number of bins we go in each direction to find the max

/*
 * MaskGeneration.cpp: Creating a mask using the threshold returned
 * from Otsu's method. The algorithm is as follows (terminology is 
 * used from both the Otsu paper and MATLAB's implementation):
 * 1. counts = histogramGeneration(inputImage)
 * 2. p = counts / sum(counts)
 * 	- sum(counts) is just the nunber of pixels, so we can just 
 * 	divide each count by the number of pixels.
 * 3. omega = cumsum(p)
 * 4. mu = cumsum(p .* (1:num_bins))
 * 5. mu_t = mu(end)
 * 6. sigma_b_squared = (mu_t * omega - mu).^2 / (omega .* (1 - omega))
 * 7. threshold = argmax(sigma_b_squared)
 * 8. iterate over image, set all pixels over threshold to 1 and all images
 *    less than threshold to zero.
 */

/*
 * histogramKernel: The name is very intuitive :-)
 * Version 0:
 * 	- Only uses global memory
 * Version 1:
 * 	- Only uses global memory
 * 	- Coalesced global memory accesses
 * Version 2 (NOT COMPLETE, COPIED FROM HW4):
 * 	- Shared memory
 */

#define HISTOGRAM_VERSION 2	// Control which version is implemented

#if HISTOGRAM_VERSION == 0
__global__ void histogramKernel(float *input, unsigned int *bins, unsigned int num_elements) {

  
}
#elif HISTOGRAM_VERSION == 1
__global__ void histogramKernel(float *input, unsigned int *bins, unsigned int num_elements) {

  	
}
#elif HISTOGRAM_VERSION == 2
__global__ void histogramKernel(float *input, unsigned int *bins, unsigned int num_elements) { 

	

}
#endif // HISTOGRAM_VERSION

/*
 * cumSumOne: Cumulative sum operation, the same as the scan operation 
 * covered in class. However this is `cumSumOne` as the first cumulative sum
 * in Otsu's method is supposed to be using the histogram divided by the sum 
 * of the entire histogram (see the MATLAB implementation of Otsu's method). 
 * Therefore, after performing the cumulative sum I divide each number by the 
 * number of pixels.
 * Version 0:
 * 	- Scan operation covered in class. 
 */

#define CUMULATIVE_SUM_ONE_VERSION 0	// Control which version is implemented

#if CUMULATIVE_SUM_ONE_VERSION == 0
__global__ void cumSumOne(unsigned int* counts, float* omega, unsigned int num_elements) {
	

}
#endif // CUMULATIVE_SUM_ONE_VERSION

/*
 * cumSumTwo: Cumulative sum operation, the same as the scan operation 
 * covered in class. However this is `cumSumTwo` as the second cumulative sum
 * in Otsu's method is supposed to be using the histogram divided by the sum 
 * of the entire histogram as well as multiply every bin by its bin ID 
 * (see the MATLAB implementation of Otsu's method). 
 * Therefore, before performing the cumulative sum, I multiply each number by 
 * its bin ID and after performing the cumulative sum I divide each number by the 
 * number of pixels.
 * Version 0:
 * 	- Scan operation covered in class. 
 */

#define CUMULATIVE_SUM_TWO_VERSION 0	// Control which version is implemented

#if CUMULATIVE_SUM_TWO_VERSION == 0
__global__ void cumSumTwo(unsigned int* counts, float* mu, unsigned int num_elements) {
	__shared__ int XY[NUM_BINS];
	

}
#endif // CUMULATIVE_SUM_TWO_VERSION

/*
 * compSigmaBSquared: Computes sigma_b_squared for each element in
 * the histogram. This value is used to determine the best threshold
 * to split between foreground and background.
 * Version 0:
 * 	- Probably the best I can do. Algorithm is data parallel making
 * 	the implementation pretty simple.
 */

#define COMP_SIGMA_B_SQUARED_VERSION 0	// Control which version is implemented

#if COMP_SIGMA_B_SQUARED_VERSION == 0
__global__ void compSigmaBSquared(float* sigma_b_squared, float* omega, float* mu) {
	

}
#endif // COMP_SIGMA_B_SQUARED_VERSION

/*
 * argmax: Returns the argmax of an array.
 * Version 0:
 * 	- Serial implementation. NOTE: Meant to be run with one thread
 * Version 1:
 * 	- Each thread pulls data into the shared memory, and then 
 * 	compares their value with the two neighboring values. If 
 * 	the current value is greater than the neightbors, it 
 * 	is returned as the maximum. NOTE: This assumes that the 
 * 	input is concave. If it is not, it will stochastically
 * 	return some local maximum.
 */

#define ARGMAX_VERSION 1	// Control which version is implemented

#if ARGMAX_VERSION == 0
__global__ void argmax(float* retId, float* input) {
	
	
}
#elif ARGMAX_VERSION == 1
__global__ void argmax(float* retId, float* input) {
	
	
}

#endif // ARGMAX_VERSION

/*
 * maskGeneration: Outputs a binary mask given an input image and a threshold.
 * Version 0:
 * 	- Pretty vanilla. Should look to see if warp divergence is an issue.
 */

#define MASK_GENERATION_VERSION 0	// Control which version is implemented

#if MASK_GENERATION_VERSION == 0
__global__ void maskGeneration(float* input, float* output, float* threshold, int width, int height, int subtractVal) {
	
}
#endif // MASK_GENERATION_VERSION

#endif // MASK_GENERATION_CPP
