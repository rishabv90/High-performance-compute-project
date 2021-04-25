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

#define HISTOGRAM_VERSION 0	// Control which version is implemented

#if HISTOGRAM_VERSION == 0
__global__ void histogramKernel(float *input, unsigned int *bins, unsigned int num_elements) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; //Get initial index based on thread id
    int stride = blockDim.x * gridDim.x; //Get stride value
    while(i < num_elements){ //Iterate over elements that this thread will process and increment histogram accordingly
        atomicAdd(&(bins[(int)input[i]]),1);
        i += stride;
    }
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
	__shared__ int XY[NUM_BINS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if(i < NUM_BINS)
        XY[tid] = counts[i];
    for(int stride = 1; stride <= tid; stride = stride * 2){
        __syncthreads();
        int in1 = XY[tid - stride];
        __syncthreads();
        XY[tid] += in1;
        
    }
    __syncthreads();
    if(i < NUM_BINS)
        omega[i] = XY[tid] / num_elements;

    __syncthreads();
    printf("OMEGA VALUES\r\n###############\r\n");
    printf("%f\r\n",omega[i]);
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if(i < NUM_BINS)
        XY[tid] = counts[i];
    for(int stride = 1; stride <= tid; stride = stride * 2){
        __syncthreads();
        int in1 = XY[tid - stride];
        __syncthreads();
        XY[tid] += in1 * (tid - stride);       
    }
    __syncthreads();
    if(i < NUM_BINS)
        mu[i] = XY[tid] / num_elements;

    __syncthreads();
    printf("MU VALUES\r\n###############\r\n");
    printf("%f\r\n",mu[i]);
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

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < NUM_BINS){
        
        sigma_b_squared[i] = pow(omega[i] - mu[i], 2) / (omega[i] * (1 - omega[i]));
    }

	/*__shared__ int shared_omega[NUM_BINS];
    __shared__ int shared_mu[NUM_BINS];
    __shared__ int shared_sigma_b_squared[NUM_BINS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if(i < NUM_BINS)
        shared_omega[tid] = omega[i];
        shared_mu[tid] = mu[i];
    for(int stride = 1; stride <= tid; stride = stride * 2){
        __syncthreads();
        int omega_val = shared_omega[tid - stride];
        int mu_val = shared_mu[tid - stride];
        __syncthreads();
        shared_sigma_b_squared[tid] += pow(omega_val - mu_val, 2) / (omega_val * (1 - omega_val));
    }
    __syncthreads();
    if(i < NUM_BINS)
        sigma_b_squared[i] = shared_sigma_b_squared[tid];

    __syncthreads();
    printf("SIGMA B SQUAred VALUES\r\n###############\r\n");
    printf("%f\r\n",sigma_b_squared[i]);*/
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

#define ARGMAX_VERSION 0	// Control which version is implemented

#if ARGMAX_VERSION == 0
__global__ void argmax(float* retId, float* input) {
    
    float max = input[0];

    for(int i = 0; i < NUM_BINS; i++){
        if(input[i] > max){
            max = input[i];
        }
    }

    *retId = max;
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
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col >= 0 && col < width && row >= 0 && row < height)
    output[row * width + col] = input[row * width + col] > *threshold ? input[row * width + col] : input[row * width + col] - subtractVal;
}
#endif // MASK_GENERATION_VERSION

#endif // MASK_GENERATION_CPP
