__global__ void sumProc5(
	float* const deviceBlockSums, 
	float* const deviceInput,
	const unsigned int size)
{
	extern __shared__ float s_out[];

	unsigned int max_elems_per_block = blockDim.x * 2;
	unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	
	// Zero out shared memory
	// Especially important when padding shmem for
	//  non-power of 2 sized input
	s_out[threadIdx.x] = 0.0;
	s_out[threadIdx.x + blockDim.x] = 0.0;

	__syncthreads();

	// Copy deviceInput to shared memory per block
	if (glbl_tid < size)
	{
		s_out[threadIdx.x] = deviceInput[glbl_tid];
		if (glbl_tid + blockDim.x < size)
			s_out[threadIdx.x + blockDim.x] = deviceInput[glbl_tid + blockDim.x];
	}
	__syncthreads();

	// Actually do the reduction
	for (unsigned int s = blockDim.x; s > 0; s >>= 1) {
		if (tid < s) {
			s_out[tid] += s_out[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0){
		deviceBlockSums[blockIdx.x] = s_out[0];
	}
}

/*__device__ void lastWarp(volatile float * sdata, int tidErShadow, int tidErLight, int tidShadowRed, int tidShadowGreen, int tidShadowBlue, int tidLightRed, int tidLightGreen, int tidLightBlue, float * deviceErodedShadowOut, float * deviceErodedLightOut, float * deviceShadowRedArrayOut, float * deviceShadowGreenArrayOut, float * deviceShadowBlueArrayOut, float * deviceLightRedArrayOut, float * deviceLightGreenArrayOut, float * deviceLightBlueArrayOut){ //this function is used to do the sums in the last warp, removing the __syncthreads()
    
}


__global__ void sumProc5SmallCalc(float * deviceErodedShadowIn, float * deviceErodedLightIn, float * deviceShadowRedArrayIn, float * deviceShadowGreenArrayIn, float * deviceShadowBlueArrayIn, float * deviceLightRedArrayIn, float * deviceLightGreenArrayIn, float * deviceLightBlueArrayIn, float * deviceErodedShadowOut, float * deviceErodedLightOut, float * deviceShadowRedArrayOut, float * deviceShadowGreenArrayOut, float * deviceShadowBlueArrayOut, float * deviceLightRedArrayOut, float * deviceLightGreenArrayOut, float * deviceLightBlueArrayOut, int size){//improved from sumProc5SmallCalcV1 by taking advantage of final warp calculations without __syncthreads(), did not use lastWarp function, wrote it directly in this function to speed up
	//assigning and summing 8 arrays, shared memory is perfectly maxed out!
  
  


}

__global__ void sumProc5SmallCalcV1(float * deviceErodedShadowIn, float * deviceErodedLightIn, float * deviceShadowRedArrayIn, float * deviceShadowGreenArrayIn, float * deviceShadowBlueArrayIn, float * deviceLightRedArrayIn, float * deviceLightGreenArrayIn, float * deviceLightBlueArrayIn, float * deviceErodedShadowOut, float * deviceErodedLightOut, float * deviceShadowRedArrayOut, float * deviceShadowGreenArrayOut, float * deviceShadowBlueArrayOut, float * deviceLightRedArrayOut, float * deviceLightGreenArrayOut, float * deviceLightBlueArrayOut, int size){//strided index, no divergence, unrolled loop, and shared memory is maxed with this version
	//assigning and summing 8 arrays in shared memory, shared memory is perfectly maxed out
  


}



__global__ void sumProc5(float * deviceErodedShadowIn, float * deviceErodedLightIn, float * deviceShadowRedArrayIn, float * deviceShadowGreenArrayIn, float * deviceShadowBlueArrayIn, float * deviceLightRedArrayIn, float * deviceLightGreenArrayIn, float * deviceLightBlueArrayIn, float * deviceErodedShadowOut, float * deviceErodedLightOut, float * deviceShadowRedArrayOut, float * deviceShadowGreenArrayOut, float * deviceShadowBlueArrayOut, float * deviceLightRedArrayOut, float * deviceLightGreenArrayOut, float * deviceLightBlueArrayOut, int size){//strided index no divergence and all threads initially add 2 so they work and unrolled loop
	//storing and summing 8 arrays in shared memory now, shared memory is perfectly maxed out!!!
 

}



__global__ void sumProc5Fabian(float * deviceErodedShadowIn, float * deviceErodedLightIn, float * deviceShadowRedArrayIn, float * deviceShadowGreenArrayIn, float * deviceShadowBlueArrayIn, float * deviceLightRedArrayIn, float * deviceLightGreenArrayIn, float * deviceLightBlueArrayIn, float * deviceErodedShadowOut, float * deviceErodedLightOut, float * deviceShadowRedArrayOut, float * deviceShadowGreenArrayOut, float * deviceShadowBlueArrayOut, float * deviceLightRedArrayOut, float * deviceLightGreenArrayOut, float * deviceLightBlueArrayOut, int size){

}






__global__ void sumProc5Reverse(float * d_out, float * deviceInput, int size){//strided index with no divergence, unrolled
  


}


__global__ void sumProc5NoDiverge(float * d_out, float * deviceInput, int size){//strided index with no divergence
 
}



__global__ void sumProc5FirstStride(float * d_out, float * deviceInput, int size){//strided index with divergence

  

		
}


__global__ void sumNaive(float * d_out, float * deviceInput, int size){//serial global version

  // naive reduction
  
    
		
}
*/


