__device__ void lastWarp(volatile float * sdata, int tidErShadow, int tidErLight, int tidShadowRed, int tidShadowGreen, int tidShadowBlue, int tidLightRed, int tidLightGreen, int tidLightBlue, float * deviceErodedShadowOut, float * deviceErodedLightOut, float * deviceShadowRedArrayOut, float * deviceShadowGreenArrayOut, float * deviceShadowBlueArrayOut, float * deviceLightRedArrayOut, float * deviceLightGreenArrayOut, float * deviceLightBlueArrayOut){ //this function is used to do the sums in the last warp, removing the __syncthreads()
    
}


__global__ void sumProc5SmallCalc/*V2*/(float * deviceErodedShadowIn, float * deviceErodedLightIn, float * deviceShadowRedArrayIn, float * deviceShadowGreenArrayIn, float * deviceShadowBlueArrayIn, float * deviceLightRedArrayIn, float * deviceLightGreenArrayIn, float * deviceLightBlueArrayIn, float * deviceErodedShadowOut, float * deviceErodedLightOut, float * deviceShadowRedArrayOut, float * deviceShadowGreenArrayOut, float * deviceShadowBlueArrayOut, float * deviceLightRedArrayOut, float * deviceLightGreenArrayOut, float * deviceLightBlueArrayOut, int size){//improved from sumProc5SmallCalcV1 by taking advantage of final warp calculations without __syncthreads(), did not use lastWarp function, wrote it directly in this function to speed up
	//assigning and summing 8 arrays, shared memory is perfectly maxed out!
  
  


}

__global__ void sumProc5SmallCalcV1(float * deviceErodedShadowIn, float * deviceErodedLightIn, float * deviceShadowRedArrayIn, float * deviceShadowGreenArrayIn, float * deviceShadowBlueArrayIn, float * deviceLightRedArrayIn, float * deviceLightGreenArrayIn, float * deviceLightBlueArrayIn, float * deviceErodedShadowOut, float * deviceErodedLightOut, float * deviceShadowRedArrayOut, float * deviceShadowGreenArrayOut, float * deviceShadowBlueArrayOut, float * deviceLightRedArrayOut, float * deviceLightGreenArrayOut, float * deviceLightBlueArrayOut, int size){//strided index, no divergence, unrolled loop, and shared memory is maxed with this version
	//assigning and summing 8 arrays in shared memory, shared memory is perfectly maxed out
  


}



__global__ void sumProc5/*ReverseCombined*/(float * deviceErodedShadowIn, float * deviceErodedLightIn, float * deviceShadowRedArrayIn, float * deviceShadowGreenArrayIn, float * deviceShadowBlueArrayIn, float * deviceLightRedArrayIn, float * deviceLightGreenArrayIn, float * deviceLightBlueArrayIn, float * deviceErodedShadowOut, float * deviceErodedLightOut, float * deviceShadowRedArrayOut, float * deviceShadowGreenArrayOut, float * deviceShadowBlueArrayOut, float * deviceLightRedArrayOut, float * deviceLightGreenArrayOut, float * deviceLightBlueArrayOut, int size){//strided index no divergence and all threads initially add 2 so they work and unrolled loop
	//storing and summing 8 arrays in shared memory now, shared memory is perfectly maxed out!!!
 


}






__global__ void sumProc5Reverse(float * d_out, float * d_in, int size){//strided index with no divergence, unrolled
  


}


__global__ void sumProc5NoDiverge(float * d_out, float * d_in, int size){//strided index with no divergence
 
}



__global__ void sumProc5FirstStride(float * d_out, float * d_in, int size){//strided index with divergence

  

		
}


__global__ void sumNaive(float * d_out, float * d_in, int size){//serial global version

  // naive reduction
  
    
		
}



