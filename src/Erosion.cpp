#ifndef EROSION_CPP
#define EROSION_CPP

/*
 * Erosion.cpp: performs a binary erosion process on a given image
 * https://en.wikipedia.org/wiki/Erosion_(morphology)#Binary_erosion
 *
 * Essentially, you define a convolution kernel (the "structural element") that is passed over the image.
 * For every entry it passes over, if the kernel matches the subset of the image contained within this window, the pixel is kept
 * Otherwise, the pixel is deleted (set to 0)
 */

#define EROSION_VERSION 1

#if EROSION_VERSION==1

#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2

/*
 * Version 1:
 *  - All global memory
 *  - Doesn't take advantage of the constant pool for the kernel mask
 */

__global__ void maskErosion(float* erodedMask, float* inputMask, float* structuralElement, int imageWidth, int imageHeight, bool doLightMask) {
  
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if(col < imageWidth && row < imageHeight){
      bool keepPixel = true;
      int start_col = col - MASK_RADIUS;
      int start_row = row - MASK_RADIUS;
      for(int j = 0; j < MASK_WIDTH; ++j){
          for(int k = 0; k < MASK_WIDTH; ++k){
              int current_row = start_row + j;
              int current_col = start_col + k;
              if(current_row > -1 && current_row < imageHeight && current_col > -1 && current_col < imageWidth){
                  float current_value = inputMask[current_row * imageWidth + current_col];
                  if(doLightMask){
                      current_value = current_value == 0 ? 1 : 0;
                  }
                  float structural_element_value = structuralElement[j * MASK_WIDTH + k];
                  if(current_value != structural_element_value){
                      keepPixel = false;
                      break;
                  }
              }
              else{
                  keepPixel = false;
                  break;
              }
          }
          if(!keepPixel){
              break;
          }
      }
      erodedMask[row * imageWidth + col] = keepPixel ? (doLightMask ? (inputMask[row * imageWidth + col] == 0 ? 1 : 0) : inputMask[row * imageWidth + col]) : 0;
  }
  
}

#endif

#if EROSION_VERSION==2

#define TILE_DIM 16
#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2
#define PADDED_DIM (TILE_DIM + MASK_WIDTH)

/*
 * Version 2:
 *  - Uses 2D-array shared memory
 *  - Doesn't take advantage of the constant pool for the kernel mask
 */

__global__ void maskErosion(float* erodedMask, float* inputMask, float* structuralElement, int imageWidth, int imageHeight, bool doLightMask) {
  
	__shared__ float sharedInputMask[PADDED_DIM][PADDED_DIM];  //block of image in shared memory

	// allocation in shared memory of image blocks
 		int dest = threadIdx.y * TILE_DIM + threadIdx.x;
 		int destY = dest/PADDED_DIM;     //row of shared memory
 		int destX = dest%PADDED_DIM;		//col of shared memory
 		int srcY = blockIdx.y *TILE_DIM + destY - MASK_RADIUS; // index to fetch data from input image
 		int srcX = blockIdx.x *TILE_DIM + destX - MASK_RADIUS; // index to fetch data from input image
 		int src = (srcY *imageWidth +srcX);   // index of input image
 		if(srcY>= 0 && srcY < imageHeight && srcX>=0 && srcX < imageWidth)
 			sharedInputMask[destY][destX] = inputMask[src];  // copy element of image in shared memory
 		else
 			sharedInputMask[destY][destX] = 0;
 		dest = threadIdx.y * TILE_DIM+ threadIdx.x + TILE_DIM * TILE_DIM;
 		destY = dest/PADDED_DIM;
		destX = dest%PADDED_DIM;
		srcY = blockIdx.y *TILE_DIM + destY - MASK_RADIUS;
		srcX = blockIdx.x *TILE_DIM + destX - MASK_RADIUS;
		src = (srcY *imageWidth +srcX);
		if(destY < PADDED_DIM){
			if(srcY>= 0 && srcY < imageHeight && srcX>=0 && srcX < imageWidth)
				sharedInputMask[destY][destX] = inputMask[src];
			else
				sharedInputMask[destY][destX] = 0;
		}

 		__syncthreads();


 		//compute kernel convolution
 		int y, x;
        bool keepPixel = true;
 		for (y= 0; y < MASK_WIDTH; y++){
 			for(x = 0; x<MASK_WIDTH; x++){
 				//accum += sharedInputMask[threadIdx.y + y][threadIdx.x + x] *structuralElement[y * MASK_WIDTH + x];
                 float current_value = sharedInputMask[threadIdx.y + y][threadIdx.x + x];
                 if(doLightMask){
                      current_value = current_value == 0 ? 1 : 0;
                  }
                if(current_value != structuralElement[y * MASK_WIDTH + x]){
                    keepPixel = false;
                    break;
                }
             }
             if(!keepPixel){
                 break;
             }
         }

 		y = blockIdx.y * TILE_DIM + threadIdx.y;
 		x = blockIdx.x * TILE_DIM + threadIdx.x;
 		if(y < imageHeight && x < imageWidth)
 			erodedMask[(y * imageWidth + x)] = keepPixel ? (doLightMask ? (sharedInputMask[threadIdx.y + MASK_RADIUS][threadIdx.x + MASK_RADIUS] == 0 ? 1 : 0) : sharedInputMask[threadIdx.y + MASK_RADIUS][threadIdx.x + MASK_RADIUS]) : 0;
 		__syncthreads();
     }
#endif

#if EROSION_VERSION==3

#define TILE_DIM 16
#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2
#define PADDED_DIM (TILE_DIM + MASK_WIDTH - 1)

/*
 * Version 3:
 *  - Uses 2D-array shared memory
 *  - Uses constant pool for kernel mask
 */

__global__ void maskErosion(float* erodedMask, float* inputMask, const float* __restrict__ structuralElement, int imageWidth, int imageHeight, bool doLightMask) {

	__shared__ float sharedInputMask[PADDED_DIM][PADDED_DIM];  //block of image in shared memory

	// allocation in shared memory of image blocks
 		int dest = threadIdx.y * TILE_DIM + threadIdx.x;
 		int destY = dest/PADDED_DIM;     //row of shared memory
 		int destX = dest%PADDED_DIM;		//col of shared memory
 		int srcY = blockIdx.y *TILE_DIM + destY - MASK_RADIUS; // index to fetch data from input image
 		int srcX = blockIdx.x *TILE_DIM + destX - MASK_RADIUS; // index to fetch data from input image
 		int src = (srcY *imageWidth +srcX);   // index of input image
 		if(srcY>= 0 && srcY < imageHeight && srcX>=0 && srcX < imageWidth)
 			sharedInputMask[destY][destX] = inputMask[src];  // copy element of image in shared memory
 		else
 			sharedInputMask[destY][destX] = 0;
 		dest = threadIdx.y * TILE_DIM+ threadIdx.x + TILE_DIM * TILE_DIM;
 		destY = dest/PADDED_DIM;
		destX = dest%PADDED_DIM;
		srcY = blockIdx.y *TILE_DIM + destY - MASK_RADIUS;
		srcX = blockIdx.x *TILE_DIM + destX - MASK_RADIUS;
		src = (srcY *imageWidth +srcX);
		if(destY < PADDED_DIM){
			if(srcY>= 0 && srcY < imageHeight && srcX>=0 && srcX < imageWidth)
				sharedInputMask[destY][destX] = inputMask[src];
			else
				sharedInputMask[destY][destX] = 0;
		}

 		__syncthreads();


 		//compute kernel convolution
 		int y, x;
        bool keepPixel = true;
 		for (y= 0; y < MASK_WIDTH; y++){
 			for(x = 0; x<MASK_WIDTH; x++){
 				//accum += sharedInputMask[threadIdx.y + y][threadIdx.x + x] *structuralElement[y * MASK_WIDTH + x];
                 float current_value = sharedInputMask[threadIdx.y + y][threadIdx.x + x];
                 if(doLightMask){
                      current_value = current_value == 0 ? 1 : 0;
                  }
                if(current_value != structuralElement[y * MASK_WIDTH + x]){
                    keepPixel = false;
                    break;
                }
             }
             if(!keepPixel){
                 break;
             }
         }

 		y = blockIdx.y * TILE_DIM + threadIdx.y;
 		x = blockIdx.x * TILE_DIM + threadIdx.x;
 		if(y < imageHeight && x < imageWidth)
 			erodedMask[(y * imageWidth + x)] = keepPixel ? (doLightMask ? (sharedInputMask[threadIdx.y + MASK_RADIUS][threadIdx.x + MASK_RADIUS] == 0 ? 1 : 0) : sharedInputMask[threadIdx.y + MASK_RADIUS][threadIdx.x + MASK_RADIUS]) : 0;
 		__syncthreads();

}

#endif

#if EROSION_VERSION==4

#define TILE_DIM 16
#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2
#define PADDED_DIM (TILE_DIM + MASK_WIDTH - 1)

/*
 * Version 4:
 *  - Uses 2D-array shared memory
 *  - Uses constant pool for kernel mask
 *  - Tries to rework inner loop operations to be more efficient
 */

__global__ void maskErosion(float* erodedMask, float* inputMask, const float* __restrict__ structuralElement, int imageWidth, int imageHeight, bool doLightMask) {

  
}

#endif

#if EROSION_VERSION==5

#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2

/*
 * Version 5:
 *  - All global memory
 *  - Takes advantage of the constant pool for the kernel mask
 */

__global__ void maskErosion(float* erodedMask, float* inputMask, const float* __restrict__ structuralElement, int imageWidth, int imageHeight, bool doLightMask) {
  
}

#endif

#if EROSION_VERSION==6
#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2

/*
 * Version 6:
 *  - All global memory
 *  - Takes advantage of the constant pool for the kernel mask
 *  - Tries to rework inner loop operations to be more efficient
 */

__global__ void maskErosion(float* erodedMask, float* inputMask, const float* __restrict__ structuralElement, int imageWidth, int imageHeight, bool doLightMask) {
  
}
#endif

#if EROSION_VERSION==7
#define TILE_DIM 16
#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2
#define PADDED_DIM (TILE_DIM + MASK_WIDTH - 1)

/*
 * Version 7:
 *  - Uses 1D-array shared memory which seems to reduce the number of loads issued by the compiler
 *  - Uses constant pool for kernel mask
 */

__global__ void maskErosion(float* erodedMask, float* inputMask, const float* __restrict__ structuralElement, int imageWidth, int imageHeight, bool doLightMask) {

  
}
#endif

#if EROSION_VERSION==8
#define TILE_DIM 16
#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2
#define PADDED_DIM (TILE_DIM + MASK_WIDTH - 1)

/*
 * Version 8:
 *  - Uses 1D-array shared memory which seems to reduce the number of loads issued by the compiler
 *  - Uses constant pool for kernel mask
 *  - Tries to rework inner loop operations to be more efficient
 */

__global__ void maskErosion(float* erodedMask, float* inputMask, const float* __restrict__ structuralElement, int imageWidth, int imageHeight, bool doLightMask) {

  
}
#endif

#undef TILE_DIM
#undef MASK_WIDTH
#undef MASK_RADIUS
#undef PADDED_DIM
#endif
