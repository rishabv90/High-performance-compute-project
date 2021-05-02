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
                  float previous_value = -1;
                  if(doLightMask){
                      previous_value = current_value;
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
#define PADDED_DIM (TILE_DIM + MASK_WIDTH - 1)

/*
 * Version 2:
 *  - Uses 2D-array shared memory
 *  - Doesn't take advantage of the constant pool for the kernel mask
 */

__global__ void maskErosion(float* erodedMask, float* inputMask, float* structuralElement, int imageWidth, int imageHeight, bool doLightMask) {
  
    __shared__ unsigned char shared_inputMask[PADDED_DIM][PADDED_DIM];

    int ty = threadIdx.y;
    int tx = threadIdx.x;


    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;


    int start_row = row - MASK_RADIUS;
    int start_col = col - MASK_RADIUS;

    if((start_row >= 0) && (start_row < imageHeight) && (start_col >= 0) && (start_col < imageWidth) ){

        shared_inputMask[ty][tx] = inputMask[ start_row * imageWidth + start_col];

    }
    else{

        shared_inputMask[ty][tx] = 0;

    }
    __syncthreads();

    if( ty < TILE_DIM && tx < TILE_DIM ){
        bool keepPixel = true;
        for(int i = 0; i < MASK_WIDTH; i++) {
            for(int j = 0; j < MASK_WIDTH; j++) {
              
                float current_value = shared_inputMask[i+ty][j+tx] ;
                if(doLightMask){
                      current_value = 1 - current_value;
                }
                float structural_element_value = structuralElement[i * MASK_WIDTH + j];
                if(current_value != structural_element_value){
                    keepPixel = false;
                    break;
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
        if(row < imageHeight && col < imageWidth)
                erodedMask[row * imageWidth + col] = keepPixel ? inputMask[row * imageWidth + col] : 0;

        }

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

   __shared__ unsigned char shared_inputMask[PADDED_DIM][PADDED_DIM];

    int ty = threadIdx.y;
    int tx = threadIdx.x;


    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;


    int start_row = row - MASK_RADIUS;
    int start_col = col - MASK_RADIUS;

    if((start_row >= 0) && (start_row < imageHeight) && (start_col >= 0) && (start_col < imageWidth) ){

        shared_inputMask[ty][tx] = inputMask[ start_row * imageWidth + start_col];

    }
    else{

        shared_inputMask[ty][tx] = 0;

    }
    __syncthreads();

    if( ty < TILE_DIM && tx < TILE_DIM ){
        bool keepPixel = true;
        for(int i = 0; i < MASK_WIDTH; i++) {
            for(int j = 0; j < MASK_WIDTH; j++) {
              
                float current_value = shared_inputMask[i+ty][j+tx] ;
                if(doLightMask){
                      current_value = 1 - current_value;
                  }
                float structural_element_value = structuralElement[i * MASK_WIDTH + j];
                if(current_value != structural_element_value){
                    keepPixel = false;
                    break;
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
        if(row < imageHeight && col < imageWidth)
                erodedMask[row * imageWidth + col] = keepPixel ? inputMask[row * imageWidth + col] : 0;

        }  
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
