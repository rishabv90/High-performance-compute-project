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
                  float structural_element_value = structuralElement[j * MASK_WIDTH + k];
                  if(doLightMask){
                      structural_element_value = 1 - structural_element_value;
                  }
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
      erodedMask[row * width + col] = keepPixel ? inputMask[row * width + col] : 0;
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
  
  __shared__ float shared_inputMask[PADDED_DIM][PADDED_DIM];

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
                  float structural_element_value = structuralElement[j * MASK_WIDTH + k];
                  if(doLightMask){
                      structural_element_value = 1 - structural_element_value;
                  }
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
      erodedMask[row * width + col] = keepPixel ? inputMask[row * width + col] : 0;
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
