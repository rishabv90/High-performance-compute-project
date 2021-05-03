#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <stdio.h>
#include <wb.h>
#include "RGBtoYUV.cpp"
#include "Greyscale.cpp"
#include "colorInvariant.cpp"
#include "Erosion.cpp"
#include "MaskGeneration.cpp"
#include "Smooth.cpp"
#include "sumProc5.cpp"
#include "map1Proc5.cpp"
#include "proc5.cpp"
#include "map2Proc5.cpp"


#define NUM_BINS 256
#define SMOOTH_KERNEL_VERSION 0	// Define Smooth Version 0 = 2D Shared Memory, 1 = 1D Shared Memory, 2 = 2D Global Memory

#define USE_STREAMING
#define USE_STREAM_EVENTS

//Canonical way to check for errors in CUDA - https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t errCode, const char *file, int line, bool abort=true) {
  if (errCode != cudaSuccess) {
    fprintf(stderr, "GPU Assertion: %s %s %d\n", cudaGetErrorString(errCode), file, line);
    if (abort) exit(errCode);
  }
}

static const int erosionStrelWidth = 5, erosionStrelHeight = 5;
static const float hostErosionStrel[erosionStrelHeight][erosionStrelWidth] = {
  {1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1}
};

static const int kernelWidth = 5, kernelHeight = 5;
static const float hostKernelData[kernelWidth][kernelHeight] = {
  {0.04, 0.04, 0.04, 0.04, 0.04},
  {0.04, 0.04, 0.04, 0.04, 0.04},
  {0.04, 0.04, 0.04, 0.04, 0.04},
  {0.04, 0.04, 0.04, 0.04, 0.04},
  {0.04, 0.04, 0.04, 0.04, 0.04}
};

void sumProc5Host(float* sum_input, unsigned int size, float* totalSum){
	// Set up number of threads and blocks
	unsigned int blockSize = 1024; 
	unsigned int maxElementsPerBlock = blockSize * 2; // due to binary tree nature of algorithm

	unsigned int gridSize = 0;
	if (size <= maxElementsPerBlock)
	{
		gridSize = (unsigned int)std::ceil(float(size) / float(maxElementsPerBlock));
	}
	else
	{
		gridSize = size / maxElementsPerBlock;
		if (size % maxElementsPerBlock != 0)
			gridSize++;
	}

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks / grid size
	float* deviceBlockSums;
	gpuErrchk(cudaMalloc(&deviceBlockSums, sizeof(float) * gridSize));
	gpuErrchk(cudaMemset(deviceBlockSums, 0, sizeof(float) * gridSize));

	// Sum data allocated for each block
	sumProc5<<<gridSize, blockSize, sizeof(float) * maxElementsPerBlock>>>(deviceBlockSums, sum_input, size);

	// Sum each block's total sums (to get global total sum)
	// Use basic implementation if number of total sums is <= 2048
	// Else, recurse on this same function
	if (gridSize <= maxElementsPerBlock)
	{
		gpuErrchk(cudaMemset(totalSum, 0, sizeof(float)));
		sumProc5<<<1, blockSize, sizeof(float) * maxElementsPerBlock>>>(totalSum, deviceBlockSums, gridSize);
	}
	else
	{
		float* sum_input_block_sums;
		gpuErrchk(cudaMalloc(&sum_input_block_sums, sizeof(float) * gridSize));
		gpuErrchk(cudaMemcpy(sum_input_block_sums, deviceBlockSums, sizeof(float) * gridSize, cudaMemcpyDeviceToDevice));
		sumProc5Host(sum_input_block_sums, gridSize, totalSum);
		gpuErrchk(cudaFree(sum_input_block_sums));
	}

	gpuErrchk(cudaFree(deviceBlockSums));
}

//-------------------------------------------------
//---------------- MAIN FUNCTION:
	// Arguments:
	// ./shadow_removal <- This program call
	// input_image      <- The input image to have shadow removed
	// output_directory <- The directory to produce output images in
	// kernel_file      <- The convolution kernel file

int main(int argc, char** argv) {
  if (argc != 3) {
	fprintf(stderr, "Argument count is: %d and needs to be 3\n", argc);
    fprintf(stderr, "Usage: ./shadow_removal input_image output_directory\n");
    fprintf(stderr, "/(-_-)\\\n");
    return 1;
  }

  //----------------------------------------------------------------------------------------//
  //-------------------------------- Host Variables ----------------------------------------//
  //----------------------------------------------------------------------------------------//
  //--------- Process 0: RGB Image Input
  int imageChannels;
  int imageWidth;
  int imageHeight;
  
  char *inputImageFile = argv[1];
  std::string baseOutputDir(argv[2]);

  if (baseOutputDir.back() != '/') {
    baseOutputDir.append("/");
  }

  wbImage_t inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  float *hostInputImageData = wbImage_getData(inputImage);

  // -------- Setting up streaming and events for synchronization
#ifdef USE_STREAMING
  cudaStream_t colorspaceStream, yuvStream, grayscaleStream, resultStream;
  cudaStreamCreate(&colorspaceStream);
  cudaStreamCreate(&yuvStream);
  cudaStreamCreate(&grayscaleStream);
  cudaStreamCreate(&resultStream);
#endif

#if defined(USE_STREAM_EVENTS) && defined(USE_STREAMING)
  cudaEvent_t colorspaceCompleteEvent, yuvCompleteEvent, grayscaleCompleteEvent;
  cudaEventCreate(&colorspaceCompleteEvent);
  cudaEventCreate(&yuvCompleteEvent);
  cudaEventCreate(&grayscaleCompleteEvent);
#endif

  //--------- Process 1: YUV Conversion
  float *hostOutputImageDataYUV = (float*) malloc(imageWidth*imageHeight*imageChannels*sizeof(float));

  //--------- Process 1: Color Invariance
  float *hostOutputImageDataColorInvariant = (float*) malloc(imageWidth*imageHeight*imageChannels*sizeof(float));

  //--------- Process 1: Greyscale Conversion
  float *hostOutputImageDataGreyScale = (float*) malloc(imageWidth*imageHeight*sizeof(float));

  //--------- Process 2: YUV Masking
  float *hostOutputImageDataCbMask = (float*) malloc(imageWidth*imageHeight*sizeof(float));

  //--------- Process 2: Greyscale Masking
  float *hostOutputImageDataGrayMask = (float*) malloc(imageWidth*imageHeight*sizeof(float));

  //--------- Process 3: Smoothing
  float *hostOutputImageDataSmooth = (float*) malloc(imageWidth*imageHeight*sizeof(float));

  //--------- Process 4: Light Mask Erosion
  float *hostOutputImageErodedLight = (float*) malloc(imageWidth*imageHeight*sizeof(float));

  //--------- Process 4: Shadow Mask Erosion
  float *hostOutputImageErodedShadow = (float*) malloc(imageWidth*imageHeight*sizeof(float));

  //--------- Process 5: Ratio & Final Image
  float *hostResultImageData = (float*) malloc(imageWidth*imageHeight*imageChannels*sizeof(float));

  //------------------------------------------------------------------------------------------//
  //-------------------------------- Device Variables ----------------------------------------//
  //------------------------------------------------------------------------------------------//
  //--------- Process 0: RGB Image Input
  float *deviceInputImageData;
  gpuErrchk(cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));

  //--------- Process 1: YUV Conversion
  float *deviceYUVOutputImageData;
  gpuErrchk(cudaMalloc((void **) &deviceYUVOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));

  //--------- Process 1: Color Invariance
  float *deviceColorInvariantOutputImageData;
  gpuErrchk(cudaMalloc((void **)&deviceColorInvariantOutputImageData, imageWidth * imageHeight * sizeof(float) * imageChannels));

  //--------- Process 1: Greyscale Conversion
  float *deviceGreyscaleOutputImageData;
  float *redData;
  float *greenData;
  float *blueData;
  gpuErrchk(cudaMalloc((void **)&deviceGreyscaleOutputImageData, imageWidth * imageHeight * sizeof(float)));
  
  //For coalesced accesses after first process
  gpuErrchk(cudaMalloc((void **)&redData, imageWidth * imageHeight * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&greenData, imageWidth * imageHeight * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&blueData, imageWidth * imageHeight * sizeof(float)));

  //--------- Process 2: YUV Masking
  float *deviceCBMaskOutputImageData;
  unsigned int *deviceCbBins;
  unsigned int *deviceCbHistogramSum;
  unsigned int *deviceGreyHistogramSum;
  float *deviceGrayOmega;
  float *deviceCbOmega;
  float *deviceGrayMu;
  float *deviceCbMu;
  float *deviceGraySigmaBSquared;
  float *deviceCbSigmaBSquared;
  float *deviceGrayThreshold;
  float *deviceCbThreshold;
  
  gpuErrchk(cudaMalloc((void **)&deviceCBMaskOutputImageData, imageWidth * imageHeight * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceCbBins, NUM_BINS * sizeof(int)));
  gpuErrchk(cudaMalloc((void **)&deviceCbHistogramSum, sizeof(int)));
  gpuErrchk(cudaMalloc((void **)&deviceGreyHistogramSum, sizeof(int)));
  gpuErrchk(cudaMalloc((void **)&deviceGrayOmega, NUM_BINS * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceCbOmega, NUM_BINS * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceGrayMu, NUM_BINS * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceCbMu, NUM_BINS * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceGraySigmaBSquared, NUM_BINS * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceCbSigmaBSquared, NUM_BINS * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceGrayThreshold, sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceCbThreshold, sizeof(float)));

  //--------- Process 2: Greyscale Masking
  float *deviceGreyMaskOutputImageData;
  unsigned int *deviceGreyBins;

  gpuErrchk(cudaMalloc((void **)&deviceGreyMaskOutputImageData, imageWidth * imageHeight * sizeof(float))); 
  gpuErrchk(cudaMalloc((void **)&deviceGreyBins, NUM_BINS * sizeof(int)));

  //--------- Process 3: Smoothing
  float *deviceSmoothOutputImageData;
  float *deviceMaskData;

  gpuErrchk(cudaMalloc((void **)&deviceSmoothOutputImageData, imageWidth * imageHeight * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceMaskData, kernelWidth * kernelHeight * sizeof(float)));

  //--------- Process 4: Light Mask Erosion
  float *deviceErodedLight;
  float *deviceStrel;

  gpuErrchk(cudaMalloc((void **)&deviceErodedLight, imageHeight * imageWidth * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceStrel, erosionStrelHeight * erosionStrelWidth * sizeof(float)));

  //--------- Process 4: Shadow Mask Erosion
  float *deviceErodedShadow;

  gpuErrchk(cudaMalloc((void **)&deviceErodedShadow, imageHeight * imageWidth * sizeof(float)));

  //--------- Process 5: Final Image
  float *deviceShadowRedArray;
  float *deviceShadowGreenArray;
  float *deviceShadowBlueArray;
  float *deviceLightRedArray;
  float *deviceLightGreenArray;
  float *deviceLightBlueArray;
  //Sums
  float *deviceErodedShadowSum;
  float *deviceErodedLightSum;
  float *deviceShadowRedArraySum;
  float *deviceShadowGreenArraySum;
  float *deviceShadowBlueArraySum;
  float *deviceLightRedArraySum;
  float *deviceLightGreenArraySum;
  float *deviceLightBlueArraySum;
  //
  float *deviceResultImageData;
  
  gpuErrchk(cudaMalloc((void **)&deviceShadowRedArray, imageHeight * imageWidth * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceShadowGreenArray, imageHeight * imageWidth * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceShadowBlueArray, imageHeight * imageWidth * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceLightRedArray, imageHeight * imageWidth * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceLightGreenArray, imageHeight * imageWidth * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceLightBlueArray, imageHeight * imageWidth * sizeof(float)));

  gpuErrchk(cudaMalloc((void **)&deviceShadowRedArraySum, sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceShadowGreenArraySum, sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceShadowBlueArraySum, sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceLightRedArraySum, sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceLightGreenArraySum, sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceLightBlueArraySum,  sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceErodedShadowSum,  sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceErodedLightSum,  sizeof(float)));
  
  gpuErrchk(cudaMalloc((void **)&deviceResultImageData, imageHeight * imageWidth * imageChannels * sizeof(float)));


  //-----------------------------------------------------------------------------------------------------//
  //-------------------------------- Data Transfer (Host -> GPU) ----------------------------------------//
  //-----------------------------------------------------------------------------------------------------//
  gpuErrchk(cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice));	 
			 
  #ifndef USE_STREAMING
  gpuErrchk(cudaMemcpy(deviceMaskData, hostKernelData,
             kernelWidth * kernelHeight * sizeof(float), 
			       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(deviceStrel, hostErosionStrel,
             erosionStrelWidth * erosionStrelHeight * sizeof(float),
             cudaMemcpyHostToDevice));
  #else
  gpuErrchk(cudaMemcpyAsync(deviceMaskData, hostKernelData,
             kernelWidth * kernelHeight * sizeof(float),
             cudaMemcpyHostToDevice, yuvStream));

  gpuErrchk(cudaMemcpyAsync(deviceStrel, hostErosionStrel,
             erosionStrelWidth * erosionStrelHeight * sizeof(float),
             cudaMemcpyHostToDevice, grayscaleStream));
  #endif


  //-------------------------------------------------------------------------------------------------//
  //-------------------------------- STOP! IT'S KERNEL TIME! ----------------------------------------//
  //-------------------------------------------------------------------------------------------------//

  // Shared grid or block dimension tuples
  dim3 dimGridHisto(2, 1, 1);
  dim3 dimBlockHisto(1024, 1, 1);

  dim3 dimGridCumSum(1, 1, 1);
  dim3 dimBlockCumSum(NUM_BINS, 1, 1);

  dim3 dimGridMasking((imageWidth - 1) / 16 + 1, (imageHeight - 1)/16 + 1, 1);
  dim3 dimBlockMasking(16, 16, 1);

  dim3 dimGridErosion((imageWidth - 1) / 16 + 1, (imageHeight - 1)/16 + 1, 1);
  dim3 dimBlockErosion(16, 16, 1);

  // Kernel Launches:
  // Color Invariant -> Grayscale processing, and YUV
  //--------- Process 1: Color Invariance
  dim3 dimGridColorInvariance((imageWidth - 1) / 16 + 1, (imageHeight - 1)/16 + 1, 1);
  dim3 dimBlockColorInvariance(16, 16, 1);

  cudaEvent_t astartEvent, astopEvent;
  float aelapsedTime;
  cudaEventCreate(&astartEvent);
  cudaEventCreate(&astopEvent);
  cudaEventRecord(astartEvent, 0);

#ifdef USE_STREAMING
  YUVandCItoGray<<<dimGridColorInvariance, dimBlockColorInvariance, 0, colorspaceStream>>>(deviceGreyscaleOutputImageData, deviceYUVOutputImageData, deviceInputImageData, redData, greenData, blueData, imageWidth, imageHeight); //implemented
#else
  YUVandCItoGray<<<dimGridColorInvariance, dimBlockColorInvariance>>>(deviceGreyscaleOutputImageData, deviceYUVOutputImageData, deviceInputImageData, redData, greenData, blueData, imageWidth, imageHeight);



#endif
cudaDeviceSynchronize();
cudaEventRecord(astopEvent, 0);
cudaEventSynchronize(astopEvent);
cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
printf("Total execution time (ms) %f for YUVandCItoGray \n",aelapsedTime);
  // Wait until this stream is done as all other streams depend on the colorspace transform
#if defined(USE_STREAM_EVENTS) && defined(USE_STREAMING)
  cudaEventRecord(colorspaceCompleteEvent, colorspaceStream);
  cudaStreamWaitEvent(grayscaleStream, colorspaceCompleteEvent, 0);
#elif defined(USE_STREAMING)
  cudaStreamSynchronize(colorspaceStream);
#endif
  
  //--------- Process 1: Greyscale Conversion, done previously


cudaEvent_t event1StartEvent, event1StopEvent;
float event1ElapsedTime;
cudaEventCreate(&event1StartEvent);
cudaEventCreate(&event1StopEvent);
cudaEventRecord(event1StartEvent, 0);
  //--------- Process 2: Greyscale Masking

#ifdef USE_STREAMING
  histogramKernel<<<dimGridHisto, dimBlockHisto, 0, grayscaleStream>>>(deviceGreyscaleOutputImageData, deviceGreyBins, imageWidth * imageHeight, true); //implemented
#else
  histogramKernel<<<dimGridHisto, dimBlockHisto>>>(deviceGreyscaleOutputImageData, deviceGreyBins, imageWidth * imageHeight, true); 
#endif

histogramSumKernel<<<dimGridCumSum, dimBlockCumSum>>>(deviceGreyBins, deviceGreyHistogramSum);

  //--------- Process 2: Greyscale Masking
#ifdef USE_STREAMING
  cumSumOne<<<dimGridCumSum, dimBlockCumSum, 0, grayscaleStream>>>(deviceGreyBins, deviceGrayOmega, imageWidth * imageHeight, deviceGreyHistogramSum); //pending
#else
  cumSumOne<<<dimGridCumSum, dimBlockCumSum>>>(deviceGreyBins, deviceGrayOmega, imageWidth * imageHeight, deviceGreyHistogramSum);
#endif
  
  //--------- Process 2: Greyscale Masking
#ifdef USE_STREAMING
  cumSumTwo<<<dimGridCumSum, dimBlockCumSum, 0, grayscaleStream>>>(deviceGreyBins, deviceGrayMu, imageWidth * imageHeight, deviceGreyHistogramSum); //pending
#else
  cumSumTwo<<<dimGridCumSum, dimBlockCumSum>>>(deviceGreyBins, deviceGrayMu, imageWidth * imageHeight, deviceGreyHistogramSum);
#endif
  
  //--------- Process 2: Greyscale Masking
#ifdef USE_STREAMING
  compSigmaBSquared<<<dimGridCumSum, dimBlockCumSum, 0, grayscaleStream>>>(deviceGraySigmaBSquared, deviceGrayOmega, deviceGrayMu);
#else
  compSigmaBSquared<<<dimGridCumSum, dimBlockCumSum>>>(deviceGraySigmaBSquared, deviceGrayOmega, deviceGrayMu);
#endif
  //--------- Process 2: Greyscale Masking
#ifdef USE_STREAMING
  argmax<<<1, 256, 0, grayscaleStream>>>(deviceGrayThreshold, deviceGraySigmaBSquared); //pending
#else
  argmax<<<1, 256>>>(deviceGrayThreshold, deviceGraySigmaBSquared);
#endif
  
  //--------- Process 2: Greyscale Masking
#ifdef USE_STREAMING
  maskGeneration<<<dimGridMasking, dimBlockMasking, 0, grayscaleStream>>>(deviceGreyscaleOutputImageData, deviceGreyMaskOutputImageData, deviceGrayThreshold, imageWidth, imageHeight, 1, true);
#else
  maskGeneration<<<dimGridMasking, dimBlockMasking>>>(deviceGreyscaleOutputImageData, deviceGreyMaskOutputImageData, deviceGrayThreshold, imageWidth, imageHeight, 1, true);
#endif
cudaDeviceSynchronize();
cudaEventRecord(event1StopEvent, 0);
cudaEventSynchronize(event1StopEvent);
cudaEventElapsedTime(&event1ElapsedTime, event1StartEvent, event1StopEvent);
printf("Total execution time (ms) %f for Greyscale Mask Generation \n",event1ElapsedTime);

cudaEvent_t event2StartEvent, event2StopEvent;
float event2ElapsedTime;
cudaEventCreate(&event2StartEvent);
cudaEventCreate(&event2StopEvent);
cudaEventRecord(event2StartEvent, 0);
  //--------- Process 4: Light Mask Erosion
#ifdef USE_STREAMING
  maskErosion<<<dimGridErosion, dimBlockErosion, 0, grayscaleStream>>>(deviceErodedLight, deviceGreyMaskOutputImageData, deviceStrel, imageWidth, imageHeight, true);
#else
  maskErosion<<<dimGridErosion, dimBlockErosion>>>(deviceErodedLight, deviceGreyMaskOutputImageData, deviceStrelLight, imageWidth, imageHeight, true);
#endif
cudaDeviceSynchronize();
cudaEventRecord(event2StopEvent, 0);
cudaEventSynchronize(event2StopEvent);
cudaEventElapsedTime(&event2ElapsedTime, event2StartEvent, event2StopEvent);
printf("Total execution time (ms) %f for light erosion \n",event2ElapsedTime);
cudaEvent_t event6StartEvent, event6StopEvent;
float event6ElapsedTime;
cudaEventCreate(&event6StartEvent);
cudaEventCreate(&event6StopEvent);
cudaEventRecord(event6StartEvent, 0);
  //--------- Process 4: Shadow Mask Erosion
#ifdef USE_STREAMING  
  maskErosion<<<dimGridErosion, dimBlockErosion, 0, grayscaleStream>>>(deviceErodedShadow, deviceGreyMaskOutputImageData, deviceStrel, imageWidth, imageHeight, false);
#else
  maskErosion<<<dimGridErosion, dimBlockErosion>>>(deviceErodedShadow, deviceGreyMaskOutputImageData, deviceStrelShadow, imageWidth, imageHeight, false);
#endif
cudaDeviceSynchronize();
  cudaEventRecord(event6StopEvent, 0);
  cudaEventSynchronize(event6StopEvent);
  cudaEventElapsedTime(&event6ElapsedTime, event6StartEvent, event6StopEvent);
  printf("Total execution time (ms) %f for shadow mask erosion \n",event6ElapsedTime);

#if defined(USE_STREAM_EVENTS) && defined(USE_STREAMING)
  cudaEventRecord(grayscaleCompleteEvent, grayscaleStream);
  cudaStreamWaitEvent(yuvStream, colorspaceCompleteEvent, 0);
#endif

  // YUV processing
  //--------- Process 1: YUV Conversion, done previously
  dim3 dimGridYUVConversion((imageWidth - 1) / 16 + 1, (imageHeight - 1)/16 + 1, 1);
  dim3 dimBlockYUVConversion(16, 16, 1);

  cudaEvent_t event3StartEvent, event3StopEvent;
  float event3ElapsedTime;
  cudaEventCreate(&event3StartEvent);
  cudaEventCreate(&event3StopEvent);
  cudaEventRecord(event3StartEvent, 0);
  //--------- Process 2: YUV Masking
#ifdef USE_STREAMING
  histogramKernel<<<dimGridHisto, dimBlockHisto, 0, yuvStream>>>(deviceYUVOutputImageData, deviceCbBins, imageWidth * imageHeight, false); //implemented
#else
  histogramKernel<<<dimGridHisto, dimBlockHisto>>>(deviceYUVOutputImageData, deviceCbBins, imageWidth * imageHeight, false);
#endif

histogramSumKernel<<<dimGridCumSum, dimBlockCumSum>>>(deviceCbBins, deviceCbHistogramSum);


  //--------- Process 2: YUV Masking
#ifdef USE_STREAMING
  cumSumOne<<<dimGridCumSum, dimBlockCumSum, 0, yuvStream>>>(deviceCbBins, deviceCbOmega, imageWidth * imageHeight, deviceCbHistogramSum);
#else
  cumSumOne<<<dimGridCumSum, dimBlockCumSum>>>(deviceCbBins, deviceCbOmega, imageWidth * imageHeight, deviceCbHistogramSum);
#endif

  //--------- Process 2: YUV Masking
#ifdef USE_STREAMING
  cumSumTwo<<<dimGridCumSum, dimBlockCumSum, 0, yuvStream>>>(deviceCbBins, deviceCbMu, imageWidth * imageHeight, deviceCbHistogramSum);
#else
  cumSumTwo<<<dimGridCumSum, dimBlockCumSum>>>(deviceCbBins, deviceCbMu, imageWidth * imageHeight, deviceCbHistogramSum);
#endif

  //--------- Process 2: YUV Masking
#ifdef USE_STREAMING
  compSigmaBSquared<<<dimGridCumSum, dimBlockCumSum, 0, yuvStream>>>(deviceCbSigmaBSquared, deviceCbOmega, deviceCbMu);
#else
  compSigmaBSquared<<<dimGridCumSum, dimBlockCumSum>>>(deviceCbSigmaBSquared, deviceCbOmega, deviceCbMu);
#endif

  //--------- Process 2: YUV Masking
#ifdef USE_STREAMING
  argmax<<<1, 256, 0, yuvStream>>>(deviceCbThreshold, deviceCbSigmaBSquared);
#else
  argmax<<<1, 256>>>(deviceCbThreshold, deviceCbSigmaBSquared);
#endif

  //--------- Process 2: YUV Masking
#ifdef USE_STREAMING
  maskGeneration<<<dimGridMasking, dimBlockMasking, 0, yuvStream>>>(deviceYUVOutputImageData, deviceCBMaskOutputImageData, deviceCbThreshold, imageWidth, imageHeight, 0, false);
#else
  maskGeneration<<<dimGridMasking, dimBlockMasking>>>(deviceYUVOutputImageData, deviceCBMaskOutputImageData, deviceCbThreshold, imageWidth, imageHeight, 0, false);
#endif
cudaDeviceSynchronize();
  cudaEventRecord(event3StopEvent, 0);
  cudaEventSynchronize(event3StopEvent);
  cudaEventElapsedTime(&event3ElapsedTime, event3StartEvent, event3StopEvent);
  printf("Total execution time (ms) %f for YUV Masking \n",event3ElapsedTime);

  //--------- Process 3: Smoothing
 /* dim3 dimGridSmoothing((imageWidth-1)/16 +1, (imageHeight-1)/16+1, 1);
  dim3 dimBlockSmoothing(16, 16, 1);*/

//=======
#ifdef USE_STREAMING
  //smooth_kernel<<<dimGridSmoothing, dimBlockSmoothing, 0, yuvStream>>>(deviceCBMaskOutputImageData, deviceSmoothOutputImageData, deviceMaskData, 1, imageWidth, imageHeight);
  #if SMOOTH_KERNEL_VERSION == 0
  // 2D Shared Memory, Smoothing Kernel
  dim3 dimGridSmoothing((imageWidth * 2 - 1)/16 +1, (imageHeight * 2 -1)/16+1, 1);
  dim3 dimBlockSmoothing(16, 16, 1);
  cudaEvent_t event4StartEvent, event4StopEvent;
float event4ElapsedTime;
cudaEventCreate(&event4StartEvent);
cudaEventCreate(&event4StopEvent);
cudaEventRecord(event4StartEvent, 0);
  smooth_kernel<<<dimGridSmoothing, dimBlockSmoothing, 0, yuvStream>>>(deviceCBMaskOutputImageData, deviceSmoothOutputImageData, deviceMaskData, 1, imageWidth, imageHeight);

  #elif SMOOTH_KERNEL_VERSION == 1
  // 1D Shared Memory, Smoothing Kernels (Row + Column)
  smooth_kernel_row<<<dimGridSmoothing, dimBlockSmoothing, 0, yuvStream>>>(deviceCBMaskOutputImageData, deviceSmoothOutputImageData, deviceMaskData, 1, imageWidth, imageHeight);

  smooth_kernel_col<<<dimGridSmoothing, dimBlockSmoothing, 0, yuvStream>>>(deviceSmoothOutputImageData, deviceSmoothOutputImageData, deviceMaskData, 1, imageWidth, imageHeight);

  #elif SMOOTH_KERNEL_VERSION == 2
  // 2D Global Memory, Smoothing Kernel
  smooth_kernel_global<<<dimGridSmoothing, dimBlockSmoothing, 0, yuvStream>>>(deviceCBMaskOutputImageData, deviceSmoothOutputImageData, deviceMaskData, 1, imageWidth, imageHeight); 
  #endif
#else
  //smooth_kernel<<<dimGridSmoothing, dimBlockSmoothing>>>(deviceCBMaskOutputImageData, deviceSmoothOutputImageData, deviceMaskData, 1, imageWidth, imageHeight);
  #if SMOOTH_KERNEL_VERSION == 0
  // 2D Shared Memory, Smoothing Kernel
  dim3 dimGridSmoothing((imageWidth * 2 - 1)/16 +1, (imageHeight * 2 -1)/16+1, 1);
  dim3 dimBlockSmoothing(16, 16, 1);
  smooth_kernel<<<dimGridSmoothing, dimBlockSmoothing>>>(deviceCBMaskOutputImageData, deviceSmoothOutputImageData, deviceMaskData, 1, imageWidth, imageHeight);

  #elif SMOOTH_KERNEL_VERSION == 1
  // 1D Shared Memory, Smoothing Kernels (Row + Column)
  smooth_kernel_row<<<dimGridSmoothing, dimBlockSmoothing>>>(deviceCBMaskOutputImageData, deviceSmoothOutputImageData, deviceMaskData, 1, imageWidth, imageHeight);

  smooth_kernel_col<<<dimGridSmoothing, dimBlockSmoothing>>>(deviceSmoothOutputImageData, deviceSmoothOutputImageData, deviceMaskData, 1, imageWidth, imageHeight);

  #elif SMOOTH_KERNEL_VERSION == 2
  // 2D Global Memory, Smoothing Kernel
  dim3 dimGridSmoothing((imageWidth-1)/16 +1, (imageHeight-1)/16+1, 1);
  dim3 dimBlockSmoothing(16, 16, 1);
  smooth_kernel_global<<<dimGridSmoothing, dimBlockSmoothing>>>(deviceCBMaskOutputImageData, deviceSmoothOutputImageData, deviceMaskData, 1, imageWidth, imageHeight); 
  #endif
#endif
cudaDeviceSynchronize();
  cudaEventRecord(event4StopEvent, 0);
  cudaEventSynchronize(event4StopEvent);
  cudaEventElapsedTime(&event4ElapsedTime, event4StartEvent, event4StopEvent);
  printf("Total execution time (ms) %f for smooth kernel\n",event4ElapsedTime);

#if defined(USE_STREAM_EVENTS) && defined(USE_STREAMING)
  cudaEventRecord(yuvCompleteEvent, yuvStream);
  cudaStreamWaitEvent(resultStream, yuvCompleteEvent, 0);
  cudaStreamWaitEvent(resultStream, grayscaleCompleteEvent, 0);
#elif defined(USE_STREAMING)
  cudaStreamSynchronize(yuvStream);
  cudaStreamSynchronize(grayscaleStream);
#endif

  //--------- Process 5: Ratio & Final Image
  //PROC 5
  cudaEvent_t event5StartEvent, event5StopEvent;
  float event5ElapsedTime;
  cudaEventCreate(&event5StartEvent);
  cudaEventCreate(&event5StopEvent);
  cudaEventRecord(event5StartEvent, 0);
#ifdef USE_STREAMING
  map1Proc5<<<dimGridYUVConversion, dimBlockYUVConversion, 0, resultStream>>>(redData, greenData, blueData, deviceErodedShadow, deviceErodedLight, deviceShadowRedArray, deviceShadowGreenArray, deviceShadowBlueArray, deviceLightRedArray, deviceLightGreenArray, deviceLightBlueArray, imageWidth, imageHeight);
#else
  map1Proc5<<<dimGridYUVConversion, dimBlockYUVConversion>>>(redData, greenData, blueData, deviceErodedShadow, deviceErodedLight, deviceShadowRedArray, deviceShadowGreenArray, deviceShadowBlueArray, deviceLightRedArray, deviceLightGreenArray, deviceLightBlueArray, imageWidth, imageHeight);
#endif
  
unsigned int size = imageHeight * imageWidth;

sumProc5Host(deviceErodedShadow, size, deviceErodedShadowSum);
sumProc5Host(deviceErodedLight, size, deviceErodedLightSum);
sumProc5Host(deviceShadowRedArray, size, deviceShadowRedArraySum);
sumProc5Host(deviceShadowGreenArray, size, deviceShadowGreenArraySum);
sumProc5Host(deviceShadowBlueArray, size, deviceShadowBlueArraySum);
sumProc5Host(deviceLightRedArray, size, deviceLightRedArraySum);
sumProc5Host(deviceLightGreenArray, size, deviceLightGreenArraySum);
sumProc5Host(deviceLightBlueArray, size, deviceLightBlueArraySum);

//

//Perform sums

	
#ifdef USE_STREAMING
  smallCalc<<<3,1, 0, resultStream>>>(deviceShadowRedArraySum, deviceShadowGreenArraySum, deviceShadowBlueArraySum, deviceLightRedArraySum, deviceLightGreenArraySum, deviceLightBlueArraySum, deviceErodedLightSum, deviceErodedShadowSum);
#else
  smallCalc<<<3,1>>>(deviceShadowRedArraySum, deviceShadowGreenArraySum, deviceShadowBlueArraySum, deviceLightRedArraySum, deviceLightGreenArraySum, deviceLightBlueArraySum, deviceErodedLightSum, deviceErodedShadowSum);
#endif


#ifdef USE_STREAMING  
  proc5<<<dimGridYUVConversion, dimBlockYUVConversion, 0, resultStream>>>(redData, greenData, blueData, deviceResultImageData, deviceSmoothOutputImageData, deviceShadowRedArraySum, deviceShadowGreenArraySum, deviceShadowBlueArraySum, imageWidth, imageHeight);
#else
  proc5<<<dimGridYUVConversion, dimBlockYUVConversion>>>(redData, greenData, blueData, deviceResultImageData, deviceSmoothOutputImageData, deviceShadowRedArraySum, deviceShadowGreenArraySum, deviceShadowBlueArraySum, imageWidth, imageHeight);
#endif
cudaDeviceSynchronize();
cudaEventRecord(event5StopEvent, 0);
cudaEventSynchronize(event5StopEvent);
cudaEventElapsedTime(&event5ElapsedTime, event5StartEvent, event5StopEvent);
printf("Total execution time (ms) %f for result integration \n",event5ElapsedTime);
  

    //cudaDeviceSynchronize(); 
  
  //-----------------------------------------------------------------------------------------------------//
  //-------------------------------- Data Transfer (GPU -> Host) ----------------------------------------//
  //-----------------------------------------------------------------------------------------------------//
	
  // YUV Data
  gpuErrchk(cudaMemcpy(hostOutputImageDataYUV, deviceYUVOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost));

  // Greyscale Data
  gpuErrchk(cudaMemcpy(hostOutputImageDataGreyScale, deviceGreyscaleOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost));

  // Color Invariant Data
  gpuErrchk(cudaMemcpy(hostOutputImageDataColorInvariant, deviceColorInvariantOutputImageData,
             imageWidth * imageHeight * sizeof(float) * imageChannels,
             cudaMemcpyDeviceToHost));

  // Mask Data
  gpuErrchk(cudaMemcpy(hostOutputImageDataGrayMask, deviceGreyMaskOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost));

  gpuErrchk(cudaMemcpy(hostOutputImageDataCbMask, deviceCBMaskOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost));

  // Smooth Image Data
  gpuErrchk(cudaMemcpy(hostOutputImageDataSmooth, deviceSmoothOutputImageData,
	           imageWidth * imageHeight * sizeof(float),
	           cudaMemcpyDeviceToHost));

  // Light-mask erosion data

  gpuErrchk(cudaMemcpy(hostOutputImageErodedLight, deviceErodedLight,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost));

  // Shadow-mask erosion data
  gpuErrchk(cudaMemcpy(hostOutputImageErodedShadow, deviceErodedShadow,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost));
			 
  // PROC 5
  // Shadow-mask erosion data
  gpuErrchk(cudaMemcpy(hostResultImageData, deviceResultImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost));

  //-----------------------------------------------------------------------------------------------------//
  //---------------------------------------- Image Export -----------------------------------------------//
  //-----------------------------------------------------------------------------------------------------//

  // wbImage_export expects to be exporting wbImages, so anything we want to export, copy its host buffer into a wbImage
  // YUV Image
  wbImage_t imgOutputYUV = wbImage_new(imageWidth, imageHeight, 3, hostOutputImageDataYUV);
	
  // Grey Scale Image
  wbImage_t imgOutputGreyscale = wbImage_new(imageWidth, imageHeight, 1, hostOutputImageDataGreyScale);
  wbImage_t imgOutputGreyMask = wbImage_new(imageWidth, imageHeight, 1, hostOutputImageDataGrayMask);
  wbImage_t imgOutputCbMask = wbImage_new(imageWidth, imageHeight, 1, hostOutputImageDataCbMask);
	
  // Color Invariant Image
  wbImage_t imgOutputColorInvariant = wbImage_new(imageWidth, imageHeight, imageChannels, hostOutputImageDataColorInvariant);
	
  // Smooth Image
  wbImage_t imgOutputSmooth = wbImage_new(imageWidth, imageHeight, 1, hostOutputImageDataSmooth);
  
  //Result Image
  wbImage_t resultImage = wbImage_new(imageWidth, imageHeight, imageChannels, hostResultImageData);

	
  
  wbImage_t imgOutputLightMaskErosion = wbImage_new(imageWidth, imageHeight, 1, hostOutputImageErodedLight);


  wbImage_t imgOutputShadowMaskErosion = wbImage_new(imageWidth, imageHeight, 1, hostOutputImageErodedShadow);



  // Output Image Path Strings
  std::string yuvOutputPath = baseOutputDir + "Proc1_OutputYUV.ppm";
  std::string greyOutputPath = baseOutputDir + "Proc1_OutputGrey.ppm";
  std::string ciOutputPath = baseOutputDir + "Proc1_OutputCI.ppm";
  
  
  std::string cbMaskOutputPath = baseOutputDir + "Proc2_OutputCbMask.ppm";
  std::string greyMaskOutputPath = baseOutputDir + "Proc2_OutputGreyMask.ppm";

  std::string smoothOutputPath = baseOutputDir + "Proc3_OutputSmooth.ppm";

  std::string erodedLightMaskOutputPath = baseOutputDir + "Proc4_OutputErodedLightMask.ppm";
  std::string erodedShadowMaskOutputPath = baseOutputDir + "Proc4_OutputErodedShadowMask.ppm";
  std::string resultImagePath = baseOutputDir + "Proc5_ResultImage.ppm";

 
  // Export Output Images
  wbExport(yuvOutputPath.c_str(), imgOutputYUV);
  wbExport(greyOutputPath.c_str(), imgOutputGreyscale);
  wbExport(ciOutputPath.c_str(), imgOutputColorInvariant);

  wbExport(cbMaskOutputPath.c_str(), imgOutputCbMask);
  wbExport(greyMaskOutputPath.c_str(), imgOutputGreyMask);
  
  wbExport(smoothOutputPath.c_str(), imgOutputSmooth);

  wbExport(erodedLightMaskOutputPath.c_str(), imgOutputLightMaskErosion);
  wbExport(erodedShadowMaskOutputPath.c_str(), imgOutputShadowMaskErosion);
  wbExport(resultImagePath.c_str(), resultImage);

  // Free Cuda Memory
  cudaFree(deviceInputImageData);
  cudaFree(deviceYUVOutputImageData);
  cudaFree(deviceGreyscaleOutputImageData);
  cudaFree(deviceColorInvariantOutputImageData);
  cudaFree(deviceGreyMaskOutputImageData);
  cudaFree(deviceCBMaskOutputImageData);
  cudaFree(deviceSmoothOutputImageData);
  cudaFree(deviceMaskData);
  cudaFree(deviceErodedLight);
  cudaFree(deviceErodedShadow);
  
  // Destroying cuda streams
#ifdef USE_STREAMING
  cudaStreamDestroy(colorspaceStream);
  cudaStreamDestroy(yuvStream);
  cudaStreamDestroy(grayscaleStream);
  cudaStreamDestroy(resultStream);
#endif

#if defined(USE_STREAM_EVENTS) && defined(USE_STREAMING)
  cudaEventDestroy(colorspaceCompleteEvent);
  cudaEventDestroy(yuvCompleteEvent);
  cudaEventDestroy(grayscaleCompleteEvent);
#endif

  // Proc 5:
  cudaFree(deviceShadowRedArray);
  cudaFree(deviceShadowGreenArray);
  cudaFree(deviceShadowBlueArray);
  cudaFree(deviceLightRedArray);
  cudaFree(deviceLightGreenArray);
  cudaFree(deviceLightBlueArray);
  
  cudaFree(deviceShadowRedArraySum);
  cudaFree(deviceShadowGreenArraySum);
  cudaFree(deviceShadowBlueArraySum);
  cudaFree(deviceLightRedArraySum);
  cudaFree(deviceLightGreenArraySum);
  cudaFree(deviceLightBlueArraySum);
  cudaFree(deviceErodedLightSum);
  cudaFree(deviceErodedShadowSum);

	// Delete the Images
  wbImage_delete(inputImage);
  wbImage_delete(imgOutputYUV);
  wbImage_delete(imgOutputGreyscale);
  wbImage_delete(imgOutputColorInvariant);
  wbImage_delete(imgOutputGreyMask);
  wbImage_delete(imgOutputCbMask);
  wbImage_delete(imgOutputSmooth);
  wbImage_delete(imgOutputLightMaskErosion);
  wbImage_delete(imgOutputShadowMaskErosion);
	
  // Print Success
  printf("<('.'<)\n");
  printf("(>'.')>\n");
  printf("\\(-_-)/\n");
  printf("/\\/\\(;;)/\\/\\\n");
  return 0;
}
