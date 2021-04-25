#ifndef SMOOTH_CPP
#define SMOOTH_CPP

//@@ DEFINE TILE SIZE AND INSERT DEVICE CODE HERE
#define TILE_HEIGHT 16
#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2
#define AREA ( MASK_WIDTH + TILE_HEIGHT -1 )
#define clamp(x) (min(max((x), 0.0), 1.0))

//-----------------------------------------------------------------------//
//
//	Smooth Kernel
//		Shared Memory, Tiled Version
//
//-----------------------------------------------------------------------//
//		In = the input image matrix
//		Out = the resulting convolution image matrix
//		K = constant memory kernel
//		imgWidth = ImageIn's & ImageOut's width
//		imgHeight = ImageIn's & ImageOut's height
__global__ void smooth_kernel(float *In, float *Out, const float* __restrict__ K, int channels, int imgWidth, int imgHeight) {
	// Using shared memory
	// Want more data than the mask area.
	// Need Area x Area of Shared Memory ( Area = mask + tile -1 )
	
}

//-----------------------------------------------------------------------//
//
//	Smooth Kernel Row
//		1D Shared Memory Row Kernel
//
//-----------------------------------------------------------------------//
//		In = the input image matrix
//		Out = the resulting convolution image matrix
//		K = constant memory kernel
//		imgWidth = ImageIn's & ImageOut's width
//		imgHeight = ImageIn's & ImageOut's height
__global__ void smooth_kernel_row(float *In, float *Out, const float* __restrict__ K, int channels, int imgWidth, int imgHeight) {

    // Shared Memory:
    // 1D Row Shared Memory must be of size tile x area.
    // We still use area because we still need the padding in all the columns, while the 
    // rows are confined to the tile itself.
	
}
//-----------------------------------------------------------------------//
//
//	Smooth Kernel Col
//		1D Shared Memory Column Kernel
//
//-----------------------------------------------------------------------//
//		In = the input image matrix
//		Out = the resulting convolution image matrix
//		K = constant memory kernel
//		imgWidth = ImageIn's & ImageOut's width
//		imgHeight = ImageIn's & ImageOut's height
// The column kernel works the same way as the row kernel except that instead of dealing with the 
__global__ void smooth_kernel_col(float *In, float *Out, const float* __restrict__ K, int channels, int imgWidth, int imgHeight) {
	// shared memory:
	
}

//-----------------------------------------------------------------------//
//
//	Smooth Kernel Global
//		Global Memory Convolution
//
//-----------------------------------------------------------------------//
//		In = the input image matrix
//		Out = the resulting convolution image matrix
//		K = constant memory kernel
//		imgWidth = ImageIn's & ImageOut's width
//		imgHeight = ImageIn's & ImageOut's height
__global__ void smooth_kernel_global(float *In, float *Out, const float* __restrict__ K, int channels, int imgWidth, int imgHeight) {
	// gLoc = our global Location.
	
}
#endif // SMOOTH_CPP
