#ifndef SMOOTH_CPP
#define SMOOTH_CPP

//@@ DEFINE TILE SIZE AND INSERT DEVICE CODE HERE
#define IN_TILE_HEIGHT 16
#define OUT_TILE_HEIGHT 8
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

    // The size of each thread block matches the size of an input tile
    __shared__ float ds_In[IN_TILE_HEIGHT][IN_TILE_HEIGHT];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row_o = blockIdx.y * OUT_TILE_HEIGHT  + ty;
    int Col_o = blockIdx.x * OUT_TILE_HEIGHT  + tx;

    int Row_i = Row_o - MASK_RADIUS;
    int Col_i = Col_o - MASK_RADIUS;


    if(Row_i >= 0 && Row_i < imgHeight && Col_i >= 0 && Col_i < imgWidth)
        ds_In[ty][tx] = In[Row_i*imgWidth + Col_i];
    else
        ds_In[ty][tx] = 0;


    __syncthreads();


    if(tx < OUT_TILE_HEIGHT && ty < OUT_TILE_HEIGHT){
        float pixVal = 0;
        for(int y=0; y < MASK_WIDTH; y++)
            for(int x=0; x < MASK_WIDTH; x++)
                pixVal += K[y*MASK_WIDTH + x] * ds_In[y+ty][x+tx];

        if(Row_o < imgHeight && Col_o < imgWidth)
            Out[Row_o*imgWidth + Col_o] = (unsigned char)(pixVal);
    } 
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
	int Col = (blockIdx.x * blockDim.x + threadIdx.x);
    int Row = (blockIdx.y * blockDim.y + threadIdx.y);
    if(Col < imgWidth && Row < imgHeight) {
        float pixVal = 0;
        int start_col = Col - (MASK_WIDTH/2);
        int start_row = Row - (MASK_WIDTH/2);
        
        for (int y=0;y<MASK_WIDTH;++y) {
            for (int x=0;x<MASK_WIDTH;++x) {
                int curRow = start_row + y;
                int curCol = start_col + x;
                if (curRow > -1 && curRow < imgHeight && curCol > -1 && curCol < imgWidth){ // >= 0
                    pixVal += In[curRow*imgWidth + curCol] * K[y*MASK_WIDTH + x];
                }                    
            }
        }
        Out[Row*imgWidth+Col] = (unsigned char)(pixVal);
    }
}
#endif // SMOOTH_CPP
