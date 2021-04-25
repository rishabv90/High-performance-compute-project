#include "catch.hpp"
#include "../src/Greyscale.cpp"

SCENARIO("We can perform a basic rgb to greyscale calculation", "[greyscale]") {
  GIVEN("A basic, statically declared array with an odd number of rows and columns") {
    float inputImage[5][5][3] = {
        {{0.9, 0.9, 0.9}, {0.1, 0.1, 0.4}, {0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}, {0.9, 0.8, 0.7}},
        {{0.9, 0.9, 0.9}, {0.1, 0.1, 0.4}, {0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}, {0.9, 0.8, 0.7}},
        {{0.9, 0.9, 0.9}, {0.1, 0.1, 0.4}, {0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}, {0.9, 0.8, 0.7}},
        {{0.9, 0.9, 0.9}, {0.1, 0.1, 0.4}, {0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}, {0.9, 0.8, 0.7}},
        {{0.9, 0.9, 0.9}, {0.1, 0.1, 0.4}, {0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}, {0.9, 0.8, 0.7}}
    };

    float expectedOutput[5][5] = {
        {0.900000, 0.134206, 0.381508, 0.681508, 0.818492},
        {0.900000, 0.134206, 0.381508, 0.681508, 0.818492},
        {0.900000, 0.134206, 0.381508, 0.681508, 0.818492},
        {0.900000, 0.134206, 0.381508, 0.681508, 0.818492},
        {0.900000, 0.134206, 0.381508, 0.681508, 0.818492}
    };

    float *d_rgbImage, *d_greyImage;
    float *h_greyImage;

    cudaMalloc((void**)&d_rgbImage, 5*5*3*sizeof(float));
    cudaMalloc((void**)&d_greyImage, 5*5*sizeof(float));
    h_greyImage = (float*)malloc(5*5*sizeof(float));

    cudaMemcpy(d_rgbImage, inputImage, 5*5*3*sizeof(float), cudaMemcpyHostToDevice);

    WHEN("The kernel is invoked as a single threadblock") {
      dim3 dimGrid(1, 1);
      dim3 dimBlock(5, 5);

      colorConvertGrey<<<dimGrid, dimBlock>>>(d_greyImage, d_rgbImage, 5, 5);
      cudaDeviceSynchronize();

      THEN("The output should match MATLAB") {
        cudaMemcpy(h_greyImage, d_greyImage, 5*5*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 5; i++) {
          for (int j = 0; j < 5; j++) {
            CAPTURE(i);
            CAPTURE(j);
            REQUIRE(fabs(expectedOutput[i][j] - h_greyImage[5*i + j]) < 0.01f);
          }
        }
      }
    }

    WHEN("The kernel is invoked as multiple threadblocks") {
      dim3 dimGrid(3, 3);
      dim3 dimBlock(2, 2);

      colorConvertGrey<<<dimGrid, dimBlock>>>(d_greyImage, d_rgbImage, 5, 5);
      cudaDeviceSynchronize();

      THEN("The output should match MATLAB") {
        cudaMemcpy(h_greyImage, d_greyImage, 5*5*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 5; i++) {
          for (int j = 0; j < 5; j++) {
            CAPTURE(i);
            CAPTURE(j);
            REQUIRE(fabs(expectedOutput[i][j] - h_greyImage[5*i + j]) < 0.01f);
          }
        }
      }
    }
  }
}
