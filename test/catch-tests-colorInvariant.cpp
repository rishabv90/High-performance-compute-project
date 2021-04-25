#include "catch.hpp"
#include "../src/colorInvariant.cpp"
#include <math.h>

SCENARIO("We can perform a basic color invariance calculation", "[color-invariance]") {
  GIVEN("A basic, statically declared array with an odd number of rows and columns") {
    float inputImage[5][5][3] = {
        {{0.9, 0.9, 0.9}, {0.1, 0.1, 0.4}, {0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}, {0.9, 0.8, 0.7}},
        {{0.9, 0.9, 0.9}, {0.1, 0.1, 0.4}, {0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}, {0.9, 0.8, 0.7}},
        {{0.9, 0.9, 0.9}, {0.1, 0.1, 0.4}, {0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}, {0.9, 0.8, 0.7}},
        {{0.9, 0.9, 0.9}, {0.1, 0.1, 0.4}, {0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}, {0.9, 0.8, 0.7}},
        {{0.9, 0.9, 0.9}, {0.1, 0.1, 0.4}, {0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}, {0.9, 0.8, 0.7}}
    };

    float expectedOutput[5][5][3] {
      {{0.785398, 0.785398, 0.785398}, {0.244979, 0.244979, 1.325818}, {0.540420, 0.674741, 0.896055}, {0.643501, 0.718830, 0.851966}, {0.844154, 0.726642, 0.661043}},
      {{0.785398, 0.785398, 0.785398}, {0.244979, 0.244979, 1.325818}, {0.540420, 0.674741, 0.896055}, {0.643501, 0.718830, 0.851966}, {0.844154, 0.726642, 0.661043}},
      {{0.785398, 0.785398, 0.785398}, {0.244979, 0.244979, 1.325818}, {0.540420, 0.674741, 0.896055}, {0.643501, 0.718830, 0.851966}, {0.844154, 0.726642, 0.661043}},
      {{0.785398, 0.785398, 0.785398}, {0.244979, 0.244979, 1.325818}, {0.540420, 0.674741, 0.896055}, {0.643501, 0.718830, 0.851966}, {0.844154, 0.726642, 0.661043}},
      {{0.785398, 0.785398, 0.785398}, {0.244979, 0.244979, 1.325818}, {0.540420, 0.674741, 0.896055}, {0.643501, 0.718830, 0.851966}, {0.844154, 0.726642, 0.661043}}
    };

    float *d_rgbImage, *d_ciImage;
    float *h_ciImage;

    cudaMalloc((void**)&d_rgbImage, 5*5*3*sizeof(float));
    cudaMalloc((void**)&d_ciImage, 5*5*3*sizeof(float));
    h_ciImage = (float*)malloc(5*5*3*sizeof(float));

    cudaMemcpy(d_rgbImage, inputImage, 5*5*3*sizeof(float), cudaMemcpyHostToDevice);

    WHEN("The kernel is invoked as a single threadblock") {
      dim3 dimGrid(1, 1);
      dim3 dimBlock(5, 5);

      colorInvariant<<<dimGrid, dimBlock>>>(d_ciImage, d_rgbImage, 5, 5);
      cudaDeviceSynchronize();

      THEN("The output should match MATLAB") {
        cudaMemcpy(h_ciImage, d_ciImage, 5*5*3*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 5; i++) {
          for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 3; k++) {
              CAPTURE(i);
              CAPTURE(j);
              CAPTURE(k);
              REQUIRE(fabs(expectedOutput[i][j][k] - h_ciImage[5*3*i + 3*j + k]) < 0.01f);
            }
          }
        }
      }
    }

    WHEN("The kernel is invoked as multiple threadblocks") {
      dim3 dimGrid(3, 3);
      dim3 dimBlock(2, 2);

      colorInvariant<<<dimGrid, dimBlock>>>(d_ciImage, d_rgbImage, 5, 5);
      cudaDeviceSynchronize();

      THEN("The output should match MATLAB") {
        cudaMemcpy(h_ciImage, d_ciImage, 5*5*3*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 5; i++) {
          for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 3; k++) {
              CAPTURE(i);
              CAPTURE(j);
              CAPTURE(k);
              REQUIRE(fabs(expectedOutput[i][j][k] - h_ciImage[5*3*i + 3*j + k]) < 0.01f);
            }
          }
        }
      }
    }
  }
}
